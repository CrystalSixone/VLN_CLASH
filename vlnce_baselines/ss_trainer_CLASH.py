import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines
import copy

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

import tqdm
from gym import Space
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens, read_eval_stats_file, update_eval_stats_file
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss, Timer
from vlnce_baselines.causal.data_utils import LoadZdict

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from vlnce_baselines.models.encoders.bert_tokens import MyRobertaTokenizer, MyBertTokenizer

import logging
from vlnce_baselines.common.log import HabitatLogger
from vlnce_baselines.common.utils import extract_best_eval_results, display_sample

from vlnce_baselines.nav_model.one_stage_prompt_manager import OneStagePromptManager
from vlnce_baselines.nav_model.chat_model import ChatModel
from vlnce_baselines.common.utils import get_camera_orientations

@baseline_registry.register_trainer(name="SS-CLASH")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) 
        self.instr_bert_model_name = config.MODEL.INSTRUCTION_ENCODER.model_name if hasattr(config.MODEL.INSTRUCTION_ENCODER, 'model_name') else 'bert'
        self.bert_tokenizer = MyBertTokenizer(model_name=self.instr_bert_model_name)

        # Init the log to save the information into the file
        log_dir = self.config.LOG_DIR
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Init the file_logger
        if self.config.run_type in ['train', 'collect_dataset']:
            train_logger_filename = os.path.join(log_dir, "train.log")
            self.logger = HabitatLogger(
                name="habitat", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=train_logger_filename
            )
            self.logger.info(f"Start! Good Luck!!!")
        
        elif self.config.run_type in ['eval']:
            log_dir = self.config.results_dir
            eval_logger_filename = os.path.join(log_dir, f"{self.config.eval.split}_eval.log")
            self.logger = HabitatLogger(
                name="habitat", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=eval_logger_filename
            )
            # Read the previous best eval results
            if os.path.exists(eval_logger_filename):
                with open(eval_logger_filename, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        last_line = lines[-1]
            if self.config.eval.split == 'val_seen_unseen':
                self.val_seen_eval_results = extract_best_eval_results(log_file=eval_logger_filename, split='val_seen')
                self.val_unseen_eval_results = extract_best_eval_results(log_file=eval_logger_filename, split='val_unseen')
            else:
                self.eval_results = extract_best_eval_results(log_file=eval_logger_filename, split=self.config.eval.split)
            self.logger.info(f"Start! Good Luck!!!")
        
        elif self.config.run_type == 'inference':
            log_dir = self.config.results_dir
            eval_logger_filename = os.path.join(log_dir, f"{self.config.inference.split}_eval.log")
            
            self.logger = HabitatLogger(
                name="habitat", level=logging.INFO, format_str="%(asctime)-15s %(message)s",
                filename=eval_logger_filename
            )
            self.eval_results = extract_best_eval_results(log_file=eval_logger_filename, split=self.config.inference.split)
        
        # Init VLM model
        if self.config.LLM.use:          
            assert self.config.run_type in ['eval', 'inference'] # Only use for eval now
            if self.config.run_type == 'inference':
                self.llm_traj_log_dir = os.path.join(self.config.LOG_DIR, 'llm_traj_log', self.config.inference.split)
            else:
                self.llm_traj_log_dir = os.path.join(self.config.LOG_DIR, 'llm_traj_log', self.config.eval.split)
            os.makedirs(self.llm_traj_log_dir, exist_ok=True)
            
            self.chat_model = ChatModel(self.config, self.config.LLM.llm_name, type = self.config.LLM.llm_type, port = self.config.LLM.llm_port)
            self.prompt_manager = OneStagePromptManager(self.config, model=self.chat_model, logger=self.logger, log_dir=self.llm_traj_log_dir)
        
        # Init pointgoal
        if self.config.PointGoal.use:
            self.logger.info(f"*Init pointgoal")
            self.pointgoal_config = self.config.pointgoal_config
            self.pointgoal_stats = {
                'reach_goal': 0,
                'total_steps': 0,
                'avg_reach_goal': 0,
            }
            
    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir(self.config.checkpoint_folder)
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.eval.save_results:
                self._make_results_dir(self.config.results_dir)

    def save_checkpoint(self, iteration: int, only_save_learnable_weights: bool = False):
        if only_save_learnable_weights:
            trainable_params = {name: param for name, param in self.policy.named_parameters() if param.requires_grad}
            state_dict = trainable_params
        else:
            state_dict = self.policy.state_dict()
        
        torch.save(
            obj={
                "state_dict": state_dict,
                # "config": self.config,
                # "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.task_config.dataset.split
        self.config.defrost()
        self.config.task_config.task.ndtw.split = self.split
        self.config.task_config.task.sdtw.split = self.split
        self.config.task_config.environment.iterator_options.max_scene_repeat_steps = -1
        self.config.simulator_gpu_ids = self.config.simulator_gpu_ids[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.rl.policy.obs_transforms.resizer_per_sensor.sizes
        crop_config = self.config.rl.policy.obs_transforms.center_cropper_per_sensor.sensor_crops
        task_config = self.config.task_config
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["rgb", "depth"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.simulator, f"{sensor_type}_sensor")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.orientation = camera_orientations[action]
                camera_config.uuid = camera_template.lower()
                setattr(task_config.simulator, camera_template, camera_config)
                task_config.simulator.agent_0.sensors.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.rl.policy.obs_transforms.resizer_per_sensor.sizes = resize_config
        self.config.rl.policy.obs_transforms.center_cropper_per_sensor.sensor_crops = crop_config
        self.config.task_config = task_config
        self.config.sensors = task_config.simulator.agent_0.sensors
        if self.config.video_option:
            self.config.task_config.task.measurements.append("TOP_DOWN_MAP_VLNCE")
            self.config.task_config.task.measurements.append("top_down_map_vlnce")
            self.config.task_config.task.measurements.append("DISTANCE_TO_GOAL")
            self.config.task_config.task.measurements.append("SUCCESS")
            self.config.task_config.task.measurements.append("SPL")
            os.makedirs(self.config.video_dir, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["rgb"]:
                sensor = getattr(self.config.task_config.simulator, f"{sensor_type}_sensor")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.width = H
                    camera_config.height = H
                    camera_config.orientation = orient
                    camera_config.uuid = camera_template.lower()
                    camera_config.hfoV = 90
                    sensor_uuids.append(camera_config.uuid)
                    setattr(self.config.task_config.simulator, camera_template, camera_config)
                    self.config.task_config.simulator.agent_0.sensors.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.gpu_numbers
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.il.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.torch_gpu_ids[self.local_rank]
            self.config.defrost()
            self.config.torch_gpu_id = self.config.torch_gpu_ids[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.task_config.seed = self.config.task_config.seed + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')

        # add aug envs
        if self.config.IL.add_dataset_aug:
            aug_config = deepcopy(self.config)
            aug_config.defrost()
            if self.config.IL.aug_dataset_type == 'scalevln':
                aug_config.task_config.dataset.data_path = self.config.task_config.dataset.scalevln_data_aug_path
            elif self.config.IL.aug_dataset_type == 'prevalent':
                aug_config.task_config.dataset.data_path = self.config.task_config.dataset.prevalent_data_aug_path
            aug_config.freeze()
            self.aug_envs = construct_envs(
                aug_config, 
                get_env_class(self.config.ENV_NAME),
                auto_reset_done=False
            )
            aug_env_num = self.aug_envs.num_envs
            aug_dataset_len = sum(self.aug_envs.number_of_episodes)
            logger.info(f'AUG ENV NUM: {aug_env_num}, DATASET LEN: {aug_dataset_len}')

            self.mp3d_env = self.envs

        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space
    
    def _setup_actor_critic_agent(self, ppo_cfg, observation_space, action_space) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        import torch.nn as nn
        from habitat_baselines.rl.ddppo.algo import DDPPO
        from habitat_baselines.rl.ppo import PPO
        from gym import spaces
        from collections import OrderedDict
        import copy
        from gym.spaces import Discrete
        
        policy = baseline_registry.get_policy(self.pointgoal_config.RL.POLICY.name)

        # depth_only
        allow_names = ['depth', 'pointgoal_with_gps_compass']
        ac_observation_space = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k in allow_names
                )
            )
        )
        # ac_observation_space.spaces['pointgoal_with_gps_compass'] = ac_observation_space.spaces['depth']
        
        
        # action_space change
        ac_action_space = copy.deepcopy(action_space)
        ac_action_space.actions_select = Discrete(4)
        ac_action_space.actions_select.n = 4
        if 'HIGHTOLOWEVAL' in ac_action_space.spaces:
            ac_action_space.spaces.pop('HIGHTOLOWEVAL')
        else:
            ac_action_space.spaces.pop('HIGHTOLOW')
        
        # Init actor_critic
        actor_critic = policy.from_config(
            self.pointgoal_config, ac_observation_space, ac_action_space
        )
        self.pointgoal_obs_space = ac_observation_space
        actor_critic.to(self.device)

        if (
            self.pointgoal_config.RL.DDPPO.pretrained_encoder
            or self.pointgoal_config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.pointgoal_config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.pointgoal_config.RL.DDPPO.pretrained:
            actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.pointgoal_config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.pointgoal_config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.pointgoal_config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(actor_critic.critic.fc.weight)
            nn.init.constant_(actor_critic.critic.fc.bias, 0)

        # pointgoal_agent = (DDPPO if self._is_distributed else PPO)
        
        pointgoal_agent = PPO(
            actor_critic=actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )
        
        # Load checkpoints
        checkpoint_path = self.pointgoal_config.EVAL_CKPT_PATH_DIR
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        incompatible_keys = pointgoal_agent.load_state_dict(ckpt_dict["state_dict"], strict=True)
        if incompatible_keys.missing_keys:
            self.logger.info(f"Missing keys: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            self.logger.info(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
        
        return pointgoal_agent

    def _initialize_policy(
        self,
        config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net_modified import BinaryDistPredictor_TRM_modified
        self.waypoint_predictor = BinaryDistPredictor_TRM_modified(args=self.config.MODEL.WAYPOINT_PRED) 
        cwp_fn = self.config.MODEL.WAYPOINT_PRED.ckpt_path
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)
        
        try:
            self.policy.to(self.device)
        except Exception as e:
            self.policy.net.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers # self.policy.net -> vlnce_baselines.models.etp

        if self.config.gpu_numbers > 1:
            print('Using', self.config.gpu_numbers,'GPU!')
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, broadcast_buffers=False)
        try:    
            self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.IL.lr)
        except Exception as e:
            self.optimizer = torch.optim.AdamW(self.policy.net.parameters(), lr=self.config.IL.lr)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
                self.logger.info(f"Loading checkpoint from {ckpt_path}")  
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            if "iteration" in ckpt_dict.keys():
                start_iter = ckpt_dict["iteration"]
            else:
                start_iter = 0

            if 'vln_bert' in ckpt_dict:
                # load the VLN model from the descrete environments
                ckpt_dict = ckpt_dict['vln_bert']
                incompatible_keys = self.policy.net.load_state_dict(ckpt_dict["state_dict"], strict=False)
                if incompatible_keys.missing_keys:
                    logger.info(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.info(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
                    
            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.gpu_numbers == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                # Print mismatched keys when loading state dict
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                if incompatible_keys.missing_keys:
                    logger.info(f"Missing keys: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.info(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
            if config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
		
        try:
            params = sum(param.numel() for param in self.policy.parameters())
            params_t = sum(
                p.numel() for p in self.policy.parameters() if p.requires_grad
            )
        except Exception as e:
            params = sum(param.numel() for param in self.policy.net.parameters())
            params_t = sum(
                p.numel() for p in self.policy.net.parameters() if p.requires_grad
            )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")
        
        #### load causal data ####
        # backdoor inference
        if self.config.model.causal.do_back_img or self.config.model.causal.do_back_txt:
            if hasattr(self.config.model.causal, 'backdoor_txt_zdict_file') and len(self.config.model.causal.backdoor_txt_zdict_file) > 0:
                txt_zdict_file = self.config.model.causal.backdoor_txt_zdict_file
            else:
                txt_zdict_file = self.config.model.causal.txt_zdict_file
            self.zdictReader = LoadZdict(
                img_zdict_file=self.config.model.causal.img_zdict_file,
                obj_zdict_file=None,
                txt_zdict_file=txt_zdict_file
            )
            self.z_dicts = defaultdict(lambda:None)
            if self.config.model.causal.do_back_img:
                self.z_dicts['img_zdict'] = self.zdictReader.load_img_tensor()
            if self.config.model.causal.do_back_txt:
                self.z_dicts['instr_zdict'] = self.zdictReader.load_instr_tensor()
        
        # frontdoor inference
        if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
            from vlnce_baselines.causal.data_utils import KMeansPicker
            self.front_feat_loader = KMeansPicker(self.config.model.causal.front_feat_file, n_clusters=self.config.model.causal.front_n_clusters)
            
            self.z_front_log_dir = os.path.join(self.config.checkpoint_folder, 'z_front_log')
            os.makedirs(self.z_front_log_dir, exist_ok=True)
        
        # pointgoal 
        if self.config.pointgoal.use:
            ppo_cfg = self.pointgoal_config.RL.PPO
            self.pointgoal_agent = self._setup_actor_critic_agent(ppo_cfg, observation_space, action_space)
            self.pointgoal_agent.to(self.device)
            self.pointgoal_agent.eval()
            self.pointgoal_actor_critic = self.pointgoal_agent.actor_critic
            self.logger.info(f"Successfully loaded pointgoal agent from {self.pointgoal_config.EVAL_CKPT_PATH_DIR}")
                
        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs, noise=None):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool) 
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            if noise is not None:
                rgb_fts.append(obs['cand_rgb'][i] * noise)
                dep_fts.append(obs['cand_depth'][i]) # no noise for depth
            else:
                rgb_fts.append(obs['cand_rgb'][i])
                dep_fts.append(obs['cand_depth'][i])
            if self.config.MODEL.CLIP_ENCODER.model == "ViT-B/32":
                loc_fts.append(obs['cand_angle_fts'][i])
            else:
                view_box_fts = np.array([[1, 1, 1]]*len(obs['cand_angle_fts'][i])).astype(np.float32)
                view_loc_fts = np.concatenate([np.array(obs['cand_angle_fts'][i]), view_box_fts], 1)
                loc_fts.append(torch.from_numpy(view_loc_fts).float())
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            if noise is not None:
                rgb_fts.append(obs['pano_rgb'][i][~cand_idxes] * noise)
                dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            else:
                rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
                dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            if self.config.MODEL.CLIP_ENCODER.model == "ViT-B/32":
                loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            else:
                view_box_fts = np.array([[1, 1, 1]] * len(obs['pano_angle_fts'][~cand_idxes])).astype(np.float32)
                view_loc_fts = np.concatenate([np.array(obs['pano_angle_fts'][~cand_idxes]), view_box_fts], 1)
                loc_fts.append(torch.from_numpy(view_loc_fts).float())
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        # env drop
        already_dropout = False if noise is None else True
        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
            'already_dropout': already_dropout,
        }
    
    def _nav_vp_variable(self, pano_embeds, cur_vp, cur_pos, cur_ori, cand_vp, cand_pos, cand_embeds, view_lens,nav_types):
        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        batch_size = len(self.gmaps)
        batch_cand_vps = []
        for i, gmap in enumerate(self.gmaps):
            current_cand_vps = gmap.current_cand_vps
            batch_cand_vps.append(current_cand_vps)
            if len(current_cand_vps) > 0:
                cur_cand_pos_fts = gmap.get_pos_fts(
                    cur_vp[i], cur_pos[i], cur_ori[i],
                    current_cand_vps
                )
            else:
                cur_cand_pos_fts = np.zeros((0, 7), dtype=np.float32)
            cur_start_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i],
                [gmap.start_vp]
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors_wgrad(batch_vp_pos_fts).to(self.device)

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)
        vp_masks = gen_seq_masks(view_lens+1)
        vp_cand_vpids = [[None]+x for x in batch_cand_vps]

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': vp_masks,
            'vp_nav_masks': vp_nav_masks,
            'vp_cand_vpids': vp_cand_vpids,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                        [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, observations, envs_to_pause, instruction_text=None):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

            observations = [obs for i, obs in enumerate(observations) if i not in envs_to_pause]

            if instruction_text is not None:
                instruction_text = [text for i, text in enumerate(instruction_text) if i not in envs_to_pause]

        return envs, batch, observations, instruction_text

    def train(self):
        self._set_config()

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        
        if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
            self.z_front_dict = self.front_feat_loader.random_pick_front_features(self.z_front_log_dir, iter=0, save_file=True)
        
        if self.local_rank < 1:
            self.logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval
            
            if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
                if idx % 3000 == 0:
                    self.z_front_dict = self.front_feat_loader.random_pick_front_features(self.z_front_log_dir, iter=0, save_file=True)

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                self.logger.info(loss_str)
                self.save_checkpoint(cur_iter, only_save_learnable_weights=True)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.loss = torch.tensor(0.0).to(self.device)

            if self.config.il.train_mode == 'sample':
                with autocast():
                    if self.config.IL.add_dataset_aug:
                        self.envs = self.mp3d_env
                    self.rollout('train', ml_weight, sample_ratio, self.envs)

                    self.scaler.scale(self.loss).backward() # self.loss.backward()
                    self.scaler.step(self.optimizer)        # self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scaler.update()
                    self.loss = torch.tensor(0.0).to(self.device)
                
                    if self.config.IL.add_dataset_aug:
                        self.envs = self.aug_envs
                        for aug_idx in range(self.config.IL.aug_times):
                            self.rollout('train', ml_weight, sample_ratio, self.envs)
                            self.scaler.scale(self.loss).backward() # self.loss.backward()
                            self.scaler.step(self.optimizer)        # self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scaler.update()
                            self.loss = torch.tensor(0.0).to(self.device)
            
            elif self.config.il.train_mode == 'dagger':
                    if self.config.IL.add_dataset_aug:
                        self.envs = self.mp3d_env
                    self.rollout('train', ml_weight=0.25, sample_ratio=1.0, envs=self.envs, add_env_dropout=False) # teacher forcing
                    self.rollout('train', ml_weight=1.0, sample_ratio=sample_ratio, envs=self.envs, add_env_dropout=True) # sample

                    if not self.config.IL.add_dataset_aug or (not self.config.IL.accumulate_grad and self.config.IL.add_dataset_aug):
                        self.loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.loss = torch.tensor(0.0).to(self.device)
                
                    if self.config.IL.add_dataset_aug:
                        self.envs = self.aug_envs
                        for aug_idx in range(self.config.IL.aug_times):
                            self.rollout('train', ml_weight=0.25, sample_ratio=1.0, envs=self.envs, add_env_dropout=False) # teacher forcing
                            self.rollout('train', ml_weight=1.0, sample_ratio=sample_ratio, envs=self.envs, add_env_dropout=True) # sample
                            
                            self.loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.loss = torch.tensor(0.0).to(self.device)
                
            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)
    
    def set_video_config(self):
        self.config.defrost()
        if self.config.video_option or self.config.LLM.use:
            self.config.task_config.task.measurements.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.task_config.task.measurements.append("top_down_map_vlnce")
            self.config.task_config.task.measurements.append("DISTANCE_TO_GOAL")
            self.config.task_config.task.measurements.append("SUCCESS")
            self.config.task_config.task.measurements.append("SPL")
            os.makedirs(self.config.video_dir, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["rgb"]:
                sensor = getattr(self.config.task_config.simulator, f"{sensor_type}_sensor")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.width = H
                    camera_config.height = H
                    camera_config.orientation = orient
                    camera_config.uuid = camera_template.lower()
                    camera_config.hfov = 90
                    sensor_uuids.append(camera_config.uuid)
                    setattr(self.config.task_config.simulator, camera_template, camera_config)
                    self.config.task_config.simulator.agent_0.sensors.append(camera_template)
        self.config.freeze()
    
    def _get_spaces(
        self, config, envs=None
    ):
        """Gets both the observation space and action space.

        Args:
            config (Config): The config specifies the observation transforms.
            envs (Any, optional): An existing Environment. If None, an
                environment is created using the config.

        Returns:
            observation space, action space
        """
        if envs is not None:
            observation_space = envs.observation_spaces[0]
            action_space = envs.action_spaces[0]

        else:
            env = get_env_class(self.config.env_name)(config=config)
            observation_space = env.observation_space
            action_space = env.action_space

        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        return observation_space, action_space

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            self.logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.task_config.environment.iterator_options.shuffle = False
        self.config.task_config.environment.iterator_options.max_scene_repeat_steps = -1
        self.config.il.ckpt_to_load = checkpoint_path
        self.config.freeze()
        if self.config.video_option or self.config.LLM.use:
            self.set_video_config()

        if self.config.eval.save_results:
            if self.config.task_config.dataset.split == 'val_seen_unseen':
                fname = os.path.join(
                    self.config.results_dir,
                    f"stats_ckpt_{checkpoint_index}_val_seen.json",
                )
            else:
                fname = os.path.join(
                    self.config.results_dir,
                    f"stats_ckpt_{checkpoint_index}_{self.config.task_config.dataset.split}.json",
                )
            if os.path.exists(fname) and not os.path.isfile(self.config.eval_ckpt_path_dir):
                print("skipping -- evaluation exists.")
                if self.config.eval.split == 'val_seen_unseen':
                    return 0, 0, 0, 0
                else:
                    return 0, 0
        
        observation_space, action_space = self._get_spaces(self.config)
        
        '''Load model'''
        checkpoint_index = self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()
        
        '''Load eval dataset'''
        if hasattr(self.config.task_config.dataset, 'EVAL_EP_IDS'):
            eval_ep_ids_path = self.config.task_config.dataset.EVAL_EP_IDS.format(split=self.config.task_config.dataset.split)
            with open(eval_ep_ids_path, 'r') as f:
                self.traj = [str(x) for x in json.load(f)]
            
        save_rank = self.local_rank 
        
        # Load already evaluated episodes
        self.stat_eps = {}
        self.config.defrost()
        self.config.eval.stats_file = os.path.join(
                    self.config.results_dir,
                    f"stats_ep_ckpt_{checkpoint_index}_{self.config.task_config.dataset.split}_r{save_rank}_w{self.world_size}.json",
                )
        self.config.freeze()
        self.loaded_stat_eps = read_eval_stats_file(self.config.eval.stats_file)
        if len(self.loaded_stat_eps) > 0:
            self.logger.info(f"Successfully loaded {len(self.loaded_stat_eps)} episodes that have been evaluated from {self.config.eval.stats_file}")
        
        if self.config.eval.split == 'val_seen_unseen':
            self.val_seen_stat_eps = {}
            self.val_unseen_stat_eps = {}
            
            self.config.defrost()
            self.config.eval.val_seen_stats_file = os.path.join(
                self.config.results_dir,
                f"stats_ep_ckpt_{checkpoint_index}_val_seen_r{save_rank}_w{self.world_size}.json",
            )
            self.config.eval.val_unseen_stats_file = os.path.join(
                self.config.results_dir,
                f"stats_ep_ckpt_{checkpoint_index}_val_unseen_r{save_rank}_w{self.world_size}.json",
            )
            self.config.freeze()
            self.val_seen_stat_eps = read_eval_stats_file(self.config.eval.val_seen_stats_file)
            self.val_unseen_stat_eps = read_eval_stats_file(self.config.eval.val_unseen_stats_file)
            
            self.loaded_stat_eps = {**self.val_seen_stat_eps, **self.val_unseen_stat_eps}
            
        # Skip already evaluated episodes
        if hasattr(self.config.task_config.dataset, 'EVAL_EP_IDS'):
            self.traj = [x for x in self.traj if int(x) not in self.loaded_stat_eps.keys()]
        
        # If debug, override the episodes allowed
        if hasattr(self.config, 'dataset') and self.config.dataset.episodes_allowed is not None: # for debug!
            self.traj = [str(x) for x in self.config.dataset.episodes_allowed]
        
        self.config.defrost()
        self.config.task_config.dataset.do_shuffle = False
        self.config.freeze()
        
        if hasattr(self.config.task_config.dataset, 'UNSEEN_LENGTH'):
            self.unseen_length = self.config.task_config.dataset.UNSEEN_LENGTH
        
        '''Construct eval envs'''
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.env_name),
            episodes_allowed=self.traj[::5] if self.config.eval.fast_eval else self.traj,
            auto_reset_done=False, 
            do_shuffle=False,
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )

        if self.config.eval.episode_count == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.eval.episode_count, sum(self.envs.number_of_episodes))

            
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None
        
        # load front file
        if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
            self.z_front_dict = self.front_feat_loader.read_tim_tsv(self.config.model.causal.front_Kmeans_file, return_dict=True)
            self.logger.info(f"Successfully load front features from {self.config.model.causal.front_Kmeans_file}")

        '''Start eval loop'''
        eval_num = 0
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
            self.loaded_stat_eps = {**self.loaded_stat_eps, **self.stat_eps}
                
            eval_num += self.config.NUM_ENVIRONMENTS
            if eval_num % self.config.NUM_ENVIRONMENTS == 0:
                # output spl and sr
                if self.config.eval.split in ['val_seen_unseen', 'train']:
                    # Val seen & unseen
                    update_eval_stats_file(self.config.eval.stats_file, self.loaded_stat_eps)
                    
                    if self.config.eval.split == 'val_seen_unseen':
                        # Val unseen
                        unseen_aggregated_states = {}
                        unseen_num_episodes = len(self.val_unseen_stat_eps)
                        if unseen_num_episodes > 0: 
                            for stat_key in next(iter(self.val_unseen_stat_eps.values())).keys():
                                for v in self.val_unseen_stat_eps.values():
                                    if stat_key == 'distances':
                                        continue
                                    unseen_aggregated_states[stat_key] = sum(v[stat_key] for v in self.val_unseen_stat_eps.values()) / unseen_num_episodes

                            if self.config.LLM.use:
                                self.logger.info('Val Unseen SPL: %.4f, SR: %.4f' % (unseen_aggregated_states['spl'], unseen_aggregated_states['success']))
                        
                        update_eval_stats_file(self.config.eval.val_unseen_stats_file, self.loaded_stat_eps)
                        
                        # Val seen
                        seen_aggregated_states = {}
                        seen_num_episodes = len(self.val_seen_stat_eps)
                        if seen_num_episodes > 0:
                            for stat_key in next(iter(self.val_seen_stat_eps.values())).keys():
                                for v in self.val_seen_stat_eps.values():
                                    if stat_key == 'distances':
                                        continue
                                    seen_aggregated_states[stat_key] = sum(v[stat_key] for v in self.val_seen_stat_eps.values()) / seen_num_episodes
                            if self.config.LLM.use:
                                self.logger.info('Val Seen SPL: %.4f, SR: %.4f' % (seen_aggregated_states['spl'], seen_aggregated_states['success']))
                        
                        update_eval_stats_file(self.config.eval.val_seen_stats_file, self.val_seen_stat_eps)
                else:
                    aggregated_states = {}
                    num_episodes = len(self.stat_eps)
                    for stat_key in next(iter(self.stat_eps.values())).keys():
                        for v in self.stat_eps.values():
                            if stat_key == 'distances':
                                continue
                            aggregated_states[stat_key] = sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
                    print('SPL: %.4f, SR: %.4f' % (aggregated_states['spl'], aggregated_states['success']))
                        
        self.envs.close()
        
        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.loaded_stat_eps)
        for stat_key in next(iter(self.loaded_stat_eps.values())).keys():
            for v in self.loaded_stat_eps.values():
                if stat_key == 'distances':
                    continue
                aggregated_states[stat_key] = sum(v[stat_key] for v in self.loaded_stat_eps.values()) / num_episodes
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.config.eval.split == 'val_seen_unseen':
            val_seen_total = len(self.val_seen_stat_eps)
            val_unseen_total = len(self.val_unseen_stat_eps)
        
        update_eval_stats_file(self.config.eval.stats_file, self.loaded_stat_eps)
        if self.config.eval.split == 'val_seen_unseen':
            update_eval_stats_file(self.config.eval.val_seen_stats_file, self.val_seen_stat_eps)
            update_eval_stats_file(self.config.eval.val_unseen_stats_file, self.val_unseen_stat_eps)
            
            unseen_aggregated_states = {}
            unseen_num_episodes = len(self.val_unseen_stat_eps)
            if unseen_num_episodes > 0:
                for stat_key in next(iter(self.val_unseen_stat_eps.values())).keys():
                    if stat_key == 'distances':
                        continue
                    unseen_aggregated_states[stat_key] = (
                        sum(v[stat_key] for v in self.val_unseen_stat_eps.values()) / unseen_num_episodes
                    )
            
            seen_aggregated_states = {}
            seen_num_episodes = len(self.val_seen_stat_eps)
            if seen_num_episodes > 0:
                for stat_key in next(iter(self.val_seen_stat_eps.values())).keys():
                    if stat_key == 'distances':
                        continue
                    seen_aggregated_states[stat_key] = (
                        sum(v[stat_key] for v in self.val_seen_stat_eps.values()) / seen_num_episodes
                    )

        if self.world_size > 1:
            if self.config.eval.split == 'val_seen_unseen':
                if unseen_num_episodes > 0:
                    self.logger.info(f"rank {save_rank}'s val unseen {unseen_num_episodes}-episode results: {unseen_aggregated_states}")
                    for k,v in unseen_aggregated_states.items():
                        v = torch.tensor(v*unseen_num_episodes).cuda()
                        cat_v = gather_list_and_concat(v,self.world_size)
                        v = (sum(cat_v)/val_unseen_total).item()
                        unseen_aggregated_states[k] = v
                
                if seen_num_episodes > 0:
                    self.logger.info(f"rank {save_rank}'s val seen {seen_num_episodes}-episode results: {seen_aggregated_states}")
                    for k,v in seen_aggregated_states.items():
                        v = torch.tensor(v*seen_num_episodes).cuda()
                        cat_v = gather_list_and_concat(v,self.world_size)
                        v = (sum(cat_v)/val_seen_total).item()
                        seen_aggregated_states[k] = v
            else:
                self.logger.info(f"rank {save_rank}'s {num_episodes}-episode results: {aggregated_states}")
                for k,v in aggregated_states.items():
                    v = torch.tensor(v*num_episodes).cuda()
                    cat_v = gather_list_and_concat(v,self.world_size)
                    v = (sum(cat_v)/total).item()
                    aggregated_states[k] = v
        
        if self.config.eval.split == 'val_seen_unseen':
            for split in ['val_unseen', 'val_seen']:
                if split == 'val_unseen' and unseen_num_episodes == 0:
                    unseen_aggregated_states['spl'] = 0
                    unseen_aggregated_states['success'] = 0
                    continue
                if split == 'val_seen' and seen_num_episodes == 0:
                    seen_aggregated_states['spl'] = 0
                    seen_aggregated_states['success'] = 0
                    continue
                
                stat_eps = self.val_unseen_stat_eps if split == 'val_unseen' else self.val_seen_stat_eps
                tmp_aggregated_states = unseen_aggregated_states if split == 'val_unseen' else seen_aggregated_states

                if self.local_rank < 1:
                    if self.config.eval.save_results:
                        fname = os.path.join(
                            self.config.results_dir,
                            f"stats_ckpt_{checkpoint_index}_{split}.json",
                        )
                        with open(fname, "w") as f:
                            json.dump(tmp_aggregated_states, f, indent=2)

                    self.logger.info(f"{split} Episodes evaluated: {len(stat_eps)}")
                    checkpoint_num = int(checkpoint_index) + 1
                    for k, v in tmp_aggregated_states.items():
                        self.logger.info(f"{split} Average episode {k}: {v:.6f}")
                        writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            
            return unseen_aggregated_states['spl'], unseen_aggregated_states['success'], seen_aggregated_states['spl'], seen_aggregated_states['success']
        
        else:
            split = self.config.task_config.dataset.split

            if self.local_rank < 1:
                if self.config.eval.save_results:
                    fname = os.path.join(
                        self.config.results_dir,
                        f"stats_ckpt_{checkpoint_index}_{split}.json",
                    )
                    with open(fname, "w") as f:
                        json.dump(aggregated_states, f, indent=2)

                self.logger.info(f"Episodes evaluated: {total}")
                checkpoint_num = int(checkpoint_index) + 1
                for k, v in aggregated_states.items():
                    self.logger.info(f"Average episode {k}: {v:.6f}")
                    writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            
            return aggregated_states['spl'], aggregated_states['success']

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.inference.ckpt_path
        self.logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.task_config.dataset.split = self.config.inference.split
        self.config.task_config.dataset.roles = ["guide"]
        self.config.task_config.dataset.languages = self.config.inference.languages
        self.config.task_config.environment.iterator_options.shuffle = False
        self.config.task_config.environment.iterator_options.max_scene_repeat_steps = -1
        self.config.task_config.task.measurements = ['POSITION_INFER']
        self.config.task_config.task.MEASUREMENTS = ['POSITION_INFER']
        self.config.task_config.task.sensors = [s for s in self.config.task_config.task.sensors if "INSTRUCTION" in s]
        self.config.simulator_gpu_ids = [self.config.simulator_gpu_ids[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.rl.policy.obs_transforms.resizer_per_sensor.sizes
        crop_config = self.config.rl.policy.obs_transforms.center_cropper_per_sensor.sensor_crops
        task_config = self.config.task_config
        if not self.config.LLM.use:
            camera_orientations = get_camera_orientations12()
        elif self.config.LLM.use:
            if self.config.LLM.add_pano_stamp:
                camera_orientations = get_camera_orientations(36)
            else:
                camera_orientations = get_camera_orientations12()
        for sensor_type in ["rgb", "depth"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.simulator, f"{sensor_type}_sensor")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}" # .0
                camera_config = deepcopy(sensor)
                camera_config.orientation = camera_orientations[action]
                camera_config.uuid = camera_template.lower()
                setattr(task_config.simulator, camera_template, camera_config)
                task_config.simulator.agent_0.sensors.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.rl.policy.obs_transforms.resizer_per_sensor.sizes = resize_config
        self.config.rl.policy.obs_transforms.center_cropper_per_sensor.sensor_crops = crop_config
        self.config.task_config = task_config
        self.config.sensors = task_config.simulator.agent_0.sensors
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.gpu_numbers
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.torch_gpu_ids[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.torch_gpu_id = self.config.torch_gpu_ids[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.env_name),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.inference.episode_count == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.inference.episode_count, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)
        
        # load front file
        if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
            self.z_front_dict = self.front_feat_loader.read_tim_tsv(self.config.model.causal.front_Kmeans_file, return_dict=True)
            self.logger.info(f"Successfully load front features from {self.config.model.causal.front_Kmeans_file}")

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.model.task_type == "r2r":
            with open(self.config.inference.predictions_file, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.inference.predictions_file}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.inference.predictions_file, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.inference.predictions_file}")
    
    def save_obs(self, obs, ep_ids, step):
        import matplotlib.pyplot as plt
        for i in range(len(obs)):
            rgb = obs[i]['rgb']
            output_file = os.path.join(self.config.video_dir, f'episode{ep_ids[i]}_step{step}.png')
            plt.imsave(output_file, rgb)
            print(f"Observation saved to {output_file}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori] # position
        ori = [x[1] for x in pos_ori] # orientation 4-dim (quaternions)
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None, envs=None, add_env_dropout=False):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        envs = self.envs if envs is None else envs
        
        envs.resume_all()
        observations = envs.reset()
        
        if mode == 'train' and 'reverie' in self.config.base_task_config_path:
            reverie_inf_indices = torch.zeros(self.envs.num_envs, dtype=torch.bool)
        
        instr_max_len = self.config.il.max_text_len 
        instr_pad_id = 0
        if self.instr_bert_model_name == 'roberta':
            instr_pad_id = 1
        instruction_text = [ep['instruction']['text'] for ep in observations]
        observations = extract_instruction_tokens(observations, self.config.task_config.task.instruction_sensor_uuid,
                                                  bert_tokenizer=self.bert_tokenizer,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms) 
        
        if mode == 'eval':
            curr_eps = envs.current_episodes()
            env_to_pause = [i for i, ep in enumerate(envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]
            gt_end_points = [ep.reference_path[-1] for ep in curr_eps]    
            envs, batch, observations, instruction_text = self._pause_envs(envs, batch, observations, env_to_pause, instruction_text)
            if envs.num_envs == 0: return
            
            new_curr_eps = []
            for i, eps in enumerate(curr_eps):
                if i in env_to_pause:
                    continue
                else:
                    new_curr_eps.append(eps)
            
            curr_eps = new_curr_eps
            
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]   
            gt_end_points = [None for _ in range(envs.num_envs)]
            envs, batch, observations, instruction_text = self._pause_envs(envs, batch, observations, env_to_pause, instruction_text)
            if envs.num_envs == 0: return
            curr_eps = envs.current_episodes()
            for i in range(envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        if self.config.LLM.use:
            self.prompt_manager.reset(envs.num_envs, instruction_text, use_landmark_mark=self.config.LLM.use_landmark_mark)
            llm_format = ['vp', 'logits', 'instruction', 'rgb0']
            if self.config.LLM.add_pano_stamp:
                llm_format.append('pano')
                llm_format.append('anno')
            
        #### causal inference ####
        # backdoor inference
        img_zdict, instr_zdict = None, None
        if self.config.model.causal.do_back_img or self.config.model.causal.do_back_txt:
            if 'img_zdict' in self.z_dicts:
                img_zdict = self.z_dicts['img_zdict']
            if 'instr_zdict' in self.z_dicts:
                instr_zdict = self.z_dicts['instr_zdict']
        
        # frontdoor inference
        front_txt_feats, front_vp_feats, front_gmap_feats = None, None, None
        if self.config.model.causal.do_front_local or self.config.model.causal.do_front_global or self.config.model.causal.do_front_txt:
            if self.config.model.causal.do_front_txt and 'txt_feats' in self.z_front_dict:
                front_txt_feats = []
                for _ in range(envs.num_envs):
                    front_txt_feats.append(self.z_front_dict['txt_feats'])
                front_txt_feats = torch.from_numpy(np.array(front_txt_feats)).to(self.device)
            if self.config.model.causal.do_front_local and 'vp_feats' in self.z_front_dict:
                front_vp_feats = []
                for _ in range(envs.num_envs):
                    front_vp_feats.append(self.z_front_dict['vp_feats'])
                front_vp_feats = torch.from_numpy(np.array(front_vp_feats)).to(self.device)
            if self.config.model.causal.do_front_global and 'gmap_feats' in self.z_front_dict:
                front_gmap_feats = []
                for _ in range(envs.num_envs):
                    front_gmap_feats.append(self.z_front_dict['gmap_feats'])
                front_gmap_feats = torch.from_numpy(np.array(front_gmap_feats)).to(self.device)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        
        # backdoor inference
        instr_z_direction_features, instr_z_direction_pzs, instr_z_landmark_features, instr_z_landmark_pzs = None, None, None, None
        if self.config.model.causal.do_back_txt:
            instr_z_direction_features = instr_zdict['instr_direction_features'].repeat(self.envs.num_envs,1).reshape(self.envs.num_envs,-1,instr_zdict['instr_direction_features'].shape[-1]) # add batch_size in the first dimension
            instr_z_direction_pzs = instr_zdict['instr_direction_pzs'].repeat(self.envs.num_envs,1).reshape(self.envs.num_envs,-1,1)
            instr_z_landmark_features = instr_zdict['instr_landmark_features'].repeat(self.envs.num_envs,1).reshape(self.envs.num_envs,-1,instr_zdict['instr_direction_features'].shape[-1])
            instr_z_landmark_pzs = instr_zdict['instr_landmark_pzs'].repeat(self.envs.num_envs,1).reshape(self.envs.num_envs,-1,1)
        
        # Ensure the dimension of front_txt_feats
        if front_txt_feats is not None:
            front_txt_feats = front_txt_feats[0].unsqueeze(0).expand(envs.num_envs, -1, -1)

        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
            instr_z_direction_features=instr_z_direction_features,
            instr_z_direction_pzs=instr_z_direction_pzs,
            instr_z_landmark_features=instr_z_landmark_features,
            instr_z_landmark_pzs=instr_z_landmark_pzs,
            front_txt_feats=front_txt_feats,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION or self.config.LLM.use)
        if mode == 'infer':
            have_real_pos = False
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug, use_llm=self.config.LLM.use) for _ in range(envs.num_envs)]
        prev_vp = [None] * envs.num_envs
        
        # Waypoint predictor direction 12 splits
        if self.config.LLM.use and self.config.LLM.add_pano_stamp:
            camera_orientations_12 = get_camera_orientations(12)
            sensor_list = []
            batch_observations = {}
            for sensor_type in ["rgb", "depth"]:
                for action, orient in camera_orientations_12.items():
                        sensor_name = f"{sensor_type}_{action}"
                        sensor_list.append(f"{sensor_type}_{action}")
                        if sensor_name == 'rgb_0.0' and 'rgb_0.0' not in batch:
                            batch_observations['rgb_0.0'] = batch['rgb']
                        elif sensor_name == 'depth_0.0' and 'depth_0.0' not in batch:
                            batch_observations['depth_0.0'] = batch['depth']
                        else:
                            batch_observations[sensor_name] = batch[sensor_name]
        else:
            batch_observations = batch
        
        # env-drop
        noise = None
        if self.config.IL.add_env_dropout and add_env_dropout:
            if isinstance(self.policy.net, DDP):
                drop_env_model = self.policy.net.module.drop_env
            else:
                drop_env_model = self.policy.net.drop_env
            noise = drop_env_model(torch.ones(self.config.MODEL.CLIP_ENCODER.output_size).to(self.device))
            
        last_embeds = None

        for stepk in range(self.max_len):
            total_actions += envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            # ***** Waypoint *****
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch_observations,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs, noise=noise)
            
            # backdoor inference
            if self.config.model.causal.do_back_img:
                z_img_features = img_zdict['img_features'].repeat(envs.num_envs,1).reshape(envs.num_envs,-1,self.config.MODEL.CLIP_ENCODER.output_size)
                z_img_pzs = img_zdict['img_pzs'].repeat(envs.num_envs,1).reshape(envs.num_envs,-1,1)
                vp_inputs['z_img_features'] = z_img_features
                vp_inputs['z_img_pzs'] = z_img_pzs
            
            vp_inputs.update({
                'mode': 'panorama',
            })
            # ***** Panorama *****
            outputs = self.policy.net(**vp_inputs)
            if len(outputs) == 3:
                pano_embeds, pano_masks, pano_fused_embeds = outputs
            else:
                pano_embeds, pano_masks = outputs
                pano_fused_embeds = None
            
            if pano_fused_embeds is not None:
                avg_pano_embeds = pano_fused_embeds
            else:
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                                torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            batch_cand_vp_info = []
            
            for i in range(envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cand_vp_info ={cand_vp_i[j] : {
                    "polar":(wp_outputs['cand_angles'][i][j], 
                            wp_outputs['cand_distances'][i][j]) 
                            } 
                            for j in range(len(cand_vp_i))} #nav_inputs['cur_ang'][i]
                batch_cand_vp_info.append(cand_vp_info)
                
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode in ['eval', 'infer']:
                traj_ids = []
                for i in range(envs.num_envs):
                    ep_id = curr_eps[i].episode_id
                    scene_id = curr_eps[i].scene_id.split('/')[-1].split('.')[-2]
                    traj_id = f"{ep_id}-{scene_id}"
                    traj_ids.append(traj_id)
            
            if mode == 'train' or self.config.VIDEO_OPTION or self.config.LLM.use:
                cand_real_pos = []
                for i in range(envs.num_envs):
                    cand_real_pos_i = [
                        envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * envs.num_envs

            infer_ghost_vp = [None] * self.envs.num_envs # for llm
            for i in range(envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                if self.config.LLM.use:
                    infer_ghost_vp[i] = self.gmaps[i].update_graph_llm(prev_vp[i], stepk+1,
                        cur_vp[i], cur_pos[i], cur_ori[i],cur_embeds,
                        cand_vp[i], cand_pos[i], cand_embeds,
                        cand_real_pos[i], batch_cand_vp_info[i], gt_end_points[i])
                else:
                    self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                            cur_vp[i], cur_pos[i], cur_embeds,
                                            cand_vp[i], cand_pos[i], cand_embeds,
                                            cand_real_pos[i])
                                            
            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            
            if self.config.model.add_local_branch:
                nav_inputs.update(
                    self._nav_vp_variable(pano_embeds, cur_vp, cur_pos, cur_ori, cand_vp, cand_pos, cand_embeds, vp_inputs['view_lens'], vp_inputs['nav_types']
                ))
            
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
            })
            
            if self.config.model.add_local_branch:
                nav_inputs['mode'] = 'navigation_with_local_branch'
            
            no_vp_left = nav_inputs.pop('no_vp_left')
            
            # frontdoor inference
            if self.config.model.causal.do_front_txt:   
                front_txt_feats = front_txt_feats[0].unsqueeze(0).expand(envs.num_envs, -1, -1)
            if self.config.model.causal.do_front_local:
                front_vp_feats = front_vp_feats[0].unsqueeze(0).expand(envs.num_envs, -1, -1)
            if self.config.model.causal.do_front_global:
                front_gmap_feats = front_gmap_feats[0].unsqueeze(0).expand(envs.num_envs, -1, -1)
            nav_inputs['front_txt_feats'] = front_txt_feats
            nav_inputs['front_vp_feats'] = front_vp_feats
            nav_inputs['front_gmap_feats'] = front_gmap_feats
            
            # ***** Navigation *****
            nav_outs = self.policy.net(**nav_inputs)
            if self.config.model.add_local_branch:
                nav_logits = nav_outs['fused_logits']
            else:
                nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            if mode == 'train' or self.config.VIDEO_OPTION or self.config.LLM.use:
                if mode == 'infer':
                    pass
                else:
                    teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train':
                teacher_actions_for_loss = teacher_actions.clone()
                loss += F.cross_entropy(nav_logits, teacher_actions_for_loss, reduction='sum', ignore_index=-100)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # Use LLM to help the decision
            if self.config.LLM.use:
                c = torch.distributions.Categorical(nav_probs)
                entropy = c.entropy()
                
                if mode == 'infer':
                    teacher_actions_for_vlm = None
                else:
                    teacher_actions_for_vlm = teacher_actions.cpu().numpy()
                
                nav_inputs.update({
                    'step_ids': nav_inputs['gmap_step_ids'],
                    'vp_ids': nav_inputs['gmap_vp_ids'],
                    'raw_instruction': instruction_text,
                    'teacher_actions': teacher_actions_for_vlm,
                    'pred_actions': cpu_a_t,
                    'nav_logits': nav_logits.clone().detach().cpu().numpy(),
                    'nav_probs': nav_probs.clone().detach().cpu().numpy(),
                    'ghost_vp': infer_ghost_vp,
                    'pred_entropy': entropy.cpu().numpy(),
                })
                nav_inputs['traj_id'] = traj_ids
                self.prompt_manager.save_llm_traj_log(observations, nav_inputs, self.gmaps, llm_format)
                
                # get multiple candidate actions from net
                batch_net_pred_cand_actions = []
                for i in range(envs.num_envs):
                    net_pred_cand_actions = []
                    current_vp_id = self.gmaps[i].current_vp
                    for j in range(len(nav_probs[i])):
                        if nav_probs[i, j].data.item() > self.config.LLM.net_prob_min_threshold:
                            net_pred_cand_actions.append(nav_inputs['gmap_vp_ids'][i][j])
                    batch_net_pred_cand_actions.append(net_pred_cand_actions)

                # get help from the llm
                is_infer = mode == 'infer'
                llm_outputs = self.prompt_manager.run_nav(nav_inputs, self.gmaps, net_pred_cand_actions=batch_net_pred_cand_actions, llm_stats=self.llm_stats, is_infer=is_infer)
                llm_decision = llm_outputs['llm_decision']
                batch_message = llm_outputs['batch_message']
                cpu_a_t = llm_outputs['new_a_t']

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control': # by default
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    if self.config.pointgoal.use:
                        # pointgoal
                        env_actions.append(
                            {
                                'action': {
                                    'act': 5,
                                    'cur_vp': cur_vp[i],
                                    'front_vp': front_vp, 'front_pos': front_pos,
                                    'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                    'back_path': back_path,
                                    'pointgoal_net': self.pointgoal_actor_critic,
                                    'max_pg_steps': self.config.pointgoal.max_pg_steps,
                                    'obs_transforms': self.obs_transforms,
                                    'cfg': self.config.pointgoal_config,
                                    'device': self.device,
                                },
                                'vis_info': vis_info,
                            }
                        )
                    else:
                        # tryout
                        env_actions.append(
                            {
                                'action': {
                                    'act': 4,
                                    'cur_vp': cur_vp[i],
                                    'front_vp': front_vp, 'front_pos': front_pos,
                                    'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                    'back_path': back_path,
                                    'tryout': use_tryout,
                                },
                                'vis_info': vis_info,
                            }
                        )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp) 
            
            outputs = envs.step(env_actions) # update the observatons
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            # calculate metric
            if mode == 'eval':
                curr_eps = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    if 'reverie' not in self.config.base_task_config_path:
                        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])           

                    if 'reverie' in self.config.base_task_config_path:
                        nan_indices = np.where(np.isnan(distances))[0]
                        inf_indices = np.where(np.isinf(distances))[0]
                        if len(nan_indices) > 0 or len(inf_indices) > 0:
                            distances = np.nan_to_num(distances, nan=4.0, posinf=4.0, neginf=4.0)
                    
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    success_distance = self.config.eval.success_distance if hasattr(self.config.eval, 'success_distance') else 3.0
                    metric['success'] = 1. if distances[-1] <= success_distance else 0.
                    metric['oracle_success'] = 1. if (distances <= success_distance).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['gt_length'] = gt_length
                    metric['distances'] = distances.tolist()
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    if 'reverie' not in self.config.base_task_config_path:
                        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                        metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    self.stat_eps[ep_id] = metric
                    if self.config.eval.split == 'val_seen_unseen':
                        if int(ep_id) < self.unseen_length:
                            self.val_unseen_stat_eps[ep_id] = metric
                        else:
                            self.val_seen_stat_eps[ep_id] = metric
                        
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = envs.current_episodes()
                for i in range(envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        instruction_text.pop(i)
                        if 'reverie' in self.config.base_task_config_path and mode == 'train':
                            reverie_inf_indices = torch.cat([reverie_inf_indices[:i], reverie_inf_indices[i+1:]])
                        
                        if self.config.LLM.use:
                            self.prompt_manager.update_batch(i)
                
                curr_eps = envs.current_episodes()

            if envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations, self.config.task_config.task.instruction_sensor_uuid,
                                                  bert_tokenizer=self.bert_tokenizer,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            
            if self.config.LLM.use and self.config.LLM.add_pano_stamp:
                for sensor_type in ["rgb", "depth"]:
                    for action, orient in camera_orientations_12.items():
                        sensor_name = f"{sensor_type}_{action}"
                        sensor_list.append(f"{sensor_type}_{action}")
                        batch_observations[sensor_name] = batch[sensor_name]
            else:
                batch_observations = batch

        if mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())