#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
import shutil

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config
from habitat_extensions.config.habitat21_default import Config as CN


def lowercase_config_keys(config):
    """Convert all uppercase keys in the config to lowercase recursively, keeping original keys."""
    for key in list(config.keys()):
        if key.isupper():
            config[key.lower()] = config[key]  # Create a new lowercase key without deleting the original
        new_key = key.lower()
        if isinstance(config[key], CN):  # Check the original key for CN type
            config[new_key] = config[key]  # Create a new lowercase key at the same level
            lowercase_config_keys(config[new_key])  # Recursively call for the new lowercase key
        elif isinstance(config[key], dict):
            config[new_key] = config[key]  # Create a new lowercase key at the same level
            lowercase_config_keys(config[new_key])  # Recursively call for the new lowercase key
        elif isinstance(config[key], list):
            config[new_key] = config[key]  # Create a new lowercase key at the same level
            for item in config[new_key]:
                if isinstance(item, CN) or isinstance(item, dict):
                    lowercase_config_keys(item)  # Recursively call for items in the list
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        # required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        # required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        # required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="debug mode",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None,
            debug: bool = False) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()    
    config.run_type = run_type
    config.debug = debug

    ###### create log directions ######
    project_name = config.NAME
    print(f"[Project Name]: {project_name}")
    config.TENSORBOARD_DIR = config.TENSORBOARD_DIR.replace("${NAME}", project_name)
    config.CHECKPOINT_FOLDER = config.CHECKPOINT_FOLDER.replace("${NAME}", project_name)
    config.EVAL_CKPT_PATH_DIR = config.EVAL_CKPT_PATH_DIR.replace("${NAME}", project_name)
    config.RESULTS_DIR = config.RESULTS_DIR.replace("${NAME}", project_name)
    config.VIDEO_DIR = config.VIDEO_DIR.replace("${NAME}", project_name)
    config.LOG_DIR = config.LOG_DIR.replace("${NAME}", project_name)
    
    config.TENSORBOARD_DIR = os.path.join(config.BASE_LOG_DIR, config.TENSORBOARD_DIR)
    config.CHECKPOINT_FOLDER = os.path.join(config.BASE_LOG_DIR, config.CHECKPOINT_FOLDER)
    config.EVAL_CKPT_PATH_DIR = os.path.join(config.BASE_LOG_DIR, config.EVAL_CKPT_PATH_DIR)
    config.RESULTS_DIR = os.path.join(config.BASE_LOG_DIR, config.RESULTS_DIR)
    config.VIDEO_DIR = os.path.join(config.BASE_LOG_DIR, config.VIDEO_DIR)
    config.LOG_DIR = os.path.join(config.BASE_LOG_DIR, config.LOG_DIR)
    config.LOG_FILE = os.path.join(config.RESULTS_DIR, 'eval.log')
    
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    
    ####################################

    ### NDTW ###
    if run_type == "train":
        config.TASK_CONFIG.TASK.NDTW.SPLIT = "train"
    elif run_type == "eval":
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        
    ### DDPPO pointGoal ###
    pointgoal_config = None
    if not hasattr(config, 'PointGoal'):
        config.PointGoal = CN()
        config.PointGoal.use = False
    else:
        if config.PointGoal.use:
            pointgoal_config = get_config(config.PointGoal.config_path, [])
            config.TASK_CONFIG.TASK.SENSORS.append('POINTGOAL_WITH_GPS_COMPASS_SENSOR')
    
    config.pointgoal_config = pointgoal_config

    config = lowercase_config_keys(config) # for habitat23
    config = lowercase_config_keys(config)

    if run_type in ["eval", "inference"]:
        if config.MODEL.causal.do_front_local or config.MODEL.causal.do_front_global or config.MODEL.causal.do_front_txt:
            if len(config.eval.ckpt_path_dir) > 0:
                eval_ckpt_dir = os.path.dirname(os.path.dirname(config.eval.ckpt_path_dir))
                config.MODEL.causal.front_Kmeans_file = os.path.join(eval_ckpt_dir, config.MODEL.causal.front_Kmeans_file)
            else:
                config.MODEL.causal.front_Kmeans_file = os.path.join(config.CHECKPOINT_FOLDER, config.MODEL.causal.front_Kmeans_file)
            
    ## copy config files to log dir
    config_basename = os.path.basename(exp_config)
    target_config_path = os.path.join(config.LOG_DIR, config_basename)
    try:
        shutil.copy2(exp_config, target_config_path)
    except Exception as e:
        logger.info(f"Error copying config file: {e}")
    logger.info(f"Config file copied to: {target_config_path}")
    
    if run_type == "inference":
        config.BASE_TASK_CONFIG_PATH = 'run_r2r/r2r_scalevln_infer.yaml'

    sim_config_basename = os.path.basename(config.BASE_TASK_CONFIG_PATH)
    target_sim_config_path = os.path.join(config.LOG_DIR, sim_config_basename)
    try:
        shutil.copy2(config.BASE_TASK_CONFIG_PATH, target_sim_config_path)
    except Exception as e:
        logger.info(f"Error copying sim config file: {e}")
    logger.info(f"Sim config file copied to: {target_sim_config_path}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"local_rank: {local_rank}")
    config.local_rank = local_rank
    config.freeze()
    logger.add_filehandler(config.LOG_FILE)
    logger.info(f"New Navigation Start!")
    logger.info(f"waypoint noise:{config.IL.loc_noise} spilt:{config.EVAL.SPLIT} episode:{config.EVAL.EPISODE_COUNT}")

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference() # -> vlnce_baselines/ss_trainer_CLASH.py

if __name__ == "__main__":
    main()
