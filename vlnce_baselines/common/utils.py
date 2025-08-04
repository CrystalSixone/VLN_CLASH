import os, sys
from typing import Any, Dict, List
import torch
import torch.distributed as dist
import numpy as np
import copy
import math
import json
from habitat_sim.utils.common import d3_40_colors_rgb
from PIL import Image
import matplotlib.pyplot as plt

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
    bert_tokenizer = None,
    is_clip_long=False,
    max_length: int = 512,
    pad_id: int = 0,
) -> Dict[str, Any]:
    """Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    if (
        instruction_sensor_uuid not in observations[0]
        or instruction_sensor_uuid == "pointgoal_with_gps_compass"
    ):
        return observations
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
        ):
            if bert_tokenizer is None:
                token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
                if len(token) < max_length:
                    token += [pad_id] * (max_length - len(token))
                observations[i][instruction_sensor_uuid] = np.array(token)
            else:
                # use bert tokenizer
                if is_clip_long:
                    tokens = bert_tokenizer(observations[i][instruction_sensor_uuid]['text'], truncate=True)[0]
                else:
                    tokens = bert_tokenizer.text_token(observations[i][instruction_sensor_uuid]['text'])['input_ids'][0]
                    
                observations[i][instruction_sensor_uuid] = tokens
            
                if max_length is not None:
                    observations[i][instruction_sensor_uuid] = observations[i][instruction_sensor_uuid][:max_length]
        else:
            break
    return observations

# def extract_instruction_tokens(
#     observations: List[Dict],
#     instruction_sensor_uuid: str,
#     tokens_uuid: str = "tokens",
#     max_length: int = 512,
#     pad_id: int = 0,
# ):
#     """Extracts instruction tokens from an instruction sensor if the tokens
#     exist and are in a dict structure."""
#     if instruction_sensor_uuid not in observations[0]:
#         return observations
#     for i in range(len(observations)):
#         if (
#             isinstance(observations[i][instruction_sensor_uuid], dict)
#             and tokens_uuid in observations[i][instruction_sensor_uuid]
#         ):
#             token = observations[i][instruction_sensor_uuid]["tokens"][:max_length]
#             if len(token) < max_length:
#                 token += [pad_id] * (max_length - len(token))
#             observations[i][instruction_sensor_uuid] = np.array(token)
#         else:
#             break
#     return observations

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def dis_to_con(path, amount=0.25):
    starts = path[:-1]
    ends = path[1:]
    new_path = [path[0]]
    for s, e in zip(starts,ends):
        vec = np.array(e) - np.array(s)
        ratio = amount/np.linalg.norm(vec[[0,2]])
        unit = vec*ratio
        times = int(1/ratio)
        for i in range(times):
            if i != times - 1:
                location = np.array(new_path[-1])+unit
                new_path.append(location.tolist())
        new_path.append(e)
    
    return new_path

def get_camera_orientations12():
    base_angle_deg = 30
    base_angle_rad = math.pi / 6
    orient_dict = {}
    for k in range(1,12):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict

def get_camera_orientations(sectors=12):
    base_angle_deg = 360 / sectors
    base_angle_rad = math.pi / (sectors / 2)
    orient_dict = {}
    for k in range(0,sectors):
        orient_dict[str(base_angle_deg*k)] = [0.0, base_angle_rad*k, 0.0]
    return orient_dict

def extract_best_eval_results(log_file, split):
    results = {'best_spl':-1, 'best_sr':-1, 
               'best_spl_index':0, 'best_sr_index':0, 
               'best_spl_sr': -1, 'best_spl_sr_index': 0,
               'best_spl_sr_spl': -1, 'best_spl_sr_sr': -1}

    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            if f"Best {split} SPL and SR" in line:
                parts = line.split()
                cur_best_spl_sr = float(parts[7])
                if cur_best_spl_sr > results['best_spl_sr']:
                    results['best_spl_sr'] = cur_best_spl_sr
                    results['best_spl_sr_index'] = int(parts[-7])
                    results['best_spl_sr_spl'] = float(parts[-4])
                    results['best_spl_sr_sr'] = float(parts[-1])

            elif f"Best {split} SPL" in line:
                parts = line.split()
                cur_best_spl = float(parts[-4])
                if cur_best_spl > results['best_spl']:
                    results['best_spl'] = cur_best_spl  # The value before "at"
                    results['best_spl_index'] = int(parts[-1])   # The index value at the end
            elif f"Best {split} SR" in line:
                parts = line.split()
                cur_best_sr = float(parts[-4])
                if cur_best_sr > results['best_sr']:
                    results['best_sr'] = cur_best_sr
                    results['best_sr_index'] = int(parts[-1])   
    else:
        print(f"Log file {log_file} does not exist.")

    return results

def check_output_path(dir,sub_dir,file_name):
    new_dir = os.path.join(dir,sub_dir)
    os.makedirs(new_dir,exist_ok=True)
    file_path = os.path.join(new_dir,file_name)
    return file_path

def read_eval_stats_file(file_path):
    eval_stats = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                eval_stats = json.load(file)
                # Convert string keys to integer keys
                eval_stats = {int(k): v for k, v in eval_stats.items()}
            except json.JSONDecodeError:
                eval_stats = {}
    return eval_stats

def update_eval_stats_file(file_path, eval_stats):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing data from the file
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
                # Convert string keys to integer keys
                existing_data = {int(k): v for k, v in existing_data.items()}
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # Update the existing data with the new eval_stats
    existing_data.update(eval_stats)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    
def display_sample(rgb_obs, semantic_obs, depth_obs, output_dir,idx=0, trajectory_id=0):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    rgb_img.save("trajectory%d_%d_rgb.png"%(trajectory_id,idx))
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    semantic_img.save(os.path.join(output_dir, "trajectory%d_%d_semantic.png"%(trajectory_id,idx)))

    depth_img = Image.fromarray((depth_obs.squeeze() / 10 * 255).astype(np.uint8), mode="L")
    depth_img.save(os.path.join(output_dir, "trajectory%d_%d_depth.png"%(trajectory_id,idx)))

    arr = [rgb_img, semantic_img, depth_img]
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.savefig(os.path.join(output_dir, "trajectory%d_%d_all.png"%(trajectory_id,idx)))