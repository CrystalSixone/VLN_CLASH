#!/bin/bash
#SBATCH --job-name=vln_clash_train
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH -p L40
#SBATCH -n 6
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate VLN_CLASH

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
MASTER_PORT=29502

mode=$1
val_split=${2:-"val_unseen"}
cfg=${3:-"run_r2r/iter_train_r2r.yaml"}

flag_train="
      --run-type train
      --exp-config $cfg
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 16
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      "

flag_eval="
      --run-type eval
      --exp-config $cfg
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.back_algo control
      EVAL.SPLIT $val_split
      "

flag_infer="
      --run-type inference
      --exp-config $cfg
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      IL.back_algo control
      EVAL.SPLIT test
      "

case $mode in 
      train)
      echo "###### train mode ######"
      python run.py $flag_train
      ;;
      eval)
      echo "###### eval mode ######"
      echo "val_split: $val_split"
      export CUDA_VISIBLE_DEVICES=0
      python run.py $flag_eval
      ;;
      infer)
      echo "###### infer mode ######"
      export CUDA_VISIBLE_DEVICES=0
      python run.py $flag_infer
      ;;
esac