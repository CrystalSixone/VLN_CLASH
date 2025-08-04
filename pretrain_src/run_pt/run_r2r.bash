#!/bin/bash
#SBATCH -o logs/w61.%j.out ### 作业名称自定义
#SBATCH -e logs/w61.%j.err ### 作业名称自定义
#SBATCH -J w61 ### 作业名称自定义
#SBATCH -p L40  ### 使用L40队列
#SBATCH -N 2 ###使用1个节点
#SBATCH -n 6 ###总共申请6个core
#SBATCH --ntasks-per-node=6   ###每个节点使用6个core
#SBATCH --gres=gpu:l40:2 ###每个节点使用1张l40，卡数可以自定义

source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate habitat21 

NODE_RANK=0
NUM_GPUS=2
outdir=pretrained/r2r_ce/mlm.sap_habitat_depth

# train
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port 8887 \
    pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
    --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
    --output_dir $outdir
