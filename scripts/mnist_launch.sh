#!/bin/bash
  
# Enable conda environment
eval "$(conda shell.bash hook)"
conda activate /fsx/myenv

NODE_LIST=$(scontrol show jobid=$SLURM_JOBID | awk -F= '/NodeList=/{print $2}' | grep -v Exc)
MASTER_NODE=$(scontrol show hostname $NODE_LIST | head -n 1)
MASTER_ADDR=$(scontrol show node=$MASTER_NODE | awk -F= '/NodeAddr=/{print $2}' | awk '{print $1}')

echo $NODE_LIST
echo $SLURM_NNODES

torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=1 \
         --node_rank=$SLURM_NODEID \
         --master-addr=$MASTER_ADDR \
         --master_port=1234 \
         /fsx/models/train_mnist.py
