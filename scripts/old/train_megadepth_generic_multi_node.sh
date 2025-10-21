#!/usr/bin/env bash
#SBATCH --job-name=reloc3r
#SBATCH --nodes=2                 
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --partition=alvis
#SBATCH --time=1200
#SBATCH --account=NAISS2025-5-255
#SBATCH --output=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_%N.out
#SBATCH --error=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_%N.err

OUTPUT_DIR="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# -------------------------------------------------------
# Distributed setup for torchrun
# -------------------------------------------------------
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_NODEID
GPUS_PER_NODE=1
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_RANK=$NODE_RANK"

# -------------------------------------------------------
# Launch with torchrun
# -------------------------------------------------------
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --train_dataset "50_000 @ MegaDepth(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "1_000 @ MegaDepth_valid(split='test', resolution=(512, 384), seed=777)" \
    --model "Reloc3rGeneric(backbone='dinov3')" \
    --pretrained "pretrained_models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 --amp 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder --log_wandb \
    --output_dir "$OUTPUT_DIR"
