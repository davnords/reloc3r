#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --partition=alvis
#SBATCH --job-name=reloc3r
#SBATCH --output=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_0_log.out
#SBATCH --error=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_0_log.err
#SBATCH --time=1200
#SBATCH --account=NAISS2025-5-255
#SBATCH --nodes=1

OUTPUT_DIR="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=1 train.py \
    --train_dataset "50_000 @ BlendedMVS(split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], transform=ColorJitter)" \
    --test_dataset "1_000 @ BlendedMVS(split='val', resolution=(512, 384), seed=777)" \
    --model "Reloc3rGeneric(backbone='dinov3')" \
    --pretrained "pretrained_models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 1e-5 --min_lr 1e-7 --warmup_epochs 5 --epochs 100 --batch_size 28 --accum_iter 1 --amp 1 \
    --save_freq 10 --keep_freq 10 --eval_freq 1 \
    --freeze_encoder --log_wandb --name "dinov3_blendedmvs" \
    --output_dir "$OUTPUT_DIR"  

