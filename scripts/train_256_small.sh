#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=A100:2
#SBATCH --partition=alvis
#SBATCH --job-name=reloc3r
#SBATCH --output=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_0_log.out
#SBATCH --error=/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/%j_0_log.err
#SBATCH --time=400
#SBATCH --account=NAISS2025-5-255
#SBATCH --nodes=1

MODEL_NAME="mum"
NAME="${MODEL_NAME}_256_finetune"
OUTPUT_DIR="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/${NAME}/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# lr for finetuning decoder is in reality 1e-5
# and min lr is 1e-7

torchrun --nproc_per_node=2 train.py \
    --train_dataset "50_000 @ MegaDepth(split='train', resolution=[(256, 256)], transform=ColorJitter)" \
    --test_dataset "1_000 @ MegaDepth_valid(split='test', resolution=(256, 256), seed=777)" \
    --vit "$MODEL_NAME" \
    --blr 1.5e-4 \
    --min_lr 1e-6 --warmup_epochs 5 --epochs 100 --batch_size 32 --accum_iter 1 --amp 1 \
    --save_freq 10 --keep_freq 50 --eval_freq 1 \
    --name "$NAME" \
    --output_dir "$OUTPUT_DIR" \
    # --log_wandb \
    # --freeze_encoder \
    # --pretrained "pretrained_models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \

