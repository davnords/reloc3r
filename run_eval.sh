#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-255
#SBATCH -o /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/slurm_outs/%x_%j.out
#SBATCH -t 0-02:00:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=alvis

source ~/.bashrc
conda activate capi

# python eval_relpose.py --model "RelposeTransformer(vit='mum')" --test_dataset "MegaDepth_valid(resolution=(256,256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/mum/5241602/checkpoint-100.pth --amp 1
# python eval_relpose.py --model "RelposeTransformer(vit='crocov2')" --test_dataset "MegaDepth_valid(resolution=(256,256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/crocov2/5241605/checkpoint-100.pth --amp 1
# python eval_relpose.py --model "RelposeTransformer(vit='dinov3')" --test_dataset "MegaDepth_valid(resolution=(256,256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/dinov3/5241603/checkpoint-100.pth --amp 1

# python eval_relpose.py --model "RelposeTransformer(vit='mum')" --test_dataset "RealEstate(split='test', resolution=(256, 256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/mum/5241602/checkpoint-100.pth --amp 1
# python eval_relpose.py --model "RelposeTransformer(vit='crocov2')" --test_dataset "RealEstate(split='test', resolution=(256, 256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/crocov2/5241605/checkpoint-100.pth --amp 1
python eval_relpose.py --model "RelposeTransformer(vit='dinov3')" --test_dataset "RealEstate(split='test', resolution=(256, 256), seed=777)" --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/dinov3/5241603/checkpoint-100.pth --amp 1