# Running evaluation for megadepth1500

# CUDA_VISIBLE_DEVICES=0 python eval_relpose.py \
#     --model "Reloc3rRelpose(img_size=512)" \
#     --test_dataset "MegaDepth_valid(resolution=(512,384), seed=777)" \


CUDA_VISIBLE_DEVICES=2 python eval_relpose_generic.py \ 
    --model "Reloc3rGeneric(backbone='dinov3')" \ 
    --test_dataset "MegaDepth_valid(resolution=(512,384), seed=777)" \
    --ckpt /mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/5205190/checkpoint-best.pth \
    --amp 1 \