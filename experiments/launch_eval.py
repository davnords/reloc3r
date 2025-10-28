import train as training
from shared.trainer import Trainer
from shared.submit import get_args_parser, submit_jobs
import sys
from pathlib import Path

ckts = {
    "mum": "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/mum/5241602/checkpoint-100.pth",
    "dinov3": "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/dinov3/5241603/checkpoint-100.pth",
    "crocov2": "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/reloc3r/output_dir/crocov2/5241605/checkpoint-100.pth",
}

dts = {
    "megadepth": "MegaDepth_valid(resolution=(256,256), seed=777)",
    "re10k": "RealEstate(split='test', resolution=(256, 256), seed=777)",
    "blendedmvs": "BlendedMVS(split='test', resolution=(256, 256), seed=777)",
    "scannet": "ScanNet1500(resolution=(256,256), seed=777)",
    "co3d": "Co3d(split='test', resolution=(256,256), seed=777)",
}

def main():
    description = "Submitit launcher for relative pose evaluation"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents, add_help=False)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    args.amp = 1

    models = ["mum"]
    datasets = ["megadepth"]
    # models = ["mum", "dinov3", "crocov2"]
    # datasets = ["megadepth", "re10k", "blendedmvs", "scannet", "co3d"]
    for model in models:
        for dataset in datasets:
            args.output_dir = Path('output_dir') / args.model / dataset
            args.ckpt = ckpts[model]
            args.test_dataset = dts[dataset]
            args.model = f"RelposeTransformer(vit='{model}')"
            submit_jobs(Trainer, args, name='reloc3r:eval')
    return 0

if __name__ == "__main__":
    sys.exit(main())