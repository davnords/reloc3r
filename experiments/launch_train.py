import train as training
from shared.trainer import Trainer
from shared.submit import get_args_parser, submit_jobs
import sys
from pathlib import Path

def main():
    description = "Submitit launcher for relative pose estimation training"
    train_args_parser = training.get_args_parser()
    
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents, add_help=False)
    args = args_parser.parse_args()
    args.training_module = training.__name__

    # args.train_dataset = "50_000 @ MegaDepth(split='train', resolution=[(256, 256)], transform=ColorJitter)"
    args.train_dataset = "50_000 @ MegaDepth(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ RealEstate(split='train', resolution=[(256, 256)])+50_000 @ DL3DV(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ WildRGBD(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ ScanNet(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ MVSSynth(split='train', resolution=[(256, 256)])+50_000 @ VKitti(split='train', resolution=[(256, 256)])"
    args.test_dataset = "1_000 @ MegaDepth_valid(split='test', resolution=(256, 256), seed=777)+1_000 @ RealEstate(split='test', resolution=(256, 256), seed=777)"
    args.blr = 1.5e-4 
    args.min_lr = 1e-6
    args.warmup_epochs = 5 
    args.epochs = 100
    args.batch_size = 32
    args.accum_iter = 1 
    args.amp = 1
    args.save_freq = 10
    args.keep_freq = 50 
    args.eval_freq = 1

    dir_name = args.name if args.name is not None else args.vit
    args.output_dir = Path('output_dir') / dir_name / "%j"
    submit_jobs(Trainer, args, name='reloc3r:train')
    return 0

if __name__ == "__main__":
    sys.exit(main())
