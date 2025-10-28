from reloc3r.datasets import get_data_loader  # noqa
import torchvision.utils as vutils
import torch

mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader

# train_dataset = "50_000 @ MegaDepth(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ RealEstate(split='test', resolution=[(256, 256)])+50_000 @ DL3DV(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ WildRGBD(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ ScanNet(split='train', resolution=[(256, 256)], transform=ColorJitter)+50_000 @ MVSSynth(split='train', resolution=[(256, 256)])+50_000 @ VKitti(split='train', resolution=[(256, 256)])"
# train_dataset = "50_000 @ DL3DV(split='train', resolution=[(256, 256)], transform=ColorJitter)"
# train_dataset = "50_000 @ WildRGBD(split='train', resolution=[(256, 256)], transform=ColorJitter)"
# train_dataset = "50_000 @ ScanNet(split='train', resolution=[(256, 256)], transform=ColorJitter)"
# train_dataset = "50_000 @ MVSSynth(split='train', resolution=[(256, 256)])"
# train_dataset = "50_000 @ VKitti(split='train', resolution=[(256, 256)])"
train_dataset = "50_000 @ Hypersim(split='train', resolution=[(256, 256)])"
# train_dataset = "50_000 @ RealEstate(split='test', resolution=[(256, 256)])"
# train_dataset = "50_000 @ BlendedMVS(split='train', resolution=[(256, 256)])"

data_loader_train = build_dataset(train_dataset, 4, 8, test=False)

if hasattr(data_loader_train, 'dataset') and hasattr(data_loader_train.dataset, 'set_epoch'):
    data_loader_train.dataset.set_epoch(0)

for i, batch in enumerate(data_loader_train):
    print(f"Batch {i}:")
    imgs1 = batch[0]['img']
    imgs2 = batch[1]['img']

    # # unnormalize: (x * std) + mean
    imgs1 = imgs1 * std + mean
    imgs2 = imgs2 * std + mean

    vutils.save_image(imgs1[0], f"test_image_{i}_0.png")
    vutils.save_image(imgs2[0], f"test_image_{i}_1.png")

    # print('Pose shape: ', batch[0]['camera_pose'].shape)
    # print('Intrinsics shape: ', batch[0]['camera_intrinsics'].shape)
    # break