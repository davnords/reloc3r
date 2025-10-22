import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import wandb
from PIL import Image
from torchvision import transforms as TF
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as tf
from .utils import patchify, unpatchify
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Get LateX font
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams["savefig.bbox"] = 'tight'

# ------------------------------------------------------------
# Multi-view visualization

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tf.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def show_grids(grids, titles, vertical=True):
    if vertical: 
        fig, axs = plt.subplots(len(grids), 1, figsize=(40, 10))
    else:
        fig, axs = plt.subplots(1, len(grids), figsize=(5 * len(grids), 5))
        fig.subplots_adjust(wspace=0.05)  # reduce horizontal space between images
    for ax, img, title in zip(axs, grids, titles):
        img = img.detach()
        img = tf.to_pil_image(img)
        ax.imshow(np.asarray(img))
        ax.set_title(title)
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def unnormalize(x):
    imagenet_mean_tensor = torch.tensor(IMAGENET_DEFAULT_MEAN).view(1,3,1,1).to(x.device, non_blocking=True)
    imagenet_std_tensor = torch.tensor(IMAGENET_DEFAULT_STD).view(1,3,1,1).to(x.device, non_blocking=True)
    x = torch.clip((x * imagenet_std_tensor + imagenet_mean_tensor) * 255, 0, 255).detach().cpu()/255
    return x 

def reconstruct_predicted_image(pred, seq, patchified, patch_size, mask, visible=True):
    B, S, C_in, H, W = seq.shape
    mean, var = patchified.mean(dim=-1, keepdim=True), patchified.var(dim=-1, keepdim=True)
    pred = pred.view(B*S, -1, patch_size**2*3)
    predicted_image = unpatchify(pred * (var + 1.e-6)**0.5 + mean, patch_size, H, W)
    if visible:
        # Replace visible patches with the original image
        image_masks = unpatchify(patchify(torch.ones_like(predicted_image), patch_size) * mask[:, :, None], patch_size, H, W)
        masked_target_image = (1 - image_masks) * seq.view(B*S, C_in, H, W)
        predicted_image = predicted_image * image_masks + masked_target_image
    predicted_image = predicted_image.view(B, S, C_in, H, W)
    return predicted_image

def visualize_sequence(seq: torch.Tensor, path:str):
    """
    Seq: torch.Tensor(B, S, 3, H, W)
    """
    seq = seq[0] # Eliminate all the unessecary elements in the batch
    S, C_in, H, W = seq.shape
    show_grids(
        [make_grid(unnormalize(seq[i]), normalize=True) for i in range(S)],
        [f'Frame {i}' for i in range(S)],
        vertical=False
    )
    plt.savefig(path)
    plt.close()

def qualitative_evaluation(model: torch.nn.Module, seq: torch.Tensor, path:str, dont_log_wandb:bool=True, visible:bool=True):
    """
    Seq: torch.Tensor(B, S, 3, H, W)
    Very awkward with all the reshapes and stuff... Probably need to simplify this to avoid always reshaping between (B*S, ...) to (B, S, ...) and vice versa
    """
    patch_size = model.patch_size
    seq = seq[0].unsqueeze(0) # Eliminate all the unessecary elements in the batch
    B, S, C_in, H, W = seq.shape
    with torch.inference_mode():
        # out, _, mask = model(seq)
        loss, out, mask = model(seq)
        mask = mask.to(dtype=torch.bool)
        mask = mask.view(B*S, -1)
        patchified = patchify(seq.view(B*S, C_in, H, W), patch_size)        
        predicted_image = reconstruct_predicted_image(out, seq, patchified, patch_size, mask, visible=visible)
        baseline_prediction = reconstruct_predicted_image(torch.zeros_like(out, device=out.device), seq, patchified, patch_size, mask, visible=visible)

        masked_images = patchified.clone()
        masked_images[mask] = masked_images.min()
        masked_images = unpatchify(masked_images, patch_size, H, W)
        show_grids(
            [
                make_grid(unnormalize(seq[0]), normalize=True),
                make_grid(unnormalize(masked_images), normalize=True),
                make_grid(unnormalize(predicted_image[0]), normalize=True),
                make_grid(unnormalize(baseline_prediction[0]), normalize=True),
            ],
            [
                'Original',
                'Masked',
                'Predicted',
                'Baseline',
            ]
        )
        plt.savefig(path)
        plt.close()

        if not dont_log_wandb:
            wandb.log({path.split('/')[-1].split('_')[0]+"_plot": wandb.Image(path)})

def showcase_scenes(model):
    # References: https://github.com/naver/croco/blob/master/demo.py and https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb#scrollTo=4573e6be-935a-4106-8c06-e467552b0e3d
    device = 'cuda'
    model = model.to(device)
    model.eval()
    scenes = [
        (
            'assets/mit/IMG_8551.JPG', 
            'assets/mit/IMG_8552.JPG', 
            # 'assets/mit/IMG_8553.JPG', 
            # 'assets/mit/IMG_8554.JPG', 
            # 'assets/mit/IMG_8555.JPG', 
            # 'assets/mit/IMG_8556.JPG', 
            # 'assets/mit/IMG_8557.JPG', 
            # 'assets/mit/IMG_8558.JPG', 
            
            'assets/mit/IMG_8565.JPG', 
            'assets/mit/IMG_8567.JPG', 
        ),
    ]
    img_size = 448
    transforms = TF.Compose([
        TF.CenterCrop((1024, 1024)),
        TF.Resize((img_size, img_size)),
        TF.ToTensor(),
        TF.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    for imgs in scenes:
        imgs = tuple(transforms(Image.open(img).convert('RGB')) for img in imgs)
        seq = torch.stack(imgs).unsqueeze(0).to(device)

        B, S, C_in, H, W = seq.shape
        with torch.inference_mode():
            out, _, mask = model(seq)
        patchified = patchify(seq.reshape(B*S, C_in, H, W), model.patch_size)  
        predicted_image = reconstruct_predicted_image(out, seq, patchified, model.patch_size, mask)
        
        masked_images = patchified.clone()
        masked_images[mask] = masked_images.min()
        masked_images = unpatchify(masked_images, model.patch_size)
        show_grids(
            [
                make_grid(unnormalize(seq[0]), normalize=False),
                make_grid(unnormalize(masked_images), normalize=False),
                make_grid(unnormalize(predicted_image[0]), normalize=False),
            ],
            [
                'Original',
                'Masked',
                'Predicted',
            ],
            vertical=False
        )
        
        plt.savefig('MIT.pdf', bbox_inches='tight')
        plt.close()    

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

def denormalize(t, mean, std):
    mean = torch.tensor(mean).view(3,1,1).to(t.device)
    std = torch.tensor(std).view(3,1,1).to(t.device)
    return t * std + mean

def visualize_attention(model, img1, img2):
    transform = transforms.Compose(
        [
            # transforms.RandomResizedCrop(
            #     448, interpolation=transforms.InterpolationMode.BICUBIC
            # ),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img1, img2 = transform(img1), transform(img2)

    model.eval()
    model.cuda()

    with torch.no_grad():
        seq = torch.stack([img1, img2], dim=0).unsqueeze(0).cuda()

        print('seq shape: ', seq.shape)

        B, S, C_in, H, W = seq.shape
        seq = seq.view(B*S, C_in, H, W)  # [B*S, C, H, W]
        x = model.forward_encoder(seq, 0,return_all_blocks=True)[-1]
        x = model.norm(x)

        x = model.decoder_embed(x)
        rope_sincos = model.rope_embed_decoder(H=H//model.patch_size, W=W//model.patch_size)

        _, P, C = x.shape

        # apply alternating attention
        for frame_block, global_block in zip(model.decoder_frame_blocks, model.decoder_global_blocks):
            # Frame-wise attention
            if x.shape != (B * S, P, C):
                x = x.view(B, S, P, C).view(B * S, P, C)
            x = frame_block(x, rope_sincos)
            
            # Global attention
            x = x.view(B, S, P, C).view(B, S * P, C)
            x = global_block(x, rope_sincos)

    q = model.decoder_global_blocks[1].attn._last_q
    k = model.decoder_global_blocks[1].attn._last_k

    attentions = q@k.transpose(-2, -1)
    attentions = attentions.sum(axis=1).squeeze(0)


    # query_patch_idx = 155
    query_patch_idx = 320
    # query_patch_idx = 462
    attention = attentions[query_patch_idx+1, attentions.shape[-1]//2:][1:]

    ph, pw = H // model.patch_size, W // model.patch_size
    attn_map = attention.reshape(ph, pw).cpu().numpy()

    # argmax in img2
    max_idx = attn_map.argmax()
    row2, col2 = divmod(max_idx, pw)

    patch_h, patch_w = H // ph, W // pw
    center_x2 = col2 * patch_w + patch_w // 2
    center_y2 = row2 * patch_h + patch_h // 2

    # query patch in img1
    row1, col1 = divmod(query_patch_idx - 1, pw)
    center_x1 = col1 * patch_w + patch_w // 2
    center_y1 = row1 * patch_h + patch_h // 2

    print(f"Query patch in img1: (row={row1}, col={col1}), pixel=({center_x1}, {center_y1})")
    print(f"Best match in img2: (row={row2}, col={col2}), pixel=({center_x2}, {center_y2})")

    # plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    img1_vis = denormalize(img1, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img2_vis = denormalize(img2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    axes[0].imshow(img1_vis.squeeze(0).permute(1, 2, 0).cpu())
    axes[0].scatter([center_x1], [center_y1], c='blue', s=80, marker='o')
    # axes[0].set_title("Query patch (img1)")
    axes[0].axis("off") 

    axes[1].imshow(img2_vis.squeeze(0).permute(1, 2, 0).cpu())
    axes[1].scatter([center_x2], [center_y2], c='red', s=80, marker='o')
    # axes[1].set_title("Best match (img2)")
    axes[1].axis("off")

    plt.savefig('attention_match.pdf', bbox_inches='tight')
    plt.show()
