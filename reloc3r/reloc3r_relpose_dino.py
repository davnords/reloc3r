from copy import deepcopy
import os
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial
import reloc3r.utils.path_to_croco
from reloc3r.patch_embed import ManyAR_PatchEmbed
from models.pos_embed import RoPE2D 
from models.blocks import Block, DecoderBlock
from models.croco import CroCoNet
from reloc3r.pose_head import PoseHead
from reloc3r.utils.misc import freeze_all_params, transpose_to_landscape
from pdb import set_trace as bb
from huggingface_hub import PyTorchModelHubMixin
from typing import Literal


# parts of the code adapted from 
# 'https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/croco.py#L21'
# 'https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/model.py#L46'
class Reloc3rRelposeDINO(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 img_size=512,          # input image size
                 patch_size=16,         # patch_size 
                 dec_embed_dim=768,     # decoder feature dimension 
                 dec_depth=12,          # decoder depth 
                 dec_num_heads=12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                 vit: Literal["dinov3", "mum"] = "dinov3",
                ):   
        super(Reloc3rRelposeDINO, self).__init__()

        # patchify and positional embedding
        enc_embed_dim=1024
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        # self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        # pose regression head
        self.pose_head = PoseHead(net=self)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        self.initialize_weights() 

        if vit == 'dinov3':
            print('Loading DINOv3 ViT-L/16 model...')
            # TODO: Split into model initialization and then model weight loading
            # self.encoder = torch.hub.load("/mimer/NOBACKUP/groups/snic2022-6-266/davnords/dinov3", "dinov3_vitl16", source='local', weights="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
            model = torch.hub.load(
                "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/dinov3",
                "dinov3_vitl16",
                source="local",
                pretrained=False  # ensure no weights are loaded
            )
            model.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
            ckpt_path = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print(model.load_state_dict(ckpt, strict=True))
            self.encoder = model
        elif vit == 'mum':
            print('Loading MuM ViT-L/16 model...')
            from mum.model import vit_large
            model = vit_large().eval()
            model.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
            pretrained_weights = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/MuM_ViTLarge_BaseDecoder_500k.pth"
            ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
            print(model.load_state_dict(ckpt['model'], strict=True))
            self.encoder = model
        else:
            raise NotImplementedError(f'ViT {vit} not implemented')

        self.encoder.eval()
        # self.pos_getter = PositionGetter(patch_size=self.patch_size)

    def initialize_weights(self):
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def freeze_encoder(self):
        freeze_all_params([
            self.encoder
        ])

    def load_state_dict(self, ckpt, **kw):
        return super().load_state_dict(ckpt, **kw)

    def _encode_image(self, image, true_shape):
        # TODO: Implement this one based on DINOv3 / MuM encoding
        B, C, H, W = image.shape
        x, pos = self.encoder.patch_embed(image, true_shape=true_shape)
        
        patch_size = self.encoder.patch_size
        assert H%patch_size==0 and W%patch_size==0, "Patch size must evenly divide image"

        rope_sincos = self.encoder.rope_embed(H=H//patch_size, W=W//patch_size)

        # height, width = true_shape.T
        # is_landscape = (width >= height)
        # is_portrait = ~is_landscape
        # sin_landscape, cos_landscape = self.encoder.rope_embed(H=H//patch_size, W=W//patch_size)
        # sin_portrait, cos_portrait   = self.encoder.rope_embed(H=W//patch_size, W=H//patch_size)  # swapped

        # sin_list, cos_list = [], []
        # for i in range(B):
        #     if is_landscape[i]:
        #         sin_list.append(sin_landscape)
        #         cos_list.append(cos_landscape)
        #     else:
        #         sin_list.append(sin_portrait)
        #         cos_list.append(cos_portrait)

        # sin = torch.stack(sin_list, dim=0)  # (B, n_tokens, dim)
        # cos = torch.stack(cos_list, dim=0)  # (B, n_tokens, dim)
        # sin, cos = sin.unsqueeze(1).repeat(1, self.encoder.num_heads, 1, 1), cos.unsqueeze(1).repeat(1, self.encoder.num_heads, 1, 1)
        # rope_sincos = (sin, cos)

        cls_tokens = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder.blocks:
            x = blk(x, rope_sincos)
        x_norm = self.encoder.norm(x)
        return x_norm[:, 1:], pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encoder(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk in self.dec_blocks:
            # img1 side
            f1, _ = blk(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape):
        B, S, D = decout[-1].shape
        return self.head(decout, img_shape)

    def forward(self, view1, view2):
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(view1, view2)  # B,S,D

        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            pose1 = self._downstream_head([tok.float() for tok in dec1], shape1)  
            pose2 = self._downstream_head([tok.float() for tok in dec2], shape2)  # relative camera pose from 2 to 1. 
            
        return pose1, pose2


def setup_reloc3r_relpose_model(model_args, device):
    if '224' in model_args:
        ckpt_path = 'siyan824/reloc3r-224'
    elif '512' in model_args:
        ckpt_path = 'siyan824/reloc3r-512'
    reloc3r_relpose = Reloc3rRelpose.from_pretrained(ckpt_path)
    reloc3r_relpose.to(device)
    reloc3r_relpose.eval()
    print('Model loaded from ', ckpt_path)
    return reloc3r_relpose


@torch.no_grad()
def inference_relpose(batch, model, device, use_amp=False): 
    # to device. 
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split():  
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    # forward. 
    view1, view2 = batch
    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        _, pose2 = model(view1, view2)
    pose2to1 = pose2["pose"]
    return pose2to1