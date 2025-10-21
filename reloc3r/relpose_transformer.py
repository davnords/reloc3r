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
class RelposeTransformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 patch_size=16,         # patch_size 
                 dec_embed_dim=768,     # decoder feature dimension 
                 dec_depth=12,          # decoder depth 
                 dec_num_heads=12,      # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder 
                 pos_embed='RoPE100',   # positional embedding (either cosine or RoPE100)
                 vit: Literal["dust3r", "crocov2", "dinov3", "mum"] = "dinov3",
                ):   
        super(RelposeTransformer, self).__init__()

        # patchify and positional embedding
        enc_embed_dim=1024
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)

        self.position_getter = PositionGetter()

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
            model = torch.hub.load("/mimer/NOBACKUP/groups/snic2022-6-266/davnords/dinov3", "dinov3_vitl16", source='local', weights="/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        elif vit == 'dust3r' or vit == 'crocov2':
            if vit == "dust3r":
                print('Loading DUSt3R ViT-L/16 model...')
                weight_path = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            else:
                print('Loading CroCo V2 ViT-L/16 model...')
                weight_path = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth"

            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            croco_kwargs = {'enc_embed_dim': 1024, 'enc_depth': 24, 'enc_num_heads': 16, 'dec_embed_dim': 768, 'dec_num_heads': 12, 'dec_depth': 12, 'pos_embed': 'RoPE100'}
            model = CroCoNet(**croco_kwargs)
            print(model.load_state_dict(ckpt['model'], strict=False))
        elif vit == 'mum':
            print('Loading MuM ViT-L/16 model...')
            from mum.model import vit_large
            model = vit_large().eval()
            pretrained_weights = "/mimer/NOBACKUP/groups/snic2022-6-266/davnords/mv-ssl/pretrained_models/MuM_ViTLarge_BaseDecoder_500k.pth"
            ckpt = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
            print(model.load_state_dict(ckpt['model'], strict=True))
        else:
            raise NotImplementedError(f'ViT {vit} not implemented')
        
        self.encoder = model
        self.encoder.eval()

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
        B, C, H, W = image.shape

        assert H%self.patch_size == 0 and W%self.patch_size == 0, f'Image size {(H,W)} not divisible by patch size {self.patch_size}'
        pos = self.position_getter(B, H//self.patch_size, W//self.patch_size, image.device)
        x = self.encoder.forward_features(image)['x_norm_patchtokens']
        return x, pos, None

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


class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos