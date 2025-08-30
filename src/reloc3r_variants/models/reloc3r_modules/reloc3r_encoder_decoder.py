import third_party.reloc3r.reloc3r.utils.path_to_croco
from models.pos_embed import RoPE2D # from croco
from models.blocks import Block, DecoderBlock # from croco

import src.reloc3r_variants.models.reloc3r_modules.path_to_reloc3r
from third_party.reloc3r.reloc3r.patch_embed import ManyAR_PatchEmbed
from third_party.reloc3r.reloc3r.reloc3r_relpose import Reloc3rRelpose
from third_party.reloc3r.reloc3r.pose_head import PoseHead
from third_party.reloc3r.reloc3r.utils.misc import transpose_to_landscape



import torch
from functools import partial
import torch.nn as nn


class Reloc3r_Encoder_Decoder(nn.Module):
    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 enc_embed_dim=1024,
                 enc_depth=24,
                 enc_num_heads=16,
                 dec_embed_dim=768,
                 dec_depth=12,
                 dec_num_heads=12,
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,
                 pos_embed='RoPE100'):
        super(Reloc3r_Encoder_Decoder, self).__init__()

        # patchify and positional embedding
        self.patch_embed = ManyAR_PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        self.pos_embed = pos_embed
        self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
        self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
        if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
        freq = float(pos_embed[len('RoPE'):])
        self.rope = RoPE2D(freq=freq)

        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)

        # ViT decoder
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)  # transfer from encoder to decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)

        self.initialize_weights()

        # record some parameters
        h, w = int(img_size * 0.75), img_size
        self.patch_grid_size = (h // patch_size, w // patch_size)
        self.Np = self.patch_grid_size[0] * self.patch_grid_size[1]

    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
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

    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
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


    def _encoder(self, img1, img2):
        B = img1.shape[0]
        shape1 = torch.tensor(img1.shape[-2:])[None].repeat(B, 1)
        shape2 = torch.tensor(img2.shape[-2:])[None].repeat(B, 1)

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

    def forward(self,
                img1: torch.Tensor,
                img2: torch.Tensor,):
        """
        :param img1: image 1, shape (B, 3, H, W)
        :param img2: image 2, shape (B, 3, H, W)
        """
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encoder(img1, img2)
        
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        tokens1 = dec1[-1]
        tokens2 = dec2[-1]

        return tokens1, tokens2


def build_reloc3r_encoder_decoder(model="Reloc3r_Encoder_Decoder(img_size=512)",
                                  ckpt_path=None,
                                  device=None,):
    enc_dec = eval(model)
    enc_dec.to(device)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "model" in checkpoint:
            checkpoint = checkpoint[:model]
        if "DUST3R" in ckpt_path:
            # Initialized from DUST3R
            modified_ckpt = {}
            for k, v in checkpoint.items():
                if k.startwith("dec_blocks."):
                    continue
                new_k = k.replace("dec_blocks2", "dec_blocks", 1)
                modified_ckpt[new_k] = v
            checkpoint = modified_ckpt

        output = enc_dec.load_state_dict(checkpoint, strict=False)
        print("Following weights are found in the checkpoint, but not loaded to Reloc3r model")
        print(output.unexpected_keys)
        del checkpoint
        torch.cuda.empty_cache()
        enc_dec.eval()
        return enc_dec
    else:
        return enc_dec
    

class PoseHead_Almost(PoseHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)

        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)

        feat = self.more_mlps(feat)  # [B, D_]

        return {
            "features": feat,
            "pose": torch.empty(1),  # dummy, used to trick tranposed function from reloc3r
        }


class Reloc3rRelpose_Almost(Reloc3rRelpose):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_head = PoseHead_Almost(net=self)
        self.head = transpose_to_landscape(self.pose_head, activate=True)

        self.initialize_weights()

    def forward(self, view1, view2):
        return super().forward(view1, view2)
        
