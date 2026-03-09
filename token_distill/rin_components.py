from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from .blocks import RINBlock
from .utils import (
    LayerNormZero,
    WeightTiedHead,
    init_pos_embed,
    init_weights,
    apply_random_masking,
)


@dataclass
class EncoderOutput:
    latent: torch.Tensor
    repa_feat: torch.Tensor
    write_attn_weights: torch.Tensor
    read_attn_weights: torch.Tensor
    masks: Optional[torch.Tensor] = None


@dataclass
class DecoderOutput:
    decoder_2d_logits: torch.Tensor
    latent: torch.Tensor
    write_attn_weights: torch.Tensor
    read_attn_weights: torch.Tensor
    repa_feat: torch.Tensor


class RINEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_2d_tokens,
        num_1d_tokens,
        patch_size,
        process_attn_depth,
        num_blocks,
        num_registers=16,
        final_norm=True,
        final_norm_context=True,
        out_proj_dim=None,
        update_with_cold_latent=True,
        use_slot_write_attn=False,
        enable_masking=False,
        masking_prob=0.9,
        min_masking_ratio=0.0,
        max_masking_ratio=0.6,
    ):
        print("instantiating RINEncoder")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.process_attn_depth = process_attn_depth
        self.num_blocks = num_blocks
        self.final_norm = final_norm
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers

        self.pos_embed_enc = init_pos_embed([num_2d_tokens], in_dim)
        self.patch_enc = nn.Sequential(
            nn.Conv2d(
                3,
                in_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.patch_enc.apply(init_weights)
        self.latent_2d_ln = nn.LayerNorm(self.patch_enc[0].out_channels)

        self.blocks = nn.ModuleList(
            [
                RINBlock(
                    in_dim,
                    latent_dim,
                    process_attn_depth,
                    final_norm=(i == num_blocks - 1) and final_norm,
                    final_norm_context=(i == num_blocks - 1) and final_norm_context,
                    use_slot_write_attn=use_slot_write_attn,
                )
                for i in range(num_blocks)
            ]
        )

        self.latent_conditioning = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, latent_dim)
        )
        self.update_with_cold_latent = update_with_cold_latent

        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio
        if enable_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))
            nn.init.trunc_normal_(self.mask_token, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        self.apply(init_weights)

    def apply_random_masking(self, x):
        masked_x, mask = apply_random_masking(
            x=x,
            mask_token=self.mask_token,
            enable_masking=self.enable_masking,
            training=self.training,
            masking_prob=self.masking_prob,
            min_masking_ratio=self.min_masking_ratio,
            max_masking_ratio=self.max_masking_ratio,
        )
        return masked_x, mask

    def forward(self, x, return_attn_weights=False):
        x = self.patch_enc(x)
        masks = None
        if self.enable_masking:
            x, masks = self.apply_random_masking(x)
        x = self.latent_2d_ln(x + self.pos_embed_enc)
        latent = self.latent_conditioning.expand(x.shape[0], -1, -1)

        for block in self.blocks:
            latent, x, write_attn_weights, read_attn_weights = block(
                latent, x, return_attn_weights
            )

        latent = self.out_proj(F.silu(latent))  # following MAGVIT
        latent = latent[:, : self.num_1d_tokens]

        return EncoderOutput(
            latent=latent,
            repa_feat=x,
            write_attn_weights=write_attn_weights,
            read_attn_weights=read_attn_weights,
            masks=masks,
        )


class RINDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        target_codebook_size,
        process_attn_depth,
        num_blocks,
        num_2d_tokens,
        num_1d_tokens,
        final_norm=False,
        final_norm_context=True,
        in_proj_dim=None,
        repa_target_block_num=None,
    ):
        print("instantiating RINDecoder")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.target_codebook_size = target_codebook_size
        self.process_attn_depth = process_attn_depth
        self.num_blocks = num_blocks
        self.final_norm = final_norm
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.repa_target_block_num = (
            num_blocks - 1 if repa_target_block_num is None else repa_target_block_num
        )

        self.blocks = nn.ModuleList(
            [
                RINBlock(
                    in_dim,
                    latent_dim,
                    process_attn_depth,
                    final_norm=(i == num_blocks - 1) and final_norm,
                    final_norm_context=(i == num_blocks - 1) and final_norm_context,
                )
                for i in range(num_blocks)
            ]
        )

        self.in_proj_dim = latent_dim if in_proj_dim is None else in_proj_dim
        self.in_proj = nn.Linear(self.in_proj_dim, latent_dim)

        self.cond_cross_attn = None
        self.latent_mlp = None
        self.latent_layer_norm = None

        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.latent_layer_norm = LayerNormZero(latent_dim)

        self.latent_init = nn.Parameter(torch.zeros(1, num_1d_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.latent_init, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        self.pos_embed_dec = init_pos_embed([num_2d_tokens], in_dim)
        self.pos_embed_dec_latent = init_pos_embed([num_1d_tokens], latent_dim)
        self.embeddings = nn.Embedding(target_codebook_size + 1, in_dim)
        self.classification_head = WeightTiedHead(self.embeddings, target_codebook_size)

        self.apply(init_weights)

    def update_latent(self, latent, latent_1d_quantized):
        latent = self.latent_layer_norm(self.latent_mlp(latent))
        latent = latent + self.latent_init
        latent = torch.cat([latent, latent_1d_quantized], dim=1)
        return latent

    def forward(
        self, x, latent_1d_quantized, prev_latent=None, return_attn_weights=False
    ):
        x = self.embeddings(x)
        x = x + self.pos_embed_dec

        latent_1d_quantized = self.in_proj(latent_1d_quantized)
        latent_1d_quantized = latent_1d_quantized + self.pos_embed_dec_latent

        if prev_latent is not None:
            latent = self.update_latent(prev_latent, latent_1d_quantized)
        else:
            latent = self.update_latent(
                torch.zeros_like(latent_1d_quantized), latent_1d_quantized
            )

        for i, block in enumerate(self.blocks):
            latent, x, write_attn_weights, read_attn_weights = block(
                latent, x, return_attn_weights
            )
            if i == self.repa_target_block_num:
                repa_feat = x

        latent = latent[:, : self.num_1d_tokens]

        return DecoderOutput(
            decoder_2d_logits=self.classification_head(x),
            latent=latent,
            write_attn_weights=write_attn_weights,
            read_attn_weights=read_attn_weights,
            repa_feat=repa_feat,
        )
