from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from diffusers.models.attention import JointTransformerBlock

from .blocks import TransformerLayer
from .utils import (
    LayerNormZero,
    WeightTiedHead,
    init_pos_embed,
    init_weights,
    apply_random_masking,
    apply_frequency_cutoff,
    IndicesToToken,
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
    intermediate_features: Optional[torch.Tensor] = None


class ViTEncoder(nn.Module):
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
        enable_freq_cutoff=False,
        cutoff_downsample_factor=2,
        cutoff_prob=0.5,
    ):
        print("instantiating ViTEncoder")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.process_attn_depth = process_attn_depth
        self.num_blocks = num_blocks
        self.final_norm = final_norm
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers

        # Image patch encoding (same as RINEncoder)
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

        assert latent_dim == in_dim, (
            "Latent dimension must be the same as input dimension"
        )

        # Transformer blocks for self-attention on concatenated sequence
        self.blocks = nn.ModuleList(
            [TransformerLayer(in_dim, norm_context=False) for _ in range(num_blocks)]
        )

        self.latent_conditioning = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, latent_dim)
        )
        self.update_with_cold_latent = update_with_cold_latent

        self.last_ln = nn.LayerNorm(latent_dim)

        # Output projection
        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio

        # Frequency cutoff parameters
        self.enable_freq_cutoff = enable_freq_cutoff
        self.cutoff_downsample_factor = cutoff_downsample_factor
        self.cutoff_prob = cutoff_prob
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
        # Apply frequency cutoff in pixel space (if enabled)
        x = apply_frequency_cutoff(
            x,
            enable_freq_cutoff=self.enable_freq_cutoff,
            training=self.training,
            cutoff_downsample_factor=self.cutoff_downsample_factor,
            cutoff_prob=self.cutoff_prob,
        )

        # Process image patches
        x = self.patch_enc(x)

        masks = None
        if self.enable_masking:
            x, masks = self.apply_random_masking(x)
        x = self.latent_2d_ln(x + self.pos_embed_enc)

        # Initialize latent
        latent = self.latent_conditioning.expand(x.shape[0], -1, -1)

        # Concatenate latent tokens with image patches (latent first)
        combined = torch.cat([latent, x], dim=1)

        # Process through transformer blocks
        for block in self.blocks:
            combined, attn_weights = block(
                combined, return_attn_weights=return_attn_weights
            )

        # Split back to latent and image portions
        latent_out, x = (
            combined[:, : self.num_1d_tokens],
            combined[:, self.num_1d_tokens + self.num_registers :],
        )

        # Apply output projection
        latent_out = self.last_ln(latent_out)
        latent = self.out_proj(F.silu(latent_out))

        return EncoderOutput(
            latent=latent,
            repa_feat=x,
            write_attn_weights=attn_weights if return_attn_weights else None,
            read_attn_weights=attn_weights if return_attn_weights else None,
            masks=masks,
        )


class DINOv3EncoderQFormer(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_2d_tokens,
        num_1d_tokens,
        dinov3_model_name="dinov3_vitb16",
        dinov3_weights_path=None,
        num_registers=16,
        final_norm=True,
        final_norm_context=True,
        out_proj_dim=None,
        enable_masking=False,
        masking_prob=0.9,
        min_masking_ratio=0.0,
        max_masking_ratio=0.6,
        freeze_dinov3=True,
        num_qformer_layers=2,
        num_attention_heads=8,
    ):
        print("instantiating DINOv3EncoderQFormer")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers
        self.freeze_dinov3 = freeze_dinov3
        self.num_qformer_layers = num_qformer_layers
        self.num_attention_heads = num_attention_heads

        # Initialize DINOv3 model (following rep_guidance.py pattern)
        if dinov3_weights_path is None:
            raise ValueError("No pretrained weights specified for DINOv3")

        # Get absolute path to dinov3 directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")

        self.dinov3_model = torch.hub.load(
            repo_or_dir=dinov3_path,
            model=dinov3_model_name,
            source="local",
            weights=dinov3_weights_path,
        )

        # Freeze DINOv3 parameters if specified
        if self.freeze_dinov3:
            for param in self.dinov3_model.parameters():
                param.requires_grad = False
            self.dinov3_model.eval()

        # Get DINOv3 feature dimension
        self.dinov3_dim = self.dinov3_model.embed_dim

        assert latent_dim == self.dinov3_dim, (
            f"Latent dimension ({latent_dim}) must match DINOv3 dimension ({self.dinov3_dim})"
        )

        # Initialize learnable query tokens for QFormer
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, self.dinov3_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Initialize learnable mask token for masking DINOv3 features
        if enable_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dinov3_dim))
            nn.init.trunc_normal_(self.mask_token, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)
        else:
            self.mask_token = None

        # QFormer layers - cross attention transformer
        self.qformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.dinov3_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=self.dinov3_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_qformer_layers)
            ]
        )

        # Output projection
        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        if self.out_proj_dim != latent_dim:
            self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)
        else:
            self.out_proj = nn.Identity()

        # Final normalization
        self.final_norm_layer = (
            nn.LayerNorm(self.dinov3_dim) if final_norm else nn.Identity()
        )

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio

        # Initialize new parameters (not DINOv3)
        if hasattr(self.out_proj, "apply"):
            self.out_proj.apply(init_weights)
        self.final_norm_layer.apply(init_weights)

    def forward(self, x, return_attn_weights=False):
        batch_size = x.shape[0]

        # Generate masks if masking is enabled
        masks = None
        if self.enable_masking and self.training:
            # Generate random masks for patches
            B, _, H, W = x.shape
            num_patches = (H // self.dinov3_model.patch_size) * (
                W // self.dinov3_model.patch_size
            )

            # Randomly decide masking ratio for this batch
            masking_ratio = (
                torch.rand(1).item() * (self.max_masking_ratio - self.min_masking_ratio)
                + self.min_masking_ratio
            )
            num_masked = int(masking_ratio * num_patches)

            # Create boolean mask
            masks = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
            for i in range(B):
                if torch.rand(1).item() < self.masking_prob:
                    indices = torch.randperm(num_patches, device=x.device)[:num_masked]
                    masks[i, indices] = True

        # Use DINOv3's built-in interface to get features
        if self.freeze_dinov3:
            with torch.no_grad():
                # Get intermediate features from DINOv3
                dinov3_features = self.dinov3_model.get_intermediate_layers(x, n=1)[0]
        else:
            dinov3_features = self.dinov3_model.get_intermediate_layers(x, n=1)[0]

        # Apply masking to DINOv3 features if enabled
        if masks is not None:
            # Use learnable mask token for masked positions
            mask_tokens = self.mask_token.expand_as(dinov3_features)
            # Apply masking
            dinov3_features = torch.where(
                masks.unsqueeze(-1).expand_as(dinov3_features),
                mask_tokens,
                dinov3_features,
            )

        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Apply QFormer layers (cross-attention from queries to DINOv3 features)
        latent_features = query_tokens
        for layer in self.qformer_layers:
            latent_features = layer(
                tgt=latent_features,  # Query tokens
                memory=dinov3_features,  # DINOv3 features as memory
            )

        # Apply final normalization
        latent_features = self.final_norm_layer(latent_features)

        # Apply output projection
        latent_features = self.out_proj(latent_features)

        # Extract the actual latent tokens (without registers)
        latent_out = latent_features[:, : self.num_1d_tokens]

        return EncoderOutput(
            latent=latent_out,
            repa_feat=dinov3_features,  # Use DINOv3 features as repa_feat
            write_attn_weights=None,  # QFormer doesn't return attention weights in this simple setup
            read_attn_weights=None,
            masks=masks,
        )


class MAEEncoderQFormer(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_2d_tokens,
        num_1d_tokens,
        mae_model_name="facebook/vit-mae-base",
        num_registers=16,
        final_norm=True,
        final_norm_context=True,
        out_proj_dim=None,
        enable_masking=False,
        masking_prob=0.9,
        min_masking_ratio=0.0,
        max_masking_ratio=0.6,
        num_qformer_layers=2,
        num_attention_heads=8,
    ):
        print("instantiating MAEEncoderQFormer")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers
        self.num_qformer_layers = num_qformer_layers
        self.num_attention_heads = num_attention_heads

        # Initialize MAE model
        from .mae import MAEwNorm

        self.mae_model = MAEwNorm(model_name=mae_model_name)

        # Get MAE feature dimension
        self.mae_dim = self.mae_model.hidden_size

        assert latent_dim == self.mae_dim, (
            f"Latent dimension ({latent_dim}) must match MAE dimension ({self.mae_dim})"
        )

        # Initialize learnable query tokens for QFormer
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, self.mae_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Initialize learnable mask token for masking MAE features
        if enable_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.mae_dim))
            nn.init.trunc_normal_(self.mask_token, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)
        else:
            self.mask_token = None

        # QFormer layers - cross attention transformer
        self.qformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.mae_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=self.mae_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_qformer_layers)
            ]
        )

        # Output projection
        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        if self.out_proj_dim != latent_dim:
            self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)
        else:
            self.out_proj = nn.Identity()

        # Final normalization
        self.final_norm_layer = (
            nn.LayerNorm(self.mae_dim) if final_norm else nn.Identity()
        )

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio

        # Initialize new parameters (not MAE)
        if hasattr(self.out_proj, "apply"):
            self.out_proj.apply(init_weights)
        self.final_norm_layer.apply(init_weights)

        for p in self.mae_model.parameters():
            p.requires_grad = False
        self.mae_model.eval()

    def forward(self, x, return_attn_weights=False):
        batch_size = x.shape[0]

        # Generate masks if masking is enabled
        masks = None
        if self.enable_masking and self.training:
            # Generate random masks for patches
            B, _, H, W = x.shape
            num_patches = (H // self.mae_model.patch_size) * (
                W // self.mae_model.patch_size
            )

            # Randomly decide masking ratio for this batch
            masking_ratio = (
                torch.rand(1).item() * (self.max_masking_ratio - self.min_masking_ratio)
                + self.min_masking_ratio
            )
            num_masked = int(masking_ratio * num_patches)

            # Create boolean mask
            masks = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
            for i in range(B):
                if torch.rand(1).item() < self.masking_prob:
                    indices = torch.randperm(num_patches, device=x.device)[:num_masked]
                    masks[i, indices] = True

        # Extract MAE features (always frozen, no grad context needed - handled by MAEwNorm)
        mae_features = self.mae_model(x)  # Shape: (B, num_patches, mae_dim)
        # Note: MAE already removes CLS token

        # Apply masking to MAE features if enabled
        if masks is not None:
            # Use learnable mask token for masked positions
            mask_tokens = self.mask_token.expand_as(mae_features)
            # Apply masking
            mae_features = torch.where(
                masks.unsqueeze(-1).expand_as(mae_features), mask_tokens, mae_features
            )

        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Apply QFormer layers (cross-attention from queries to MAE features)
        latent_features = query_tokens
        for layer in self.qformer_layers:
            latent_features = layer(
                tgt=latent_features,  # Query tokens
                memory=mae_features,  # MAE features as memory
            )

        # Apply final normalization
        latent_features = self.final_norm_layer(latent_features)

        # Apply output projection
        latent_features = self.out_proj(latent_features)

        # Extract the actual latent tokens (without registers)
        latent_out = latent_features[:, : self.num_1d_tokens]

        return EncoderOutput(
            latent=latent_out,
            repa_feat=mae_features,  # Use MAE features as repa_feat
            write_attn_weights=None,  # QFormer doesn't return attention weights in this simple setup
            read_attn_weights=None,
            masks=masks,
        )


class SigLIPEncoderQFormer(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_2d_tokens,
        num_1d_tokens,
        siglip_model_name="google/siglip-base-patch16-224",
        num_registers=16,
        final_norm=True,
        final_norm_context=True,
        out_proj_dim=None,
        enable_masking=False,
        masking_prob=0.9,
        min_masking_ratio=0.0,
        max_masking_ratio=0.6,
        num_qformer_layers=2,
        num_attention_heads=8,
    ):
        print("instantiating SigLIPEncoderQFormer")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers
        self.num_qformer_layers = num_qformer_layers
        self.num_attention_heads = num_attention_heads

        # Initialize SigLIP model
        from .siglipv2 import SigLIP2wNorm

        self.siglip_model = SigLIP2wNorm(
            model_name=siglip_model_name, num_tokens=num_2d_tokens
        )

        # Get SigLIP feature dimension
        self.siglip_dim = self.siglip_model.hidden_size

        assert latent_dim == self.siglip_dim, (
            f"Latent dimension ({latent_dim}) must match SigLIP dimension ({self.siglip_dim})"
        )

        # Initialize learnable query tokens for QFormer
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, self.siglip_dim)
        )
        nn.init.trunc_normal_(self.query_tokens, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Initialize learnable mask token for masking SigLIP features
        if enable_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.siglip_dim))
            nn.init.trunc_normal_(self.mask_token, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)
        else:
            self.mask_token = None

        # QFormer layers - cross attention transformer
        self.qformer_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.siglip_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=self.siglip_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(num_qformer_layers)
            ]
        )

        # Output projection
        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        if self.out_proj_dim != latent_dim:
            self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)
        else:
            self.out_proj = nn.Identity()

        # Final normalization
        self.final_norm_layer = (
            nn.LayerNorm(self.siglip_dim) if final_norm else nn.Identity()
        )

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio

        # Initialize new parameters (not SigLIP)
        if hasattr(self.out_proj, "apply"):
            self.out_proj.apply(init_weights)
        self.final_norm_layer.apply(init_weights)

        for p in self.siglip_model.parameters():
            p.requires_grad = False
        self.siglip_model.eval()

    def forward(self, x, return_attn_weights=False):
        batch_size = x.shape[0]

        # Generate masks if masking is enabled
        masks = None
        if self.enable_masking and self.training:
            # Generate random masks for patches
            B, _, H, W = x.shape
            num_patches = (H // self.siglip_model.patch_size) * (
                W // self.siglip_model.patch_size
            )

            # Randomly decide masking ratio for this batch
            masking_ratio = (
                torch.rand(1).item() * (self.max_masking_ratio - self.min_masking_ratio)
                + self.min_masking_ratio
            )
            num_masked = int(masking_ratio * num_patches)

            # Create boolean mask
            masks = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
            for i in range(B):
                if torch.rand(1).item() < self.masking_prob:
                    indices = torch.randperm(num_patches, device=x.device)[:num_masked]
                    masks[i, indices] = True

        # Extract SigLIP features (always frozen, no grad context needed - handled by SigLIP2wNorm)
        siglip_features = self.siglip_model(x)  # Shape: (B, num_patches, siglip_dim)

        # Apply masking to SigLIP features if enabled
        if masks is not None:
            # Use learnable mask token for masked positions
            mask_tokens = self.mask_token.expand_as(siglip_features)
            # Apply masking
            siglip_features = torch.where(
                masks.unsqueeze(-1).expand_as(siglip_features),
                mask_tokens,
                siglip_features,
            )

        # Expand query tokens for the batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)

        # Apply QFormer layers (cross-attention from queries to SigLIP features)
        latent_features = query_tokens
        for layer in self.qformer_layers:
            latent_features = layer(
                tgt=latent_features,  # Query tokens
                memory=siglip_features,  # SigLIP features as memory
            )

        # Apply final normalization
        latent_features = self.final_norm_layer(latent_features)

        # Apply output projection
        latent_features = self.out_proj(latent_features)

        # Extract the actual latent tokens (without registers)
        latent_out = latent_features[:, : self.num_1d_tokens]

        return EncoderOutput(
            latent=latent_out,
            repa_feat=siglip_features,  # Use SigLIP features as repa_feat
            write_attn_weights=None,  # QFormer doesn't return attention weights in this simple setup
            read_attn_weights=None,
            masks=masks,
        )


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        num_2d_tokens,
        num_1d_tokens,
        dinov3_model_name="dinov3_vitb16",
        dinov3_weights_path=None,
        num_registers=16,
        final_norm=True,
        final_norm_context=True,
        out_proj_dim=None,
        enable_masking=False,
        masking_prob=0.9,
        min_masking_ratio=0.0,
        max_masking_ratio=0.6,
        freeze_dinov3=True,
        separate_patch_norm=False,
    ):
        print("instantiating DINOv3Encoder")
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.num_2d_tokens = num_2d_tokens
        self.num_1d_tokens = num_1d_tokens
        self.num_registers = num_registers
        self.freeze_dinov3 = freeze_dinov3
        self.separate_patch_norm = separate_patch_norm

        # Initialize weights if no pretrained weights specified
        if dinov3_weights_path is None:
            raise ValueError("No pretrained weights specified for DINOv3")

        # Get absolute path to dinov3 directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")

        self.dinov3_model = torch.hub.load(
            repo_or_dir=dinov3_path,
            model=dinov3_model_name,
            source="local",
            weights=dinov3_weights_path,
        )

        # Freeze DINOv3 parameters if specified
        if self.freeze_dinov3:
            for param in self.dinov3_model.parameters():
                param.requires_grad = False
            self.dinov3_model.eval()

        # Get DINOv3 feature dimension (768 for ViT-B, 1024 for ViT-L, etc.)
        self.dinov3_dim = self.dinov3_model.embed_dim

        assert latent_dim == self.dinov3_dim, (
            f"Latent dimension ({latent_dim}) must match DINOv3 dimension ({self.dinov3_dim})"
        )

        # Initialize learnable latent tokens
        self.latent_tokens = nn.Parameter(
            torch.zeros(1, num_1d_tokens + num_registers, self.dinov3_dim)
        )
        nn.init.trunc_normal_(self.latent_tokens, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Output projection
        self.out_proj_dim = latent_dim if out_proj_dim is None else out_proj_dim
        if self.out_proj_dim != latent_dim:
            self.out_proj = nn.Linear(latent_dim, self.out_proj_dim)
        else:
            self.out_proj = nn.Identity()

        self.patch_norm = None
        if self.separate_patch_norm:
            self.patch_norm = nn.LayerNorm(self.dinov3_dim)
            self.patch_norm.apply(init_weights)

        # Random masking parameters
        self.enable_masking = enable_masking
        self.masking_prob = masking_prob
        self.min_masking_ratio = min_masking_ratio
        self.max_masking_ratio = max_masking_ratio

        # Initialize new parameters (not DINOv3)
        if hasattr(self.out_proj, "apply"):
            self.out_proj.apply(init_weights)

        self.dinov3_model.norm.apply(init_weights)  # reinitialize last norm layer

    def forward(self, x, return_attn_weights=False):
        batch_size = x.shape[0]

        # Generate masks if masking is enabled
        masks = None
        if self.enable_masking and self.training:
            # Generate random masks for patches
            B, _, H, W = x.shape
            num_patches = (H // self.dinov3_model.patch_size) * (
                W // self.dinov3_model.patch_size
            )

            # Randomly decide masking ratio for this batch
            masking_ratio = (
                torch.rand(1).item() * (self.max_masking_ratio - self.min_masking_ratio)
                + self.min_masking_ratio
            )
            num_masked = int(masking_ratio * num_patches)

            # Create boolean mask
            masks = torch.zeros(B, num_patches, dtype=torch.bool, device=x.device)
            for i in range(B):
                if torch.rand(1).item() < self.masking_prob:
                    indices = torch.randperm(num_patches, device=x.device)[:num_masked]
                    masks[i, indices] = True

        # Use DINOv3's prepare_tokens_with_masks which handles patch embedding + masking
        # This prepends CLS token and storage tokens which are essential for DINOv3's pretrained weights
        tokens, (H, W) = self.dinov3_model.prepare_tokens_with_masks(x, masks)

        # Expand learnable latent tokens for the batch
        latent_tokens = self.latent_tokens.expand(batch_size, -1, -1)

        # Concatenate our latent tokens with DINOv3's tokens (keeping CLS and storage tokens)
        # Format: [our_latent_tokens, CLS, storage_tokens, patch_features]
        combined = torch.cat([latent_tokens, tokens], dim=1)

        # Process through DINOv3's transformer blocks
        def process_blocks(x):
            for block in self.dinov3_model.blocks:
                # Get RoPE embeddings if the model uses them
                rope_sincos = (
                    self.dinov3_model.rope_embed(H=H, W=W)
                    if hasattr(self.dinov3_model, "rope_embed")
                    and self.dinov3_model.rope_embed is not None
                    else None
                )
                x = block(x, rope_sincos)
            return x

        if self.freeze_dinov3:
            with torch.no_grad():
                combined = process_blocks(combined)
        else:
            combined = process_blocks(combined)

        # Split the output: our latent tokens and DINOv3's tokens
        our_latent_start = 0
        our_latent_end = self.num_1d_tokens + self.num_registers
        dinov3_start = our_latent_end
        dinov3_special_end = dinov3_start + 1 + self.dinov3_model.n_storage_tokens

        # Extract our latent tokens
        latent = combined[:, our_latent_start:our_latent_end]

        # Extract patch features (excluding CLS and storage tokens from DINOv3)
        patch_features_out = combined[:, dinov3_special_end:]

        latent = self.dinov3_model.norm(latent)
        latent = self.out_proj(latent)
        if self.separate_patch_norm:
            patch_features_out = self.patch_norm(patch_features_out)
        else:
            patch_features_out = self.dinov3_model.norm(patch_features_out)

        # Extract the actual latent tokens (without registers)
        latent_out = latent[:, : self.num_1d_tokens]

        return EncoderOutput(
            latent=latent_out,
            repa_feat=patch_features_out,
            write_attn_weights=None,  # DINOv3 doesn't return attention weights in this simple setup
            read_attn_weights=None,
            masks=masks,
        )


class ViTDecoder(nn.Module):
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
        print("instantiating ViTDecoder")
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

        # Transformer blocks for self-attention on concatenated sequence
        self.blocks = nn.ModuleList(
            [TransformerLayer(in_dim, norm_context=False) for _ in range(num_blocks)]
        )

        # Input projection for latent
        self.in_proj_dim = latent_dim if in_proj_dim is None else in_proj_dim
        self.in_proj = nn.Linear(self.in_proj_dim, latent_dim)

        assert latent_dim == in_dim, (
            "Latent dimension must be the same as input dimension"
        )

        # Latent update components
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.latent_layer_norm = LayerNormZero(latent_dim)

        # Initialize latent parameters
        self.latent_init = nn.Parameter(torch.zeros(1, num_1d_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.latent_init, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Positional embeddings
        self.pos_embed_dec = init_pos_embed([num_2d_tokens], in_dim)
        self.pos_embed_dec_latent = init_pos_embed([num_1d_tokens], latent_dim)

        # Token embeddings and classification head
        self.embeddings = nn.Embedding(target_codebook_size + 1, in_dim)
        self.classification_head = WeightTiedHead(self.embeddings, target_codebook_size)
        self.last_ln = nn.LayerNorm(latent_dim)

        self.apply(init_weights)

    def update_latent(self, latent, latent_1d_quantized):
        latent = self.latent_layer_norm(self.latent_mlp(latent) + latent)
        latent = latent + self.latent_init
        latent = torch.cat([latent, latent_1d_quantized], dim=1)
        return latent

    def forward(
        self,
        x,
        latent_1d_quantized,
        prev_latent=None,
        return_attn_weights=False,
        capture_intermediate=False,
        intermediate_block_idx=-1,
    ):
        # Process target tokens
        x = self.embeddings(x)
        x = x + self.pos_embed_dec

        # Process input latent
        latent_1d_quantized = self.in_proj(latent_1d_quantized)
        latent_1d_quantized = latent_1d_quantized + self.pos_embed_dec_latent

        # Update latent
        if prev_latent is not None:
            latent = self.update_latent(prev_latent, latent_1d_quantized)
        else:
            latent = self.update_latent(
                torch.zeros_like(latent_1d_quantized), latent_1d_quantized
            )

        # Concatenate latent tokens with target tokens (latent first)
        x_start_idx = latent.shape[1]
        combined = torch.cat([latent, x], dim=1)

        # Initialize intermediate features
        intermediate_features = None

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            combined, attn_weights = block(
                combined, return_attn_weights=return_attn_weights
            )

            # Capture intermediate features if requested
            if capture_intermediate:
                block_idx = (
                    intermediate_block_idx
                    if intermediate_block_idx >= 0
                    else self.num_blocks - 1
                )
                if i == block_idx:
                    # Extract target token features (excluding latent tokens)
                    intermediate_features = combined[:, x_start_idx:]

            # Extract repa_feat at specified block
            if i == self.repa_target_block_num:
                _, repa_feat = (
                    combined[:, : self.num_1d_tokens],
                    combined[:, self.num_1d_tokens :],
                )

        # Split back to latent and target portions
        latent, x = combined[:, : self.num_1d_tokens], combined[:, x_start_idx:]
        x = self.last_ln(x)

        return DecoderOutput(
            decoder_2d_logits=self.classification_head(x),
            latent=latent,
            write_attn_weights=attn_weights if return_attn_weights else None,
            read_attn_weights=attn_weights if return_attn_weights else None,
            repa_feat=repa_feat,
            intermediate_features=intermediate_features,
        )


class ViTDecoderWithoutSelfCondition(nn.Module):
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
        linear_indices_to_token=False,
    ):
        print("instantiating ViTDecoderWithoutSelfCondition")
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

        # Transformer blocks for self-attention on concatenated sequence
        self.blocks = nn.ModuleList(
            [TransformerLayer(in_dim, norm_context=False) for _ in range(num_blocks)]
        )

        # Input projection for latent
        self.in_proj_dim = latent_dim if in_proj_dim is None else in_proj_dim
        self.in_proj = nn.Linear(self.in_proj_dim, latent_dim)

        assert latent_dim == in_dim, (
            "Latent dimension must be the same as input dimension"
        )

        # Initialize latent parameters
        self.latent_init = nn.Parameter(torch.zeros(1, num_1d_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.latent_init, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Positional embeddings
        self.pos_embed_dec = init_pos_embed([num_2d_tokens], in_dim)
        self.pos_embed_dec_latent = init_pos_embed([num_1d_tokens], latent_dim)

        # Token embeddings and classification head
        if linear_indices_to_token:
            self.embeddings = IndicesToToken(target_codebook_size + 1, in_dim)
            self.classification_head = nn.Linear(in_dim, target_codebook_size)
        else:
            self.embeddings = nn.Embedding(target_codebook_size + 1, in_dim)
            self.classification_head = WeightTiedHead(
                self.embeddings, target_codebook_size
            )
        self.last_ln = nn.LayerNorm(latent_dim)
        self.apply(init_weights)

    def forward(
        self,
        x,
        latent_1d_quantized,
        return_attn_weights=False,
        capture_intermediate=False,
        intermediate_block_idx=-1,
    ):
        # Process target tokens
        x = self.embeddings(x)
        x = x + self.pos_embed_dec

        # Process input latent
        latent_1d_quantized = self.in_proj(latent_1d_quantized)
        latent_1d_quantized = latent_1d_quantized + self.pos_embed_dec_latent

        registers = self.latent_init.expand(x.shape[0], -1, -1)
        latent = torch.cat([latent_1d_quantized, registers], dim=1)

        # Concatenate latent tokens with target tokens (latent first)
        x_start_idx = latent.shape[1]
        combined = torch.cat([latent, x], dim=1)

        # Initialize intermediate features
        intermediate_features = None

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            combined, attn_weights = block(
                combined, return_attn_weights=return_attn_weights
            )

            # Capture intermediate features if requested
            if capture_intermediate:
                block_idx = (
                    intermediate_block_idx
                    if intermediate_block_idx >= 0
                    else self.num_blocks - 1
                )
                if i == block_idx:
                    # Extract target token features (excluding latent tokens)
                    intermediate_features = combined[:, x_start_idx:]

            # Extract repa_feat at specified block
            if i == self.repa_target_block_num:
                _, repa_feat = (
                    combined[:, : self.num_1d_tokens],
                    combined[:, self.num_1d_tokens :],
                )

        # Split back to latent and target portions
        latent, x = combined[:, : self.num_1d_tokens], combined[:, x_start_idx:]
        x = self.last_ln(x)

        return DecoderOutput(
            decoder_2d_logits=self.classification_head(x),
            latent=latent,
            write_attn_weights=attn_weights if return_attn_weights else None,
            read_attn_weights=attn_weights if return_attn_weights else None,
            repa_feat=repa_feat,
            intermediate_features=intermediate_features,
        )


class DiTDecoderBitwise(nn.Module):
    """
    MMDiT-based decoder using JointTransformerBlock from diffusers for joint attention processing.
    This version uses linear indices to token mapping (IndicesToToken) instead of embeddings.
    """

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
        num_attention_heads=8,
        attention_head_dim=None,
    ):
        print("instantiating MMDiT DiTDecoderBitwise")
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

        # Calculate attention dimensions
        if attention_head_dim is None:
            attention_head_dim = in_dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Input projection for latent
        self.in_proj_dim = latent_dim if in_proj_dim is None else in_proj_dim
        self.in_proj = nn.Linear(self.in_proj_dim, latent_dim)

        assert latent_dim == in_dim, (
            "Latent dimension must be the same as input dimension"
        )

        # MMDiT transformer blocks using JointTransformerBlock from diffusers
        self.blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=in_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i
                    == num_blocks - 1,  # Last block is context pre only
                )
                for i in range(num_blocks)
            ]
        )

        # Initialize latent parameters
        self.latent_init = nn.Parameter(torch.zeros(1, num_1d_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.latent_init, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Positional embeddings - make flexible for variable sequence lengths
        self.pos_embed_dec = init_pos_embed([num_2d_tokens], in_dim)
        self.pos_embed_dec_latent = init_pos_embed([num_1d_tokens], latent_dim)

        # Token embeddings and classification head - using linear indices to token
        self.embeddings = IndicesToToken(target_codebook_size + 1, in_dim)
        self.bits = self.embeddings.bits
        self.classification_head = nn.Sequential(
            nn.Linear(in_dim, self.bits * 2),
            Rearrange("... (bits d) -> ... bits d", d=2),
        )

        self.last_ln = nn.LayerNorm(latent_dim)
        self.apply(init_weights)

    def forward(
        self,
        x,
        latent_1d_quantized,
        return_attn_weights=False,
        capture_intermediate=False,
        intermediate_block_idx=-1,
    ):
        # Process target tokens
        x = self.embeddings(x)
        x = x + self.pos_embed_dec

        # Process input latent to use as cross-attention conditioning
        latent_1d_quantized = self.in_proj(latent_1d_quantized)
        latent_1d_quantized = latent_1d_quantized + self.pos_embed_dec_latent

        registers = self.latent_init.expand(x.shape[0], -1, -1)
        conditioning = torch.cat([latent_1d_quantized, registers], dim=1)

        # Process through MMDiT transformer blocks with joint attention
        attn_weights = None
        repa_feat = None
        intermediate_features = None

        # Create time embeddings from latent_1d_quantized
        temb = torch.mean(latent_1d_quantized, dim=1)

        # Initialize with separate conditioning and target tokens
        hidden_states = x  # Target tokens
        encoder_hidden_states = conditioning  # Conditioning tokens

        for i, block in enumerate(self.blocks):
            # JointTransformerBlock processes both streams and returns (encoder_out, hidden_out)
            block_output = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,  # Time embedding (dummy zeros for our use case)
            )

            encoder_hidden_states, hidden_states = block_output

            # Capture intermediate features if requested
            if capture_intermediate:
                block_idx = (
                    intermediate_block_idx
                    if intermediate_block_idx >= 0
                    else self.num_blocks - 1
                )
                if i == block_idx:
                    # Capture the hidden states (target tokens) as intermediate features
                    intermediate_features = hidden_states

            # Extract repa_feat at specified block
            if i == self.repa_target_block_num:
                repa_feat = hidden_states

        # Output processing - use the last valid encoder hidden states for latent
        latent = None  # Do not return latent, JointTransformerBlock does not return it
        x = hidden_states  # Target tokens from hidden states
        x = self.last_ln(x)

        return DecoderOutput(
            decoder_2d_logits=self.classification_head(x),
            latent=latent,
            write_attn_weights=attn_weights if return_attn_weights else None,
            read_attn_weights=attn_weights if return_attn_weights else None,
            repa_feat=repa_feat,
            intermediate_features=intermediate_features,
        )


class DiTDecoder(nn.Module):
    """
    MMDiT-based decoder using JointTransformerBlock from diffusers for joint attention processing.
    """

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
        num_attention_heads=8,
        attention_head_dim=None,
    ):
        print("instantiating MMDiT DiTDecoder")
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

        # Calculate attention dimensions
        if attention_head_dim is None:
            attention_head_dim = in_dim // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Input projection for latent
        self.in_proj_dim = latent_dim if in_proj_dim is None else in_proj_dim
        self.in_proj = nn.Linear(self.in_proj_dim, latent_dim)

        assert latent_dim == in_dim, (
            "Latent dimension must be the same as input dimension"
        )

        # MMDiT transformer blocks using JointTransformerBlock from diffusers
        self.blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=in_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i
                    == num_blocks - 1,  # Last block is context pre only
                )
                for i in range(num_blocks)
            ]
        )

        # Initialize latent parameters
        self.latent_init = nn.Parameter(torch.zeros(1, num_1d_tokens, self.latent_dim))
        nn.init.trunc_normal_(self.latent_init, 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02)

        # Positional embeddings - make flexible for variable sequence lengths
        self.pos_embed_dec = init_pos_embed([num_2d_tokens], in_dim)
        self.pos_embed_dec_latent = init_pos_embed([num_1d_tokens], latent_dim)

        # Token embeddings and classification head
        self.embeddings = nn.Embedding(target_codebook_size + 1, in_dim)
        self.classification_head = WeightTiedHead(self.embeddings, target_codebook_size)

        self.last_ln = nn.LayerNorm(latent_dim)
        self.apply(init_weights)

    def forward(
        self,
        x,
        latent_1d_quantized,
        return_attn_weights=False,
        capture_intermediate=False,
        intermediate_block_idx=-1,
    ):
        # Process target tokens
        x = self.embeddings(x)
        x = x + self.pos_embed_dec

        # Process input latent to use as cross-attention conditioning
        latent_1d_quantized = self.in_proj(latent_1d_quantized)
        latent_1d_quantized = latent_1d_quantized + self.pos_embed_dec_latent

        registers = self.latent_init.expand(x.shape[0], -1, -1)
        conditioning = torch.cat([latent_1d_quantized, registers], dim=1)

        # Process through MMDiT transformer blocks with joint attention
        attn_weights = None
        repa_feat = None
        intermediate_features = None

        # Create time embeddings from latent_1d_quantized
        temb = torch.mean(latent_1d_quantized, dim=1)

        # Initialize with separate conditioning and target tokens
        hidden_states = x  # Target tokens
        encoder_hidden_states = conditioning  # Conditioning tokens

        for i, block in enumerate(self.blocks):
            # JointTransformerBlock processes both streams and returns (encoder_out, hidden_out)
            block_output = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,  # Time embedding (dummy zeros for our use case)
            )

            encoder_hidden_states, hidden_states = block_output

            # Capture intermediate features if requested
            if capture_intermediate:
                block_idx = (
                    intermediate_block_idx
                    if intermediate_block_idx >= 0
                    else self.num_blocks - 1
                )
                if i == block_idx:
                    # Capture the hidden states (target tokens) as intermediate features
                    intermediate_features = hidden_states

            # Extract repa_feat at specified block
            if i == self.repa_target_block_num:
                repa_feat = hidden_states

        # Output processing - use the last valid encoder hidden states for latent
        latent = None  # Do not return latent, JointTransformerBlock does not return it
        x = hidden_states  # Target tokens from hidden states
        x = self.last_ln(x)

        return DecoderOutput(
            decoder_2d_logits=self.classification_head(x),
            latent=latent,
            write_attn_weights=attn_weights if return_attn_weights else None,
            read_attn_weights=attn_weights if return_attn_weights else None,
            repa_feat=repa_feat,
            intermediate_features=intermediate_features,
        )
