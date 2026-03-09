from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class LayerNormZero(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-6)


class WeightTiedHead(nn.Module):
    def __init__(self, embeddings, target_codebook_size):
        super().__init__()
        self.weight = embeddings.weight
        self.target_codebook_size = target_codebook_size

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        # Get the weights for the target codebook size
        weight = self.weight[
            : self.target_codebook_size
        ]  # Shape: [target_codebook_size, embed_dim]
        # Compute the logits by matrix multiplication
        logits = torch.matmul(
            x, weight.t()
        )  # Shape: [batch_size, seq_len, target_codebook_size]
        return logits


class IndicesToToken(nn.Module):
    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.bits = math.log2(num_embeddings - 1)
        assert self.bits.is_integer(), "Target codebook size must be a power of 2"
        self.bits = int(self.bits)
        self.linear = nn.Linear(self.bits, dim)
        self.num_embeddings = num_embeddings
        self.mask_indices = num_embeddings - 1
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def indices_to_bit(self, idx):
        mask = 2 ** torch.arange(self.bits, device=idx.device)
        bits = (idx.unsqueeze(-1) & mask) > 0
        return bits.float() * 2.0 - 1.0

    def forward(self, x):
        x_bit = self.indices_to_bit(x).to(dtype=self.linear.weight.dtype)
        return torch.where(
            x.unsqueeze(-1) == self.mask_indices, self.mask_token, self.linear(x_bit)
        )


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        module.weight.data = nn.init.trunc_normal_(
            module.weight.data, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(
            module.weight.data, mean=0.0, std=0.02, a=-2 * 0.02, b=2 * 0.02
        )
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)
    elif isinstance(module, LayerNormZero):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.zero_()


def init_pos_embed(shape, dim):
    return nn.init.trunc_normal_(
        nn.Parameter(torch.zeros(1, *shape, dim)), 0.0, 0.02, a=-2 * 0.02, b=2 * 0.02
    )


def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns:
        Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


def apply_random_masking(
    x: torch.Tensor,
    mask_token: torch.Tensor,
    enable_masking: bool = True,
    training: bool = True,
    masking_prob: float = 0.9,
    min_masking_ratio: float = 0.0,
    max_masking_ratio: float = 0.6,
) -> torch.Tensor:
    """
    Apply random masking to patch embeddings using uniform distribution.

    Args:
        x: Input patch embeddings [batch_size, seq_len, dim]
        mask_token: Learnable mask token [1, 1, dim]
        enable_masking: Whether masking is enabled
        training: Whether in training mode
        masking_prob: Probability of applying masking (0.9 means 90% chance of masking)
        min_masking_ratio: Minimum masking ratio (0.0 means no minimum)
        max_masking_ratio: Maximum masking ratio (0.6 means at most 60% masked)

    Returns:
        Masked patch embeddings
    """
    if not enable_masking or not training:
        return x, None

    batch_size, seq_len, dim = x.shape

    # Determine which samples in the batch should be masked
    should_mask = torch.rand(batch_size, device=x.device) < masking_prob

    # Sample masking ratio from uniform distribution [min_masking_ratio, max_masking_ratio] for samples that should be masked
    mask_ratio = (
        torch.rand(batch_size, device=x.device)
        * (max_masking_ratio - min_masking_ratio)
        + min_masking_ratio
    )
    mask_ratio = torch.clamp(mask_ratio, min_masking_ratio, max_masking_ratio)

    # Set mask ratio to 0 for samples that should not be masked
    mask_ratio = torch.where(should_mask, mask_ratio, torch.zeros_like(mask_ratio))

    # Calculate number of patches to mask per sample
    mask_len = torch.floor(seq_len * mask_ratio).long()
    mask_len = torch.clamp(mask_len, min=0)  # Allow 0 for unmasked samples

    # Create random mask for each sample
    batch_rand = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=-1)
    mask = batch_rand < mask_len.unsqueeze(-1)

    # Apply masking using learnable mask token
    mask_token_expanded = mask_token.expand(batch_size, seq_len, dim)
    masked_x = torch.where(mask.unsqueeze(-1), mask_token_expanded, x)

    return masked_x, mask


def apply_frequency_cutoff(
    image: torch.Tensor,
    enable_freq_cutoff: bool = True,
    training: bool = True,
    cutoff_downsample_factor: int = 2,
    cutoff_prob: float = 0.5,
) -> torch.Tensor:
    """
    Apply high-frequency cutoff through downsampling-upsampling in pixel space.

    Args:
        image: Input image tensor of shape (B, C, H, W)
        enable_freq_cutoff: Whether frequency cutoff is enabled
        training: Whether in training mode
        cutoff_downsample_factor: Downsampling factor for frequency cutoff
        cutoff_prob: Probability of applying cutoff during training

    Returns:
        Processed image tensor with same shape (B, C, H, W)
    """
    if not enable_freq_cutoff or not training:
        return image

    # Apply cutoff with specified probability
    if torch.rand(1).item() > cutoff_prob:
        return image

    if torch.rand(1).item() > 0.5:
        cutoff_downsample_factor *= 2

    _, _, height, width = image.shape

    # Calculate downsampled size
    down_height = height // cutoff_downsample_factor
    down_width = width // cutoff_downsample_factor

    # Downsample to remove high frequencies
    image_downsampled = F.interpolate(
        image, size=(down_height, down_width), mode="bilinear", align_corners=False
    )

    # Upsample back to original size
    image_filtered = F.interpolate(
        image_downsampled, size=(height, width), mode="bilinear", align_corners=False
    )

    return image_filtered
