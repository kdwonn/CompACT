from einops import rearrange, parse_shape
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from .compact_tokenizer import CompactTokenizerOutput


@dataclass
class TokenDistillOutput:
    diffusion_loss: torch.Tensor
    masks: torch.Tensor
    latent_1d_quantized: torch.Tensor
    indices: torch.Tensor
    commit_loss: torch.Tensor
    base_2d_logits: Optional[torch.Tensor] = None
    base_2d_ids: Optional[torch.Tensor] = None
    base_2d_ids_from_logits: Optional[torch.Tensor] = None
    image_recon: Optional[torch.Tensor] = None
    repa_feat: Optional[torch.Tensor] = None
    repa_masks: Optional[torch.Tensor] = None
    intermediate_features: Optional[torch.Tensor] = None  # For bisim guidance


class TokenDistillModel(nn.Module):
    def __init__(
        self,
        base_tokenizer: nn.Module,
        compact_tokenizer: nn.Module,
        codebook_size: int,
        num_tokens: int,
        normalize_mean: list,
        normalize_std: list,
        intermediate_block_idx: int = -1,  # Which block to extract features from (-1 = last)
    ):
        super().__init__()
        self.base_tokenizer = base_tokenizer
        self.compact_tokenizer = compact_tokenizer
        self.codebook_size = codebook_size
        self.num_tokens = num_tokens
        self.intermediate_block_idx = intermediate_block_idx

        # Store normalization parameters for denormalization
        self.register_buffer(
            "normalize_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "normalize_std", torch.tensor(normalize_std).view(1, 3, 1, 1)
        )

        for param in self.base_tokenizer.parameters():
            param.requires_grad = False

    def base_decode(self, decoder_2d_logits, recon_image=False):
        if recon_image:
            if len(decoder_2d_logits.shape) == 3:
                decoder_2d_ids = decoder_2d_logits.argmax(dim=-1)
                image_recon = self.base_tokenizer.decode_indices(decoder_2d_ids)
            elif len(decoder_2d_logits.shape) == 2:  # ids, not logits
                decoder_2d_ids = decoder_2d_logits
                image_recon = self.base_tokenizer.decode_indices(decoder_2d_ids)
            elif len(decoder_2d_logits.shape) == 4:  # bitwise
                decoder_2d_ids = self.compact_tokenizer.diffusion.bits_to_indices(
                    decoder_2d_logits.argmax(dim=-1)
                )
                image_recon = self.base_tokenizer.decode_indices(decoder_2d_ids)
            else:
                raise ValueError(
                    f"Invalid decoder_2d_logits shape: {decoder_2d_logits.shape}"
                )
            # Don't clamp here - let denormalization handle the range conversion
        else:
            image_recon = None
        return image_recon

    @torch.no_grad()
    def base_encode(self, image, pre_encoded_indices: Optional[torch.Tensor] = None):
        # If pre-encoded indices are provided, use them directly
        if pre_encoded_indices is not None:
            # Reshape pre-encoded indices to the expected format [batch, seq_len]
            batch_size = image.shape[0]
            if len(pre_encoded_indices.shape) == 3:  # [batch, H, W]
                base_2d_ids = rearrange(pre_encoded_indices, "b h w -> b (h w)")
            elif len(pre_encoded_indices.shape) == 2:  # [batch, seq_len]
                base_2d_ids = pre_encoded_indices
            else:
                raise ValueError(
                    f"Unexpected pre_encoded_indices shape: {pre_encoded_indices.shape}"
                )

            # For pre-encoded case, we set dummy values for hidden states and quantized
            # These are only needed for reconstruction, which isn't typically done during training
            seq_len = base_2d_ids.shape[1]
            spatial_size = int(seq_len**0.5)  # Assume square spatial layout
            base_2d_hidden_shape = {"h": spatial_size, "w": spatial_size}

            # Create dummy tensors (will be replaced by full base_tokenizer.encode if needed for reconstruction)
            base_2d_hidden = torch.zeros(
                batch_size, seq_len, 512, device=image.device, dtype=image.dtype
            )  # Dummy hidden
            base_2d_quantized = torch.zeros(
                batch_size,
                512,
                spatial_size,
                spatial_size,
                device=image.device,
                dtype=image.dtype,
            )  # Dummy quantized

            return base_2d_hidden, base_2d_hidden_shape, base_2d_quantized, base_2d_ids

        # Standard path: call base tokenizer encode
        base_2d_hidden, base_2d_quantized, _, base_2d_info = self.base_tokenizer.encode(
            image
        )

        # Handle different formats of base_2d_info
        if isinstance(base_2d_info, tuple):
            # MAGE VQGAN returns (perplexity, min_encodings, min_encoding_indices)
            _, _, base_2d_ids = base_2d_info
        else:
            # OpenMagvitV2 returns indices directly
            base_2d_ids = base_2d_info

        base_2d_hidden_shape = parse_shape(base_2d_hidden, "_ _ h w")
        base_2d_hidden = rearrange(base_2d_hidden, "b c h w -> b (h w) c")
        if len(base_2d_ids.shape) == 1:
            base_2d_ids = rearrange(base_2d_ids, "(b l) -> b l", b=image.shape[0])

        return base_2d_hidden, base_2d_hidden_shape, base_2d_quantized, base_2d_ids

    def denormalize(self, normalized_image):
        """
        Denormalize images from normalized space back to [0, 1] range.
        Args:
            normalized_image: Tensor of shape (B, C, H, W) in normalized space
        Returns:
            Tensor of shape (B, C, H, W) in [0, 1] range
        """
        denormalized = normalized_image * self.normalize_std + self.normalize_mean
        return denormalized.clamp(0, 1)

    def forward(
        self,
        image,
        recon_image: bool = False,
        pre_encoded_indices: Optional[torch.Tensor] = None,
        compact_encoder_image: Optional[torch.Tensor] = None,
        capture_intermediate: bool = False,
    ) -> TokenDistillOutput:
        base_2d_hidden, base_2d_hidden_shape, base_2d_quantized, base_2d_ids = (
            self.base_encode(image, pre_encoded_indices=pre_encoded_indices)
        )

        # Use provided compact_encoder_image or default to the main image
        encoder_image = (
            compact_encoder_image if compact_encoder_image is not None else image
        )

        compact_tokenizer_output: CompactTokenizerOutput = self.compact_tokenizer(
            base_2d_ids=base_2d_ids,
            image=encoder_image,
            capture_intermediate=capture_intermediate,
            intermediate_block_idx=self.intermediate_block_idx,
        )

        image_recon = self.base_decode(
            compact_tokenizer_output.latent_2d_decoded, recon_image=recon_image
        )

        if compact_tokenizer_output.latent_2d_decoded.ndim == 4:
            base_2d_ids_from_logits = self.compact_tokenizer.diffusion.bits_to_indices(
                compact_tokenizer_output.latent_2d_decoded.argmax(dim=-1)
            )
        else:
            base_2d_ids_from_logits = compact_tokenizer_output.latent_2d_decoded.argmax(
                dim=-1
            )

        return TokenDistillOutput(
            diffusion_loss=compact_tokenizer_output.diffusion_loss,
            masks=compact_tokenizer_output.masks,
            latent_1d_quantized=compact_tokenizer_output.latent_1d_quantized,
            indices=compact_tokenizer_output.indices,
            commit_loss=compact_tokenizer_output.commit_loss.sum(),
            base_2d_logits=compact_tokenizer_output.latent_2d_decoded,
            base_2d_ids=base_2d_ids,
            base_2d_ids_from_logits=base_2d_ids_from_logits,
            image_recon=image_recon,
            repa_feat=compact_tokenizer_output.repa_feat,
            repa_masks=compact_tokenizer_output.repa_masks,
            intermediate_features=compact_tokenizer_output.intermediate_features,
        )

    @torch.no_grad()
    def encode(self, image):
        _, indices, _, _ = self.compact_tokenizer.encode(
            image, return_attn_weights=False
        )

        return indices

    @torch.no_grad()
    def decode(self, indices, denormalize=True, **kwargs):
        latent_1d_quantized = self.compact_tokenizer.quantizer.indices_to_codes(indices)

        latent_2d_decoded = self.compact_tokenizer.decode(latent_1d_quantized, **kwargs)

        image_recon = self.base_decode(latent_2d_decoded, recon_image=True)

        if denormalize:
            image_recon = self.denormalize(image_recon)
        else:
            # If not denormalizing, clamp to [0,1] for consistency
            image_recon = image_recon.clamp(0, 1)

        return image_recon

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        return self.compact_tokenizer.quantizer.indices_to_codes(latent)

    @torch.no_grad()
    def unnormalize_image(self, image):
        return self.denormalize(image)
