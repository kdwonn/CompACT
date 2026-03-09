from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:

        ATTENTION_MODE = "xformers"
    except:
        ATTENTION_MODE = "math"
print(f"attention mode is {ATTENTION_MODE}")
print(f"torch version is {torch.__version__}")
print(f"cuda version is {torch.version.cuda}")


@dataclass
class CompactTokenizerOutput:
    diffusion_loss: torch.Tensor
    masks: torch.Tensor
    latent_1d_quantized: torch.Tensor
    latent_2d_decoded: torch.Tensor
    latent_2d_target: torch.Tensor
    indices: torch.Tensor
    commit_loss: torch.Tensor
    repa_feat: torch.Tensor
    repa_masks: Optional[torch.Tensor] = None
    encoder_write_attn_weights: Optional[torch.Tensor] = None
    encoder_read_attn_weights: Optional[torch.Tensor] = None
    decoder_write_attn_weights: Optional[torch.Tensor] = None
    decoder_read_attn_weights: Optional[torch.Tensor] = None
    intermediate_features: Optional[torch.Tensor] = None


class CompactTokenizer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        diffusion: nn.Module,
        num_1d_tokens: int,
        num_2d_tokens: int,
        target_codebook_size: int,
        masking_type: str = "arccos",
        latent_self_cond_prob: float = 0.9,
        repa_on_enc: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.num_1d_tokens = num_1d_tokens
        self.num_2d_tokens = num_2d_tokens
        self.target_codebook_size = target_codebook_size
        self.mask_token_id = target_codebook_size
        self.masking_type = masking_type

        self.quantizer = quantizer
        self.latent_self_cond_prob = latent_self_cond_prob
        self.repa_on_enc = repa_on_enc

    def encode(self, encoder_input, return_attn_weights=False):
        encoder_output = self.encoder(
            encoder_input, return_attn_weights=return_attn_weights
        )
        latent_1d_quantized, indices, commit_loss = self.quantizer(
            encoder_output.latent
        )
        return (
            latent_1d_quantized,
            indices,
            commit_loss,
            encoder_output,
        )

    def decode(
        self,
        latent_1d_quantized,
        return_attn_weights=False,
    ):
        batch_size = latent_1d_quantized.shape[0]
        return self.diffusion.p_sample_loop(
            self.decoder,
            shape=(batch_size, self.num_2d_tokens),
            condition=latent_1d_quantized,
            model_kwargs={
                "return_attn_weights": return_attn_weights,
            },
        )

    def forward(
        self,
        base_2d_ids,
        image,
        return_attn_weights=False,
        capture_intermediate=False,
        intermediate_block_idx=-1,
    ):
        latent_1d_quantized, indices, commit_loss, encoder_output = self.encode(
            image, return_attn_weights
        )

        diffusion_output = self.diffusion.training_losses(
            self.decoder,
            base_2d_ids,
            condition=latent_1d_quantized,
            model_kwargs={
                "return_attn_weights": return_attn_weights,
                "capture_intermediate": capture_intermediate,
                "intermediate_block_idx": intermediate_block_idx,
            },
        )
        decoder_output = diffusion_output["model_output"]
        repa_feat = (
            encoder_output.repa_feat if self.repa_on_enc else decoder_output.repa_feat
        )
        repa_masks = (
            encoder_output.masks if self.repa_on_enc else diffusion_output["mask"]
        )

        return CompactTokenizerOutput(
            diffusion_loss=diffusion_output["loss"],
            masks=diffusion_output["mask"],
            latent_1d_quantized=latent_1d_quantized,
            latent_2d_decoded=decoder_output.decoder_2d_logits,
            latent_2d_target=base_2d_ids,
            indices=indices,
            commit_loss=commit_loss,
            encoder_write_attn_weights=encoder_output.write_attn_weights,
            encoder_read_attn_weights=encoder_output.read_attn_weights,
            decoder_write_attn_weights=decoder_output.write_attn_weights,
            decoder_read_attn_weights=decoder_output.read_attn_weights,
            repa_feat=repa_feat,
            repa_masks=repa_masks,
            intermediate_features=decoder_output.intermediate_features,
        )
