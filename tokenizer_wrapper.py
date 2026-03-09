import logging

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from flextok.flextok_wrapper import FlexTok, FlexTokFromHub
from token_distill.mage_vqgan.vqgan import VQModel

logger = logging.getLogger(__name__)


class TokenizerWrapper(nn.Module):
    """
    Base class for tokenizer wrappers that provides a common interface for different tokenizers.
    """

    def __init__(self, normalize_mean=None, normalize_std=None):
        super().__init__()
        # Store normalization parameters for unnormalization
        if normalize_mean is not None and normalize_std is not None:
            self.register_buffer(
                "normalize_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "normalize_std", torch.tensor(normalize_std).view(1, 3, 1, 1)
            )
        else:
            self.normalize_mean = None
            self.normalize_std = None

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method")

    def encode(self, x):
        raise NotImplementedError("Subclasses must implement encode method")

    def decode(self, x, denormalize=False):
        raise NotImplementedError("Subclasses must implement decode method")

    def unnormalize_image(self, normalized_image):
        """
        Unnormalize images from normalized space back to [0, 1] range.
        Args:
            normalized_image: Tensor of shape (B, C, H, W) in normalized space
        Returns:
            Tensor of shape (B, C, H, W) in [0, 1] range
        """
        if self.normalize_mean is None or self.normalize_std is None:
            logger.warning("Normalization parameters not set, returning image as-is")
            return normalized_image

        denormalized = normalized_image * self.normalize_std + self.normalize_mean
        return denormalized.clamp(0, 1)

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        raise NotImplementedError(
            "Subclasses must implement get_code_from_latent method"
        )


class VAEWrapper(TokenizerWrapper):
    def __init__(self, model_path: str, normalize_mean=None, normalize_std=None):
        super().__init__(normalize_mean=normalize_mean, normalize_std=normalize_std)
        self.vae = AutoencoderKL.from_pretrained(model_path).eval()
        self.scaling_factor = 0.18215

    def forward(self, x):
        return self.vae.forward(x)

    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    def decode(self, x, denormalize=False):
        decoded = self.vae.decode(x / self.scaling_factor).sample
        if denormalize:
            decoded = self.unnormalize_image(decoded)
        return decoded

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        return latent


class FlexTokenWrapper(TokenizerWrapper):
    def __init__(
        self,
        model_path: str,
        num_tokens: int,
        codebook_size: int,
        vae_image_size=28,
        normalize_mean=None,
        normalize_std=None,
    ):
        super().__init__(normalize_mean=normalize_mean, normalize_std=normalize_std)
        self.tokenizer: FlexTok = FlexTokFromHub.from_pretrained(model_path).eval()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.vae_image_size = vae_image_size
        assert self.codebook_size == self.tokenizer.regularizer.codebook_size

    def forward(self, x):
        return self.tokenizer.forward(x)

    def encode(self, x):
        tokens = self.tokenizer.tokenize(x)
        tokens = torch.cat(tokens, dim=0)[:, : self.num_tokens]
        return tokens

    def decode(self, x, denormalize=False):
        x = torch.split(x, 1, dim=0)
        decoded = self.tokenizer.detokenize(
            x,
            timesteps=20,  # Number of denoising steps
            guidance_scale=7.5,  # Classifier-free guidance scale
            perform_norm_guidance=True,  # See https://arxiv.org/abs/2410.02416
            vae_image_sizes=self.vae_image_size,
        )
        if denormalize:
            decoded = self.unnormalize_image(decoded)
        return decoded

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        return self.tokenizer.regularizer.indices_to_embedding(latent)


class MAGEWrapper(TokenizerWrapper):
    def __init__(
        self,
        encoder_config,
        decoder_config,
        n_embed=1024,
        embed_dim=256,
        ckpt_path=None,
        normalize_mean=None,
        normalize_std=None,
    ):
        super().__init__(normalize_mean=normalize_mean, normalize_std=normalize_std)

        # When called with _recursive_=True, encoder_config and decoder_config
        # are already instantiated as Encoder and Decoder objects
        encoder = encoder_config
        decoder = decoder_config

        # Create VQModel
        self.vqgan = VQModel(
            encoder=encoder,
            decoder=decoder,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
        ).eval()

    def forward(self, x):
        return self.vqgan.forward(x)

    def encode(self, x):
        """Encode images to quantized latent representation."""
        _, quant, _, _ = self.vqgan.encode(x)
        return quant

    def decode(self, x, denormalize=False):
        """Decode quantized representation back to images."""
        decoded = self.vqgan.decode(x)
        if denormalize:
            decoded = self.unnormalize_image(decoded)
        return decoded

    @torch.no_grad()
    def get_code_from_latent(self, latent):
        return latent
