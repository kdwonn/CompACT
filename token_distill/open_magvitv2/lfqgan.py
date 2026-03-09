import torch
import torch.nn as nn

from token_distill.open_magvitv2.enc_dec import Encoder, Decoder
from token_distill.open_magvitv2.lfq import LFQ


class OpenMagvitV2(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        lfq: LFQ,
        n_embed: int,
        embed_dim: int,
        ckpt_path: str = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantize = lfq
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            keys_to_remove = [
                key for key in state_dict.keys() if key.startswith("model_ema")
            ]
            for key in keys_to_remove:
                state_dict.pop(key)
            self.load_state_dict(state_dict)

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, indices), loss_breakdown = self.quantize(
            h, return_loss_breakdown=True
        )
        return h, quant, emb_loss, indices

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_indices(self, indices):
        # Handle indices shape - LFQ returns flat indices [seq_len] for single batch
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)  # Add batch dimension

        batch_size = indices.shape[0]
        seq_len = indices.shape[1]
        h = w = int(seq_len**0.5)
        shape = (batch_size, h, w, self.embed_dim)
        quant_b = self.quantize.get_codebook_entry(indices, shape, None)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        h, quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    @property
    def z_dim(self):
        return self.encoder.z_channels
