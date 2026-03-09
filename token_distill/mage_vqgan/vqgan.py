import torch
import torch.nn as nn
from token_distill.mage_vqgan.vq import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        remap=None,
        sane_index_shape=False,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = encoder
        self.decoder = decoder
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("MAGE VQGAN: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        print("MAGE VQGAN: Strict load")
        self.load_state_dict(sd, strict=True)
        print(f"MAGE VQGAN: Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        quant, emb_loss, info = self.quantize(h)
        return h, quant, emb_loss, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_indices(self, indices):
        # Handle indices shape - VQ might return flat indices [seq_len] for single batch
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)  # Add batch dimension

        batch_size = indices.shape[0]
        seq_len = indices.shape[1]
        h = w = int(seq_len**0.5)
        quant_b = self.quantize.get_codebook_entry(
            indices, (batch_size, h, w, self.quantize.e_dim)
        )
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        _, quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def z_dim(self):
        return self.quantize.e_dim
