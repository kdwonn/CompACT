import torch.nn as nn
from einops import einsum, rearrange


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=8, ff_mult=4, norm_context=True):
        super().__init__()

        # Multi-head attention with optimized implementation from PyTorch 2.0+
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.ff_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model) if norm_context else nn.Identity()

    def forward(self, x, context=None, return_attn_weights=False):
        # Multi-head attention block (with residual connection and layer norm)
        normed_x = self.norm(x)
        if context is not None:
            kv = self.norm_context(context)
        else:
            kv = normed_x

        attn_output, attn_weights = self.self_attn(
            normed_x,
            kv,
            kv,
            need_weights=return_attn_weights,
        )
        x = x + attn_output

        # Feed-forward block (with residual connection and layer norm)
        ff_output = self.ff_mlp(self.norm_ff(x))
        x = x + ff_output

        return x, attn_weights


class SlotAttention(nn.Module):
    def __init__(self, d_model, nhead=8, ff_mult=4, norm_context=True):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model * 2)
        self.ff_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.eps = 1e-6

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.norm_context = nn.LayerNorm(d_model) if norm_context else nn.Identity()
        self.nhead = nhead
        self.proj = nn.Linear(d_model, d_model)

    def inverted_attention(self, q, k, v):
        dots = einsum(q, k, "b k h d, b l h d -> b k h l")
        attn = dots.softmax(dim=1)
        attn_before_reweighting = attn
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
        out = einsum(attn, v, "b k h l, b l h d -> b k h d")
        out = rearrange(out, "b k h d -> b k (h d)")
        out = self.proj(out)
        return out, attn_before_reweighting

    def forward(self, x, context=None, return_attn_weights=False):
        kv = self.kv(self.norm(context))
        kv = rearrange(kv, "b l (kv h d) -> kv b l h d", kv=2, h=self.nhead)
        k, v = kv[0], kv[1]

        q = self.q(self.norm(x))
        q = rearrange(q, "b k (h d) -> b k h d", h=self.nhead)

        out, attn_before_reweighting = self.inverted_attention(q, k, v)
        out = out + x

        out = out + self.ff_mlp(self.norm_ff(out))

        return out, attn_before_reweighting


class RINBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        latent_dim,
        process_attn_depth,
        final_norm=True,
        final_norm_context=True,
        use_slot_write_attn=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.process_attn_depth = process_attn_depth

        self.read_cross_attn = TransformerLayer(in_dim)
        if use_slot_write_attn:
            self.write_cross_attn = SlotAttention(latent_dim)
        else:
            self.write_cross_attn = TransformerLayer(latent_dim)

        self.process_attn_depth = nn.ModuleList(
            [
                TransformerLayer(latent_dim, norm_context=False)
                for _ in range(process_attn_depth)
            ]
        )

        self.final_norm = nn.LayerNorm(latent_dim) if final_norm else nn.Identity()
        self.final_norm_context = (
            nn.LayerNorm(latent_dim) if final_norm_context else nn.Identity()
        )

    def forward(self, latents, context, return_attn_weights=False):
        latents, write_attn_weights = self.write_cross_attn(
            latents, context, return_attn_weights
        )

        for layer in self.process_attn_depth:
            latents, _ = layer(latents, return_attn_weights=False)
        context, read_attn_weights = self.read_cross_attn(
            context, latents, return_attn_weights
        )

        latents = self.final_norm(latents)
        context = self.final_norm_context(context)

        return latents, context, write_attn_weights, read_attn_weights
