import torch
import torch.nn as nn
from models import (
    CDiTBlock,
    TimestepEmbedder,
    ActionEmbedder,
    modulate,
)
import torch.nn.functional as F


def initialize_discrete_cdit_weights(model):
    """Shared weight initialization for discrete CDiT models"""

    # Apply basic initialization first to all linear layers
    def _basic_init(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    model.apply(_basic_init)

    # Initialize (and freeze) pos_embed by sin-cos embedding:
    nn.init.normal_(model.pos_embed, std=0.02)

    # Initialize action embedding:
    nn.init.normal_(model.y_embedder.x_emb.mlp[0].weight, std=0.02)
    nn.init.normal_(model.y_embedder.x_emb.mlp[2].weight, std=0.02)

    nn.init.normal_(model.y_embedder.y_emb.mlp[0].weight, std=0.02)
    nn.init.normal_(model.y_embedder.y_emb.mlp[2].weight, std=0.02)

    nn.init.normal_(model.y_embedder.angle_emb.mlp[0].weight, std=0.02)
    nn.init.normal_(model.y_embedder.angle_emb.mlp[2].weight, std=0.02)

    nn.init.normal_(model.time_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(model.time_embedder.mlp[2].weight, std=0.02)

    # Zero-out adaLN modulation layers in DiT blocks:
    for block in model.blocks:
        nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    # Zero-out output layers:
    nn.init.constant_(model.final_layer.adaLN_modulation[-1].weight, 0)
    nn.init.constant_(model.final_layer.adaLN_modulation[-1].bias, 0)


class WeightTiedHead(nn.Module):
    def __init__(self, hidden_size, embeddings):
        super().__init__()
        self.weight = embeddings.weight
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = F.linear(x, self.weight)
        return x


class FSQIndicesToToken(nn.Module):
    """Convert FSQ indices to tokens using proper FSQ level structure"""

    def __init__(self, fsq_levels, dim):
        super().__init__()
        self.fsq_levels = fsq_levels
        self.num_fsq_dims = len(fsq_levels)
        self.codebook_size = 1
        for level in fsq_levels:
            self.codebook_size *= level

        # Create basis for converting flat indices to per-level indices
        # This is the same as FSQ's _basis
        basis = [1]
        for level in fsq_levels[:-1]:
            basis.append(basis[-1] * level)
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.int32))
        self.register_buffer("levels", torch.tensor(fsq_levels, dtype=torch.int32))

        # Linear projection from FSQ coordinates to embedding dimension
        self.linear = nn.Linear(self.num_fsq_dims, dim)

        # Mask token for special mask index
        self.mask_indices = self.codebook_size  # Use codebook_size as mask index
        self.mask_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def indices_to_level_indices(self, indices):
        """Convert flat FSQ indices to per-level indices (same as FSQ.indices_to_level_indices)"""
        indices = indices.unsqueeze(-1)  # (..., 1)
        level_indices = (indices // self.basis) % self.levels  # (..., num_fsq_dims)
        return level_indices

    def level_indices_to_codes(self, level_indices):
        """Convert level indices to normalized FSQ codes [-1, 1] range"""
        # Normalize to [-1, 1] range (same as FSQ._scale_and_shift_inverse)
        half_width = self.levels.float() // 2
        codes = (level_indices.float() - half_width) / half_width
        return codes

    def forward(self, x):
        """
        x: (...,) tensor of FSQ indices
        Returns: (..., dim) tensor of embeddings
        """
        # Handle mask tokens
        is_mask = x == self.mask_indices

        # Convert indices to level indices, then to codes
        level_indices = self.indices_to_level_indices(x)  # (..., num_fsq_dims)
        codes = self.level_indices_to_codes(level_indices)  # (..., num_fsq_dims)

        # Project to embedding dimension
        embeddings = self.linear(codes.to(dtype=self.linear.weight.dtype))

        # Replace mask positions with mask token
        embeddings = torch.where(is_mask.unsqueeze(-1), self.mask_token, embeddings)

        return embeddings


class FSQFinalLayer(nn.Module):
    def __init__(self, hidden_size, fsq_levels):
        super().__init__()
        self.fsq_levels = fsq_levels
        self.num_fsq_dims = len(fsq_levels)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Create separate output heads for each FSQ dimension
        # Each dimension has its own number of levels
        self.output_heads = nn.Linear(hidden_size, sum(fsq_levels))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)

        # Apply output head to get logits for each FSQ dimension
        outputs = self.output_heads(x)

        # Return logits for each FSQ dimension
        # outputs has shape (batch, seq, sum(fsq_levels))
        return outputs


class DiscreteCDiT(nn.Module):
    def __init__(
        self,
        input_size=64,
        codebook_size=1024,
        context_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_action_cfg=False,
    ):
        super().__init__()
        self.context_size = context_size
        self.in_channels = codebook_size
        self.out_channels = codebook_size
        self.num_heads = num_heads
        self.use_action_cfg = use_action_cfg

        self.x_embedder = nn.Embedding(codebook_size + 1, hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        # Learnable null action embedding for classifier-free guidance (optional)
        if use_action_cfg:
            self.null_action_embedding = nn.Parameter(torch.randn(1, hidden_size))

        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, input_size, hidden_size),
            requires_grad=True,
        )  # for context and for predicted frame

        self.blocks = nn.ModuleList(
            [
                CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = WeightTiedHead(hidden_size, self.x_embedder)
        self.initialize_weights()

    def initialize_weights(self):
        initialize_discrete_cdit_weights(self)

        # Model-specific initialization
        nn.init.normal_(self.x_embedder.weight, std=0.02)
        if self.use_action_cfg:
            nn.init.normal_(self.null_action_embedding, std=0.02)

    def forward(self, x, t, y, x_cond, rel_t, action_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        action_mask: (N,) optional boolean tensor. True = use real action, False = use null action
        """
        x = self.x_embedder(x) + self.pos_embed[self.context_size :]
        x_cond = (
            self.x_embedder(x_cond.flatten(0, 1)).unflatten(
                0, (x_cond.shape[0], x_cond.shape[1])
            )
            + self.pos_embed[: self.context_size]
        )  # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        x_cond = x_cond.flatten(1, 2)
        y = self.y_embedder(y)

        # Apply action CFG: replace with null embedding when action_mask is False
        if action_mask is not None and self.use_action_cfg:
            null_y = self.null_action_embedding.expand(y.shape[0], -1)
            y = torch.where(action_mask.unsqueeze(-1), y, null_y)

        time_emb = self.time_embedder(rel_t[..., None])
        c = time_emb + y  # if training on unlabeled data, dont add y.

        for block in self.blocks:
            x = block(x, c, x_cond)
        x = self.final_layer(x, c)
        return x


class FSQDiscreteCDiT(nn.Module):
    def __init__(
        self,
        input_size=64,
        fsq_levels=[8, 5, 5, 5],  # FSQ levels for each dimension
        context_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        use_action_cfg=False,
    ):
        super().__init__()
        self.context_size = context_size
        self.fsq_levels = fsq_levels
        self.num_fsq_dims = len(fsq_levels)
        self.num_heads = num_heads
        self.use_action_cfg = use_action_cfg

        # Calculate codebook size from FSQ levels
        self.codebook_size = 1
        for level in fsq_levels:
            self.codebook_size *= level

        self.in_channels = self.codebook_size
        self.out_channels = self.codebook_size

        # Use FSQIndicesToToken for embedding FSQ indices
        self.x_embedder = FSQIndicesToToken(fsq_levels, hidden_size)

        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = TimestepEmbedder(hidden_size)

        # Learnable null action embedding for classifier-free guidance (optional)
        if use_action_cfg:
            self.null_action_embedding = nn.Parameter(torch.randn(1, hidden_size))

        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, input_size, hidden_size),
            requires_grad=True,
        )  # for context and for predicted frame

        self.blocks = nn.ModuleList(
            [
                CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )

        self.final_layer = FSQFinalLayer(hidden_size, fsq_levels)
        self.initialize_weights()

    def initialize_weights(self):
        initialize_discrete_cdit_weights(self)

        # FSQ-specific initialization
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        if self.x_embedder.linear.bias is not None:
            nn.init.constant_(self.x_embedder.linear.bias, 0)
        nn.init.normal_(self.x_embedder.mask_token, std=0.02)
        if self.use_action_cfg:
            nn.init.normal_(self.null_action_embedding, std=0.02)

        # Initialize FSQ output heads
        nn.init.constant_(self.final_layer.output_heads.weight, 0)
        nn.init.constant_(self.final_layer.output_heads.bias, 0)

    def forward(self, x, t, y, x_cond, rel_t, action_mask=None):
        """
        Forward pass of FSQ DiT.
        x: (N, H*W) tensor of FSQ indices
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of action labels
        x_cond: (N, context_size, H*W) tensor of context FSQ indices
        rel_t: (N,) tensor of relative timesteps
        action_mask: (N,) optional boolean tensor. True = use real action, False = use null action
        """
        # Embed FSQ indices directly using FSQIndicesToToken
        x = self.x_embedder(x) + self.pos_embed[self.context_size :]
        x_cond = (
            self.x_embedder(x_cond.flatten(0, 1)).unflatten(
                0, (x_cond.shape[0], x_cond.shape[1])
            )
            + self.pos_embed[: self.context_size]
        )  # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        x_cond = x_cond.flatten(1, 2)
        y = self.y_embedder(y)

        # Apply action CFG: replace with null embedding when action_mask is False
        if action_mask is not None and self.use_action_cfg:
            null_y = self.null_action_embedding.expand(y.shape[0], -1)
            y = torch.where(action_mask.unsqueeze(-1), y, null_y)

        time_emb = self.time_embedder(rel_t[..., None])
        c = time_emb + y  # if training on unlabeled data, dont add y.

        for block in self.blocks:
            x = block(x, c, x_cond)
        x = self.final_layer(x, c)
        return x
