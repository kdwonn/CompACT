import math
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List


def get_masking_ratio(progress, mode="arccos") -> torch.Tensor:
    """Get masking ratio."""
    if not isinstance(progress, torch.Tensor):
        r = torch.tensor(progress)
    else:
        r = progress

    if mode == "root":
        val_to_mask = 1 - (r**0.5)
    elif mode == "square":
        val_to_mask = 1 - (r**2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    elif mode.startswith("geometric"):
        beta_min = 10 ** (-5)
        beta_max = 20
        if "-" in mode:
            try:
                beta_min = float(mode.split("-")[1])
                beta_max = float(mode.split("-")[2])
            except (ValueError, IndexError):
                raise ValueError(
                    f"Invalid beta_min and beta_max for geometric mode: {mode}"
                )
        val_to_mask = torch.exp(-(beta_min ** (1 - r)) * beta_max**r)
    elif mode == "all":
        val_to_mask = torch.ones_like(r)
    elif mode == "none":
        val_to_mask = torch.zeros_like(r)
    else:
        raise ValueError("Invalid masking mode.")
    return val_to_mask


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def add_gumbel_noise(t, temperature):
    return t + temperature * gumbel_noise(t)


def top_k_filtering(logits: torch.Tensor, top_k: int, dim: int = -1) -> torch.Tensor:
    """
    Filter logits to keep only top-k values, setting others to -inf.

    Args:
        logits: Input logits tensor
        top_k: Number of top values to keep
        dim: Dimension to apply filtering on

    Returns:
        Filtered logits with non-top-k values set to -inf
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(dim))
    values, indices = logits.topk(top_k, dim=dim)
    filtered_logits = torch.full_like(logits, -np.inf)
    filtered_logits.scatter_(dim, indices, values)
    return filtered_logits


def top_p_filtering(logits: torch.Tensor, top_p: float, dim: int = -1) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Input logits tensor
        top_p: Cumulative probability threshold
        dim: Dimension to apply filtering on

    Returns:
        Filtered logits with tokens beyond cumulative prob threshold set to -inf
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=dim)
    sorted_probs = torch.softmax(sorted_logits, dim=dim)
    cumulative_probs = torch.cumsum(sorted_probs, dim=dim)

    # Find where cumulative probability exceeds top_p
    cutoff_mask = cumulative_probs > top_p
    # Always keep at least one token
    if dim == -1:
        cutoff_mask[..., 0] = False
    else:
        cutoff_mask.index_fill_(
            dim, torch.tensor([0], device=cutoff_mask.device), False
        )

    # Set filtered positions to -inf
    sorted_logits[cutoff_mask] = -np.inf

    # Restore original order
    filtered_logits = sorted_logits.gather(dim, sorted_indices.argsort(dim))
    return filtered_logits


def mask_tokens(
    x: torch.Tensor, t: torch.Tensor, schedule_type: str, mask_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mask tokens based on timestep t (mask probability = relative timestep)

    Args:
        x: Original token IDs [batch_size, seq_len]
        t: Timestep indices [batch_size]
        schedule_type: Masking schedule type
        mask_token_id: Token ID used for masking

    Returns:
        Tuple of (masked tokens, mask)
    """
    batch_size, seq_len = x.shape

    # Get mask probability for each sample's timestep
    mask_ratio = get_masking_ratio(t, schedule_type)
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.0)

    mask_len = torch.floor(seq_len * mask_ratio).to(x.device)
    mask_len = torch.clamp(mask_len, min=1)

    batch_rand = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=-1)
    mask = batch_rand < mask_len.unsqueeze(-1)

    masked_x = torch.where(mask, mask_token_id, x)

    return masked_x, mask


def mask_condition_for_cfg(condition: torch.Tensor, cfg_prob: float) -> torch.Tensor:
    """
    Mask condition tensor for classifier-free guidance training.

    Args:
        condition: Condition tensor of shape [batch_size, ...]
        cfg_prob: Probability of masking each batch sample

    Returns:
        Masked condition tensor with same shape as input
    """
    if cfg_prob <= 0.0:
        return condition

    batch_size = condition.shape[0]
    # Create random mask for each batch sample
    mask = torch.rand(batch_size, device=condition.device) < cfg_prob

    # Create zero tensor with same shape as condition
    zero_condition = torch.zeros_like(condition, device=condition.device)

    # Apply mask: replace masked samples with zeros
    mask = mask.view(-1, *[1] * (condition.ndim - 1))
    masked_condition = torch.where(mask, zero_condition, condition)

    return masked_condition


def compute_masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss only on masked positions.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        mask: Binary mask indicating positions to compute loss [batch_size, seq_len]

    Returns:
        Scalar loss value (weighted by mask)
    """
    logits = rearrange(logits, "b n c -> b c n")
    loss = F.cross_entropy(logits, targets, reduction="none")

    mask_weight = mask.float()
    if mask_weight.sum() > 0:
        loss = (loss * mask_weight).sum() / mask_weight.sum()
    else:
        loss = loss.mean() * 0.0  # No masked tokens, zero loss

    return loss


def compute_fsq_masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    fsq_levels: List[int],
) -> torch.Tensor:
    """
    Compute per-dimension cross-entropy loss for FSQ radix-wise embedding.

    Args:
        logits: Model output logits [batch_size, seq_len, sum(fsq_levels)]
        targets: Target token IDs (flat FSQ indices) [batch_size, seq_len]
        mask: Binary mask indicating positions to compute loss [batch_size, seq_len]
        fsq_levels: FSQ levels for each dimension (e.g., [8, 8, 8, 6, 5])

    Returns:
        Scalar loss value (weighted by mask)
    """
    batch_size, seq_len = targets.shape
    device = targets.device
    num_fsq_dims = len(fsq_levels)

    # Convert flat indices to per-dimension indices
    # Replicate FSQIndicesToToken.indices_to_level_indices logic
    basis = [1]
    for level in fsq_levels[:-1]:
        basis.append(basis[-1] * level)
    basis = torch.tensor(basis, dtype=torch.int32, device=device)
    levels = torch.tensor(fsq_levels, dtype=torch.int32, device=device)

    targets_expanded = targets.unsqueeze(-1)  # [B, N, 1]
    level_targets = (targets_expanded // basis) % levels  # [B, N, num_fsq_dims]

    # Split logits into per-dimension logits
    logits_split = torch.split(
        logits, list(fsq_levels), dim=-1
    )  # List of [B, N, level_i]

    # Compute cross-entropy for each dimension
    total_loss = 0.0
    for dim_idx in range(num_fsq_dims):
        dim_logits = logits_split[dim_idx]  # [B, N, level_i]
        dim_targets = level_targets[..., dim_idx]  # [B, N]

        # Compute cross-entropy: transpose to [B, level_i, N] for F.cross_entropy
        dim_logits_t = dim_logits.transpose(1, 2)
        dim_loss = F.cross_entropy(
            dim_logits_t, dim_targets, reduction="none"
        )  # [B, N]

        # Apply mask and accumulate
        mask_weight = mask.float()
        if mask_weight.sum() > 0:
            dim_loss_masked = (dim_loss * mask_weight).sum() / mask_weight.sum()
        else:
            dim_loss_masked = dim_loss.mean() * 0.0

        total_loss += dim_loss_masked

    # Average across dimensions
    return total_loss


def extract_logits(model_output, output_key: Optional[str] = None) -> torch.Tensor:
    """
    Extract logits from model output (handles dict/object/tensor).

    Args:
        model_output: Model output (can be tensor, dict, or object with attributes)
        output_key: Optional key to extract from output

    Returns:
        Extracted logits tensor
    """
    if output_key is None:
        return model_output

    # Try attribute access (for objects like TokenDistillOutput)
    if hasattr(model_output, output_key):
        return getattr(model_output, output_key)

    # Try dict access
    if isinstance(model_output, dict) and output_key in model_output:
        return model_output[output_key]

    # Fallback error
    raise ValueError(f"Model output does not contain key '{output_key}'")


def sample_tokens(
    logits: torch.Tensor,
    current_x: torch.Tensor,
    mask_token_id: int,
    temperature: float,
    sampling_method: str = "gumbel",
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample token IDs from logits with filtering and method selection.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size]
        current_x: Current token sequence [batch_size, seq_len]
        mask_token_id: Token ID used for masking
        temperature: Sampling temperature
        sampling_method: 'gumbel' or 'argmax'
        top_k: Optional top-k filtering
        top_p: Optional top-p (nucleus) filtering

    Returns:
        Tuple of (sampled_ids, sampled_logits)
    """
    # Apply temperature
    logits = logits / max(temperature, 1e-4)

    # Apply top-k and/or top-p filtering if specified
    filtered_logits = logits.clone()
    if top_k is not None and top_k > 0:
        filtered_logits = top_k_filtering(filtered_logits, top_k, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        filtered_logits = top_p_filtering(filtered_logits, top_p, dim=-1)

    # Sample based on the specified method
    if sampling_method == "gumbel":
        sampled_ids = torch.where(
            current_x == mask_token_id,
            add_gumbel_noise(filtered_logits, temperature).argmax(dim=-1),
            current_x,
        )
    elif sampling_method == "argmax":
        sampled_ids = torch.where(
            current_x == mask_token_id,
            filtered_logits.argmax(dim=-1),
            current_x,
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # Get sampled logits
    sampled_logits = torch.where(
        current_x == mask_token_id,
        logits.gather(dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1),
        +np.inf,
    )

    return sampled_ids, sampled_logits


def compute_next_mask(
    sampled_logits: torch.Tensor,
    current_x: torch.Tensor,
    sampled_ids: torch.Tensor,
    ratio: float,
    schedule_type: str,
    seq_len: int,
    mask_token_id: int,
    temperature: float,
    device: torch.device,
    final_step: bool = False,
) -> torch.Tensor:
    """
    Compute confidence-based mask for next iteration.

    Args:
        sampled_logits: Logits for sampled tokens [batch_size, seq_len]
        current_x: Current token sequence [batch_size, seq_len]
        sampled_ids: Newly sampled token IDs [batch_size, seq_len]
        ratio: Progress ratio (0 to 1)
        schedule_type: Masking schedule type
        seq_len: Sequence length
        mask_token_id: Token ID used for masking
        temperature: Temperature for confidence noise
        device: Device for tensor operations
        final_step: Whether this is the final sampling step

    Returns:
        Next x (masked or final sampled_ids)
    """
    if final_step:
        return sampled_ids

    # Calculate next masking ratio
    next_masking_ratio = get_masking_ratio(ratio, schedule_type)
    current_mask_len = (current_x == mask_token_id).sum(dim=-1)
    next_mask_len = torch.floor(seq_len * next_masking_ratio).to(device)
    next_mask_len = torch.clamp(next_mask_len, max=current_mask_len - 1)

    # Compute confidence scores
    confidence = add_gumbel_noise(sampled_logits, temperature)
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    cut_off = sorted_confidence[:, next_mask_len[0].long() - 1].unsqueeze(-1)
    next_mask = confidence <= cut_off

    # Apply mask
    return torch.where(next_mask, mask_token_id, sampled_ids)


class DiscreteDiffusion:
    def __init__(
        self,
        mask_token_id: int = 0,
        num_timesteps: int = 16,
        schedule_type: str = "cosine",
        temperature: float = 1.0,
        sampling_type: str = "maskgit",
        output_key: Optional[str] = None,
        feed_timestep: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
    ):
        """
        DiscreteDiffusion - Discrete diffusion model with masking as absorbing state

        Args:
            model: Model to predict original tokens from masked tokens
            num_timesteps: Number of diffusion timesteps
            schedule_type: Masking schedule type ('linear' or 'cosine')
            mask_token_id: Token ID used for masking
        """
        print("instantiating DiscreteDiffusion")
        super().__init__()
        self.mask_token_id = mask_token_id
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.temperature = temperature
        self.sampling_type = sampling_type
        self.output_key = output_key
        self.feed_timestep = feed_timestep
        self.top_k = top_k
        self.top_p = top_p
        self.sampling_method = sampling_method

    def model_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        if self.feed_timestep:
            return model(x, t, **model_kwargs)
        else:
            return model(x, **model_kwargs)

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for computing loss during training

        Args:
            model: Model to compute predictions
            x_start: Original token IDs [batch_size, seq_len]
            t: Optional specific timesteps, otherwise randomly sampled
            model_kwargs: Additional keyword arguments for model
            output_key: Optional key to extract from model output if it returns a complex object

        Returns:
            Dictionary with loss and other metrics
        """
        batch_size, seq_len = x_start.shape
        x_start = x_start.to(torch.long)

        if model_kwargs is None:
            model_kwargs = {}

        t = torch.rand(batch_size, device=x_start.device)

        masked_x, mask = mask_tokens(x_start, t, self.schedule_type, self.mask_token_id)
        model_output = self.model_forward(model, masked_x, t, model_kwargs)
        logits = extract_logits(model_output, self.output_key)
        loss = compute_masked_cross_entropy(logits, x_start, mask)

        return {
            "loss": loss,
            "masked_inputs": masked_x,
            "mask": mask,
            "model_output": model_output,
        }

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        progress=False,
        device: torch.device = None,
    ) -> torch.Tensor:
        if self.sampling_type == "maskgit":
            return self.sample_maskgit(
                model,
                shape,
                model_kwargs,
                noise,
                progress,
                device,
                self.top_k,
                self.top_p,
                self.sampling_method,
            )
        else:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")

    @torch.no_grad()
    def sample_maskgit(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        progress=False,
        device: torch.device = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
    ) -> torch.Tensor:
        """
        MaskGIT-style sampling with iterative confidence-based unmasking

        Returns:
            Generated token sequences
        """
        if device is None:
            device = next(model.parameters()).device

        # Start with all masked tokens
        x = torch.full(shape, self.mask_token_id, device=device)
        seq_len = shape[1]

        for t in range(self.num_timesteps):
            ratio = (t + 1) / self.num_timesteps
            annealed_temp = self.temperature * (1 - ratio)

            model_output = self.model_forward(model, x, ratio, model_kwargs)
            logits = extract_logits(model_output, self.output_key)
            sampled_ids, sampled_logits = sample_tokens(
                logits,
                x,
                self.mask_token_id,
                annealed_temp,
                sampling_method,
                top_k,
                top_p,
            )

            final_step = t == self.num_timesteps - 1
            x = compute_next_mask(
                sampled_logits,
                x,
                sampled_ids,
                ratio,
                self.schedule_type,
                seq_len,
                self.mask_token_id,
                annealed_temp,
                device,
                final_step,
            )

        return x

    @torch.no_grad()
    def sample_diffusion(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        temperature: float = 1.0,
        callback=None,
    ) -> torch.Tensor:
        raise NotImplementedError("Sampling not yet implemented for discrete diffusion")


class DiscreteDiffusionWithCFG:
    def __init__(
        self,
        mask_token_id: int = 0,
        num_timesteps: int = 16,
        schedule_type: str = "cosine",
        temperature: float = 1.0,
        sampling_type: str = "maskgit",
        output_key: Optional[str] = None,
        cfg_prob: float = 0.1,
        condition_key: Optional[str] = None,
        cfg_scale: float = 2.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
        for_eval: bool = False,  # dummy parameter to match the interface
        bitwise: bool = False,
        num_bits: Optional[int] = None,
        use_fsq: bool = False,
        fsq_levels: Optional[List[int]] = None,
    ):
        print("instantiating DiscreteDiffusionWithCFG")
        super().__init__()
        self.mask_token_id = mask_token_id
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.temperature = temperature
        self.sampling_type = sampling_type
        self.output_key = output_key
        self.cfg_prob = cfg_prob
        self.condition_key = condition_key
        self.cfg_scale = cfg_scale
        self.top_k = top_k
        self.top_p = top_p
        self.sampling_method = sampling_method
        self.bitwise = bitwise
        self.num_bits = num_bits
        self.use_fsq = use_fsq
        self.fsq_levels = fsq_levels

        # Validate bitwise configuration
        if self.bitwise and self.num_bits is None:
            raise ValueError("num_bits must be provided when bitwise=True")
        if self.bitwise:
            print(f"Bitwise mode enabled with {self.num_bits} bits")

        # Validate FSQ configuration
        if self.use_fsq and self.fsq_levels is None:
            raise ValueError("fsq_levels must be provided when use_fsq=True")
        if self.use_fsq:
            print(f"FSQ mode enabled with levels {self.fsq_levels}")

    def _extract_condition_from_kwargs(
        self, model_kwargs: Optional[Dict]
    ) -> Optional[torch.Tensor]:
        """
        Extract condition tensor from model_kwargs for CFG masking.

        Args:
            model_kwargs: Dictionary containing model arguments

        Returns:
            Condition tensor if found, None otherwise
        """
        if model_kwargs is None:
            return None

        # If condition_key is specified, use that key
        if self.condition_key is not None:
            return model_kwargs.get(self.condition_key)

        return None

    def indices_to_bits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Convert token indices to bit representations.

        Args:
            idx: Token indices of shape [batch_size, seq_len]

        Returns:
            Bit representations of shape [batch_size, seq_len, num_bits]
        """
        mask = 2 ** torch.arange(self.num_bits, device=idx.device)
        # Shape: [batch_size, seq_len, num_bits]
        bits = (idx.unsqueeze(-1) & mask) > 0
        return bits.float()

    def bits_to_indices(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Convert bit predictions back to token indices.

        Args:
            bits: Bit predictions of shape [batch_size, seq_len, num_bits]
                  Values should be 0 or 1 (or probabilities that will be rounded)

        Returns:
            Token indices of shape [batch_size, seq_len]
        """
        # Round to get binary values if needed
        bits = torch.round(bits).long()
        # Convert binary to decimal
        powers = 2 ** torch.arange(self.num_bits, device=bits.device)
        indices = (bits * powers).sum(dim=-1)
        return indices

    def fsq_level_indices_to_flat(self, level_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert per-dimension FSQ indices to flat codebook indices.

        Args:
            level_indices: Per-dimension indices [batch, seq_len, num_fsq_dims]
                          Each dimension in range [0, fsq_levels[i]-1]

        Returns:
            Flat indices [batch, seq_len] in range [0, codebook_size-1]

        Example:
            fsq_levels = [8, 8, 8, 6, 5]
            level_indices = [[7, 7, 7, 5, 4]]  # Max values
            flat = 7*1 + 7*8 + 7*64 + 5*512 + 4*3072 = 15359
        """
        # Compute basis for multi-radix conversion
        # basis[i] = product of all levels before position i
        basis = [1]
        for level in self.fsq_levels[:-1]:
            basis.append(basis[-1] * level)
        basis = torch.tensor(basis, dtype=torch.long, device=level_indices.device)

        # Convert: flat_idx = sum(level_indices[i] * basis[i])
        flat_indices = (level_indices * basis).sum(dim=-1)
        return flat_indices

    def model_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor,
        model_kwargs: Optional[Dict] = None,
    ):
        return model(x, condition, **model_kwargs)

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = x_start.shape
        x_start = x_start.to(torch.long)

        condition_from_kwargs = condition is None
        if model_kwargs is None:
            model_kwargs = {}

        if condition_from_kwargs:
            condition = self._extract_condition_from_kwargs(model_kwargs)

        if condition_from_kwargs and condition is None:
            raise ValueError("No condition found in explicit parameter or model_kwargs")

        t = torch.rand(batch_size, device=x_start.device)

        masked_x, mask = mask_tokens(x_start, t, self.schedule_type, self.mask_token_id)
        masked_condition = mask_condition_for_cfg(condition, self.cfg_prob)

        updated_model_kwargs = model_kwargs.copy()
        if condition_from_kwargs:
            updated_model_kwargs[self.condition_key] = masked_condition

        model_output = self.model_forward(
            model,
            masked_x,
            None if condition_from_kwargs else masked_condition,
            model_kwargs=updated_model_kwargs,
        )
        logits = extract_logits(model_output, self.output_key)

        if self.use_fsq:
            # FSQ mode: per-dimension cross-entropy
            loss = compute_fsq_masked_cross_entropy(
                logits, x_start, mask, self.fsq_levels
            )
        elif self.bitwise:
            # Bitwise mode: logits shape is [batch, seq_len, bits, 2]
            # Convert x_start to bit representation
            x_start_bits = self.indices_to_bits(x_start)  # [batch, seq_len, bits]

            # Compute binary cross-entropy for each bit
            # Reshape for cross_entropy: [batch*seq_len*bits, 2]
            logits_reshaped = logits.reshape(-1, 2)
            # Target shape: [batch*seq_len*bits]
            targets_reshaped = x_start_bits.long().reshape(-1)

            loss = F.cross_entropy(logits_reshaped, targets_reshaped, reduction="none")

            # Reshape back to [batch, seq_len, bits] and sum over bits
            # (since these are log probabilities, we sum them)
            loss = loss.reshape(batch_size, seq_len, self.num_bits).sum(dim=-1)

            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            # Standard mode: logits shape is [batch, seq_len, vocab_size]
            loss = compute_masked_cross_entropy(logits, x_start, mask)

        return {
            "loss": loss,
            "masked_inputs": masked_x,
            "mask": mask,
            "model_output": model_output,
        }

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        condition=None,
        model_kwargs=None,
        device: torch.device = None,
        progress: bool = False,  # dummy parameter to match the interface
        cfg_decay: str = "cosine",
    ) -> torch.Tensor:
        if self.sampling_type == "maskgit":
            return self.sample_maskgit(
                model,
                shape,
                condition,
                model_kwargs,
                device,
                self.cfg_scale,
                cfg_decay,
                self.top_k,
                self.top_p,
                self.sampling_method,
            )
        else:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")

    @torch.no_grad()
    def sample_maskgit(
        self,
        model,
        shape,
        condition=None,
        model_kwargs=None,
        device: torch.device = None,
        cfg_scale: float = 2.0,
        cfg_decay: str = "cosine",
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
    ) -> torch.Tensor:
        """
        MaskGIT-style sampling with iterative confidence-based unmasking with CFG support

        Returns:
            Generated token sequences
        """
        condition_from_kwargs = condition is None
        if condition is None:
            condition = self._extract_condition_from_kwargs(model_kwargs)

        if condition is None:
            raise ValueError("No condition found in explicit parameter or model_kwargs")

        if device is None:
            device = next(model.parameters()).device

        # Start with all masked tokens
        x = torch.full(shape, self.mask_token_id, device=device)
        seq_len = shape[1]
        initial_cfg_scale = cfg_scale
        use_cfg = initial_cfg_scale >= 1.0

        for t in range(self.num_timesteps):
            ratio = (t + 1) / self.num_timesteps
            annealed_temp = self.temperature * (1 - ratio)

            if cfg_decay == "cosine" and use_cfg:
                scale_step = (
                    (1 - torch.cos(torch.ones((1), device=device) * torch.pi)) * 1 / 2
                )
                cfg_scale = (initial_cfg_scale - 1) * scale_step + 1
            elif cfg_decay == "linear" and use_cfg:
                cfg_scale = (initial_cfg_scale - 1) * ratio + 1
            else:
                cfg_scale = initial_cfg_scale

            if use_cfg:
                # Classifier-free guidance: batched inference for efficiency
                zero_condition = torch.zeros_like(condition, device=condition.device)
                batched_condition = torch.cat([condition, zero_condition], dim=0)

                # Update model_kwargs with batched tensors for CFG
                updated_model_kwargs = {}
                if model_kwargs is not None:
                    for key, value in model_kwargs.items():
                        if isinstance(value, torch.Tensor):
                            # Batch all tensors to match the batched input
                            updated_model_kwargs[key] = torch.cat([value, value], dim=0)
                        else:
                            # Keep non-tensor values as is
                            updated_model_kwargs[key] = value

                    # Update the condition key specifically with our batched condition
                    if condition_from_kwargs:
                        updated_model_kwargs[self.condition_key] = batched_condition

                # Single batched forward pass
                batched_output = self.model_forward(
                    model,
                    torch.cat([x, x], dim=0),
                    None if condition_from_kwargs else batched_condition,
                    model_kwargs=updated_model_kwargs,
                )
                batched_logits = extract_logits(batched_output, self.output_key)

                # Split batched results: first half is conditional, second half is unconditional
                batch_size = x.shape[0]
                cond_logits, uncond_logits = (
                    batched_logits[:batch_size],
                    batched_logits[batch_size:],
                )

                # CFG interpolation
                logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)

            else:
                model_output = self.model_forward(
                    model,
                    x,
                    None if condition_from_kwargs else condition,
                    model_kwargs=model_kwargs,
                )
                logits = extract_logits(model_output, self.output_key)

            if self.use_fsq:
                # FSQ mode: logits shape is [batch, seq_len, sum(fsq_levels)]
                # Split into per-dimension logits
                logits_split = torch.split(logits, list(self.fsq_levels), dim=-1)
                # logits_split is list of [batch, seq_len, level_i] tensors

                # Apply temperature to each dimension
                logits_split = [
                    dim_logits / max(annealed_temp, 1e-4) for dim_logits in logits_split
                ]

                # Apply top-k and top-p filtering per dimension (if specified)
                if top_k is not None and top_k > 0:
                    logits_split = [
                        top_k_filtering(dim_logits, top_k, dim=-1)
                        for dim_logits in logits_split
                    ]
                if top_p is not None and 0.0 < top_p < 1.0:
                    logits_split = [
                        top_p_filtering(dim_logits, top_p, dim=-1)
                        for dim_logits in logits_split
                    ]

                # Sample each dimension independently
                dim_samples = []
                dim_log_probs = []

                for dim_logits in logits_split:
                    # Sample based on method
                    if sampling_method == "gumbel":
                        dim_sample = add_gumbel_noise(dim_logits, annealed_temp).argmax(
                            dim=-1
                        )
                    elif sampling_method == "argmax":
                        dim_sample = dim_logits.argmax(dim=-1)
                    else:
                        raise ValueError(f"Unknown sampling method: {sampling_method}")

                    dim_samples.append(dim_sample)  # [batch, seq_len]

                    # Calculate log probability of selected sample for confidence
                    log_prob = torch.log_softmax(
                        dim_logits, dim=-1
                    )  # [batch, seq_len, level_i]
                    selected_log_prob = log_prob.gather(
                        dim=-1, index=dim_sample.unsqueeze(-1)
                    ).squeeze(-1)  # [batch, seq_len]
                    dim_log_probs.append(selected_log_prob)

                # Stack per-dimension samples: [batch, seq_len, num_fsq_dims]
                level_indices = torch.stack(dim_samples, dim=-1)

                # Convert to flat FSQ indices
                sampled_ids_from_fsq = self.fsq_level_indices_to_flat(level_indices)

                # Only update masked positions
                sampled_ids = torch.where(
                    x == self.mask_token_id,
                    sampled_ids_from_fsq,
                    x,
                )

                # Calculate token confidence from per-dimension probabilities
                # Token confidence = product of dimension probs = sum in log space
                token_log_confidence = torch.stack(dim_log_probs, dim=-1).sum(
                    dim=-1
                )  # [batch, seq_len]

                sampled_logits = torch.where(
                    x == self.mask_token_id,
                    token_log_confidence,  # Already in log space
                    +np.inf,  # Non-masked positions get infinite confidence (never re-masked)
                )
            elif self.bitwise:
                # Bitwise mode: logits shape is [batch, seq_len, bits, 2]
                logits = logits / max(annealed_temp, 1e-4)

                # Sample or argmax to get bit predictions
                if sampling_method == "gumbel":
                    bit_predictions = add_gumbel_noise(logits, annealed_temp).argmax(
                        dim=-1
                    )  # [batch, seq_len, bits]
                elif sampling_method == "argmax":
                    bit_predictions = logits.argmax(dim=-1)  # [batch, seq_len, bits]
                else:
                    raise ValueError(f"Unknown sampling method: {sampling_method}")

                # Convert bit predictions to token indices
                sampled_ids_from_bits = self.bits_to_indices(bit_predictions.float())

                # Only update masked positions
                sampled_ids = torch.where(
                    x == self.mask_token_id,
                    sampled_ids_from_bits,
                    x,
                )

                # Calculate token confidence from bit probabilities
                # First convert logits to log probabilities
                log_probs = torch.log_softmax(
                    logits, dim=-1
                )  # [batch, seq_len, bits, 2]

                # Get the log probability of the selected bit for each position
                selected_bit_log_probs = log_probs.gather(
                    dim=-1, index=bit_predictions.unsqueeze(-1)
                ).squeeze(-1)  # [batch, seq_len, bits]

                # Token confidence is the product of all bit probabilities (or sum in log space)
                # log P(token) = log P(bit_0) + log P(bit_1) + ... + log P(bit_n)
                token_log_confidence = selected_bit_log_probs.sum(
                    dim=-1
                )  # [batch, seq_len]

                sampled_logits = torch.where(
                    x == self.mask_token_id,
                    token_log_confidence,  # Already in log space
                    +np.inf,
                )
            else:
                # Standard mode: logits shape is [batch, seq_len, vocab_size]
                sampled_ids, sampled_logits = sample_tokens(
                    logits,
                    x,
                    self.mask_token_id,
                    annealed_temp,
                    sampling_method,
                    top_k,
                    top_p,
                )

            final_step = t == self.num_timesteps - 1
            x = compute_next_mask(
                sampled_logits,
                x,
                sampled_ids,
                ratio,
                self.schedule_type,
                seq_len,
                self.mask_token_id,
                annealed_temp,
                device,
                final_step,
            )

        return x


class DiscreteDiffusionWithHistoryMasking:
    def __init__(
        self,
        mask_token_id: int = 0,
        num_timesteps: int = 16,
        schedule_type: str = "linear",
        temperature: float = 1.0,
        sampling_type: str = "maskgit",
        output_key: Optional[str] = None,
        history_key: str = "x_cond",
        history_sample_t: float = 0.8,  # 1 means no masking
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
        for_eval: bool = False,  # dummy parameter to match the interface
        cfg_prob: float = 0.0,  # Probability of masking action for CFG training (0 = disabled)
        cfg_scale: float = 0.0,  # CFG scale for sampling (0 = no CFG, >0 = amplified guidance)
        without_history_masking: bool = False,
    ):
        """
        DiscreteDiffusionWithHistoryMasking - Discrete diffusion with schedule-based history frame masking and action CFG

        During training, masks tokens in history frames using random timesteps from the schedule.
        During sampling, uses a fixed timestep for consistent light masking.
        Also supports classifier-free guidance on actions using learnable null action embedding.

        Args:
            mask_token_id: Token ID used for masking
            num_timesteps: Number of diffusion timesteps
            schedule_type: Masking schedule type ('linear', 'cosine', etc.)
            temperature: Sampling temperature
            sampling_type: Sampling algorithm ('maskgit')
            output_key: Optional key to extract from model output
            history_key: Key in model_kwargs for history frames (e.g., "x_cond")
            history_sample_t: Fixed timestep for history masking during sampling (e.g., 0.2)
            top_k: Optional top-k filtering during sampling
            top_p: Optional top-p (nucleus) filtering during sampling
            sampling_method: Sampling method ('gumbel' or 'argmax')
            cfg_prob: Probability of masking action for CFG training (0 = disabled)
            cfg_scale: CFG scale for sampling (0 = no CFG, >0 = amplified guidance)
        """
        print("instantiating DiscreteDiffusionWithHistoryMasking")
        super().__init__()

        # Validate CFG parameters
        assert 0.0 <= cfg_prob <= 1.0, f"cfg_prob must be in [0, 1], got {cfg_prob}"
        assert cfg_scale >= 0.0, f"cfg_scale must be >= 0, got {cfg_scale}"

        self.mask_token_id = mask_token_id
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.temperature = temperature
        self.sampling_type = sampling_type
        self.output_key = output_key
        self.history_key = history_key
        self.history_sample_t = history_sample_t
        self.top_k = top_k
        self.top_p = top_p
        self.sampling_method = sampling_method
        self.cfg_prob = cfg_prob
        self.cfg_scale = cfg_scale
        self.without_history_masking = without_history_masking

    def _extract_history_from_kwargs(
        self, model_kwargs: Optional[Dict]
    ) -> Optional[torch.Tensor]:
        """
        Extract history frames tensor from model_kwargs.

        Args:
            model_kwargs: Dictionary containing model arguments

        Returns:
            History tensor if found, None otherwise
        """
        if model_kwargs is None:
            return None
        return model_kwargs.get(self.history_key)

    def _mask_history_frames(
        self, history: torch.Tensor, fixed_t: Optional[float] = None
    ) -> torch.Tensor:
        """
        Randomly mask tokens in history frames using the masking schedule.
        Each frame gets an independent random timestep from the schedule (training),
        or uses a fixed timestep (sampling).

        Args:
            history: History frames tensor of shape [batch, context_size, seq_len]
            fixed_t: If provided, use this fixed timestep for all frames (sampling mode)

        Returns:
            Masked history frames with same shape
        """
        batch_size, context_size, seq_len = history.shape

        if fixed_t is not None:
            # Sampling mode: use fixed timestep for all frames
            t = torch.full((batch_size, context_size), fixed_t, device=history.device)
        else:
            # Training mode: sample random timesteps for each frame
            t = torch.rand(batch_size, context_size, device=history.device)

        # Get masking ratio for each frame: [batch, context_size]
        mask_ratio = get_masking_ratio(t, self.schedule_type)
        mask_ratio = torch.clamp(mask_ratio, 0.0, 1.0)

        # Calculate how many tokens to mask per frame: [batch, context_size]
        mask_len = torch.floor(seq_len * mask_ratio)
        mask_len = torch.clamp(mask_len, min=0, max=seq_len)

        # For each frame, randomly select which tokens to mask
        batch_rand = torch.rand(
            batch_size, context_size, seq_len, device=history.device
        ).argsort(dim=-1)
        mask = batch_rand < mask_len.unsqueeze(-1)

        masked_history = torch.where(mask, self.mask_token_id, history)

        return masked_history

    def model_forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ):
        return model(x, t, **model_kwargs)

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for computing loss during training with history masking

        Args:
            model: Model to compute predictions
            x_start: Original token IDs [batch_size, seq_len]
            t: Optional specific timesteps, otherwise randomly sampled
            model_kwargs: Additional keyword arguments for model (must contain history_key)

        Returns:
            Dictionary with loss and other metrics
        """
        batch_size, seq_len = x_start.shape
        x_start = x_start.to(torch.long)

        if model_kwargs is None:
            model_kwargs = {}

        history = self._extract_history_from_kwargs(model_kwargs)
        if history is None:
            raise ValueError(
                f"No history found in model_kwargs with key '{self.history_key}'"
            )

        masked_history = self._mask_history_frames(
            history, fixed_t=1 if self.without_history_masking else None
        )

        updated_model_kwargs = model_kwargs.copy()
        updated_model_kwargs[self.history_key] = masked_history

        # Action CFG: randomly mask actions during training (only if cfg_prob > 0)
        if self.cfg_prob > 0:
            # action_mask: True = use real action, False = use null action
            action_mask = torch.rand(batch_size, device=x_start.device) >= self.cfg_prob
            updated_model_kwargs["action_mask"] = action_mask

        t = torch.rand(batch_size, device=x_start.device)

        masked_x, mask = mask_tokens(x_start, t, self.schedule_type, self.mask_token_id)
        model_output = self.model_forward(model, masked_x, t, updated_model_kwargs)
        logits = extract_logits(model_output, self.output_key)
        loss = compute_masked_cross_entropy(logits, x_start, mask)

        return {
            "loss": loss,
            "masked_inputs": masked_x,
            "mask": mask,
            "model_output": model_output,
        }

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        progress=False,
        device: torch.device = None,
    ) -> torch.Tensor:
        if self.sampling_type == "maskgit":
            return self.sample_maskgit(
                model,
                shape,
                model_kwargs,
                noise,
                progress,
                device,
                self.top_k,
                self.top_p,
                self.sampling_method,
            )
        else:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")

    @torch.no_grad()
    def sample_maskgit(
        self,
        model,
        shape,
        model_kwargs=None,
        noise=None,
        progress=False,
        device: torch.device = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        sampling_method: str = "gumbel",
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        MaskGIT-style sampling with iterative confidence-based unmasking
        with random masking applied to history frames and action CFG

        Returns:
            Generated token sequences
        """
        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        # Use provided cfg_scale or default to instance variable
        if cfg_scale is None:
            cfg_scale = self.cfg_scale

        history = self._extract_history_from_kwargs(model_kwargs)
        if history is None:
            raise ValueError(
                f"No history found in model_kwargs with key '{self.history_key}'"
            )

        masked_history = self._mask_history_frames(
            history, fixed_t=self.history_sample_t
        )

        updated_model_kwargs = model_kwargs.copy()
        updated_model_kwargs[self.history_key] = masked_history

        x = torch.full(shape, self.mask_token_id, device=device)
        seq_len = shape[1]
        # Only use CFG if cfg_prob > 0 (model was trained with CFG) and cfg_scale > 0
        use_cfg = self.cfg_prob > 0 and cfg_scale > 0

        for t in range(self.num_timesteps):
            ratio = (t + 1) / self.num_timesteps
            annealed_temp = self.temperature * (1 - ratio)

            if use_cfg:
                # Batched CFG inference for efficiency
                batch_size = x.shape[0]

                # Create action masks: first half conditional (True), second half unconditional (False)
                conditional_mask = torch.ones(
                    batch_size, dtype=torch.bool, device=device
                )
                unconditional_mask = torch.zeros(
                    batch_size, dtype=torch.bool, device=device
                )
                batched_action_mask = torch.cat(
                    [conditional_mask, unconditional_mask], dim=0
                )

                # Batch all tensors in model_kwargs for CFG
                batched_model_kwargs = {}
                for key, value in updated_model_kwargs.items():
                    if isinstance(value, torch.Tensor):
                        batched_model_kwargs[key] = torch.cat([value, value], dim=0)
                    else:
                        batched_model_kwargs[key] = value

                # Add batched action mask
                batched_model_kwargs["action_mask"] = batched_action_mask

                # Single batched forward pass
                batched_t = torch.full((batch_size * 2,), ratio, device=device)
                batched_output = self.model_forward(
                    model, torch.cat([x, x], dim=0), batched_t, batched_model_kwargs
                )
                batched_logits = extract_logits(batched_output, self.output_key)

                # Split batched results
                cond_logits, uncond_logits = (
                    batched_logits[:batch_size],
                    batched_logits[batch_size:],
                )

                # CFG interpolation: cond + scale * (cond - uncond)
                # scale = 0: pure conditional, scale > 0: amplified guidance
                logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                # No CFG: standard forward pass (action_mask=None defaults to conditional)
                model_output = self.model_forward(model, x, ratio, updated_model_kwargs)
                logits = extract_logits(model_output, self.output_key)

            sampled_ids, sampled_logits = sample_tokens(
                logits,
                x,
                self.mask_token_id,
                annealed_temp,
                sampling_method,
                top_k,
                top_p,
            )

            final_step = t == self.num_timesteps - 1
            x = compute_next_mask(
                sampled_logits,
                x,
                sampled_ids,
                ratio,
                self.schedule_type,
                seq_len,
                self.mask_token_id,
                annealed_temp,
                device,
                final_step,
            )

        return x
