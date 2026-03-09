"""
Tokenizer Metrics Module

This module implements reconstruction FID (rFID) and other metrics for tokenizer evaluation.
rFID measures the distribution distance between original images and reconstructions using
Fréchet Inception Distance.
"""

from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from cleanfid import fid

from PIL import Image
import numpy as np
import os
import torch
import torch.nn.functional as F
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_batch_features
from cleanfid.resize import build_resizer
from scipy.stats import entropy


class ReconstructionFID(Metric):
    def __init__(self, mode="clean"):
        super().__init__()
        self.mode = mode
        self.feat_model = build_feature_extractor(mode)
        self.fn_resize = build_resizer(mode)

        self.add_state("fake_features", default=[], dist_reduce_fx="cat")
        self.add_state("real_features", default=[], dist_reduce_fx="cat")

    """
    Funtion that takes an image (PIL.Image or np.array or torch.tensor)
    and returns the corresponding feature embedding vector.
    The image x is expected to be in range [0, 255]
    """

    def compute_features(self, x):
        # if x is a PIL Image
        if isinstance(x, Image.Image):
            x_np = np.array(x)
            x_np_resized = self.fn_resize(x_np)
            x_t = torch.tensor(x_np_resized.transpose((2, 0, 1))).unsqueeze(0)
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, np.ndarray):
            x_np_resized = self.fn_resize(x)
            x_t = (
                torch.tensor(x_np_resized.transpose((2, 0, 1)))
                .unsqueeze(0)
                .to(self.device)
            )
            # normalization happens inside the self.feat_model, expected image range here is [0,255]
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, torch.Tensor):
            # pdb.set_trace()
            # add the batch dimension if x is passed in as C,H,W
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            b, c, h, w = x.shape
            # convert back to np array and resize
            l_x_np_resized = []
            for _ in range(b):
                x_np = x[_].cpu().numpy().transpose((1, 2, 0))
                l_x_np_resized.append(self.fn_resize(x_np)[None,])
            x_np_resized = np.concatenate(l_x_np_resized)
            x_t = torch.tensor(x_np_resized.transpose((0, 3, 1, 2))).to(self.device)
            # normalization happens inside the self.feat_model, expected image range here is [0,255]
            with torch.no_grad():
                x_feat = (
                    self.feat_model(x_t.float().to(self.device))
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
        else:
            raise ValueError("image type could not be inferred")
        return x_feat

    """
    Extract the faetures from x and add to the list of generated images
    """

    def update(self, real, fake):
        x_feat_real = self.compute_features(real.float())
        x_feat_fake = self.compute_features(fake.float())
        self.fake_features.append(
            torch.tensor(x_feat_fake, device=self.device, dtype=torch.float32)
        )
        self.real_features.append(
            torch.tensor(x_feat_real, device=self.device, dtype=torch.float32)
        )

    def compute(self):
        feats1 = dim_zero_cat(self.real_features).cpu().numpy()
        feats2 = dim_zero_cat(self.fake_features).cpu().numpy()
        mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
        mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
        return fid.frechet_distance(mu1, sig1, mu2, sig2)


class ReconstructionIS(Metric):
    """
    Inception Score (IS) metric for measuring quality and diversity of reconstructed images.

    IS measures both the quality (each image should be confidently classified) and diversity
    (images should span multiple classes) of generated/reconstructed images.

    IS = exp(E_x[KL(p(y|x) || p(y))])
    where p(y|x) is the conditional class distribution and p(y) is the marginal distribution.
    """

    def __init__(self, epsilon: float = 1e-16):
        """
        Args:
            epsilon: Small constant for numerical stability in log computation
        """
        super().__init__()
        self.epsilon = epsilon

        # Load Inception model for IS computation
        from token_distill.inception import get_inception_model

        self.inception_model = get_inception_model()
        self.inception_model.eval()

        # Number of ImageNet classes (logits_unbiased dimension)
        self._num_classes = 1008

        # Register metric states for distributed training
        # prob_total: accumulated sum of probabilities across all images
        self.add_state(
            "prob_total",
            default=torch.zeros(self._num_classes, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        # total_kl_d: accumulated entropy term for KL divergence computation
        self.add_state(
            "total_kl_d",
            default=torch.zeros(self._num_classes, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        # num_examples: total number of images processed
        self.add_state(
            "num_examples",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(self, reconstructed) -> None:
        """
        Update Inception Score metric with reconstructed images.

        Args:
            reconstructed: Tensor of reconstructed images in range [0, 1],
                          shape (batch_size, channels, height, width)
        """
        batch_size = reconstructed.shape[0]

        # Extract features and compute probabilities
        with torch.no_grad():
            features = self.inception_model(
                reconstructed.to(self.device, dtype=torch.uint8)
            )
            logits = features["logits_unbiased"]
            probs = F.softmax(logits, dim=-1)

        prob_sum = torch.sum(probs, dim=0, dtype=torch.float64)

        log_probs = torch.log(probs + self.epsilon)
        if log_probs.dtype != probs.dtype:
            log_probs = log_probs.to(probs.dtype)
        kl_sum = torch.sum(probs * log_probs, dim=0, dtype=torch.float64)

        # Update states
        self.prob_total += prob_sum
        self.total_kl_d += kl_sum
        self.num_examples += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute Inception Score from accumulated statistics.

        IS = exp(E_x[KL(p(y|x) || p(y))])

        Returns:
            Inception Score as a tensor
        """
        if self.num_examples == 0:
            return torch.tensor(0.0, device=self.device)

        mean_probs = self.prob_total / self.num_examples

        # Compute log of marginal distribution
        log_mean_probs = torch.log(mean_probs + self.epsilon)
        if log_mean_probs.dtype != self.prob_total.dtype:
            log_mean_probs = log_mean_probs.to(self.prob_total.dtype)

        excess_entropy = self.prob_total * log_mean_probs

        avg_kl_d = torch.sum(self.total_kl_d - excess_entropy) / self.num_examples

        # Inception Score
        inception_score = torch.exp(avg_kl_d)

        return inception_score


class GenerationFID(Metric):
    """
    FID metric for generated images using precomputed real statistics.

    This metric computes FID between generated images and a reference dataset
    whose statistics (mu, sigma) have been precomputed and saved to an .npz file.

    Typical use case: Computing FID against ImageNet validation set for class-conditional generation.
    """

    def __init__(self, stats_path: str, mode: str = "clean"):
        """
        Args:
            stats_path: Path to .npz file containing precomputed 'mu' and 'sigma' keys
            mode: Feature extraction mode ("clean" for InceptionV3, "legacy_pytorch" etc.)
        """
        super().__init__()
        self.mode = mode
        self.stats_path = stats_path

        # Build feature extractor
        self.feat_model = build_feature_extractor(mode)
        self.fn_resize = build_resizer(mode)

        # Load precomputed real statistics
        self.mu_real, self.sigma_real = self._load_precomputed_stats()

        self._feature_dim = self.mu_real.shape[0]
        # Track generated statistics in a streaming manner to avoid storing all features.
        self.add_state(
            "generated_count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "generated_sum",
            default=torch.zeros(self._feature_dim, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "generated_sum_outer",
            default=torch.zeros(
                (self._feature_dim, self._feature_dim), dtype=torch.float64
            ),
            dist_reduce_fx="sum",
        )

    def _load_precomputed_stats(self):
        """Load precomputed mu and sigma from .npz file."""
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Precomputed stats not found: {self.stats_path}")

        stats = np.load(self.stats_path)
        if "mu" not in stats or "sigma" not in stats:
            raise ValueError(
                f"Stats file must contain 'mu' and 'sigma' keys. Found: {list(stats.keys())}"
            )

        mu = torch.tensor(stats["mu"], dtype=torch.float32)
        sigma = torch.tensor(stats["sigma"], dtype=torch.float32)

        return mu, sigma

    def compute_features(self, x):
        """
        Compute features for images.

        Args:
            x: Images as PIL.Image, np.ndarray, or torch.Tensor in range [0, 255]

        Returns:
            Feature array
        """
        # Reuse the same feature computation logic as ReconstructionFID
        if isinstance(x, Image.Image):
            x_np = np.array(x)
            x_np_resized = self.fn_resize(x_np)
            x_t = torch.tensor(x_np_resized.transpose((2, 0, 1))).unsqueeze(0)
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, np.ndarray):
            x_np_resized = self.fn_resize(x)
            x_t = (
                torch.tensor(x_np_resized.transpose((2, 0, 1)))
                .unsqueeze(0)
                .to(self.device)
            )
            x_feat = get_batch_features(x_t, self.feat_model, self.device)
        elif isinstance(x, torch.Tensor):
            # Add batch dimension if needed
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            b, c, h, w = x.shape

            # Resize each image in the batch
            l_x_np_resized = []
            for i in range(b):
                x_np = x[i].cpu().numpy().transpose((1, 2, 0))
                l_x_np_resized.append(self.fn_resize(x_np)[None,])
            x_np_resized = np.concatenate(l_x_np_resized)
            x_t = torch.tensor(x_np_resized.transpose((0, 3, 1, 2))).to(self.device)

            # Extract features
            with torch.no_grad():
                x_feat = (
                    self.feat_model(x_t.float().to(self.device))
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
        else:
            raise ValueError("Image type could not be inferred")

        return x_feat

    def update(self, generated: torch.Tensor):
        """
        Add generated images to the metric.

        Args:
            generated: Generated images as tensor in range [0, 255]
        """
        gen_features = self.compute_features(generated.float())
        if isinstance(gen_features, np.ndarray):
            features = torch.from_numpy(gen_features)
        else:
            features = torch.tensor(gen_features)
        features = features.to(dtype=torch.float64, device=self.device)

        self.generated_count += features.shape[0]
        self.generated_sum += features.sum(dim=0)
        self.generated_sum_outer += features.T @ features

    def compute(self) -> torch.Tensor:
        """
        Compute FID between accumulated generated features and precomputed real statistics.

        Returns:
            FID score
        """
        if self.generated_count.item() == 0:
            return torch.tensor(float("nan"), device=self.device)

        mean = self.generated_sum / self.generated_count

        if self.generated_count.item() > 1:
            sum_outer = self.generated_sum_outer
            cov = (sum_outer - self.generated_count * torch.outer(mean, mean)) / (
                self.generated_count - 1
            )
        else:
            cov = torch.zeros(
                (self._feature_dim, self._feature_dim), dtype=torch.float64
            )

        mu_gen = mean.cpu().numpy()
        sigma_gen = cov.cpu().numpy()

        # Compute Frechet distance
        fid_score = fid.frechet_distance(
            self.mu_real.cpu().numpy(), self.sigma_real.cpu().numpy(), mu_gen, sigma_gen
        )

        return torch.tensor(fid_score, dtype=torch.float32, device=self.device)


class Perplexity(Metric):
    """
    Perplexity metric for measuring codebook utilization efficiency.

    Computes perplexity based on the distribution of codebook indices.
    Lower perplexity indicates better vocabulary utilization.
    """

    def __init__(self, pool_size: int, base: int = 2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.base = base
        self.add_state(
            "density", default=torch.zeros((self.pool_size)), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.zeros((1)), dist_reduce_fx="sum")

    def update(self, indices: torch.Tensor) -> None:
        """
        Update the perplexity metric with new codebook indices.

        Args:
            indices: Tensor of codebook indices, shape (batch_size, sequence_length)
        """
        # Flatten indices and convert to cpu for histogram computation
        indices_flat = indices.flatten().cpu().data.numpy()

        # Compute histogram of indices
        density, _ = np.histogram(
            indices_flat, bins=self.pool_size, range=(0, self.pool_size), density=True
        )

        # Update state
        self.density += torch.tensor(density, dtype=torch.float32, device=self.device)
        self.count += 1

    def compute(self) -> torch.Tensor:
        """
        Compute the perplexity from accumulated density statistics.

        Returns:
            Perplexity value as a tensor
        """
        if self.count == 0:
            return torch.tensor(0.0, device=self.device)
        else:
            # Compute average density
            density = self.density.cpu().data.numpy()
            count = self.count.cpu().data.numpy()
            avg_density = density / count

            # Compute perplexity using entropy
            perplexity = self.base ** entropy(avg_density, base=self.base)
            return torch.tensor(perplexity, dtype=torch.float32, device=self.device)

    @staticmethod
    def compute_direct(
        indices: torch.Tensor, pool_size: int, base: int = 2
    ) -> torch.Tensor:
        """
        Compute perplexity directly from current batch indices without accumulation.

        Args:
            indices: Tensor of codebook indices, shape (batch_size, sequence_length)
            pool_size: Size of the codebook (number of possible indices)
            base: Base for entropy computation (default: 2)

        Returns:
            Perplexity value as a tensor
        """
        if indices is None:
            return torch.tensor(0.0)

        # Flatten indices and convert to cpu for histogram computation
        indices_flat = indices.flatten().cpu().data.numpy()

        # Compute histogram of indices
        density, _ = np.histogram(
            indices_flat, bins=pool_size, range=(0, pool_size), density=True
        )

        # Compute perplexity using entropy
        perplexity = base ** entropy(density, base=base)
        return torch.tensor(perplexity, dtype=torch.float32, device=indices.device)


class CorrectTokens(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct_tokens", default=torch.zeros((1)), dist_reduce_fx="sum")
        self.add_state("total_tokens", default=torch.zeros((1)), dist_reduce_fx="sum")

    def update(self, inputs, targets, masks) -> None:
        correct_tokens = ((inputs == targets) * masks).sum(dim=1)
        self.correct_tokens += correct_tokens.sum()
        self.total_tokens += masks.sum()

    def compute(self) -> torch.Tensor:
        if self.total_tokens == 0:
            return 0
        else:
            return self.correct_tokens / self.total_tokens

    @staticmethod
    def compute_direct(
        inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        correct_tokens = ((inputs == targets) * masks).sum(dim=1)
        return correct_tokens.sum() / masks.sum()