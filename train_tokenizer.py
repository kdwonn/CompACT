"""
Tokenizer Training Script using PyTorch Lightning and Hydra

This script provides a comprehensive boilerplate for training various tokenizers
(CompactTokenizer, FlexTok, VAE, etc.) using PyTorch Lightning and Hydra configuration management.
"""

import logging
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import lightning as L
from PIL import Image
import numpy as np
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.profilers import PyTorchProfiler
from torchmetrics import MeanMetric
import wandb

# Import project modules
from token_distill.rep_guidance import DummyRepGuidanceLoss
from hydra_utils import save_config
from datasets import (
    WebDatasetDataModule,
    DebugDatasetDataModule,
    JSONWebDatasetDataModule,
)
from train_utils import (
    setup_tokenizer,
    setup_diffusion,
    setup_optimizer,
    setup_scheduler,
)
from tokenizer_metrics import ReconstructionFID, Perplexity, CorrectTokens
from ema_pytorch import EMA

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


class TokenizerLightningModule(L.LightningModule):
    """
    Lightning Module for tokenizer training that handles the training loop,
    optimization, and logging.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

        # Initialize tokenizer model
        self.tokenizer = self._setup_tokenizer()
        self.ema_tokenizer = EMA(
            self.tokenizer.compact_tokenizer,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
        )

        # Training configuration
        self.learning_rate = config.training.optimizer.lr
        self.weight_decay = config.training.optimizer.weight_decay
        self.grad_clip_val = config.training.get("grad_clip_val", 1.0)

        # Logging configuration
        self.log_metrics_every_n_steps = config.training.get(
            "log_metrics_every_n_steps", 50
        )

        # Loss weights
        self.commit_loss_weight = config.training.get("commit_loss_weight", 0.25)
        self.diffusion_loss_weight = config.training.get("diffusion_loss_weight", 1.0)
        self.rep_guidance_loss_weight = config.training.get(
            "rep_guidance_loss_weight", 0.1
        )

        # Metrics - using TorchMetrics for efficient computation
        self.train_commit_loss = MeanMetric()
        self.train_diffusion_loss = MeanMetric()
        self.train_rep_guidance_loss = MeanMetric()
        self.val_commit_loss = MeanMetric()
        self.val_diffusion_loss = MeanMetric()
        self.val_rep_guidance_loss = MeanMetric()
        self.val_rfid = ReconstructionFID()

        # Initialize perplexity metric with codebook size from config
        self.codebook_size = self.tokenizer.codebook_size
        self.val_perplexity = Perplexity(pool_size=self.codebook_size, base=2)

        # Initialize rep_guidance loss
        self.rep_guidance_loss = self._setup_rep_guidance_loss()

        # Local image logging configuration
        self.log_images_locally = config.training.get("log_images_locally", False)
        self.local_image_log_dir = config.training.get("local_image_log_dir", None)
        if self.log_images_locally and self.local_image_log_dir is None:
            # Default to results_dir/images
            self.local_image_log_dir = os.path.join(
                config.training.results_dir, config.training.run_name, "images"
            )

    def _setup_tokenizer(self) -> nn.Module:
        """Setup the tokenizer model based on configuration."""
        # Use the helper function from train_utils.py
        return setup_tokenizer(self.config, self.device)

    def _setup_diffusion(self):
        """Setup the diffusion model based on configuration."""
        # Use the helper function from train_utils.py
        return setup_diffusion(self.config, for_eval=False, device=self.device)

    def _setup_rep_guidance_loss(self):
        """Setup the rep_guidance loss based on configuration."""
        if self.rep_guidance_loss_weight > 0.0 and "rep_guidance" in self.config.model:
            try:
                return instantiate(self.config.model.rep_guidance)
            except Exception as e:
                logger.warning(
                    f"Failed to initialize rep_guidance loss: {e}. Using dummy loss."
                )
                return DummyRepGuidanceLoss()
        else:
            return DummyRepGuidanceLoss()

    def forward(self, x):
        """Forward pass through the tokenizer."""
        return self.tokenizer(x)

    def _compute_losses(self, batch, batch_idx, stage="train"):
        """Compute all losses for the given batch."""
        # Check if dual normalization is enabled
        use_dual_normalization = self.config.dataset.get(
            "use_dual_normalization", False
        )

        # Check if we should use DINOv2 images for compact tokenizer
        use_dino_image_for_compact = self.config.dataset.get(
            "use_dino_image_for_compact", False
        )

        # Validate configuration
        if use_dino_image_for_compact and not use_dual_normalization:
            raise ValueError(
                "use_dino_image_for_compact=True requires use_dual_normalization=True"
            )

        # Check if we're using pre-encoded indices (only during training)
        use_pre_encoded_indices = stage == "train" and self.config.dataset.get(
            "use_pre_encoded_indices", False
        )

        # Unpack batch based on format and stage
        if not isinstance(batch, (list, tuple)):
            raise ValueError(f"Invalid batch format: {batch}")
        try:
            # Batch unpacking for 4 cases:
            # 1. Standard single norm: (images, labels, keys) -> dinov2_images=images, pre_encoded_indices=None
            # 2. Standard dual norm: (images, dinov2_images, labels, keys) -> dinov2_images=batch[1], pre_encoded_indices=None
            # 3. JSON single norm: (images, indices, labels, keys) -> dinov2_images=images, pre_encoded_indices=batch[1]
            # 4. JSON dual norm: (images, dinov2_images, indices, labels, keys) -> dinov2_images=batch[1], pre_encoded_indices=batch[2]
            images = batch[0]
            dinov2_images = images
            if use_dual_normalization:
                dinov2_images = batch[1]
            pre_encoded_indices = None
            if use_pre_encoded_indices:
                pre_encoded_indices = batch[2 if use_dual_normalization else 1]
        except Exception:
            raise ValueError(f"Invalid batch format: {batch}")

        # Determine which image to use for compact encoder
        compact_encoder_image = None
        if use_dual_normalization and use_dino_image_for_compact:
            compact_encoder_image = dinov2_images

        # Forward pass through tokenizer with optional pre-encoded indices and compact encoder image
        tokenizer_output = self.tokenizer(
            images,
            recon_image=False,
            pre_encoded_indices=pre_encoded_indices,
            compact_encoder_image=compact_encoder_image,
        )

        # Extract relevant outputs
        if (
            hasattr(tokenizer_output, "image_recon")
            and tokenizer_output.image_recon is not None
        ):
            reconstructed = tokenizer_output.image_recon
        elif hasattr(tokenizer_output, "decoder_2d_pred"):
            reconstructed = tokenizer_output.decoder_2d_pred
        elif isinstance(tokenizer_output, dict):
            reconstructed = tokenizer_output.get(
                "reconstruction", tokenizer_output.get("decoder_2d_pred", images)
            )
        else:
            reconstructed = tokenizer_output

        commit_loss = torch.tensor(0.0, device=images.device)
        diffusion_loss = torch.tensor(0.0, device=images.device)
        rep_guidance_loss = torch.tensor(0.0, device=images.device)

        # Extract commit loss if available
        commit_loss = tokenizer_output.commit_loss

        # Compute diffusion loss if diffusion model is available
        diffusion_loss = tokenizer_output.diffusion_loss

        # Compute rep_guidance loss if repa_feat is available
        if (
            hasattr(tokenizer_output, "repa_feat")
            and tokenizer_output.repa_feat is not None
        ):
            rep_guidance_loss = self.rep_guidance_loss(
                tokenizer_output.repa_feat,
                dinov2_images,  # Use DINOv2-normalized images for RepGuidanceLoss
                tokenizer_output.repa_masks,
            )

        # Total loss
        total_loss = (
            self.commit_loss_weight * commit_loss
            + self.diffusion_loss_weight * diffusion_loss
            + self.rep_guidance_loss_weight * rep_guidance_loss
        )

        # Log losses - conditional logging to reduce DDP sync overhead
        if (stage == "val") or (self.global_step % self.log_metrics_every_n_steps == 0):
            # Only log to progress bar for total loss to reduce clutter
            self.log(f"{stage}/commit_loss", commit_loss, on_step=True, on_epoch=False)
            self.log(
                f"{stage}/diffusion_loss",
                diffusion_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                f"{stage}/rep_guidance_loss",
                rep_guidance_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                f"{stage}/total_loss",
                total_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        # Log perplexity/correct tokens for training
        if stage == "train" and (
            self.global_step % self.log_metrics_every_n_steps == 0
        ):
            indices = getattr(tokenizer_output, "indices", None)
            if indices is not None:
                perplexity_score = Perplexity.compute_direct(
                    indices, self.codebook_size, base=2
                )
                self.log(
                    f"{stage}/perplexity",
                    perplexity_score,
                    on_step=True,
                    on_epoch=False,
                )

            correct_tokens_score = CorrectTokens.compute_direct(
                tokenizer_output.base_2d_ids_from_logits,
                tokenizer_output.base_2d_ids,
                tokenizer_output.masks,
            )
            self.log(
                f"{stage}/correct_tokens",
                correct_tokens_score,
                on_step=True,
                on_epoch=False,
            )

        # Update metrics for epoch-end logging
        if stage == "train":
            self.train_commit_loss.update(commit_loss.detach())
            self.train_diffusion_loss.update(diffusion_loss.detach())
            self.train_rep_guidance_loss.update(rep_guidance_loss.detach())
        else:
            self.val_commit_loss.update(commit_loss.detach())
            self.val_diffusion_loss.update(diffusion_loss.detach())
            self.val_rep_guidance_loss.update(rep_guidance_loss.detach())

        return {
            "loss": total_loss,
            "commit_loss": commit_loss,
            "diffusion_loss": diffusion_loss,
            "rep_guidance_loss": rep_guidance_loss,
            "reconstructed": reconstructed,
            "indices": getattr(tokenizer_output, "indices", None),
        }

    def _update_rfid_metric(self, batch, loss_dict):
        """Update rFID metric with denormalized original and reconstructed images."""
        # Check if dual normalization is enabled
        use_dual_normalization = self.config.dataset.get(
            "use_dual_normalization", False
        )
        use_dino_image_for_compact = self.config.dataset.get(
            "use_dino_image_for_compact", False
        )

        # Unpack batch
        images = batch[0]  # Tokenizer-normalized images
        dinov2_images = (
            batch[1] if use_dual_normalization and len(batch) > 1 else images
        )

        # Determine which images to use for encoding based on configuration
        if use_dual_normalization and use_dino_image_for_compact:
            images_to_encode = dinov2_images
        else:
            images_to_encode = images

        # Properly reconstruct images using encode/decode (no masking)
        with torch.no_grad():
            # Encode the full images to get indices (no masking)
            indices = self.tokenizer.encode(images_to_encode)

            # Decode indices back to images (in normalized space)
            reconstructed = self.tokenizer.decode(indices, denormalize=False)

            # Denormalize both original and reconstructed images to [0,255] range
            original_denormalized = self.tokenizer.denormalize(images) * 255.0
            reconstructed_denormalized = (
                self.tokenizer.denormalize(reconstructed) * 255.0
            )

            # Update rFID metric
            self.val_rfid.update(original_denormalized, reconstructed_denormalized)

    def _update_perplexity_metric(self, batch, loss_dict):
        """Update validation perplexity metric with codebook indices."""
        # Get indices from tokenizer output
        indices = loss_dict.get("indices")
        if indices is None:
            return  # Skip if no indices available

        # Update perplexity metric with indices
        self.val_perplexity.update(indices)

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss_dict = self._compute_losses(batch, batch_idx, "train")

        # Log original and reconstructed images to wandb every log_every * 100 steps
        log_every = self.trainer.log_every_n_steps
        if self.global_step % (log_every * 10) == 0 and isinstance(
            self.logger, WandbLogger
        ):
            self._log_reconstruction_images(batch, batch_idx, stage="train")

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss_dict = self._compute_losses(batch, batch_idx, "val")

        # Compute rFID metric
        self._update_rfid_metric(batch, loss_dict)

        # Update perplexity metric with indices
        self._update_perplexity_metric(batch, loss_dict)

        # Log original and reconstructed images to wandb every 100 steps
        if batch_idx % 1000 == 0 and isinstance(self.logger, WandbLogger):
            self._log_reconstruction_images(batch, batch_idx, stage="val")

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._compute_losses(batch, batch_idx, "test")["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called at the end of training batch."""
        self.ema_tokenizer.update()

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Compute epoch averages using TorchMetrics
        avg_commit_loss = self.train_commit_loss.compute()
        avg_diffusion_loss = self.train_diffusion_loss.compute()
        avg_rep_guidance_loss = self.train_rep_guidance_loss.compute()

        # Log epoch averages
        self.log("train/epoch_commit_loss", avg_commit_loss, sync_dist=True)
        self.log("train/epoch_diffusion_loss", avg_diffusion_loss, sync_dist=True)
        self.log("train/epoch_rep_guidance_loss", avg_rep_guidance_loss, sync_dist=True)

        # Reset metrics for next epoch
        self.train_commit_loss.reset()
        self.train_diffusion_loss.reset()
        self.train_rep_guidance_loss.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Compute epoch averages using TorchMetrics
        avg_commit_loss = self.val_commit_loss.compute()
        avg_diffusion_loss = self.val_diffusion_loss.compute()
        avg_rep_guidance_loss = self.val_rep_guidance_loss.compute()
        rfid_score = self.val_rfid.compute()
        perplexity_score = self.val_perplexity.compute()

        # Log epoch averages
        self.log("val/epoch_commit_loss", avg_commit_loss, sync_dist=True)
        self.log("val/epoch_diffusion_loss", avg_diffusion_loss, sync_dist=True)
        self.log("val/epoch_rep_guidance_loss", avg_rep_guidance_loss, sync_dist=True)
        self.log("val/rfid", rfid_score, sync_dist=True)
        self.log("val/perplexity", perplexity_score, sync_dist=True)

        # Reset metrics for next epoch
        self.val_commit_loss.reset()
        self.val_diffusion_loss.reset()
        self.val_rep_guidance_loss.reset()
        self.val_rfid.reset()
        self.val_perplexity.reset()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Use helper functions from train_utils.py
        # Filter parameters to only include trainable ones (excludes frozen DINOv2)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = setup_optimizer(self.config, trainable_params)

        # Scheduler (optional)
        scheduler = setup_scheduler(self.config, optimizer)

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }

        return optimizer

    @torch.no_grad()
    def _log_reconstruction_images(self, batch, batch_idx, stage="val"):
        """Log original and reconstructed images to wandb and/or local filesystem."""
        # Check if dual normalization is enabled
        use_dual_normalization = self.config.dataset.get(
            "use_dual_normalization", False
        )
        use_dino_image_for_compact = self.config.dataset.get(
            "use_dino_image_for_compact", False
        )

        # Save the original training state
        was_training = self.tokenizer.training
        if was_training:
            self.tokenizer.eval()

        # Unpack batch
        if use_dual_normalization:
            if isinstance(batch, (list, tuple)) and len(batch) >= 4:
                images = batch[0]  # Tokenizer-normalized images
                dinov2_images = batch[1]  # DINOv2-normalized images
            else:
                return
        else:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images = batch[0]  # Images from WebDataset
                dinov2_images = images  # Use same images if no dual normalization
            else:
                return

        # Take only first few images for logging
        num_images = min(4, images.size(0))

        # Determine which images to use for encoding based on configuration
        if use_dual_normalization and use_dino_image_for_compact:
            # Use DINOv2 images for compact encoder
            images_to_encode = dinov2_images[:num_images]
        else:
            # Use regular tokenizer-normalized images
            images_to_encode = images[:num_images]

        # Always use original tokenizer-normalized images for visualization comparison
        images_to_log = images[:num_images]

        # Use encode/decode methods for proper reconstruction
        with torch.no_grad():
            # Encode images to get indices
            indices = self.tokenizer.encode(images_to_encode)

            # Decode indices back to images with denormalization
            reconstructed = self.tokenizer.decode(indices, denormalize=True)

            # Denormalize original images for proper visualization
            original_denormalized = self.tokenizer.denormalize(images_to_log)

        # Convert images to numpy format
        orig_imgs_np = []
        recon_imgs_np = []

        for i in range(num_images):
            # Convert to numpy and transpose (CHW -> HWC)
            orig_img = (
                original_denormalized[i].detach().cpu().numpy().transpose(1, 2, 0)
            )
            recon_img = (
                reconstructed[i].detach().cpu().float().numpy().transpose(1, 2, 0)
            )

            # Clip values to [0, 1] range
            orig_img = orig_img.clip(0, 1)
            recon_img = recon_img.clip(0, 1)

            orig_imgs_np.append(orig_img)
            recon_imgs_np.append(recon_img)

        # Log to wandb if available
        if isinstance(self.logger, WandbLogger):
            original_images = []
            reconstructed_images = []

            for i, (orig_img, recon_img) in enumerate(zip(orig_imgs_np, recon_imgs_np)):
                original_images.append(wandb.Image(orig_img, caption=f"Original {i}"))
                reconstructed_images.append(
                    wandb.Image(recon_img, caption=f"Reconstructed {i}")
                )

            # Log to wandb
            self.logger.experiment.log(
                {
                    f"{stage}/original_images": original_images,
                    f"{stage}/reconstructed_images": reconstructed_images,
                    f"{stage}/batch_idx": batch_idx,
                    "global_step": self.global_step,
                }
            )

        # Log to local filesystem if enabled
        if self.log_images_locally:
            self._save_images_locally(orig_imgs_np, recon_imgs_np, batch_idx, stage)

        # Restore the original training state
        if was_training:
            self.tokenizer.train()

    def _save_images_locally(self, orig_imgs, recon_imgs, batch_idx, stage):
        """Save original and reconstructed images to local filesystem."""
        # Create directory structure: {local_image_log_dir}/{stage}/epoch_{epoch}_step_{global_step}/
        epoch = self.current_epoch
        step = self.global_step

        save_dir = os.path.join(
            self.local_image_log_dir, stage, f"epoch_{epoch:03d}_step_{step:06d}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Save original and reconstructed images
        for i, (orig_img, recon_img) in enumerate(zip(orig_imgs, recon_imgs)):
            # Convert to PIL format (scale to 0-255 and convert to uint8)
            orig_pil = Image.fromarray((orig_img * 255).astype(np.uint8))
            recon_pil = Image.fromarray((recon_img * 255).astype(np.uint8))

            # Save with descriptive filenames
            orig_filename = f"original_{i:02d}_batch_{batch_idx:04d}.png"
            recon_filename = f"reconstructed_{i:02d}_batch_{batch_idx:04d}.png"

            orig_pil.save(os.path.join(save_dir, orig_filename))
            recon_pil.save(os.path.join(save_dir, recon_filename))

        # Save a comparison grid if we have multiple images
        if len(orig_imgs) > 1:
            self._save_comparison_grid(orig_imgs, recon_imgs, save_dir, batch_idx)

    def _save_comparison_grid(self, orig_imgs, recon_imgs, save_dir, batch_idx):
        """Save a comparison grid showing original vs reconstructed images."""
        n_images = len(orig_imgs)

        # Create a grid with original images on top, reconstructed below
        if orig_imgs[0].shape[2] == 3:  # RGB
            grid_height = orig_imgs[0].shape[0] * 2  # 2 rows
            grid_width = orig_imgs[0].shape[1] * n_images  # n_images columns
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            for i, (orig_img, recon_img) in enumerate(zip(orig_imgs, recon_imgs)):
                h, w = orig_img.shape[:2]

                # Place original image in top row
                grid[0:h, i * w : (i + 1) * w] = (orig_img * 255).astype(np.uint8)

                # Place reconstructed image in bottom row
                grid[h : 2 * h, i * w : (i + 1) * w] = (recon_img * 255).astype(
                    np.uint8
                )

            # Save the grid
            grid_pil = Image.fromarray(grid)
            grid_filename = f"comparison_grid_batch_{batch_idx:04d}.png"
            grid_pil.save(os.path.join(save_dir, grid_filename))


def setup_callbacks(config: DictConfig) -> list:
    """Setup Lightning callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            config.training.results_dir, config.training.run_name, "checkpoints"
        ),
        verbose=True,
        every_n_train_steps=10000,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Model summary with configurable depth
    model_summary_depth = config.training.get("model_summary_depth", 3)
    model_summary = ModelSummary(max_depth=model_summary_depth)
    callbacks.append(model_summary)

    return callbacks


def setup_logger(config: DictConfig):
    """Setup experiment logger."""
    run_dir = os.path.join(config.training.results_dir, config.training.run_name)

    # Try to use WandB if available, fallback to TensorBoard
    try:
        logger = WandbLogger(
            project=config.training.get("wandb_project", "tokenizer-training"),
            notes=config.training.get("notes", None),
            save_dir=run_dir,
            config=OmegaConf.to_container(config, resolve=True),
        )

        # Create symlink to wandb folder named after wandb run name
        _create_wandb_symlink(logger, run_dir, config.training.results_dir)

        # Create notes.txt file in local folder if notes are provided
        _create_notes_file(config, run_dir)

    except Exception as e:
        print(f"WandB logger failed: {e}. Using TensorBoard logger.")
        logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard_logs")

        # Create notes.txt file even when using TensorBoard logger
        _create_notes_file(config, run_dir)

    return logger


def setup_profiler(config: DictConfig):
    """Setup profiler based on configuration."""
    profiler_config = config.training.get("profiler", {})

    if not profiler_config.get("enabled", False):
        return None

    profiler_type = profiler_config.get("type", "pytorch")

    if profiler_type == "pytorch":
        schedule_config = profiler_config.get("schedule", {})

        return PyTorchProfiler(
            record_module_names=profiler_config.get("record_module_names", True),
            file_name=profiler_config.get("file_name", "debug"),
            export_to_chrome=profiler_config.get("export_to_chrome", True),
            sort_by_key=profiler_config.get("sort_by_key", "cpu_time"),
            row_limit=profiler_config.get("row_limit", 100),
            schedule=torch.profiler.schedule(
                wait=schedule_config.get("wait", 10),
                warmup=schedule_config.get("warmup", 20),
                active=schedule_config.get("active", 10),
                repeat=schedule_config.get("repeat", 2),
            ),
            profile_memory=profiler_config.get("profile_memory", True),
            with_stack=profiler_config.get("with_stack", True),
        )
    else:
        logger.warning(f"Unknown profiler type: {profiler_type}. Profiler disabled.")
        return None


def _create_wandb_symlink(logger: WandbLogger, run_dir: str, results_dir: str):
    """Create symlink to wandb folder named after wandb run name in ./wandb directory."""
    try:
        # Get the wandb run name/id
        wandb_run_name = (
            logger.experiment.name
            if hasattr(logger.experiment, "name")
            else logger.experiment.id
        )

        central_wandb_dir = os.path.join(results_dir, "wandb")
        os.makedirs(central_wandb_dir, exist_ok=True)

        # Construct symlink path in central wandb directory
        symlink_path = os.path.join(central_wandb_dir, wandb_run_name)

        # Create relative path from central wandb dir to the actual wandb folder
        relative_target = os.path.relpath(run_dir, central_wandb_dir)

        # Remove existing symlink if it exists
        if os.path.islink(symlink_path):
            os.unlink(symlink_path)
        elif os.path.exists(symlink_path):
            print(
                f"Warning: {symlink_path} exists but is not a symlink. Skipping symlink creation."
            )
            return

        # Create the symlink
        os.symlink(relative_target, symlink_path)
        print(f"Created symlink: {symlink_path} -> {relative_target}")

    except Exception as e:
        print(f"Failed to create wandb symlink: {e}")
        # Don't fail the entire training process if symlink creation fails


def _create_notes_file(config: DictConfig, run_dir: str):
    """Create notes.txt file in local folder if notes are provided."""
    try:
        notes = config.training.get("notes", None)
        if notes is not None:
            # Create run directory if it doesn't exist
            os.makedirs(run_dir, exist_ok=True)

            # Create notes.txt file
            notes_file_path = os.path.join(run_dir, "notes.txt")
            with open(notes_file_path, "w", encoding="utf-8") as f:
                f.write("Experiment Notes\n")
                f.write("================\n\n")
                f.write(f"{notes}\n\n")
                f.write(f"Created: {config.training.run_name}\n")

            print(f"Created notes file: {notes_file_path}")

    except Exception as e:
        print(f"Failed to create notes file: {e}")
        # Don't fail the entire training process if notes file creation fails


@hydra.main(version_base=None, config_path="conf_tokenizer", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting tokenizer training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

    # Set seeds for reproducibility
    L.seed_everything(config.get("seed", 42), workers=True)

    # Setup data module - use debug dataset if specified
    if config.dataset.get("use_debug_dataset", False):
        debug_image_path = config.dataset.get("debug_image_path", None)
        if debug_image_path is None:
            raise ValueError(
                "debug_image_path must be specified when use_debug_dataset is True"
            )
        logger.info(f"Using debug dataset with image: {debug_image_path}")
        data_module = DebugDatasetDataModule(config, debug_image_path)
    elif config.dataset.get("use_pre_encoded_indices", False):
        logger.info("Using JSON WebDataset with pre-encoded indices for training")
        data_module = JSONWebDatasetDataModule(config)
    else:
        logger.info("Using standard WebDataset for training")
        data_module = WebDatasetDataModule(config)

    # Setup model
    model = TokenizerLightningModule(config)

    # Log image logging configuration
    if model.log_images_locally:
        logger.info(
            f"Local image logging enabled. Images will be saved to: {model.local_image_log_dir}"
        )
    else:
        logger.info(
            "Local image logging disabled. Images will only be logged to WandB (if available)."
        )

    # Setup callbacks and logger
    callbacks = setup_callbacks(config)
    experiment_logger = setup_logger(config)

    # Add wandb.watch() for gradient and parameter tracking
    if isinstance(experiment_logger, WandbLogger):
        # Watch model with gradients and parameters
        # log_freq controls how often to log (in batches)
        experiment_logger.watch(
            model,
            log="all",  # log gradients and parameters
            log_freq=config.training.get(
                "wandb_watch_log_freq", 2500
            ),  # log every 100 batches by default
            log_graph=False,  # set to True if you want to log the model graph
        )
        logger.info(
            f"Added wandb.watch() for model monitoring (log_freq={config.training.get('wandb_watch_log_freq', 2500)})"
        )

    # Setup trainer
    trainer = L.Trainer(
        max_steps=config.get("max_steps", 500000),
        devices=config.get("devices", "auto"),
        precision="bf16-mixed" if config.get("bfloat16", True) else 32,
        gradient_clip_val=config.training.get("grad_clip_val", 1.0),
        accumulate_grad_batches=config.training.get("accumulate_grad_batches", 1),
        log_every_n_steps=config.get("log_every", 500),
        val_check_interval=config.get("eval_every", 1000),
        check_val_every_n_epoch=None,
        callbacks=callbacks,
        logger=experiment_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        profiler=setup_profiler(config),
    )

    # Resume from checkpoint if specified
    ckpt_path = None
    if config.training.get("from_checkpoint"):
        ckpt_path = config.training.from_checkpoint
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

    # Train the model
    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # Test the model
    if hasattr(data_module, "test_dataset") and data_module.test_dataset is not None:
        trainer.test(model, data_module)

    # Save final configuration
    save_dir = os.path.join(config.training.results_dir, config.training.run_name)
    save_config(config, os.path.join(save_dir, "final_config.yaml"))

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
