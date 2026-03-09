#!/usr/bin/env python3
r"""
Script to load tokenizer checkpoints and configs from tokenizer training experiments.

This script provides utilities to:
1. Load experiment configurations from various locations
2. Discover and load tokenizer checkpoints
3. Run basic reconstruction tests
4. Provide a CLI interface for easy usage

Usage:
    python load_tokenizer_checkpoint.py logs/tokenizer_training/20250719_213533_rin\ with\ reg\ 0
    python load_tokenizer_checkpoint.py logs/tokenizer_training/20250719_213533_rin\ with\ reg\ 0 --checkpoint epoch=0-step=10000.ckpt
"""

import os
import argparse
import logging
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf

# Import project modules
from hydra_utils import load_experiment_config
from train_utils import setup_tokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[\\033[34m%(asctime)s\\033[0m] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_checkpoints(exp_dir: str) -> List[Path]:
    """
    Find all available checkpoint files in the experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        List of checkpoint file paths, sorted by modification time (newest first)
    """
    checkpoint_dir = Path(exp_dir) / "checkpoints"

    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
        return []

    # Find all .ckpt files
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))

    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return []

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    logger.info(f"Found {len(checkpoint_files)} checkpoint(s):")
    for i, ckpt in enumerate(checkpoint_files):
        logger.info(f"  {i + 1}. {ckpt.name}")

    return checkpoint_files


def load_experiment_config_with_fallback(exp_dir: str) -> Optional[DictConfig]:
    """
    Load experiment configuration with multiple fallback options.

    Tries in order:
    1. .hydra/config.yaml (primary)
    2. wandb/run-*/files/config.yaml (fallback)
    3. Any .yaml file in root directory (last resort)

    Args:
        exp_dir: Path to experiment directory

    Returns:
        DictConfig if found, None otherwise
    """
    exp_path = Path(exp_dir)

    # Try primary location first
    config = load_experiment_config(str(exp_path))
    if config is not None:
        logger.info("Loaded config from .hydra/config.yaml")
        return config

    # Try wandb files as fallback
    wandb_config_pattern = exp_path / "wandb" / "run-*" / "files" / "config.yaml"
    wandb_configs = glob.glob(str(wandb_config_pattern))

    if wandb_configs:
        try:
            config_path = wandb_configs[0]  # Take the first match
            config = OmegaConf.load(config_path)
            logger.info(f"Loaded config from wandb: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load wandb config: {e}")

    # Last resort: any yaml file in the directory
    yaml_files = list(exp_path.glob("*.yaml"))
    if yaml_files:
        try:
            config_path = yaml_files[0]
            config = OmegaConf.load(config_path)
            logger.info(f"Loaded config from: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")

    logger.error(f"No valid configuration found in {exp_dir}")
    return None


def load_tokenizer_from_experiment(
    exp_dir: str, ckpt_name: Optional[str] = None, device: str = "cuda"
) -> Tuple[nn.Module, DictConfig, Dict[str, Any]]:
    """
    Load tokenizer model from experiment directory.

    Args:
        exp_dir: Path to experiment directory
        ckpt_name: Specific checkpoint name (if None, uses latest)
        device: Device to load model on

    Returns:
        Tuple of (tokenizer_model, config, checkpoint_info)

    Raises:
        ValueError: If experiment directory or checkpoints not found
        RuntimeError: If model loading fails
    """
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        raise ValueError(f"Experiment directory not found: {exp_dir}")

    # Load configuration
    config = load_experiment_config_with_fallback(exp_dir)
    if config is None:
        raise ValueError(f"Could not load configuration from {exp_dir}")

    # Find checkpoints
    checkpoint_files = find_checkpoints(exp_dir)
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found in {exp_dir}")

    # Select checkpoint
    if ckpt_name is not None:
        # Find specific checkpoint
        checkpoint_path = None
        for ckpt_path in checkpoint_files:
            if ckpt_path.name == ckpt_name:
                checkpoint_path = ckpt_path
                break

        if checkpoint_path is None:
            available_names = [ckpt.name for ckpt in checkpoint_files]
            raise ValueError(
                f"Checkpoint '{ckpt_name}' not found. Available: {available_names}"
            )
    else:
        # Use latest checkpoint
        checkpoint_path = checkpoint_files[0]
        logger.info(f"Using latest checkpoint: {checkpoint_path.name}")

    # Setup device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    device = torch.device(device)

    # Load Lightning checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        lightning_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")

    # Setup tokenizer using the loaded config
    logger.info("Setting up tokenizer model...")
    try:
        tokenizer = setup_tokenizer(config, device)
    except Exception as e:
        raise RuntimeError(f"Failed to setup tokenizer: {e}")

    # Load tokenizer weights from Lightning checkpoint
    logger.info("Loading tokenizer weights...")
    try:
        # The Lightning module stores the tokenizer under 'tokenizer' key in state_dict
        lightning_state_dict = lightning_checkpoint.get("state_dict", {})

        # Extract tokenizer state dict (remove 'tokenizer.' prefix)
        tokenizer_state_dict = {}
        tokenizer_prefix = "tokenizer."

        for key, value in lightning_state_dict.items():
            if key.startswith(tokenizer_prefix):
                new_key = key[len(tokenizer_prefix) :]
                tokenizer_state_dict[new_key] = value

        if not tokenizer_state_dict:
            # Try loading entire state dict if no tokenizer prefix found
            logger.warning("No 'tokenizer.' prefix found, trying entire state dict")
            tokenizer_state_dict = lightning_state_dict

        # Load the state dict
        missing_keys, unexpected_keys = tokenizer.load_state_dict(
            tokenizer_state_dict, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys in tokenizer: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in tokenizer: {unexpected_keys}")

        logger.info("Tokenizer weights loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer weights: {e}")

    # Set model to eval mode
    tokenizer.eval()
    tokenizer.to(device)

    # Prepare checkpoint info
    checkpoint_info = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "epoch": lightning_checkpoint.get("epoch", "unknown"),
        "global_step": lightning_checkpoint.get("global_step", "unknown"),
        "device": str(device),
    }

    # Log model info
    num_params = sum(p.numel() for p in tokenizer.parameters())
    logger.info(f"Tokenizer loaded: {num_params:,} parameters")
    logger.info(f"Checkpoint info: {checkpoint_info}")

    return tokenizer, config, checkpoint_info


def test_reconstruction(
    tokenizer: nn.Module,
    config: DictConfig,
    test_image_path: Optional[str] = None,
    save_results: bool = True,
    output_dir: str = "reconstruction_test",
) -> Dict[str, Any]:
    """
    Test tokenizer reconstruction with a sample image.

    Args:
        tokenizer: Loaded tokenizer model
        config: Experiment configuration
        test_image_path: Path to test image (if None, creates a random image)
        save_results: Whether to save reconstruction results
        output_dir: Directory to save results

    Returns:
        Dictionary with reconstruction results and metrics
    """
    logger.info("Running reconstruction test...")

    device = next(tokenizer.parameters()).device

    # Prepare test image
    if test_image_path is not None and os.path.exists(test_image_path):
        # Load image from file
        logger.info(f"Using test image: {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")

        # Get image size from config
        target_size = getattr(config.dataset, "image_size", 256)
        if isinstance(target_size, (list, tuple)):
            target_size = target_size[0]

        # Resize image
        image = image.resize((target_size, target_size), Image.BICUBIC)

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

        # Apply tokenizer normalization if available
        if hasattr(tokenizer, "normalize"):
            image_tensor = tokenizer.normalize(image_tensor)
    else:
        # Create random test image
        logger.info("Creating random test image")
        target_size = getattr(config.dataset, "image_size", 256)
        if isinstance(target_size, (list, tuple)):
            target_size = target_size[0]

        # Create random RGB image
        image_tensor = torch.randn(1, 3, target_size, target_size)

    image_tensor = image_tensor.to(device)

    # Test reconstruction
    with torch.no_grad():
        try:
            # Encode to indices
            logger.info("Encoding image...")
            indices = tokenizer.encode(image_tensor)
            logger.info(f"Encoded to indices shape: {indices.shape}")

            # Decode back to image
            logger.info("Decoding image...")
            reconstructed = tokenizer.decode(indices, denormalize=True)
            logger.info(f"Reconstructed image shape: {reconstructed.shape}")

            # Denormalize original for comparison
            if hasattr(tokenizer, "denormalize"):
                original_denorm = tokenizer.denormalize(image_tensor)
            else:
                original_denorm = image_tensor

            # Calculate reconstruction error
            mse = torch.mean((original_denorm - reconstructed) ** 2).item()

            # Prepare results
            results = {
                "original_shape": tuple(image_tensor.shape),
                "indices_shape": tuple(indices.shape),
                "reconstructed_shape": tuple(reconstructed.shape),
                "mse": mse,
                "encoding_successful": True,
                "decoding_successful": True,
            }

            # Save results if requested
            if save_results:
                os.makedirs(output_dir, exist_ok=True)

                # Save original image
                orig_img = original_denorm[0].detach().cpu().clamp(0, 1)
                orig_img = (orig_img * 255).byte().permute(1, 2, 0).numpy()
                orig_pil = Image.fromarray(orig_img)
                orig_path = os.path.join(output_dir, "original.png")
                orig_pil.save(orig_path)

                # Save reconstructed image
                recon_img = reconstructed[0].detach().cpu().clamp(0, 1)
                recon_img = (recon_img * 255).byte().permute(1, 2, 0).numpy()
                recon_pil = Image.fromarray(recon_img)
                recon_path = os.path.join(output_dir, "reconstructed.png")
                recon_pil.save(recon_path)

                logger.info(f"Saved reconstruction results to {output_dir}")
                results["saved_original"] = orig_path
                results["saved_reconstructed"] = recon_path

            logger.info(f"Reconstruction test completed. MSE: {mse:.6f}")

        except Exception as e:
            logger.error(f"Reconstruction test failed: {e}")
            results = {
                "encoding_successful": False,
                "decoding_successful": False,
                "error": str(e),
            }

    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Load tokenizer checkpoints and run reconstruction tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  %(prog)s logs/tokenizer_training/20250719_213533_rin,\ with\ reg\ 0
  %(prog)s logs/tokenizer_training/20250719_213533_rin,\ with\ reg\ 0 --checkpoint epoch=0-step=10000.ckpt
  %(prog)s logs/tokenizer_training/20250719_213533_rin,\ with\ reg\ 0 --test-image /path/to/image.jpg
        """,
    )

    parser.add_argument("exp_dir", type=str, help="Path to experiment directory")

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Specific checkpoint name (if not provided, uses latest)",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to load model on (default: cuda)",
    )

    parser.add_argument(
        "--test-image",
        "-t",
        type=str,
        default=None,
        help="Path to test image for reconstruction",
    )

    parser.add_argument(
        "--no-test", action="store_true", help="Skip reconstruction test"
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="reconstruction_test",
        help="Output directory for reconstruction test results",
    )

    args = parser.parse_args()

    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer from experiment: {args.exp_dir}")
        tokenizer, config, checkpoint_info = load_tokenizer_from_experiment(
            args.exp_dir, ckpt_name=args.checkpoint, device=args.device
        )

        logger.info("Tokenizer loaded successfully!")

        # Print configuration summary
        logger.info("Configuration summary:")
        if hasattr(config.model, "tokenizer") and hasattr(
            config.model.tokenizer, "_target_"
        ):
            logger.info(f"  Model type: {config.model.tokenizer._target_}")
        if hasattr(config.model, "compact_tokenizer") and hasattr(
            config.model.compact_tokenizer, "_target_"
        ):
            logger.info(
                f"  Compact tokenizer: {config.model.compact_tokenizer._target_}"
            )
        if hasattr(config, "dataset") and hasattr(config.dataset, "_target_"):
            logger.info(f"  Dataset: {config.dataset._target_}")
        elif hasattr(config, "dataset"):
            logger.info("  Dataset config available (no _target_ field)")

        # Run reconstruction test if requested
        if not args.no_test:
            test_results = test_reconstruction(
                tokenizer,
                config,
                test_image_path=args.test_image,
                save_results=True,
                output_dir=args.output_dir,
            )

            if test_results.get("encoding_successful", False):
                logger.info("Reconstruction test passed!")
            else:
                logger.error("Reconstruction test failed!")
                if "error" in test_results:
                    logger.error(f"Error: {test_results['error']}")

        logger.info("Script completed successfully!")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
