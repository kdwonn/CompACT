"""Utility functions for hydra configuration and object instantiation."""

import glob
from time import time
from einops import repeat
import torch
import torch.distributed as dist
import logging
import os
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate, get_original_cwd

logger = logging.getLogger(__name__)

# Global cache for evaluation model to prevent memory leaks
_eval_model_cache = None


def setup_tokenizer(config: DictConfig, device: torch.device):
    tokenizer_path = config.get("tokenizer_path", None)

    if tokenizer_path is not None:
        # Handle relative paths by resolving against original working directory
        if not os.path.isabs(tokenizer_path):
            tokenizer_path = os.path.join(get_original_cwd(), tokenizer_path)

        ckp_paths = glob.glob(os.path.join(tokenizer_path, "checkpoints", "*.ckpt"))
        ckp_path = max(ckp_paths, key=os.path.getmtime)

        # Try different config locations
        config_path = os.path.join(tokenizer_path, ".hydra", "config.yaml")
        config_merged_path = os.path.join(
            tokenizer_path, ".hydra", "config_merged.yaml"
        )
        if os.path.exists(config_merged_path):
            config_path = config_merged_path

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config file in {config_path}")

        logger.info(f"Loading compact tokenizer config from {config_path}")
        logger.info(f"Loading compact tokenizer from {ckp_path}")

        # Load the experiment config and extract tokenizer config
        experiment_config = OmegaConf.load(config_path)

        # Resolve all variable references in the config before extracting tokenizer config
        OmegaConf.resolve(experiment_config)
        tokenizer_config = experiment_config.model.tokenizer

        # override the tokenizer config with the training config if theres any (e.g. CFG weight, sampling nums, etc.)
        if config.model.get("tokenizer", None):
            tokenizer_config = OmegaConf.merge(tokenizer_config, config.model.tokenizer)

        # override the tokenizer config with the experiment config
        with open_dict(config):
            config.model.tokenizer = tokenizer_config

        # Extract normalization parameters from tokenizer experiment config
        # Update the CDiT training config to use the same normalization as the tokenizer
        exp_dataset = experiment_config.dataset
        assert exp_dataset.mean is not None and exp_dataset.std is not None, (
            "Tokenizer experiment config has null mean/std values"
        )
        # Update the CDiT training config with tokenizer's normalization
        with open_dict(config):
            if not hasattr(config, "dataset"):
                config.dataset = {}
            if experiment_config.dataset.get("use_dino_image_for_compact", False):
                config.dataset.mean = exp_dataset.dinov2_mean
                config.dataset.std = exp_dataset.dinov2_std
            else:
                config.dataset.mean = exp_dataset.mean
                config.dataset.std = exp_dataset.std
        logger.info(
            f"Updated CDiT config with tokenizer experiment normalization: mean={config.dataset.mean}, std={config.dataset.std}"
        )

        # Instantiate tokenizer from loaded config
        tokenizer = instantiate(tokenizer_config, _recursive_=True).to(device)

        # Load checkpoint weights
        state_dict = torch.load(ckp_path, map_location="cpu", weights_only=False)
        state_dict = {
            k[len("tokenizer.") :]: v
            for k, v in state_dict["state_dict"].items()
            if k.startswith("tokenizer.")
        }
        tokenizer.load_state_dict(state_dict, strict=True)

    else:
        # Fallback to config-based tokenizer (legacy behavior)
        tokenizer_config = config.model.tokenizer
        tokenizer = instantiate(tokenizer_config, _recursive_=True).to(device)

    return tokenizer


def setup_model(config: DictConfig, device: torch.device):
    """Setup CDiT model using hydra instantiation."""
    # Extract model name to get the appropriate class from CDiT_models
    # Instantiate model using hydra
    model = instantiate(
        config.model.generator,
        context_size=config.dataset.context_size,
        _recursive_=False,  # Don't recursively instantiate nested configs
    ).to(device)

    return model


def setup_diffusion(config: DictConfig, for_eval: bool, device: torch.device):
    diffusion_config = config.model.diffusion
    diffusion = instantiate(diffusion_config, for_eval=for_eval, _recursive_=False)
    return diffusion


def setup_optimizer(config: DictConfig, model_params):
    """Setup optimizer using hydra instantiation."""
    return instantiate(config.training.optimizer, params=model_params)


def setup_scheduler(config: DictConfig, optimizer):
    """Setup learning rate scheduler if specified."""
    if config.scheduler.scheduler._target_ is not None:
        return instantiate(config.scheduler.scheduler, optimizer=optimizer)
    return None


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    # Build a lookup only when needed (for torch.compile compatibility)
    ema_params_dict = dict(ema_model.named_parameters())

    for name, param in model.named_parameters():
        # Handle torch.compile naming convention which adds "_orig_mod." prefix
        ema_name = name.replace("_orig_mod.", "")
        if ema_name in ema_params_dict:
            # In-place update of EMA parameter
            ema_params_dict[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)


def print_config(config: DictConfig):
    """Print the config in a readable format."""
    formatted_config = OmegaConf.to_yaml(config)
    print("\n" + "=" * 80)
    print("CONFIGURATION:")
    print(formatted_config)
    print("=" * 80 + "\n")


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_checkpoint(
    checkpoint_dir,
    config,
    model,
    ema,
    opt,
    scheduler=None,
    bfloat_enable=False,
    scaler=None,
):
    """Load model checkpoint if available."""
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    print("Searching for model from ", checkpoint_dir)
    start_epoch = 0
    train_steps = 0

    if os.path.isfile(latest_path) or config.training.from_checkpoint:
        if os.path.isfile(latest_path) and config.training.from_checkpoint:
            raise ValueError(
                "Resuming from checkpoint, this might override latest.pth.tar!!"
            )

        latest_path = (
            latest_path
            if os.path.isfile(latest_path)
            else config.training.from_checkpoint
        )
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=False
        )

        if "model" in latest_checkpoint:
            model_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["model"].items()
            }
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["ema"].items()
            }
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(
                ema, model, decay=0
            )  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {
                k.replace("_orig_mod.", ""): v
                for k, v in latest_checkpoint["opt"].items()
            }
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")

        if "epoch" in latest_checkpoint:
            start_epoch = latest_checkpoint["epoch"] + 1

        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]

        if "scaler" in latest_checkpoint and bfloat_enable and scaler is not None:
            scaler.load_state_dict(latest_checkpoint["scaler"])

        if "scheduler" in latest_checkpoint and scheduler is not None:
            scheduler.load_state_dict(latest_checkpoint["scheduler"])

    return start_epoch, train_steps


def save_checkpoint(
    model,
    ema,
    opt,
    config,
    epoch,
    train_steps,
    bfloat_enable,
    scheduler,
    scaler,
    checkpoint_dir,
):
    """Save model checkpoint."""
    import gc

    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
        "epoch": epoch,
        "train_steps": train_steps,
    }
    if bfloat_enable and scheduler is not None:
        checkpoint.update(
            {"scaler": scaler.state_dict(), "scheduler": scheduler.state_dict()}
        )
    elif bfloat_enable:
        checkpoint.update({"scaler": scaler.state_dict()})
    elif scheduler is not None:
        checkpoint.update({"scheduler": scheduler.state_dict()})

    # Save latest checkpoint
    checkpoint_path = f"{checkpoint_dir}/latest.pth.tar"
    torch.save(checkpoint, checkpoint_path)

    # Save numbered checkpoint if needed
    if train_steps % (10 * config.ckpt_every) == 0 and train_steps > 0:
        numbered_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pth.tar"
        torch.save(checkpoint, numbered_checkpoint_path)
        checkpoint_path = numbered_checkpoint_path  # Return the numbered path

    # Explicitly delete checkpoint dict and force garbage collection
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    return checkpoint_path


@torch.no_grad()
def evaluate(
    model,
    vae,
    diffusion,
    eval_loader,
    rank,
    latent_size,
    device,
    save_dir,
    seed,
    bfloat_enable,
    num_cond,
    unnormalize_fn,
):
    """Evaluate model on test dataset."""
    from isolated_nwm_infer import model_forward_wrapper

    global _eval_model_cache

    # Use the pre-created evaluation dataloader
    loader = eval_loader
    # Update sampler epoch for proper distributed evaluation
    loader.sampler.set_epoch(seed)

    # Use cached evaluation model to prevent memory leaks
    if _eval_model_cache is None:
        from dreamsim import dreamsim

        _eval_model_cache, _ = dreamsim(
            pretrained=True, cache_dir=os.path.join(get_original_cwd(), "models")
        )
        _eval_model_cache = _eval_model_cache.to(device)
        logger.info("Created and cached DreamSim evaluation model")

    eval_model = _eval_model_cache
    score = torch.tensor(0.0).to(device)
    n_samples = torch.tensor(0).to(device)

    # Run for 1 step
    x, y, rel_t = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    rel_t = rel_t.to(device).flatten(0, 1)
    with torch.amp.autocast("cuda", enabled=bfloat_enable, dtype=torch.bfloat16):
        B, T = x.shape[:2]
        num_goals = T - num_cond

        start_time = time()
        samples = model_forward_wrapper(
            (model, diffusion, vae),
            x,
            y,
            num_timesteps=None,
            latent_size=latent_size,
            device=device,
            num_cond=num_cond,
            num_goals=num_goals,
            rel_t=rel_t,
        )
        logger.info(
            f"Time taken for generating {samples.shape}: {time() - start_time:.2f} seconds"
        )

        x_start_pixels = x[:, num_cond:].flatten(0, 1)
        x_cond_pixels = (
            x[:, :num_cond]
            .unsqueeze(1)
            .expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4])
            .flatten(0, 1)
        )
        # Unnormalize pixels directly using tokenizer's unnormalization
        # samples = vae.unnormalize_image(samples)
        x_start_pixels = unnormalize_fn(x_start_pixels)
        x_cond_pixels = unnormalize_fn(x_cond_pixels)

        res = eval_model(x_start_pixels, samples)

        score += res.sum()
        n_samples += len(res)

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        for i in range(min(samples.shape[0], 10)):
            fig, ax = plt.subplots(1, 3, dpi=256)
            ax[0].imshow(
                (x_cond_pixels[i, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(
                    "uint8"
                )
            )
            ax[1].imshow(
                (x_start_pixels[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            ax[2].imshow(
                (samples[i].permute(1, 2, 0).cpu().float().numpy() * 255).astype(
                    "uint8"
                )
            )
            plt.savefig(f"{save_dir}/{i}.png")
            plt.clf()  # Clear current figure
            plt.close(fig)  # Close specific figure
            del fig  # Explicitly delete figure reference

    dist.all_reduce(score)
    dist.all_reduce(n_samples)
    sim_score = score / n_samples

    # Explicitly delete large tensors and clear cache
    import gc

    try:
        del x, y, rel_t, samples, x_start_pixels, x_cond_pixels
        if "res" in locals():
            del res
        if "ax" in locals():
            del ax
    except Exception:
        pass

    # Force garbage collection and CUDA cache cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Additional cleanup for matplotlib backend
    if rank == 0:
        plt.close("all")  # Close any remaining matplotlib figures
        gc.collect()

    return sim_score


def train_step(
    model,
    ema,
    diffusion,
    tokenizer,
    batch,
    device,
    opt,
    scheduler,
    bfloat_enable,
    scaler,
    config,
):
    """Execute a single training step."""
    x, y, rel_t = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    rel_t = rel_t.to(device, non_blocking=True)

    with torch.amp.autocast("cuda", enabled=bfloat_enable, dtype=torch.bfloat16):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
            x = tokenizer.encode(x)
            x = x.unflatten(0, (B, T))

        num_goals = T - config.dataset.context_size
        x_start = x[:, config.dataset.context_size :].flatten(0, 1)
        x_cond = repeat(
            x[:, : config.dataset.context_size], "b t ... -> (b g) t ...", g=num_goals
        )
        y = y.flatten(0, 1)
        rel_t = rel_t.flatten(0, 1)

        t = torch.randint(
            0, diffusion.num_timesteps, (x_start.shape[0],), device=device
        )
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
        loss_dict = diffusion.training_losses(model, x_start, t, model_kwargs)
        loss = loss_dict["loss"].mean()

    if not bfloat_enable:
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:
        opt.zero_grad()
        scaler.scale(loss).backward()
        if config.training.grad_clip_val > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.training.grad_clip_val
            )
        scaler.step(opt)
        scaler.update()

    if scheduler is not None:
        scheduler.step()

    update_ema(ema, model.module)

    return loss.detach().item()
