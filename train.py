# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import torch
import wandb
from datetime import datetime
import json

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib

matplotlib.use("Agg")
from copy import deepcopy
from time import time
import os
import gc
import hydra
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import init_distributed
from train_utils import (
    setup_tokenizer,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    requires_grad,
    print_config,
    create_logger,
    cleanup,
    save_checkpoint,
    evaluate,
    train_step,
    load_checkpoint,
    setup_diffusion,
)
from data_utils import (
    create_dataloader,
    prepare_datasets,
    MemoryMonitor,
    find_dataloader_workers,
)
import hydra_utils  # noqa: F401 — registers OmegaConf resolvers
from misc import seed_everything, get_unnormalize

#################################################################################
#                                  Training Loop                                #
#################################################################################


def setup_memory_monitoring(dataloader, rank):
    """Setup memory monitoring for current process and workers."""
    import time

    # Initialize monitor with main process
    monitor = MemoryMonitor()

    if rank == 0:
        print(f"Rank {rank}: Setting up memory monitoring...")
        print(
            f"DataLoader config: num_workers={dataloader.num_workers}, persistent_workers={dataloader.persistent_workers}"
        )

    # Force workers to start by getting the first batch iterator
    if dataloader.num_workers > 0:
        if rank == 0:
            print("Triggering DataLoader worker spawn by creating iterator...")
        dataloader_iter = iter(dataloader)
        time.sleep(3)  # Give workers time to fully initialize

        # Now find workers
        worker_pids = find_dataloader_workers()

        # Add worker PIDs to monitor
        for pid in worker_pids:
            monitor.add_pid(pid)
    else:
        if rank == 0:
            print("DataLoader has num_workers=0, no worker processes to monitor")
        worker_pids = []

    if rank == 0:
        print(
            f"Rank {rank}: Monitoring PIDs - Main: {os.getpid()}, Workers: {worker_pids}"
        )

    return monitor


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """
    Trains a new CDiT model using Hydra for configuration.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # Initialize distributed; get local GPU index and create a proper torch.device
    _, rank, local_gpu, _ = init_distributed()
    torch_device = torch.device(f"cuda:{local_gpu}")
    seed = config.seed * dist.get_world_size() + rank
    seed_everything(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Print config for debugging
    if dist.get_rank() == 0:
        print_config(config)

    # Use Hydra's current working directory as experiment directory
    # Hydra has already created the timestamped directory based on run_name
    experiment_dir = os.getcwd()

    if rank == 0:
        logger = create_logger(experiment_dir)
        logger.info(f"Using Hydra experiment directory: {experiment_dir}")

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize WandB logging (only on rank 0)
        wandb_run_name = None
        if config.training.get("wandb_enabled", False):
            try:
                wandb.init(
                    project=config.training.get("wandb_project", "nwm-training"),
                    notes=config.training.get("wandb_notes", None),
                    tags=config.training.get("wandb_tags", []),
                    config=OmegaConf.to_container(config, resolve=False),
                    dir=experiment_dir,
                )
                wandb_run_name = wandb.run.name
                logger.info(
                    f"WandB logging initialized with run name: {wandb_run_name}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize WandB: {e}. Continuing without WandB logging."
                )

        # Create experiment note with wandb run name, date, and wandb_notes
        current_date_for_note = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_note = {
            "date": current_date_for_note,
            "wandb_run_name": wandb_run_name,
            "wandb_notes": config.training.get("wandb_notes", None),
            "experiment_dir": experiment_dir,
            "config_name": config.get("_target_", "unknown"),
            "model_type": config.model.generator.get("_target_", "unknown"),
        }

        experiment_note_path = os.path.join(experiment_dir, "experiment_note.json")
        with open(experiment_note_path, "w") as f:
            json.dump(experiment_note, f, indent=2)
        logger.info(f"Experiment note saved to {experiment_note_path}")
    else:
        logger = create_logger(None)

    # Create model using the utility functions
    tokenizer = setup_tokenizer(config, torch_device)
    tokenizer.eval()
    latent_size = config.model.generator.input_size
    assert config.dataset.image_size % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder)."
    )

    model = setup_model(config, torch_device)
    ema = deepcopy(model).to(
        torch_device
    )  # Create an EMA of the model for use after training
    for p in ema.parameters():
        p.detach_()
    requires_grad(ema, False)

    # Setup optimizer using Hydra instantiation
    opt = setup_optimizer(config, model.parameters())
    scheduler = setup_scheduler(config, opt)

    bfloat_enable = bool(config.bfloat16)
    scaler = None
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # Load existing checkpoint
    start_epoch, train_steps = load_checkpoint(
        checkpoint_dir,
        config,
        model,
        ema,
        opt,
        scheduler=scheduler,
        bfloat_enable=bfloat_enable,
        scaler=scaler if bfloat_enable else None,
    )

    # ~40% speedup but might leads to worse performance depending on pytorch version
    if config.torch_compile:
        model = torch.compile(model)

    # Wrap model for DDP using the local GPU index; keep a torch.device for tensor moves
    model = DDP(model, device_ids=[local_gpu])
    diffusion = setup_diffusion(config, for_eval=False, device=torch_device)
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset, test_dataset = prepare_datasets(config)
    loader, sampler = create_dataloader(train_dataset, config, rank, is_train=True)
    eval_loader, eval_sampler = create_dataloader(
        test_dataset, config, rank, is_train=False
    )

    logger.info(f"Dataset contains {len(train_dataset):,} images")
    logger.info(f"Eval dataset contains {len(test_dataset):,} images")

    # Setup memory monitoring
    memory_monitor = None
    memory_log_file = None
    if config.training.get("memory_monitoring", False):
        memory_monitor = setup_memory_monitoring(loader, rank)
        memory_log_file = os.path.join(experiment_dir, f"memory_rank_{rank}.log")
        if rank == 0:
            logger.info(f"Memory monitoring enabled, logging to {memory_log_file}")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {config.epochs} epochs...")
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for batch_idx, batch in enumerate(loader):
            loss = train_step(
                model,
                ema,
                diffusion,
                tokenizer,
                batch,
                torch_device,
                opt,
                scheduler,
                bfloat_enable,
                scaler,
                config,
            )

            # Memory monitoring - log every 100 batches
            if memory_monitor is not None and batch_idx % 10 == 0 and rank == 0:
                try:
                    mem_info = memory_monitor.table()
                    analysis = memory_monitor.analyze_high_memory_workers()

                    with open(memory_log_file, "a") as f:
                        f.write(f"\nEpoch {epoch}, Batch {batch_idx}:\n")
                        f.write(mem_info + "\n")
                        f.write(analysis + "\n")
                        f.write("=" * 80 + "\n")
                except Exception as e:
                    logger.warning(f"Failed to log memory usage: {e}")

            # Log loss values:
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % config.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = (
                    dist.get_world_size()
                    * batch[0].shape[0]
                    * (batch[0].shape[1] - config.dataset.context_size)
                    * steps_per_sec
                )
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=torch_device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}"
                )
                gc.collect()

                # Log to WandB (only on rank 0)
                if rank == 0 and config.training.get("wandb_enabled", False):
                    try:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/step": train_steps,
                                "epoch": epoch,
                            },
                            step=train_steps,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log to WandB: {e}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % config.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = save_checkpoint(
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
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    # Additional garbage collection after checkpoint save
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                dist.barrier()

            if (
                train_steps % config.eval_every == 0 and train_steps >= 0
            ) or train_steps == 1:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, "viz", str(train_steps))
                sim_score = evaluate(
                    ema,
                    tokenizer,
                    diffusion,
                    eval_loader,
                    rank,
                    latent_size,
                    torch_device,
                    save_dir,
                    config.seed,
                    bfloat_enable,
                    config.dataset.context_size,
                    get_unnormalize(config.dataset.mean, config.dataset.std),
                )
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(
                    f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}"
                )

                # Log to WandB (only on rank 0)
                if rank == 0 and config.training.get("wandb_enabled", False):
                    try:
                        wandb.log(
                            {
                                "eval/perceptual_loss": sim_score,
                                "eval/eval_time": eval_time,
                                "eval/step": train_steps,
                                "epoch": epoch,
                            },
                            step=train_steps,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log evaluation metrics to WandB: {e}"
                        )

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

    # Finish WandB run (only on rank 0)
    if rank == 0 and config.training.get("wandb_enabled", False):
        try:
            wandb.finish()
            logger.info("WandB logging finished")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    cleanup()


if __name__ == "__main__":
    main()
