# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from einops import repeat
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

import misc
import distributed as dist
from datasets import EvalDataset
from PIL import Image
from hydra_utils import load_experiment_config, merge_configs
from train_utils import (
    setup_tokenizer,
    setup_model,
    print_config,
    create_logger,
    setup_diffusion,
)


def save_image(output_file, img):
    img = img.detach().cpu()
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB")

    image.save(output_file)


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config.dataset.datasets[dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index = None

    dataset = EvalDataset(
        data_folder=data_config.data_folder,
        data_split_folder=data_config.test,
        dataset_name=dataset_name,
        image_size=config.dataset.image_size,
        min_dist_cat=config.eval_distance.eval_min_dist_cat,
        max_dist_cat=config.eval_distance.eval_max_dist_cat,
        len_traj_pred=config.eval_len_traj_pred,
        traj_stride=config.traj_stride,
        context_size=config.eval_context_size,
        normalize=config.dataset.normalize,
        action_stats=config.dataset.action_stats,
        waypoint_spacing=data_config.metric_waypoint_spacing,
        transform=misc.get_transform(
            config.dataset.image_size, config.dataset.mean, config.dataset.std
        ),
        goals_per_obs=4,
        predefined_index=predefined_index,
        traj_names="traj_names.txt",
    )

    return dataset


@torch.no_grad()
def model_forward_wrapper(
    all_models,
    curr_obs,
    curr_delta,
    num_timesteps,
    latent_size,
    device,
    num_cond,
    num_goals=1,
    rel_t=None,
    progress=False,
    skip_tokenizer=False,
):
    model, diffusion, tokenizer = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B) * (1.0 / 128.0)).to(device)
            rel_t *= num_timesteps

        if not skip_tokenizer:
            x = x.flatten(0, 1)
            x = tokenizer.encode(x).unflatten(0, (B, T))

        x_cond = repeat(x[:, :num_cond], "b t ... -> (b g) t ...", g=num_goals)
        z = torch.randn(B * num_goals, *x.shape[2:], device=device)
        y = y.flatten(0, 1)

        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)
        samples = diffusion.p_sample_loop(
            model.forward,
            z.shape,
            model_kwargs=model_kwargs,
            progress=progress,
            device=device,
        )

        if not skip_tokenizer:
            samples = tokenizer.decode(
                samples, denormalize=True
            )  # Unnormalize to [0, 1] range
            return samples  # Already in [0, 1] range after unnormalization

        return samples


def generate_rollout_efficient(
    config,
    output_dir,
    rollout_fps,
    idxs,
    all_models,
    obs_image,
    gt_image,
    delta,
    num_cond,
    device,
):
    """
    Efficient autoregressive rollout that operates in latent space.

    This version encodes observations once and performs the entire rollout
    in latent space, only decoding when needed for visualization.
    This is significantly faster than encoding/decoding at each step.
    """
    model, diffusion, tokenizer = all_models
    rollout_stride = config.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1 :: rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)

    if config.gt:
        # Ground truth mode - just visualize GT frames
        for i in range(gt_image.shape[1]):
            x_pred_pixels = gt_image[:, i].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
            visualize_preds(output_dir, idxs, i, x_pred_pixels)
    else:
        # Encode initial observations to latent space
        curr_obs = obs_image.clone().to(device)
        B, T = curr_obs.shape[:2]
        curr_obs_latents = tokenizer.encode(curr_obs.flatten(0, 1)).unflatten(0, (B, T))

        # Perform rollout in latent space
        for i in range(gt_image.shape[1]):
            curr_delta = delta[:, i : i + 1].to(device)

            # Generate next frame in latent space
            x_pred_latents = model_forward_wrapper(
                all_models,
                curr_obs_latents,
                curr_delta,
                rollout_stride,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                skip_tokenizer=True,  # Work directly with latents
            )

            # Decode for visualization
            x_pred_pixels = tokenizer.decode(x_pred_latents, denormalize=True)

            # Update latent observation window
            x_pred_latents = x_pred_latents.unsqueeze(1)
            curr_obs_latents = torch.cat((curr_obs_latents, x_pred_latents), dim=1)
            curr_obs_latents = curr_obs_latents[:, 1:]  # Remove first observation

            # Visualize the decoded prediction
            visualize_preds(output_dir, idxs, i, x_pred_pixels)


def generate_rollout(
    config,
    output_dir,
    rollout_fps,
    idxs,
    all_models,
    obs_image,
    gt_image,
    delta,
    num_cond,
    device,
):
    """
    Produce an autoregressive rollout video at a downsampled frame rate.

    Starting from the context frames, the function repeatedly predicts the *next*
    time-step image (or copies ground-truth when `config.gt` is set), feeds
    that prediction back into the sliding window of observations, and writes each
    visualised frame to disk.
    In short: **step-by-step future synthesis, so error can accumulate and be
    observed over an entire trajectory**.

    Note: Consider using generate_rollout_efficient for better performance.
    """
    rollout_stride = config.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1 :: rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i : i + 1].to(device)
        if config.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                rollout_stride,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
            )

        curr_obs = torch.cat(
            (curr_obs, x_pred_pixels.unsqueeze(1)), dim=1
        )  # append current prediction
        curr_obs = curr_obs[
            :, 1:
        ]  # remove first observation => moving window of context
        visualize_preds(
            output_dir,
            idxs,
            i,
            x_pred_pixels,
        )


def generate_time(
    config,
    output_dir,
    idxs,
    all_models,
    obs_image,
    gt_output,
    delta,
    secs,
    num_cond,
    device,
):
    """
    Predict future *snapshots* at a list of absolute times given in seconds.

    For every timestep within the requested horizon (hz is the input fps), the function sums all motion deltas from the
    start up to that instant, performs a single model call (or fetches the
    ground-truth frame), and saves the resulting image.
    In short: **one-shot forecasts at specific future times, without feeding
    predictions back into the model**.

    when config.gt is set, the function will use the ground truth frames as the future frames
    """
    eval_timesteps = [sec * config.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if config.gt:
            x_pred_pixels = gt_output[:, timestep - 1].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                obs_image,
                curr_delta,
                timestep,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
            )
        visualize_preds(
            output_dir,
            idxs,
            sec,
            x_pred_pixels,
        )


def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.squeeze()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f"id_{sample_idx}")
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f"{sec}.png")
        save_image(image_file, x_pred_pixels[batch_idx])


@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="infer_config")
def main(config: DictConfig):
    # Restore original working directory
    os.chdir(get_original_cwd())

    # Setup distributed environment
    _, rank, device, _ = dist.init_distributed()
    # Create logger
    logger = create_logger(os.getcwd())

    # Print configuration
    if rank == 0:
        print_config(config)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    # Validate required parameters
    if config.output_dir is None:
        raise ValueError("output_dir must be specified")
    if config.exp_dir is None and not config.gt:
        raise ValueError("exp_dir must be specified")
    if config.datasets_to_eval is None:
        raise ValueError("datasets must be specified")
    if config.eval_type is None:
        raise ValueError("eval_type must be specified (must be 'time' or 'rollout')")

    # Process config values
    rollout_fps_values = config.rollout_fps_values
    dataset_names = config.datasets_to_eval

    # Output directory setup
    if config.gt:
        save_output_dir = os.path.join(config.output_dir, "gt")
    else:
        save_output_dir = os.path.join(get_original_cwd(), config.exp_dir, "results")
        save_output_dir = save_output_dir + f"_{config.ckp}"

    os.makedirs(save_output_dir, exist_ok=True)

    # Load experiment configuration
    exp_config = None
    if os.path.exists(config.exp_dir):
        exp_config = load_experiment_config(config.exp_dir)
        if exp_config:
            logger.info(
                f"Loaded experiment config from {config.exp_dir}/.hydra/config.yaml"
            )
            config = merge_configs(exp_config, config)
            logger.info("Experiment configuration merged with inference config")
    else:
        raise ValueError(f"Experiment directory {config.exp_dir} does not exist")

    # Get number of context frames
    num_cond = config.dataset.context_size

    # Load model if not generating ground truth
    model_lst = (None, None, None)
    if not config.gt:
        logger.info("Setting up model and tokenizer...")

        # Setup tokenizer using helper function
        vae = setup_tokenizer(config, device)
        # Setup model using helper function
        model = setup_model(config, device)
        # Load checkpoint
        checkpoint_path = f"{config.exp_dir}/checkpoints/{config.ckp}.pth.tar"
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
        ckp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logger.info(
            f"Loading model weights: {model.load_state_dict(ckp['ema'], strict=True)}"
        )
        model.eval()
        model.to(device)
        model = torch.compile(model)

        # Create diffusion model
        diffusion = setup_diffusion(config, for_eval=True, device=device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )
        model_lst = (model, diffusion, vae)
        logger.info(
            f"Model setup complete. Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # Print configuration
    if rank == 0:
        print_config(config)

    # Set latent size
    with open_dict(config):
        config.latent_size = config.model.generator.input_size

    # Load datasets
    datasets = {}
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset_val = get_dataset_eval(
            config, dataset_name, config.eval_type, predefined_index=True
        )

        if len(dataset_val) % num_tasks != 0:
            logger.warning(
                "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )

        curr_data_loader = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        datasets[dataset_name] = curr_data_loader
        logger.info(f"Dataset {dataset_name} loaded, size: {len(dataset_val)}")

    print_freq = 1
    header = "Inference: "
    metric_logger = dist.MetricLogger(delimiter="  ")

    # Run inference for each dataset
    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        logger.info(f"Running inference on {dataset_name}...")

        for data_iter_step, (idxs, obs_image, gt_image, delta) in enumerate(
            metric_logger.log_every(curr_data_loader, print_freq, header)
        ):
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)

                # NOTE Evaluation is done only in forward direction (negative min bound offset is not used)
                if config.eval_type == "rollout":
                    for rollout_fps in rollout_fps_values:
                        # Following experiment in the paper; rollout is evaluated at 1 and 4 fps
                        curr_rollout_output_dir = os.path.join(
                            dataset_save_output_dir, f"rollout_{rollout_fps}fps"
                        )
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)

                        # Use efficient rollout if configured (default: True for better performance)
                        use_efficient_rollout = getattr(
                            config, "use_efficient_rollout", True
                        )
                        rollout_fn = (
                            generate_rollout_efficient
                            if use_efficient_rollout
                            else generate_rollout
                        )

                        rollout_fn(
                            config,
                            curr_rollout_output_dir,
                            rollout_fps,
                            idxs,
                            model_lst,
                            obs_image,
                            gt_image,
                            delta,
                            num_cond,
                            device,
                        )

                elif config.eval_type == "time":
                    # secs = [1, 2, 4, 8, 16] when num_sec_eval = 5
                    # not sure why, but time is evaluated at exponent of 2 within +-16 seconds
                    # (it is not randomly sampled within this range since it is using EvalDataset)
                    secs = np.array([2**i for i in range(0, config.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, "time")
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(
                        config,
                        curr_time_output_dir,
                        idxs,
                        model_lst,
                        obs_image,
                        gt_image,
                        delta,
                        secs,
                        num_cond,
                        device,
                    )
                else:
                    raise ValueError(
                        f"Unknown eval_type: {config.eval_type}, must be 'time' or 'rollout'"
                    )

    logger.info(f"Inference completed. Results saved to {save_output_dir}")


if __name__ == "__main__":
    main()
