# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# Eval
import lpips
from dreamsim import dreamsim
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
import distributed as dist

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == "lpips":
        general_lpips_loss_fn = lpips.LPIPS(net="alex").to(device)

        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []

            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(
                    device
                )  # RGB image from [-1,1]
                img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)

                img0_list.append(img0)
                img1_list.append(img1)

            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)

            dist = general_lpips_loss_fn.forward(all_img0, all_img1)
            dist_avg = dist.mean()

            return dist_avg
    elif loss_fn_type == "dreamsim":
        dreamsim_loss_fn, preprocess = dreamsim(pretrained=True, device=device)

        def loss_fn(img0_paths, img1_paths):
            img0_list = []
            img1_list = []

            for img0_path, img1_path in zip(img0_paths, img1_paths):
                img0 = preprocess(Image.open(img0_path)).to(device)
                img1 = preprocess(Image.open(img1_path)).to(device)

                img0_list.append(img0)
                img1_list.append(img1)

            all_img0 = torch.cat(img0_list, dim=0)
            all_img1 = torch.cat(img1_list, dim=0)

            dist = dreamsim_loss_fn(all_img0, all_img1)
            dist_mean = dist.mean()

            return dist_mean
    elif loss_fn_type == "fid":
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)

        return fid_metrics
    else:
        raise NotImplementedError

    return loss_fn


def evaluate(
    config,
    dataset_name,
    eval_type,
    metric_logger,
    loss_fns,
    gt_dir,
    exp_dir,
    secs,
    rollout_fps=None,
):
    lpips_loss_fn, dreamsim_loss_fn, fid_loss_fn = loss_fns

    if eval_type == "rollout":
        eval_name = f"rollout_{rollout_fps}fps"
        image_idxs = (secs * rollout_fps) - 1
    else:  # eval_type == 'time'
        eval_name = eval_type
        image_idxs = secs.copy()

    eps = os.listdir(gt_dir)

    for batch_start in tqdm(
        range(0, len(eps), config.batch_size),
        total=(len(eps) + config.batch_size - 1) // config.batch_size,
    ):
        batch_eps = eps[batch_start : batch_start + config.batch_size]

        gt_batch, exp_batch = {}, {}
        gt_paths_batch, exp_paths_batch = {}, {}
        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []
            gt_paths_batch[sec] = []
            exp_paths_batch[sec] = []

        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)

            if not os.path.isdir(gt_ep_dir) and not os.path.isdir(exp_ep_dir):
                continue

            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f"{image_idx}.png")
                gt_sec_img = transforms.ToTensor()(
                    Image.open(gt_sec_img_path).convert("RGB")
                ).unsqueeze(0)
                exp_sec_img_path = os.path.join(exp_ep_dir, f"{image_idx}.png")
                exp_sec_img = transforms.ToTensor()(
                    Image.open(exp_sec_img_path).convert("RGB")
                ).unsqueeze(0)

                gt_batch[sec].append(gt_sec_img)
                gt_paths_batch[sec].append(gt_sec_img_path)
                exp_batch[sec].append(exp_sec_img)
                exp_paths_batch[sec].append(exp_sec_img_path)

        for sec in secs:
            lpips_dists = lpips_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])
            dreamsim_dists = dreamsim_loss_fn(gt_paths_batch[sec], exp_paths_batch[sec])

            metric_logger.meters[f"{dataset_name}_{eval_name}_lpips_{sec}s"].update(
                lpips_dists, n=1
            )
            metric_logger.meters[f"{dataset_name}_{eval_name}_dreamsim_{sec}s"].update(
                dreamsim_dists, n=1
            )

            sec_gt_batch = torch.cat(gt_batch[sec], dim=0)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0)

            fid_loss_fn[sec].update(images=sec_gt_batch, is_real=True)
            fid_loss_fn[sec].update(images=sec_exp_batch, is_real=False)

    for sec in secs:
        metric_logger.meters[f"{dataset_name}_{eval_name}_fid_{sec}s"].update(
            fid_loss_fn[sec].compute().item(), n=1
        )


def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    log_stats = {
        k: float(meter.global_avg) for k, meter in metric_logger.meters.items()
    }
    with open(log_p, "w") as json_file:
        json.dump(
            log_stats, json_file, indent=4
        )  # indent=4 adds indentation for readability


def validate_dirs(gt_dir, exp_dir):
    """Validate that required directories exist and return their paths."""
    if not os.path.exists(gt_dir):
        logger.warning(f"Ground truth directory does not exist: {gt_dir}")
        return None, None

    if not os.path.exists(exp_dir):
        logger.warning(f"Experiment directory does not exist: {exp_dir}")
        return None, None

    return gt_dir, exp_dir


def run_evaluation(
    config, dataset_name, eval_type, gt_base_dir, exp_base_dir, secs, rollout_fps=None
):
    """Run a single evaluation of the specified type and save results."""
    if eval_type == "rollout":
        eval_name = f"rollout_{rollout_fps}fps"
    else:  # eval_type == 'time'
        eval_name = "time"

    # Prepare directories
    gt_dir = os.path.join(gt_base_dir, eval_name)
    exp_dir = os.path.join(exp_base_dir, eval_name)

    # Validate directories
    gt_dir, exp_dir = validate_dirs(gt_dir, exp_dir)
    if gt_dir is None or exp_dir is None:
        return None

    try:
        # Setup metrics and loss functions
        metric_logger = dist.MetricLogger(delimiter="  ")
        device = "cuda"
        lpips_loss_fn = get_loss_fn("lpips", secs, device)
        dreamsim_loss_fn = get_loss_fn("dreamsim", secs, device)
        fid_loss_fn = get_loss_fn("fid", secs, device)
        loss_fns = (lpips_loss_fn, dreamsim_loss_fn, fid_loss_fn)

        # Run evaluation
        logger.info(f"Evaluating {eval_name} on {dataset_name}")
        with torch.no_grad():
            evaluate(
                config,
                dataset_name,
                eval_type,
                metric_logger,
                loss_fns,
                gt_dir,
                exp_dir,
                secs,
                rollout_fps,
            )

        # Save results
        output_fn = os.path.join(config.exp_dir, f"{dataset_name}_{eval_name}.json")
        save_metric_to_disk(metric_logger, output_fn)
        logger.info(f"Saved metrics to {output_fn}")

        # Return metrics for summary
        return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}

    except Exception as e:
        logger.error(f"Error during {eval_name} evaluation: {e}")
        return None


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(config: DictConfig):
    # Restore the original working directory since Hydra changes it
    os.chdir(get_original_cwd())

    # Validate required parameters
    if config.gt_dir is None:
        raise ValueError("gt_dir must be specified")
    if config.exp_dir is None:
        raise ValueError("exp_dir must be specified")
    if config.datasets_to_eval is None:
        raise ValueError("datasets_to_eval must be specified")

    # Process config values
    rollout_fps_values = [int(fps) for fps in config.rollout_fps_values.split(",")]
    eval_types = config.eval_types.split(",")

    logger.info("Evaluation configuration:")
    logger.info(f"  Datasets: {config.datasets_to_eval}")
    logger.info(f"  Evaluation types: {eval_types}")
    logger.info(f"  Ground truth directory: {config.gt_dir}")
    logger.info(f"  Experiment directory: {config.exp_dir}")
    logger.info(f"  Batch size: {config.batch_size}")
    if "rollout" in eval_types:
        logger.info(f"  Rollout FPS values: {rollout_fps_values}")

    # Use exponential time intervals for evaluation
    secs = np.array([2**i for i in range(0, config.num_sec_eval)])
    logger.info(f"  Evaluation at seconds: {secs}")

    # Check if ground truth directory exists
    if not os.path.exists(config.gt_dir):
        raise ValueError(f"Ground truth directory {config.gt_dir} does not exist")

    results_summary = {}
    for dataset_name in config.datasets_to_eval:
        gt_dataset_dir = os.path.join(config.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(config.exp_dir, dataset_name)

        # Validate dataset directories
        if not os.path.exists(gt_dataset_dir):
            logger.warning(
                f"Ground truth directory for dataset {dataset_name} does not exist: {gt_dataset_dir}"
            )
            continue

        if not os.path.exists(exp_dataset_dir):
            logger.warning(
                f"Experiment directory for dataset {dataset_name} does not exist: {exp_dataset_dir}"
            )
            continue

        dataset_results = {}

        # Run rollout evaluations if needed
        if "rollout" in eval_types:
            for rollout_fps in rollout_fps_values:
                metrics = run_evaluation(
                    config=config,
                    dataset_name=dataset_name,
                    eval_type="rollout",
                    gt_base_dir=gt_dataset_dir,
                    exp_base_dir=exp_dataset_dir,
                    secs=secs,
                    rollout_fps=rollout_fps,
                )
                if metrics:
                    dataset_results[f"rollout_{rollout_fps}fps"] = metrics

        # Run time-based evaluation if needed
        if "time" in eval_types:
            metrics = run_evaluation(
                config=config,
                dataset_name=dataset_name,
                eval_type="time",
                gt_base_dir=gt_dataset_dir,
                exp_base_dir=exp_dataset_dir,
                secs=secs,
            )
            if metrics:
                dataset_results["time"] = metrics

        if dataset_results:
            results_summary[dataset_name] = dataset_results

    # Save overall results summary
    summary_path = os.path.join(config.exp_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Saved evaluation summary to {summary_path}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
