# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import numpy as np
import lpips
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from einops import reduce, repeat, rearrange

# Import utility functions from train_utils
from train_utils import (
    setup_tokenizer,
    setup_model,
    print_config,
    cleanup,
    setup_diffusion,
)

### evo evaluation library ###
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync, metrics
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

from datasets import TrajectoryEvalDataset
from isolated_nwm_infer import model_forward_wrapper
from misc import (
    calculate_delta_yaw,
    get_action_torch,
    save_planning_pred,
    log_viz_single,
    get_transform,
    get_unnormalize,
    unnormalize_data,
    get_normalize,
)
from isolated_nwm_eval import save_metric_to_disk
import distributed as dist
from hydra_utils import load_experiment_config, merge_configs
from misc import seed_everything

logger = logging.getLogger(__name__)


def plot_images_with_losses(
    preds, losses, save_path="predictions_with_losses.png", plot_losses=True
):
    # Denormalize images from [-1, 1] to [0, 1]
    ncol = int(preds.size(0) ** 0.5)
    nrow = preds.size(0) // ncol
    if ncol * nrow < preds.size(0):
        nrow += 1
    grid_img = vutils.make_grid(preds, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(50, 50))
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // nrow, np_grid.shape[1] // ncol

    if plot_losses:
        # Overlay the losses on each image
        for idx, loss in enumerate(losses):
            row = idx // ncol
            col = idx % ncol
            x = col * img_width
            y = row * img_height
            if idx == 0:
                text = "GT Goal"
            else:
                text = f"ID: {idx - 1}  Loss: {loss:.2f}"
            ax.text(
                x + img_width / 2,
                y + 15,
                text,
                color="white",
                ha="center",
                va="top",
                fontsize=10,
                backgroundcolor="black",
            )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_batch_final(
    init_imgs,
    pred_imgs,
    goal_imgs,
    idxs,
    losses,
    save_path="final_plan.png",
):
    # images are (B, c, h, w)
    imgs_for_plotting = torch.cat([init_imgs, pred_imgs, goal_imgs])
    ncol = init_imgs.shape[0]
    grid_img = vutils.make_grid(imgs_for_plotting, nrow=ncol, padding=2)
    np_grid = grid_img.to(torch.float32).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(ncol * 10, 30))  # Adjust size as needed
    ax.imshow(np_grid)
    ax.axis("off")

    img_height, img_width = np_grid.shape[0] // 3, np_grid.shape[1] // ncol

    # Overlay the IDs and losses on each image pair in the grid
    for i in range(ncol):
        x = i * img_width
        y_pred = img_height
        ax.text(
            x + img_width / 2,
            y_pred + 15,
            f"ID: {int(idxs[i].item())} Loss: {losses[i]:.2f}",
            color="white",
            ha="center",
            va="top",
            fontsize=40,
            backgroundcolor="black",
        )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def get_dataset_eval(config, dataset_name, predefined_index=True):
    data_config = config.dataset.datasets[dataset_name]
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/navigation_eval.pkl"
    else:
        predefined_index = None

    dataset = TrajectoryEvalDataset(
        data_folder=data_config.data_folder,
        data_split_folder=data_config.test,
        dataset_name=dataset_name,
        image_size=config.dataset.image_size,
        min_dist_cat=config.trajectory_eval_distance.min_dist_cat,
        max_dist_cat=config.trajectory_eval_distance.max_dist_cat,
        len_traj_pred=config.trajectory_eval_len_traj_pred,
        traj_stride=config.traj_stride,
        context_size=config.trajectory_eval_context_size,
        normalize=config.dataset.normalize,
        transform=get_transform(
            config.dataset.image_size, config.dataset.mean, config.dataset.std
        ),
        action_stats=config.dataset.action_stats,
        waypoint_spacing=data_config.metric_waypoint_spacing,
        predefined_index=predefined_index,
        traj_names="rollout_traj_names.txt",
    )

    return dataset


def repeat_along_first_dim(tensor, num_repeat):
    return repeat(tensor, "b ... -> (repeat b) ...", repeat=num_repeat)


class WM_Planning_Evaluator:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.exp_dir = config.exp_dir
        _, _, device, _ = dist.init_distributed()
        self.device = torch.device(device)

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()

        # Setting up Config
        self.exp_eval = self.exp_dir
        self.get_eval_name()

        # Load experiment configuration
        self.logger = logging.getLogger(__name__)
        exp_config = None
        if os.path.exists(self.exp_dir):
            exp_config = load_experiment_config(self.exp_dir)
            if exp_config:
                self.logger.info(
                    f"Loaded experiment config from {self.exp_dir}/.hydra/config.yaml"
                )
                self.config = merge_configs(exp_config, self.config)
                self.logger.info("Experiment configuration merged with planning config")
        else:
            self.logger.warning(
                f"Experiment file {self.exp_dir} does not exist, using default configuration"
            )

        # Print configuration
        if global_rank == 0:
            print_config(self.config)

        # Loading Model
        self.logger.info("Loading model...")
        # Use setup_tokenizer for VAE initialization
        self.vae = setup_tokenizer(self.config, self.device)
        # Use setup_model from train_utils to create the model
        self.model = setup_model(self.config, self.device)
        # Load checkpoint using train_utils helper
        checkpoint_path = f"{self.config.exp_dir}/checkpoints/{self.config.ckp}.pth.tar"
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
        ckp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.logger.info(
            f"Loading self.model weights: {self.model.load_state_dict(ckp['ema'], strict=True)}"
        )
        self.vae.eval()
        self.model.eval()
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        self.diffusion = setup_diffusion(self.config, for_eval=True, device=self.device)

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[self.device], find_unused_parameters=False
        )
        self.logger.info(
            f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # logging directory - okay in DDP since exist_ok=True
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Loading Datasets
        self.dataset_names = self.config.datasets_to_eval
        self.datasets = {}
        for dataset_name in self.dataset_names:
            dataset_val = get_dataset_eval(
                self.config, dataset_name, predefined_index=True
            )

            if len(dataset_val) % num_tasks != 0:
                self.logger.info(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )

            curr_data_loader = torch.utils.data.DataLoader(
                dataset_val,
                sampler=sampler_val,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            self.datasets[dataset_name] = curr_data_loader

        self.latent_size = self.config.model.generator.input_size
        self.num_cond = self.config.eval_context_size
        self.mode = "cem"  # assume CEM for planning
        self.num_samples = self.config.num_samples
        self.topk = self.config.topk
        self.opt_steps = self.config.opt_steps
        self.num_repeat_eval = self.config.num_repeat_eval
        self.action_dim = 3  # hardcoded (delta_x, delta_y, delta_yaw)
        self.unnormalize_fn = get_unnormalize(
            self.config.dataset.mean, self.config.dataset.std
        )

        self.cost_fn = self.config.cost_fn
        if self.cost_fn == "lpips":
            self.lpips = lpips.LPIPS(net="alex").to(self.device)
            self.lpips_normalize = get_normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        elif self.cost_fn == "dino":
            self.dino = torch.hub.load(
                repo_or_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "dinov3"
                ),
                model="dinov3_vitb16",
                source="local",
                weights=os.path.join(
                    os.environ.get("BASE_TOKENIZER_CKPT"),
                    "/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
                ),
            ).to(self.device)
            self.dino_normalize = get_normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        self.compute_cost_with_recon = self.config.compute_cost_with_recon
        logger.info(
            f"Computing cost with reconstruction: {self.compute_cost_with_recon}"
        )

        self.action_stats = {}
        for key in self.config.dataset.action_stats:
            self.action_stats[key] = torch.tensor(self.config.dataset.action_stats[key])

    def compute_cost(self, pred_image, goal_image, pred_code, goal_code):
        if self.cost_fn == "lpips":
            pred_image = self.lpips_normalize(pred_image)
            goal_image = self.lpips_normalize(goal_image)
            return self.lpips(pred_image, goal_image).flatten(0)
        elif self.cost_fn == "dino":
            pred_image = self.dino_normalize(pred_image)
            goal_image = self.dino_normalize(goal_image)
            dino_pred_image = self.dino.get_intermediate_layers(pred_image, n=1)[0]
            dino_goal_image = self.dino.get_intermediate_layers(goal_image, n=1)[0]
            cost = 1 - F.cosine_similarity(
                dino_pred_image, dino_goal_image, dim=-1
            ).mean(dim=-1)
            return cost
        elif self.cost_fn == "code":
            cost = (pred_code - goal_code).norm(dim=-1)
            cost = reduce(cost, "b l -> b", "mean")
            return cost
        else:
            raise ValueError(f"Unknown cost function: {self.cost_fn}")

    def init_mu_sigma(self, obs_0, traj_len, dataset_name):
        n_evals = obs_0.shape[0]
        mu = torch.zeros(n_evals, self.action_dim)
        mu[:,] = torch.tensor(self.config.plan_datasets_hyperparams[dataset_name]["mu"])
        sigma = torch.ones([n_evals, self.action_dim])
        sigma[:,] = torch.tensor(
            self.config.plan_datasets_hyperparams[dataset_name]["var_scale"]
        )
        return mu, sigma

    def generate_actions(
        self,
        dataset_save_output_dir,
        dataset_name,
        idxs,
        obs_image,
        goal_image,
        gt_actions,
        len_traj_pred,
    ):
        idx_string = "_".join(map(str, idxs.flatten().int().tolist()))
        image_plot_dir = os.path.join(dataset_save_output_dir, "plots")
        os.makedirs(image_plot_dir, exist_ok=True)

        n_evals = obs_image.shape[0]
        mu, sigma = self.init_mu_sigma(obs_image, len_traj_pred, dataset_name)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        goal_indices = self.vae.encode(
            rearrange(goal_image, "b t ... -> (b t) ...").to(self.device)
        )
        goal_code = self.vae.get_code_from_latent(goal_indices)

        goal_image = self.unnormalize_fn(goal_image)
        if self.compute_cost_with_recon:
            goal_image = self.vae.decode(goal_indices, denormalize=True).to(
                torch.float32
            )

        for i in range(self.opt_steps):
            losses = []
            for traj in range(n_evals):
                traj_id = int(idxs.flatten()[traj].item())
                logger.info(f"Trajectory {traj_id}")
                sample = (
                    torch.randn(self.num_samples, self.action_dim).to(self.device)
                    * sigma[traj]
                    + mu[traj]
                )
                single_delta = sample[:, :2]
                deltas = single_delta.unsqueeze(1).repeat(1, len_traj_pred, 1)
                unnorm_deltas = unnormalize_data(deltas, self.action_stats)
                delta_yaw = calculate_delta_yaw(unnorm_deltas)
                deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
                deltas[:, -1, -1] += sample[:, -1] * np.pi

                # TODO: implement unnormalize for the cur_obs and cur_goal (when goal image is not from the recon)
                cur_obs_image = repeat_along_first_dim(
                    obs_image[traj].unsqueeze(0), self.num_samples
                )
                cur_goal_image = repeat_along_first_dim(
                    goal_image[traj].unsqueeze(0), self.num_samples
                ).squeeze(1)
                cur_goal_code = repeat_along_first_dim(goal_code, self.num_samples)

                # WM is stochastic, so we can repeat the evaluation of each trajectory and average to reduce variance
                if self.num_repeat_eval * self.num_samples > 120:
                    cur_losses = []
                    for r in range(self.num_repeat_eval):
                        preds, pred_codes = (
                            self.autoregressive_rollout_without_intermediate_decoding(
                                cur_obs_image,
                                deltas,
                                self.config.rollout_stride,
                                only_final=True,
                                return_codes=True,
                            )
                        )
                        preds = preds[:, -1]  # take the last predicted image
                        loss = self.compute_cost(
                            preds.to(self.device),
                            cur_goal_image.to(self.device),
                            pred_codes.to(self.device),
                            cur_goal_code.to(self.device),
                        )
                        cur_losses.append(loss)

                    loss = torch.stack(cur_losses).mean(dim=0)
                else:
                    expanded_deltas = repeat_along_first_dim(
                        deltas, self.num_repeat_eval
                    )
                    expanded_obs_image = repeat_along_first_dim(
                        cur_obs_image, self.num_repeat_eval
                    )
                    expanded_goal_image = repeat_along_first_dim(
                        cur_goal_image, self.num_repeat_eval
                    )
                    expanded_goal_code = repeat_along_first_dim(
                        cur_goal_code, self.num_repeat_eval
                    )

                    preds, pred_codes = (
                        self.autoregressive_rollout_without_intermediate_decoding(
                            expanded_obs_image,
                            expanded_deltas,
                            self.config.rollout_stride,
                            only_final=True,
                            return_codes=True,
                        )
                    )
                    preds = preds[:, -1]

                    loss = self.compute_cost(
                        preds.to(self.device),
                        expanded_goal_image.to(self.device),
                        pred_codes.to(self.device),
                        expanded_goal_code.to(self.device),
                    )
                    loss = loss.view(self.num_repeat_eval, -1)
                    loss = loss.mean(dim=0)

                    preds = preds[: self.config.num_samples]

                sorted_idx = torch.argsort(loss)
                topk_idx = sorted_idx[: self.topk]
                topk_action = deltas[topk_idx][:, -1]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                if self.config.plot:
                    cur_obs_image = self.unnormalize_fn(cur_obs_image)
                    self.visualize_trajectories(
                        dataset_name,
                        gt_actions,
                        image_plot_dir,
                        i,
                        traj,
                        traj_id,
                        deltas,
                        cur_obs_image,
                        cur_goal_image,
                        preds,
                        loss,
                        topk_idx,
                    )

        # Final rollout (single sample)
        deltas = mu[:, :2]
        deltas = deltas.unsqueeze(1).repeat(1, len_traj_pred, 1)

        # Calculate yaws
        unnorm_deltas = unnormalize_data(deltas, self.action_stats)
        delta_yaw = calculate_delta_yaw(unnorm_deltas)
        deltas = torch.cat((deltas, delta_yaw.to(deltas.device)), dim=-1)
        deltas[:, -1, -1] += mu[:, -1] * np.pi

        if self.config.save_preds or self.config.plot:
            # Re-run the rollout with the final actions (fitted using topk actions)
            preds, pred_codes = self.autoregressive_rollout_without_intermediate_decoding(
                obs_image,
                deltas,
                self.config.rollout_stride,
                only_final=True,
                return_codes=True,
            )
            preds = preds[:, -1]  # take the last predicted image

            loss = self.compute_cost(
                preds.to(self.device),
                goal_image.squeeze(1).to(self.device),
                pred_codes.to(self.device),
                goal_code.to(self.device),
            )

            if self.config.save_preds:
                save_planning_pred(
                    dataset_save_output_dir,
                    n_evals,
                    idxs,
                    obs_image,
                    goal_image,
                    preds,
                    deltas,
                    loss,
                    gt_actions,
                )

            if self.config.plot:
                img_name = os.path.join(image_plot_dir, f"FINAL_{idx_string}.png")
                plot_batch_final(
                    self.unnormalize_fn(obs_image[:, -1]).to(self.device),
                    preds,
                    goal_image.squeeze(1).to(self.device),
                    idxs,
                    losses,
                    save_path=img_name,
                )

        pred_actions = get_action_torch(deltas[:, :, :2], self.action_stats)
        pred_yaw = deltas[:, :, -1].sum(1)
        return pred_actions, pred_yaw

    def visualize_trajectories(
        self,
        dataset_name,
        gt_actions,
        image_plot_dir,
        i,
        traj,
        traj_id,
        deltas,
        cur_obs_image,
        cur_goal_image,
        preds,
        loss,
        topk_idx,
    ):
        img_for_plotting = torch.cat([cur_goal_image[0:1].to(self.device), preds])
        loss_for_plotting = torch.cat((torch.tensor([0]).to(self.device), loss))
        img_name = os.path.join(image_plot_dir, f"idx{traj_id}_iter{i}.png")
        plot_images_with_losses(
            img_for_plotting,
            loss_for_plotting,
            save_path=img_name,
        )
        plot_name = os.path.join(image_plot_dir, f"idx{traj_id}_iter{i}_trajs.png")
        num_plot = self.config.num_samples
        log_viz_single(
            self.config.dataset.datasets[dataset_name]["metric_waypoint_spacing"],
            cur_obs_image[0],
            cur_goal_image[0],
            preds[:num_plot],
            deltas[:num_plot],
            loss[:num_plot],
            topk_idx[0:1],
            gt_actions[traj],
            self.action_stats,
            plan_iter=i,
            output_dir=plot_name,
        )

    def autoregressive_rollout(self, obs_image, deltas, rollout_stride):
        deltas = deltas.unflatten(1, (-1, rollout_stride)).sum(2)
        preds = []
        curr_obs = obs_image.clone().to(self.device)

        for i in tqdm(range(deltas.shape[1])):
            curr_delta = deltas[:, i : i + 1]
            all_models = self.model, self.diffusion, self.vae
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                self.config.rollout_stride,
                self.latent_size,
                num_cond=self.num_cond,
                device=self.device,
            )
            x_pred_pixels = x_pred_pixels.unsqueeze(1)
            curr_obs = torch.cat(
                (curr_obs, x_pred_pixels), dim=1
            )  # append current prediction
            curr_obs = curr_obs[:, 1:]  # remove first observation
            preds.append(x_pred_pixels)

        # (B, T, C, H, W) - B: num_samples (sampled from the proposal distribution), T: num_timesteps (planning horizon - last one is the goal), C: num_channels, H: height, W: width
        preds = torch.cat(preds, 1)
        return preds

    def autoregressive_rollout_without_intermediate_decoding(
        self,
        obs_image,
        deltas,
        rollout_stride,
        only_final=False,
        return_codes=False,
    ):
        batch_size = 80  # batch size hardcoded, need to be configurable
        num_batches = obs_image.shape[0] // batch_size

        if num_batches * batch_size < obs_image.shape[0]:
            num_batches += 1

        ret = []
        ret_codes = []

        for i in range(num_batches):
            batch_deltas = deltas[i * batch_size : (i + 1) * batch_size]
            batch_obs_image = obs_image[i * batch_size : (i + 1) * batch_size]

            batch_deltas = batch_deltas.unflatten(1, (-1, rollout_stride)).sum(2)
            preds_latents = []

            batch_obs = batch_obs_image.to(self.device)
            B, T = batch_obs.shape[:2]
            batch_obs_latents = self.vae.encode(batch_obs.flatten(0, 1)).unflatten(
                0, (B, T)
            )

            for i in tqdm(range(batch_deltas.shape[1])):
                batch_delta = batch_deltas[:, i : i + 1]
                all_models = self.model, self.diffusion, self.vae
                x_pred_latents = model_forward_wrapper(
                    all_models,
                    batch_obs_latents,
                    batch_delta,
                    self.config.rollout_stride,
                    self.latent_size,
                    num_cond=self.num_cond,
                    device=self.device,
                    skip_tokenizer=True,
                )
                x_pred_latents = x_pred_latents.unsqueeze(1)

                batch_obs_latents = torch.cat(
                    (batch_obs_latents, x_pred_latents), dim=1
                )  # append current prediction
                batch_obs_latents = batch_obs_latents[:, 1:]  # remove first observation
                preds_latents.append(x_pred_latents)

            preds_latents = torch.cat(preds_latents, 1)
            B, T = preds_latents.shape[:2]
            # (B, T, C, H, W) - B: num_samples (sampled from the proposal distribution), T: num_timesteps (planning horizon - last one is the goal), C: num_channels, H: height, W: width
            if only_final:
                preds_pixels = self.vae.decode(
                    preds_latents[:, -1], denormalize=True
                ).unsqueeze(1)
                pred_codes = self.vae.get_code_from_latent(preds_latents[:, -1])
            else:
                preds_pixels = self.vae.decode(
                    preds_latents.flatten(0, 1), denormalize=True
                ).unflatten(0, (B, T))
                pred_codes = self.vae.get_code_from_latent(
                    preds_latents.flatten(0, 1)
                ).unflatten(0, (B, T))
            preds_pixels = torch.clip(preds_pixels, -1.0, 1.0)
            ret.append(preds_pixels)
            ret_codes.append(pred_codes)

        ret = torch.cat(ret, 0)
        ret_codes = torch.cat(ret_codes, 0)
        return ret, ret_codes if return_codes else None

    def get_eval_name(self):
        # Get evaluation name for logging. Should overwrite for specific experiments
        self.eval_name = f"CEM_N{self.config.num_samples}_K{self.config.topk}_RS{self.config.rollout_stride}_rep{self.config.num_repeat_eval}_OPT{self.config.opt_steps}_COST-{self.config.cost_fn}-RECON-{self.config.compute_cost_with_recon}"

        if override := OmegaConf.select(
            self.config, "model.diffusion.num_timesteps", default=None
        ):
            self.eval_name += f"_SAMPLING-STEPS{override}"

        if override := OmegaConf.select(
            self.config, "model.diffusion.history_sample_t", default=None
        ):
            self.eval_name += f"_HISTORY-SAMPLE-T{override}"

        if override := OmegaConf.select(
            self.config, "model.tokenizer.compact_tokenizer.diffusion", default=None
        ):
            for k, v in override.items():
                self.eval_name += f"_TOK-DIFF_{k}{v}"

    def actions_to_traj(self, actions):
        positions_xyz = torch.zeros((actions.shape[0], 3))
        positions_xyz[:, :2] = actions
        orientations_quat_wxyz = torch.zeros(
            (actions.shape[0], 4)
        )  # Define identity quaternion
        orientations_quat_wxyz[:, -1] = 1  # Define identity quaternion
        timestamps = torch.arange(actions.shape[0], dtype=torch.float64)
        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz,
            orientations_quat_wxyz=orientations_quat_wxyz,
            timestamps=timestamps,
        )
        return traj

    @torch.no_grad()
    def evaluate(self):
        for dataset_name in self.dataset_names:
            metric_logger = dist.MetricLogger(delimiter="  ")
            header = "Test:"

            dataset_save_output_dir = os.path.join(self.config.output_dir, dataset_name)
            os.makedirs(dataset_save_output_dir, exist_ok=True)
            eval_save_output_dir = os.path.join(dataset_save_output_dir, self.eval_name)
            os.makedirs(eval_save_output_dir, exist_ok=True)

            curr_data_loader = self.datasets[dataset_name]
            for (
                idxs,
                obs_image,
                goal_image,
                gt_actions,
                goal_pos,
            ) in metric_logger.log_every(curr_data_loader, 1, header):
                obs_image = obs_image[:, -self.num_cond :]
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    pred_actions, pred_yaw = self.generate_actions(
                        eval_save_output_dir,
                        dataset_name,
                        idxs,
                        obs_image,
                        goal_image,
                        gt_actions,
                        self.config.trajectory_eval_len_traj_pred,
                    )
                for i in range(len(obs_image)):
                    pred_traj_i = self.actions_to_traj(pred_actions[i, :, :2])
                    gt_traj_i = self.actions_to_traj(gt_actions[i, :, :2])

                    ate, rpe_trans, _ = self.eval_metrics(gt_traj_i, pred_traj_i)

                    pred_final_pos = pred_actions[i, -1, :2].to("cpu")  # (2,)
                    pred_final_yaw = pred_yaw[i].to("cpu")  #
                    goal_final_pos = goal_pos[i, 0, :2]  # (2,)
                    goal_final_yaw = goal_pos[i, 0, -1]  # (B,)
                    pos_diff_norm = torch.norm(pred_final_pos - goal_final_pos)
                    yaw_diff = pred_final_yaw - goal_final_yaw  #
                    yaw_diff_norm = torch.atan2(
                        torch.sin(yaw_diff), torch.cos(yaw_diff)
                    ).abs()

                    metric_logger.meters["{}_ate".format(dataset_name)].update(ate, n=1)
                    metric_logger.meters["{}_rpe_trans".format(dataset_name)].update(
                        rpe_trans, n=1
                    )
                    metric_logger.meters[
                        "{}_pos_diff_norm".format(dataset_name)
                    ].update(pos_diff_norm, n=1)
                    metric_logger.meters[
                        "{}_yaw_diff_norm".format(dataset_name)
                    ].update(yaw_diff_norm, n=1)
                    break

            output_fn = os.path.join(
                self.config.output_dir, f"{dataset_name}_{self.eval_name}.json"
            )
            save_metric_to_disk(metric_logger, output_fn)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()

        # Cleanup DDP resources when done
        cleanup()

    def eval_metrics(self, traj_ref, traj_pred):
        traj_ref, traj_pred = sync.associate_trajectories(traj_ref, traj_pred)

        result = main_ape.ape(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
        )
        ate = result.stats["rmse"]

        result = main_rpe.rpe(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=False,
            correct_scale=False,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            rel_delta_tol=0.1,
        )
        rpe_rot = result.stats["rmse"]

        result = main_rpe.rpe(
            traj_ref,
            traj_pred,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False,
            delta=1.0,
            delta_unit=metrics.Unit.frames,
            rel_delta_tol=0.1,
        )
        rpe_trans = result.stats["rmse"]

        return ate, rpe_trans, rpe_rot


@hydra.main(version_base=None, config_path="conf", config_name="plan_config")
def main(config: DictConfig):
    # Restore the original working directory
    os.chdir(get_original_cwd())

    seed_everything(42)

    # Create evaluator instance with Hydra config
    evaluator = WM_Planning_Evaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
