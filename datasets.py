# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import logging
import random
import numpy as np
import torch
import os
from PIL import Image
from typing import Tuple
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import (
    angle_difference,
    get_data_path,
    get_delta_np,
    normalize_data,
    to_local_coords,
)
from hydra.utils import get_original_cwd
import lightning as L
from typing import List, Sequence, Union, Any
import webdataset as wds
from torchvision import transforms
from omegaconf import DictConfig
import json
import base64

logger = logging.getLogger(__name__)


def load_image(path):
    """Load an image and return a copy, ensuring file handle is properly closed."""
    with Image.open(path) as img:
        # Convert to RGB and copy to ensure data is loaded into memory
        return img.convert("RGB").copy()


class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


class DualNormalizationTransform:
    """
    Transform class that applies dual normalization to images:
    - One normalization for the base tokenizer
    - Another normalization for DINOv2 (always ImageNet stats)

    Returns a dictionary with both normalized versions.
    """

    def __init__(
        self,
        image_size: int,
        dinov2_image_size: int,
        use_augmentation: bool = False,
        tokenizer_mean: List[float] = [0.5, 0.5, 0.5],
        tokenizer_std: List[float] = [0.5, 0.5, 0.5],
        dinov2_mean: List[float] = [0.485, 0.456, 0.406],
        dinov2_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.image_size = image_size
        self.dinov2_image_size = dinov2_image_size
        self.use_augmentation = use_augmentation

        # Determine the larger size to avoid upsampling
        max_size = max(image_size, dinov2_image_size)

        # Common transforms applied once (including random augmentations)
        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    max_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                lambda image: image.clamp(0, 1),
                transforms.RandomCrop(max_size)
                if use_augmentation
                else transforms.CenterCrop(max_size),
                transforms.RandomHorizontalFlip(0.5)
                if use_augmentation
                else lambda image: image,
            ]
        )

        # Tokenizer-specific resize (if needed) and normalization
        self.tokenizer_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
                if image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=tokenizer_mean, std=tokenizer_std),
            ]
        )

        # DINOv2-specific resize (if needed) and normalization
        self.dinov2_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    dinov2_image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
                if dinov2_image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=dinov2_mean, std=dinov2_std),
            ]
        )

    def __call__(self, image):
        """Apply dual normalization to the image."""
        # Apply common transforms once (including random augmentations)
        augmented_image = self.common_transforms(image)

        # Apply specific resizing (if needed) and normalizations to the same augmented image
        tokenizer_image = self.tokenizer_post_transform(augmented_image.clone())
        dinov2_image = self.dinov2_post_transform(augmented_image.clone())

        # Return the tokenizer-normalized image as the main output
        # Store the DINOv2 version as additional metadata
        return {"image": tokenizer_image, "dinov2_image": dinov2_image}


class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = os.path.join(get_original_cwd(), data_folder)
        self.data_split_folder = os.path.join(get_original_cwd(), data_split_folder)
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs
        self.missing_trajectories = []

        traj_names_file = os.path.join(self.data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        # Scan for existing trajectories
        self.existing_traj_names = []
        logger.info(f"Scanning for existing trajectories in {self.dataset_name}...")
        for traj_name in tqdm.tqdm(self.traj_names):
            traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
            if os.path.exists(traj_path):
                self.existing_traj_names.append(traj_name)
            else:
                self.missing_trajectories.append(traj_name)

        logger.info(
            f"Found {len(self.existing_traj_names)}/{len(self.traj_names)} trajectories in {self.dataset_name}"
        )

        # Save missing trajectories to a file
        if self.missing_trajectories:
            missing_trajs_file = os.path.join(
                self.data_split_folder, f"missing_trajectories_{self.dataset_name}.txt"
            )
            with open(missing_trajs_file, "w") as f:
                for traj in self.missing_trajectories:
                    f.write(f"{traj}\n")
            logger.info(
                f"Saved {len(self.missing_trajectories)} missing trajectories to {missing_trajs_file}"
            )

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # use this index to retrieve the dataset name from the data_config.yaml
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in action_stats:
            self.ACTION_STATS[key] = np.expand_dims(action_stats[key], axis=0)
        self.WAYPOINT_SPACING = waypoint_spacing

    def _is_index_valid(
        self, traj_name, curr_state_time, min_offset_bound, max_offset_bound
    ):
        traj = self._get_trajectory(traj_name)
        min_bound = curr_state_time + min_offset_bound
        max_bound = curr_state_time + max_offset_bound
        is_index_valid = min_bound >= 0 and max_bound < len(traj["position"])
        return is_index_valid

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, curr_state_time, min_offset_bound, max_offset_bound) for each observation in the dataset
        """
        if predefined_index:
            logger.info(
                f"****** Using a predefined evaluation index... {predefined_index}******"
            )
            with open(predefined_index, "rb") as f:
                raw_index_to_data = pickle.load(f)
            index_to_data = [
                x for x in raw_index_to_data if x[0] not in self.missing_trajectories
            ]
            invalid_indices = [x for x in index_to_data if not self._is_index_valid(*x)]
            self.index_to_data = TorchSerializedList(
                [x for x in index_to_data if x not in invalid_indices]
            )

            logger.info(
                f"Found {len(self.index_to_data)} / {len(raw_index_to_data)} valid indices"
            )
            # Save invalid indices to a file
            invalid_indices_file = os.path.join(
                self.data_split_folder,
                f"invalid_indices_in_{os.path.splitext(os.path.basename(predefined_index))[0]}.txt",
            )
            with open(invalid_indices_file, "w") as f:
                for idx in invalid_indices:
                    f.write(f"{idx[0]}, {idx[1]}, {idx[2]}, {idx[3]}\n")
            logger.info(
                f"Saved {len(invalid_indices)} invalid indices to {invalid_indices_file}"
            )
        else:
            logger.info("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.data_split_folder,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )

            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        NOTE: This function is only used when pre-defined index is not provided; meaning that it is only called once before the training, and doesnt used in evaluation since index is already provided.

        Build an index consisting of tuples (obs_traj_name, curr_state_time, min_offset_bound, max_offset_bound)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(
            self.existing_traj_names, disable=not use_tqdm, dynamic_ncols=True
        ):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append(
                    (traj_name, curr_time, min_goal_distance, max_goal_distance)
                )

        return TorchSerializedList(samples_index), TorchSerializedList(goals_index)

    def _get_trajectory(self, trajectory_name):
        traj_path = os.path.join(self.data_folder, trajectory_name, "traj_data.pkl")
        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)
        for k, v in traj_data.items():
            traj_data[k] = v.astype("float", copy=False)
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]

        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)

        if self.normalize:
            actions[:, :2] /= self.WAYPOINT_SPACING
            goal_pos[:, :2] /= self.WAYPOINT_SPACING

        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos


class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str = "traj_names.txt",
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_curr = str(f_curr)
            curr_time, min_goal_dist, max_goal_dist = (
                int(curr_time),
                int(min_goal_dist),
                int(max_goal_dist),
            )

            goal_offset = np.random.randint(
                min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs)
            )
            goal_time = (curr_time + goal_offset).astype("int")
            rel_time = (goal_offset).astype("float") / (
                128.0
            )  # TODO: refactor, currently a fixed const

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            context = [(f_curr, t) for t in context_times] + [
                (f_curr, t) for t in goal_time
            ]

            obs_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in context
                ]
            )

            # Compute actions
            curr_traj_data = self._get_trajectory(f_curr)
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            # Goal min max bound offset is not used in EvalDataset
            f_curr, curr_time, _, _ = self.index_to_data[i]
            f_curr = str(f_curr)
            curr_time = int(curr_time)

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))

            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in context
                ]
            )
            pred_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in pred
                ]
            )

            # Compute actions
            curr_traj_data = self._get_trajectory(f_curr)
            actions, _ = self._compute_actions(
                curr_traj_data, curr_time, np.array([curr_time + 1])
            )  # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            logger.error(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )

    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]

            f_goal, goal_time, _ = self._sample_goal(
                f_curr, curr_time, min_goal_dist, max_goal_dist
            )

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in context
                ]
            )
            goal_image = self.transform(
                load_image(get_data_path(self.data_folder, f_goal, goal_time))
            ).unsqueeze(0)

            curr_traj_data = self._get_trajectory(f_curr)
            actions, goal_pos = self._compute_actions(
                curr_traj_data, curr_time, np.array([goal_time])
            )

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class DebugDatasetDataModule(L.LightningDataModule):
    """
    Lightning DataModule for debugging that returns the same single image for all batches.
    Useful for debugging model behavior with consistent input.
    """

    def __init__(
        self,
        config: DictConfig,
        debug_image_path: str,
    ):
        super().__init__()
        self.config = config
        self.dataset_config = config.dataset
        self.debug_image_path = debug_image_path

        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers

        # Load the debug image once
        self.debug_image = Image.open(debug_image_path)

        # Initialize transforms
        self.transform = self._get_transforms()

        # Process the debug image through transforms
        self.processed_image = self.transform(self.debug_image)

        # If dual normalization is used, extract both versions
        if isinstance(self.processed_image, dict):
            self.debug_batch_data = (
                self.processed_image["image"],
                self.processed_image["dinov2_image"],
                0,  # dummy class
                "debug_image",  # dummy key
            )
        else:
            self.debug_batch_data = (
                self.processed_image,
                0,  # dummy class
                "debug_image",  # dummy key
            )

    def _get_transforms(self):
        """Get transform objects for the debug dataset."""
        image_size = self.dataset_config.image_size
        mean = self.dataset_config.get("mean", [0.5, 0.5, 0.5])
        std = self.dataset_config.get("std", [0.5, 0.5, 0.5])

        # Check if dual normalization is enabled
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        if use_dual_normalization:
            dinov2_image_size = self.dataset_config.get("dinov2_image_size", 224)
            dinov2_mean = self.dataset_config.get("dinov2_mean", [0.485, 0.456, 0.406])
            dinov2_std = self.dataset_config.get("dinov2_std", [0.229, 0.224, 0.225])

            return DualNormalizationTransform(
                image_size=image_size,
                dinov2_image_size=dinov2_image_size,
                use_augmentation=False,  # No augmentation for debug
                tokenizer_mean=mean,
                tokenizer_std=std,
                dinov2_mean=dinov2_mean,
                dinov2_std=dinov2_std,
            )
        else:
            transform_list = [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=mean, std=std),
            ]
            return transforms.Compose(transform_list)

    def _create_debug_dataset(self, size: int = 1000):
        """Create a dataset that returns the same image repeatedly."""

        class DebugDataset(torch.utils.data.Dataset):
            def __init__(self, debug_data, batch_size):
                self.debug_data = debug_data
                self.batch_size = batch_size
                # Create a batch of the same image
                if len(debug_data) == 4:  # dual normalization case
                    self.batch = (
                        debug_data[0].unsqueeze(0).repeat(batch_size, 1, 1, 1),
                        debug_data[1].unsqueeze(0).repeat(batch_size, 1, 1, 1),
                        torch.tensor([debug_data[2]] * batch_size),
                        [debug_data[3]] * batch_size,
                    )
                else:  # single normalization case
                    self.batch = (
                        debug_data[0].unsqueeze(0).repeat(batch_size, 1, 1, 1),
                        torch.tensor([debug_data[1]] * batch_size),
                        [debug_data[2]] * batch_size,
                    )

            def __len__(self):
                return size  # Return enough samples for debugging

            def __getitem__(self, idx):
                return self.batch

        return DebugDataset(self.debug_batch_data, self.batch_size)

    def train_dataloader(self):
        """Create training dataloader that returns the same batch repeatedly."""
        dataset = self._create_debug_dataset(size=1000)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Already batched
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader that returns the same batch repeatedly."""
        dataset = self._create_debug_dataset(size=1)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Already batched
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader that returns the same batch repeatedly."""
        dataset = self._create_debug_dataset(size=1)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Already batched
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class WebDatasetDataModule(L.LightningDataModule):
    """
    Lightning DataModule for tokenizer training using WebDataset.
    Handles sharded data loading with efficient distributed training support.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.dataset_config = config.dataset

        self.batch_size = config.dataset.batch_size
        self.num_workers = config.dataset.num_workers

        # WebDataset specific settings
        self.shuffle_buffer_size = config.dataset.get("shuffle_buffer_size", 5000)
        self.shuffle_initial = config.dataset.get("shuffle_initial", 1000)

        # Keys to keep from the dataset
        self.keys_to_keep = config.dataset.get(
            "keys_to_keep", ("image", "cls", "__key__")
        )

        # Will be set during setup
        self.train_urls = config.dataset.train_shards
        self.val_urls = config.dataset.val_shards
        self.test_urls = config.dataset.test_shards
        self.estimated_train_size = config.dataset.estimated_train_size
        self.estimated_val_size = config.dataset.estimated_val_size
        self.estimated_test_size = config.dataset.estimated_test_size

    # Helper functions for WebDataset processing
    def _truncate_extensions(self, sample):
        """Remove file extensions from keys."""
        truncated = {}
        for key, value in sample.items():
            if isinstance(key, str) and "." in key:
                new_key = key.split(".")[0]
                truncated[new_key] = value
            else:
                truncated[key] = value
        return truncated

    def _rename_extensions(self, d: dict) -> dict:
        new_d = {}
        for k, v in d.items():
            if k.lower() in ["jpg", "jpeg", "png", "webp"]:
                new_d["image"] = v
            else:
                new_d[k] = v
        return new_d

    def _filter_keys(self, sample, keys_to_keep: Tuple[str] = tuple()):
        """Filter sample to keep only specified keys."""
        if not keys_to_keep:
            return sample

        filtered = {}
        for key in keys_to_keep:
            if key in sample:
                filtered[key] = sample[key]
        return filtered

    def _process_dual_normalization(self, sample):
        """Process dual normalization output from DualNormalizationTransform."""
        # The transforms return a dictionary with 'image' and 'dinov2_image'
        if isinstance(sample.get("image"), dict):
            image_dict = sample["image"]
            # Replace the image key with the tokenizer-normalized version
            sample["image"] = image_dict["image"]
            # Add the DINOv2-normalized version as a separate key
            sample["dinov2_image"] = image_dict["dinov2_image"]
        return sample

    def _get_transforms(self, is_training: bool = False):
        """Get transform objects for the WebDataset pipeline."""
        image_size = self.dataset_config.image_size
        use_augmentation = self.dataset_config.get("use_augmentation", False)
        mean = self.dataset_config.get("mean", [0.5, 0.5, 0.5])
        std = self.dataset_config.get("std", [0.5, 0.5, 0.5])

        # Check if dual normalization is enabled
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        if use_dual_normalization:
            # Return a custom transform that handles dual normalization
            dinov2_image_size = self.dataset_config.get("dinov2_image_size", 224)
            dinov2_mean = self.dataset_config.get("dinov2_mean", [0.485, 0.456, 0.406])
            dinov2_std = self.dataset_config.get("dinov2_std", [0.229, 0.224, 0.225])

            return DualNormalizationTransform(
                image_size=image_size,
                dinov2_image_size=dinov2_image_size,
                use_augmentation=use_augmentation and is_training,
                tokenizer_mean=mean,
                tokenizer_std=std,
                dinov2_mean=dinov2_mean,
                dinov2_std=dinov2_std,
            )
        else:
            # Original single normalization pipeline
            transform_list = [transforms.ToTensor()]

            # Add augmentations for training
            if use_augmentation and is_training:
                transform_list.extend(
                    [
                        transforms.RandomHorizontalFlip(0.5),
                    ]
                )

            # Add basic transforms
            transform_list.extend(
                [
                    transforms.Resize(
                        image_size, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    lambda image: image.clamp(0, 1),
                    transforms.RandomCrop(image_size)
                    if use_augmentation and is_training
                    else transforms.CenterCrop(image_size),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            return transforms.Compose(transform_list)

    def _create_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        shuffle=False,
        keys_to_keep: Tuple[str] = tuple(),
        transforms: Sequence[Any] = tuple(),
        n_datapoints: int = None,
    ):
        """Create a WebDataset using pipeline approach."""
        # Use keys_to_keep from config if not provided
        if not keys_to_keep:
            keys_to_keep = self.keys_to_keep

        # Get transform objects
        if not transforms:
            transforms = self._get_transforms(is_training=shuffle)

        # Create transform pipeline
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        if use_dual_normalization:
            # Handle dual normalization with custom processing
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    )
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map_dict(image=transforms),
                wds.map(lambda x: self._process_dual_normalization(x)),
                wds.to_tuple(
                    "image", "dinov2_image", "cls", "__key__"
                ),  # Include both image versions
            ]
        else:
            # Original single normalization pipeline
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    )
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map_dict(image=transforms),
                wds.to_tuple("image", "cls", "__key__"),  # Convert dict to tuple format
            ]

        # Create main pipeline
        pipeline = [
            wds.ResampledShards(uri_expression)
            if shuffle
            else wds.SimpleShardList(uri_expression),
            wds.map(lambda x: x) if shuffle else wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=self.shuffle_buffer_size, initial=self.shuffle_initial)
            if shuffle
            else wds.map(lambda x: x),
            *transform_pipeline,
            wds.batched(self.batch_size, partial=not shuffle),
        ]

        if shuffle:
            return wds.DataPipeline(*pipeline)
        else:
            return wds.DataPipeline(*pipeline)

    def train_dataloader(self):
        """Create training dataloader using WebDataset pipeline."""
        dataset = self._create_webdataset(
            uri_expression=self.train_urls,
            shuffle=True,
            n_datapoints=self.estimated_train_size,
        )

        # Calculate number of batches for epoch management
        num_batches = (
            self.estimated_train_size // self.batch_size
            if self.estimated_train_size
            else 1000
        )

        # Create WebLoader with proper epoch handling for DDP
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,  # Shuffling handled in pipeline
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # Set epoch length for proper training loop management
        loader = loader.with_epoch(num_batches)

        return loader

    def val_dataloader(self):
        """Create validation dataloader using WebDataset pipeline."""
        dataset = self._create_webdataset(
            uri_expression=self.val_urls,
            shuffle=False,
            n_datapoints=self.estimated_val_size,
        )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader

    def test_dataloader(self):
        """Create test dataloader using WebDataset pipeline."""
        dataset = self._create_webdataset(
            uri_expression=self.test_urls,
            shuffle=False,
            n_datapoints=self.estimated_test_size,
        )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader


class JSONWebDatasetDataModule(WebDatasetDataModule):
    """
    Lightning DataModule for tokenizer training using pre-encoded JSON indices for training
    and standard WebDataset for validation to ensure dataset diversity.

    This module optimizes training by loading pre-encoded base tokenizer indices from JSON
    files with base64-encoded arrays while still loading images for the compact tokenizer encoder.
    Validation uses the standard WebDataset pipeline for proper validation across diverse datasets.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config)

        # JSON configuration for training data
        self.use_pre_encoded_indices = config.dataset.get(
            "use_pre_encoded_indices", False
        )
        self.json_train_path = config.dataset.get("json_train_path", None)
        self.base_tokenizer_name = config.dataset.get(
            "base_tokenizer_name", "open-magvit-v2"
        )
        self.json_data = None  # Cache for JSON data

        if self.use_pre_encoded_indices and not self.json_train_path:
            raise ValueError(
                "json_train_path must be specified when use_pre_encoded_indices is True"
            )

    def _load_json_data(self, json_path: str):
        """Load JSON data into memory for fast access."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            print(f"Loaded JSON data from {json_path}")
            print(f"Available splits: {[k for k in data.keys() if k != 'metadata']}")
            if "metadata" in data:
                print(
                    f"Total samples: {data['metadata'].get('total_samples', 'unknown')}"
                )
            return data
        except Exception as e:
            print(f"Error loading JSON data from {json_path}: {e}")
            return None

    def _decode_base64_indices(
        self, base64_string: str, shape: list, dtype: str = "int32"
    ) -> torch.Tensor:
        """
        Decode base64 string back to torch tensor.

        Args:
            base64_string: Base64-encoded binary string
            shape: Original shape of the array
            dtype: NumPy data type

        Returns:
            Torch tensor with original shape
        """
        try:
            # Decode base64 to bytes
            array_bytes = base64.b64decode(base64_string.encode("ascii"))

            # Convert bytes to numpy array and reshape
            np_array = np.frombuffer(array_bytes, dtype=getattr(np, dtype))
            np_array = np_array.reshape(shape)

            # Convert to torch tensor
            return torch.from_numpy(np_array).long()

        except Exception as e:
            print(f"Error decoding base64 indices: {e}")
            # Return dummy tensor as fallback
            return torch.zeros(shape, dtype=torch.long)

    def _process_json_sample(self, sample, json_data, split="train"):
        """Process sample to add pre-encoded indices from JSON data."""
        key = sample.get("__key__", "")

        if json_data is not None and split in json_data:
            if key in json_data[split]:
                try:
                    # Get the stored data for this key
                    stored_data = json_data[split][key]

                    # Decode base64 indices
                    indices = self._decode_base64_indices(
                        stored_data["indices"],
                        stored_data["shape"],
                        stored_data.get("dtype", "int32"),
                    )

                    sample["indices"] = indices

                    # Update class info if available from JSON
                    if stored_data.get("class_id", -1) != -1:
                        sample["cls"] = stored_data["class_id"]

                except Exception as e:
                    print(f"Error processing JSON entry for key {key}: {e}")
                    sample["indices"] = torch.zeros(
                        (16, 16), dtype=torch.long
                    )  # Dummy fallback
            else:
                print(
                    f"Warning: No pre-encoded indices found for key {key} in {split} split"
                )
                sample["indices"] = torch.zeros(
                    (16, 16), dtype=torch.long
                )  # Dummy fallback
        else:
            print(f"Warning: JSON data not available for split {split}")
            sample["indices"] = torch.zeros(
                (16, 16), dtype=torch.long
            )  # Dummy fallback

        return sample

    def _create_json_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        json_data: dict,
        split: str = "train",
        shuffle=False,
        keys_to_keep: Tuple[str] = tuple(),
        transforms: Sequence[Any] = tuple(),
    ):
        """Create a WebDataset pipeline that includes pre-encoded indices from JSON."""
        # Use keys_to_keep from config if not provided, add 'indices' for JSON mode
        if not keys_to_keep:
            keys_to_keep = (*self.keys_to_keep, "indices")
        elif "indices" not in keys_to_keep:
            keys_to_keep = (*keys_to_keep, "indices")

        # Get transform objects
        if not transforms:
            transforms = self._get_transforms(is_training=shuffle)

        # Create transform pipeline - same as parent but with JSON processing
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        if use_dual_normalization:
            # Handle dual normalization with JSON indices
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    )
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map(
                    lambda x: self._process_json_sample(x, json_data, split)
                ),  # Add JSON indices
                wds.map_dict(image=transforms),
                wds.map(lambda x: self._process_dual_normalization(x)),
                wds.to_tuple(
                    "image", "dinov2_image", "indices", "cls", "__key__"
                ),  # Include indices
            ]
        else:
            # Single normalization with JSON indices
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    )
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map(
                    lambda x: self._process_json_sample(x, json_data, split)
                ),  # Add JSON indices
                wds.map_dict(image=transforms),
                wds.to_tuple("image", "indices", "cls", "__key__"),  # Include indices
            ]

        # Create main pipeline - same as parent
        pipeline = [
            wds.ResampledShards(uri_expression)
            if shuffle
            else wds.SimpleShardList(uri_expression),
            wds.map(lambda x: x) if shuffle else wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=self.shuffle_buffer_size, initial=self.shuffle_initial)
            if shuffle
            else wds.map(lambda x: x),
            *transform_pipeline,
            wds.batched(self.batch_size, partial=not shuffle),
        ]

        if shuffle:
            return wds.DataPipeline(*pipeline)
        else:
            return wds.DataPipeline(*pipeline)

    def setup(self, stage=None):
        """Setup method to load JSON data once during initialization."""
        if self.use_pre_encoded_indices and self.json_data is None:
            self.json_data = self._load_json_data(self.json_train_path)

    def train_dataloader(self):
        """Create training dataloader using JSON indices + WebDataset images."""
        # Ensure JSON data is loaded
        if self.use_pre_encoded_indices and self.json_data is None:
            self.setup()

        if self.use_pre_encoded_indices and self.json_data is not None:
            # Use JSON + WebDataset for training
            dataset = self._create_json_webdataset(
                uri_expression=self.train_urls,
                json_data=self.json_data,
                split="train",
                shuffle=True,
            )
        else:
            # Fallback to standard WebDataset
            dataset = self._create_webdataset(
                uri_expression=self.train_urls,
                shuffle=True,
                n_datapoints=self.estimated_train_size,
            )

        # Calculate number of batches for epoch management
        num_batches = (
            self.estimated_train_size // self.batch_size
            if self.estimated_train_size
            else 1000
        )

        # Create WebLoader with proper epoch handling for DDP
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,  # Shuffling handled in pipeline
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # Set epoch length for proper training loop management
        loader = loader.with_epoch(num_batches)

        return loader

    def val_dataloader(self):
        """Create validation dataloader using standard WebDataset (no JSON) for dataset diversity."""
        # Always use standard WebDataset for validation to ensure dataset diversity
        dataset = self._create_webdataset(
            uri_expression=self.val_urls,
            shuffle=False,
            n_datapoints=self.estimated_val_size,
        )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader

    def test_dataloader(self):
        """Create test dataloader using standard WebDataset (no JSON)."""
        # Always use standard WebDataset for testing
        dataset = self._create_webdataset(
            uri_expression=self.test_urls,
            shuffle=False,
            n_datapoints=self.estimated_test_size,
        )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader


class FramePairsNormalizationTransform:
    """
    Transform class that applies dual normalization to robotics frame pairs:
    - frame_0: dual normalization (tokenizer + DINOv2)
    - frame_1: single normalization (tokenizer only)

    Returns a dictionary with normalized frame pairs.
    """

    def __init__(
        self,
        image_size: int,
        dinov2_image_size: int,
        use_augmentation: bool = False,
        tokenizer_mean: List[float] = [0.5, 0.5, 0.5],
        tokenizer_std: List[float] = [0.5, 0.5, 0.5],
        dinov2_mean: List[float] = [0.485, 0.456, 0.406],
        dinov2_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.image_size = image_size
        self.dinov2_image_size = dinov2_image_size
        self.use_augmentation = use_augmentation

        ## Determine the larger size to avoid upsampling
        max_size = max(image_size, dinov2_image_size)

        # Common transforms applied once (including random augmentations)
        # Lets not use random horizontal flip here, since it will results to a unrealistic dynamics between the two frames
        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    max_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                lambda image: image.clamp(0, 1),
                transforms.RandomCrop(max_size)
                if use_augmentation
                else transforms.CenterCrop(max_size),
                # transforms.RandomHorizontalFlip(0.5) if use_augmentation else lambda image: image
            ]
        )

        self.common_transforms_with_flip = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    max_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                lambda image: image.clamp(0, 1),
                transforms.RandomCrop(max_size)
                if use_augmentation
                else transforms.CenterCrop(max_size),
                transforms.RandomHorizontalFlip(1.0),
            ]
        )

        # Tokenizer-specific resize (if needed) and normalization
        self.tokenizer_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
                if image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=tokenizer_mean, std=tokenizer_std),
            ]
        )

        # DINOv2-specific resize (if needed) and normalization
        self.dinov2_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    dinov2_image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
                if dinov2_image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=dinov2_mean, std=dinov2_std),
            ]
        )

    def __call__(self, frame_pair):
        """Apply transforms to frame pair."""
        frame_0, frame_1 = frame_pair
        common_transform_fn = (
            self.common_transforms_with_flip
            if random.random() < 0.5 and self.use_augmentation
            else self.common_transforms
        )
        frame_0 = common_transform_fn(frame_0)
        frame_1 = common_transform_fn(frame_1)

        # Apply dual normalization to frame_0
        frame_0_tokenizer = self.tokenizer_post_transform(frame_0.clone())
        frame_0_dinov2 = self.dinov2_post_transform(frame_0.clone())

        # Apply dual normalization to frame_1
        frame_1_tokenizer = self.tokenizer_post_transform(frame_1.clone())
        frame_1_dinov2 = self.dinov2_post_transform(frame_1.clone())

        return {
            "frame_0_tokenizer": frame_0_tokenizer,
            "frame_0_dinov2": frame_0_dinov2,
            "frame_1_tokenizer": frame_1_tokenizer,
            "frame_1_dinov2": frame_1_dinov2,
        }


class FramePairsWebDatasetDataModule(WebDatasetDataModule):
    """
    Lightning DataModule for frame pair training using WebDataset.
    Handles both:
    1. Frame pairs (0.webp, 1.webp) with metadata (json) from frame pair datasets
    2. Single images (jpg/png/webp) which are duplicated to create frame pairs

    - frame_0: applies dual normalization (tokenizer + DINOv2)
    - frame_1: applies dual normalization (tokenizer + DINOv2)
    - Minimal augmentation since images are pre-processed to 224x224
    - For single image datasets, the same image is used for both frames
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config)

        # Keys for frame pairs webdataset: frame_0, frame_1, metadata
        self.train_keys_to_keep = ("frame_0", "frame_1", "metadata", "__key__")
        self.keys_to_keep = ("image", "cls", "__key__")

        # Configuration for validation dataset format
        self.val_has_frame_pairs = config.dataset.get("val_has_frame_pairs", True)

    def _rename_frame_pair_extensions(self, d: dict) -> dict:
        """Rename frame pair webdataset extensions to meaningful keys.
        Handles both frame pairs (0, 1) and single images."""
        new_d = {}

        # Check if this is a frame pair dataset or single image dataset
        has_frame_pair = "0" in d and "1" in d
        has_single_image = any(
            k.lower() in ["jpg", "jpeg", "png", "webp"] for k in d.keys()
        )

        for k, v in d.items():
            if k == "0":
                new_d["frame_0"] = v
            elif k == "1":
                new_d["frame_1"] = v
            elif k == "json":
                new_d["metadata"] = v
            elif k.lower() in ["jpg", "jpeg", "png", "webp"] and not has_frame_pair:
                # Single image dataset - duplicate for both frames
                new_d["frame_0"] = v
                new_d["frame_1"] = v
            else:
                new_d[k] = v
        return new_d

    def _process_frame_pair_metadata(self, sample):
        """Process and parse JSON metadata."""
        if "metadata" in sample and isinstance(sample["metadata"], bytes):
            try:
                sample["metadata"] = json.loads(sample["metadata"].decode("utf-8"))
            except Exception as e:
                key = sample.get("__key__", "unknown")
                logger.warning(f"Failed to parse metadata for {key}: {e}")
                sample["metadata"] = {}
        elif "metadata" not in sample:
            # No metadata present (e.g., single image datasets) - create empty metadata
            sample["metadata"] = {}
        return sample

    def _get_frame_pair_transforms(self, is_training: bool = False):
        """Get transform objects for frame pairs."""
        image_size = self.dataset_config.image_size
        use_augmentation = self.dataset_config.get("use_augmentation", False)
        mean = self.dataset_config.get("mean", [0.5, 0.5, 0.5])
        std = self.dataset_config.get("std", [0.5, 0.5, 0.5])

        # Check if dual normalization is enabled
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        if use_dual_normalization:
            dinov2_image_size = self.dataset_config.get("dinov2_image_size", 224)
            dinov2_mean = self.dataset_config.get("dinov2_mean", [0.485, 0.456, 0.406])
            dinov2_std = self.dataset_config.get("dinov2_std", [0.229, 0.224, 0.225])

            return FramePairsNormalizationTransform(
                image_size=image_size,
                dinov2_image_size=dinov2_image_size,
                use_augmentation=use_augmentation and is_training,
                tokenizer_mean=mean,
                tokenizer_std=std,
                dinov2_mean=dinov2_mean,
                dinov2_std=dinov2_std,
            )
        else:
            # Single normalization for both frames
            transform_list = [
                transforms.ToTensor(),
                lambda image: image.clamp(0, 1),
            ]

            # Add augmentations for training
            if use_augmentation and is_training:
                transform_list.append(transforms.RandomHorizontalFlip(0.5))

            # Add normalization
            transform_list.append(transforms.Normalize(mean=mean, std=std))

            return transforms.Compose(transform_list)

    def _process_frame_pair_dual_normalization(self, sample):
        """Process dual normalization output from FramePairsNormalizationTransform."""
        if isinstance(sample.get("frames"), dict):
            frames_dict = sample["frames"]
            # Extract the different normalized versions
            sample["frame_0_tokenizer"] = frames_dict["frame_0_tokenizer"]
            sample["frame_0_dinov2"] = frames_dict["frame_0_dinov2"]
            sample["frame_1_tokenizer"] = frames_dict["frame_1_tokenizer"]
            sample["frame_1_dinov2"] = frames_dict["frame_1_dinov2"]
            # Remove the original frames dict
            del sample["frames"]
        return sample

    def _create_frame_pair_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        shuffle=False,
        keys_to_keep: Tuple[str] = tuple(),
        transforms: Sequence[Any] = tuple(),
        n_datapoints: int = None,
    ):
        """Create a WebDataset using pipeline approach for frame pairs."""
        # Use keys_to_keep from config if not provided
        if not keys_to_keep:
            keys_to_keep = self.train_keys_to_keep

        # Get transform objects
        if not transforms:
            transforms = self._get_frame_pair_transforms(is_training=shuffle)

        # Create transform pipeline for frame pair data
        use_dual_normalization = self.dataset_config.get(
            "use_dual_normalization", False
        )

        def decoding_error_handler(exn):
            if isinstance(exn, wds.autodecode.DecodingError):
                logger.error(
                    f"Decoding error: {exn}, url: {exn.url}, key: {exn.key}, k: {exn.k}"
                )
                logger.error("Skipping sample due to decoding error")
                return True  # Skip this sample and continue
            return False  # Re-raise other exceptions

        if use_dual_normalization:
            # Handle dual normalization with custom processing
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    ),
                    handler=decoding_error_handler,
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_frame_pair_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map(lambda x: self._process_frame_pair_metadata(x)),
                wds.map(
                    lambda x: {**x, "frames": (x["frame_0"], x["frame_1"])}
                ),  # Create frame pair
                wds.map_dict(frames=transforms),
                wds.map(lambda x: self._process_frame_pair_dual_normalization(x)),
                wds.to_tuple(
                    "frame_0_tokenizer",
                    "frame_0_dinov2",
                    "frame_1_tokenizer",
                    "frame_1_dinov2",
                    "metadata",
                    "__key__",
                ),
            ]
        else:
            # Single normalization pipeline
            transform_pipeline = [
                wds.decode(
                    wds.autodecode.ImageHandler(
                        "rgb8", extensions=["webp", "png", "jpg", "jpeg"]
                    ),
                    handler=decoding_error_handler,
                ),
                wds.map(lambda x: self._truncate_extensions(x)),
                wds.map(lambda x: self._rename_frame_pair_extensions(x)),
                wds.map(lambda x: self._filter_keys(x, keys_to_keep)),
                wds.map(lambda x: self._process_frame_pair_metadata(x)),
                wds.map_dict(frame_0=transforms, frame_1=transforms),
                wds.to_tuple("frame_0", "frame_1", "metadata", "__key__"),
            ]

        # Create main pipeline
        pipeline = [
            wds.ResampledShards(uri_expression)
            if shuffle
            else wds.SimpleShardList(uri_expression),
            wds.map(lambda x: x) if shuffle else wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=self.shuffle_buffer_size, initial=self.shuffle_initial)
            if shuffle
            else wds.map(lambda x: x),
            *transform_pipeline,
            wds.batched(self.batch_size, partial=not shuffle),
        ]

        return wds.DataPipeline(*pipeline)

    def train_dataloader(self):
        """Create training dataloader using frame pairs WebDataset pipeline."""
        dataset = self._create_frame_pair_webdataset(
            uri_expression=self.train_urls,
            shuffle=True,
            n_datapoints=self.estimated_train_size,
        )

        # Calculate number of batches for epoch management
        num_batches = (
            self.estimated_train_size // self.batch_size
            if self.estimated_train_size
            else 1000
        )

        # Create WebLoader with proper epoch handling for DDP
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,  # Shuffling handled in pipeline
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        # Set epoch length for proper training loop management
        loader = loader.with_epoch(num_batches)

        return loader

    def val_dataloader(self):
        """Create validation dataloader using frame pairs or single image WebDataset pipeline."""
        if self.val_has_frame_pairs:
            # Use frame pairs for validation (enables bisim distance metric)
            dataset = self._create_frame_pair_webdataset(
                uri_expression=self.val_urls,
                shuffle=False,
                n_datapoints=self.estimated_val_size,
            )
        else:
            # Fallback to single images for validation (bisim distance metric will be skipped)
            dataset = self._create_webdataset(
                uri_expression=self.val_urls,
                shuffle=False,
                n_datapoints=self.estimated_val_size,
            )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader

    def test_dataloader(self):
        """Create test dataloader using frame pairs WebDataset pipeline."""
        dataset = self._create_webdataset(
            uri_expression=self.test_urls,
            shuffle=False,
            n_datapoints=self.estimated_test_size,
        )

        # Create WebLoader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Already batched in pipeline
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return loader
