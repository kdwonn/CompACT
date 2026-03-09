from collections import defaultdict
import os
from omegaconf import DictConfig
import psutil
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from datasets import TrainingDataset
from misc import get_transform
from tabulate import tabulate
import time


def prepare_datasets(config: DictConfig):
    """Create and prepare training and test datasets."""
    train_dataset = []
    test_dataset = []

    for dataset_name in config.dataset.datasets:
        data_config = config.dataset.datasets[dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                goals_per_obs = int(data_config["goals_per_obs"])
                if data_split_type == "test":
                    goals_per_obs = 4  # standardize testing

                min_dist_cat = data_config.get("distance", {}).get(
                    "min_dist_cat", config.dataset.distance.min_dist_cat
                )
                max_dist_cat = data_config.get("distance", {}).get(
                    "max_dist_cat", config.dataset.distance.max_dist_cat
                )
                len_traj_pred = data_config.get(
                    "len_traj_pred", config.dataset.len_traj_pred
                )

                dataset = TrainingDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config.dataset.image_size,
                    min_dist_cat=min_dist_cat,
                    max_dist_cat=max_dist_cat,
                    len_traj_pred=len_traj_pred,
                    context_size=config.dataset.context_size,
                    normalize=config.dataset.normalize,
                    goals_per_obs=goals_per_obs,
                    transform=get_transform(
                        config.dataset.image_size,
                        config.dataset.mean,
                        config.dataset.std,
                    ),
                    action_stats=config.dataset.action_stats,
                    waypoint_spacing=data_config.metric_waypoint_spacing,
                    predefined_index=None,
                    traj_stride=1,
                )

                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    test_dataset.append(dataset)

                print(
                    f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}"
                )

    # Combine all datasets from different robots
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset)

    return train_dataset, test_dataset


def create_dataloader(dataset, config: DictConfig, rank, is_train=True):
    """Create dataloader for training or testing."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=is_train,
        seed=config.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )

    return loader, sampler


def get_mem_info(pid: int) -> dict[str, int]:
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res["rss"] += mmap.rss
        res["pss"] += mmap.pss
        res["uss"] += mmap.private_clean + mmap.private_dirty
        res["shared"] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith("/"):
            res["shared_file"] += mmap.shared_clean + mmap.shared_dirty
    return res


def find_dataloader_workers(parent_pid: int = None) -> list[int]:
    """Find PIDs of DataLoader worker processes."""
    if parent_pid is None:
        parent_pid = os.getpid()

    worker_pids = []
    try:
        parent = psutil.Process(parent_pid)
        all_children = parent.children(recursive=True)
        print(f"Parent PID {parent_pid} has {len(all_children)} child processes:")

        for child in all_children:
            try:
                child_name = child.name()
                child_cmdline = " ".join(child.cmdline())
                print(
                    f"  Child PID {child.pid}: name='{child_name}', cmdline='{child_cmdline}'"
                )

                # More flexible detection for DataLoader workers
                if (
                    "python" in child_name.lower()
                    or "dataloader" in child_cmdline.lower()
                    or child.pid != parent_pid
                ):  # Any child process
                    worker_pids.append(child.pid)
                    print("    -> Added as worker")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    except psutil.NoSuchProcess:
        print(f"Parent process {parent_pid} not found")

    print(f"Found {len(worker_pids)} worker PIDs: {worker_pids}")
    return worker_pids


class MemoryMonitor:
    def __init__(self, pids: list[int] = None):
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    def analyze_high_memory_workers(self) -> str:
        """Analyze what's in high memory workers."""
        self._refresh()

        result = ["=== High Memory Worker Analysis ==="]
        high_mem_pids = []

        # Find high memory workers (USS > 5GB)
        for pid in self.pids:
            if pid in self.data and self.data[pid]["uss"] > 5 * 1024**3:
                high_mem_pids.append(pid)

        result.append(f"High memory workers: {high_mem_pids}")
        result.append("")

        # Analyze each high memory worker
        for pid in high_mem_pids:
            try:
                proc = psutil.Process(pid)
                result.append(f"--- Worker PID {pid} ---")
                result.append(f"Status: {proc.status()}")
                result.append(f"CPU %: {proc.cpu_percent():.1f}")
                result.append(f"Threads: {proc.num_threads()}")

                # Memory maps analysis
                maps = proc.memory_maps()
                heap_rss = sum(m.rss for m in maps if "[heap]" in m.path)
                anon_rss = sum(m.rss for m in maps if m.path == "")

                result.append(f"Heap memory: {self.format(heap_rss)}")
                result.append(f"Anonymous memory: {self.format(anon_rss)}")
                result.append(f"Total memory maps: {len(maps)}")
                result.append("")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                result.append(f"PID {pid}: Error - {e}")

        return "\n".join(result)

    @staticmethod
    def format(size: int) -> str:
        for unit in ("", "K", "M", "G"):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
