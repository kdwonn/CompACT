# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
import torch
import torch.distributed as dist
from torcheval.metrics import FrechetInceptionDistance

from collections import defaultdict, deque
import os
import datetime
import builtins
from logging import getLogger
import pickle
import time

logger = getLogger()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    dist_url = "env://"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(port))
    print("Using port", os.environ["MASTER_PORT"])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            logger.info("torchrun env vars not sets")

    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            gpu = rank % torch.cuda.device_count()
            if "HOSTNAME" in os.environ:
                os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
            else:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
        except Exception:
            logger.info("SLURM vars not set")

    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, init_method=dist_url
    )

    # setup_for_distributed(rank == 0)
    return world_size, rank, gpu, True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )
        self.update(total_time=total_time)


def sync_fid_loss_fns(fid_loss_fn, device="cuda"):
    """
    Synchronizes FID loss function metrics across all processes.

    Args:
        fid_loss_fn (dict): Local FID loss function metrics on each process.
        device (str): Device to move the merged FID metrics to.

    Returns:
        final_fid_loss_fn (dict): Merged FID loss function metrics on all processes.
    """
    if not is_dist_avail_and_initialized():
        return fid_loss_fn

    serialized_fid_loss_fn = pickle.dumps(fid_loss_fn)
    gathered_fid_loss_fn = [None] * dist.get_world_size()

    dist.barrier()

    dist.all_gather_object(gathered_fid_loss_fn, serialized_fid_loss_fn)

    final_fid_loss_fn = {
        1: FrechetInceptionDistance(feature_dim=2048).to(device),
        2: FrechetInceptionDistance(feature_dim=2048).to(device),
        4: FrechetInceptionDistance(feature_dim=2048).to(device),
        8: FrechetInceptionDistance(feature_dim=2048).to(device),
        16: FrechetInceptionDistance(feature_dim=2048).to(device),
    }

    for serialized_fid_loss_fn in gathered_fid_loss_fn:
        curr_fid_loss_fn = pickle.loads(serialized_fid_loss_fn)
        for sec in [1, 2, 4, 8, 16]:
            sec_fid_loss_fn = curr_fid_loss_fn[sec]
            final_fid_loss_fn[sec].merge_state([sec_fid_loss_fn])

    return final_fid_loss_fn
