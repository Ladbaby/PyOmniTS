"""various utils"""
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import random
import numpy as np


def delta_time(start, end, rank=None):
    """subtracts 2 datetime objects - make it readable: e.g. hours, minutes"""
    # if rank == 0 or rank == "cpu":
    t_ = (end - start).total_seconds() / 60
    if t_ > 60:
        t_ /= 60
        t2 = "hours"
    elif t_ < 1:
        t_ *= 60
        t2 = "secs"
    else:
        t2 = "mins"
    return "{:.2f} {}".format(t_, t2)


def setup(rank, world_size):
    """Setup distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    rank = rank if torch.cuda.is_available() else 0
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def run_demo(demo_fn, world_size):
    """Run distributed environment"""
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
