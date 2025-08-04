import math
from inspect import isfunction
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


class DisableDistributed:
    """Context manager that temporarily disables distributed functions (replaces with no-ops)"""

    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def spawn(func: Callable, world_size: int, *args, **kwargs):
    args = (world_size,) + args + tuple(kwargs.values())
    mp.spawn(func, args=args, nprocs=world_size, join=True)


def int_divide(a, b):
    assert a % b == 0
    return a // b


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)
    return nn.Parameter(
        torch.randn(num_inputs, num_outputs, device=get_device(rank))
        / math.sqrt(num_inputs)
    )


def summarize_tensor(tensor: torch.Tensor) -> str:
    return (
        "x".join(map(str, tensor.shape))
        + "["
        + str(round(tensor.view(-1)[0].item(), 4))
        + "...]"
    )
