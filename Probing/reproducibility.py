"""Small reproducibility helpers shared by probing tasks.

These helpers keep the existing experiment flow intact while making all
task-local randomness explicit. We avoid forcing deterministic algorithms by
default because some CUDA ops can error when PyTorch cannot provide a
deterministic implementation.
"""

from __future__ import annotations

import inspect
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    ``PYTHONHASHSEED`` is most effective when exported before Python starts;
    setting it here still records the intended seed for child processes.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed Python/NumPy RNGs inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded independently of global torch RNG."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def dataloader_seed_kwargs(seed: int, stream: int = 0) -> dict:
    """Return DataLoader kwargs for deterministic shuffle/worker seeding.

    ``generator`` is included only when supported by the installed PyTorch,
    which keeps these scripts compatible with older cluster environments.
    """
    kwargs = {"worker_init_fn": seed_worker}
    if "generator" in inspect.signature(DataLoader).parameters:
        kwargs["generator"] = make_generator(seed + stream)
    return kwargs
