import sys
import subprocess

from os import environ as env

import numpy as np

__all__ = ["get_gpu_count", "get_freer_gpu", "get_total_gpu_memory"]


def get_gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def get_freer_gpu(strict: bool = True) -> int:
    import torch

    if sys.platform == "win32":
        return 0

    gpu_count = get_gpu_count()
    if gpu_count == 0:
        raise RuntimeError("GPU device not found!")

    memory_used = []
    for idx in range(gpu_count):
        mem = torch.cuda.mem_get_info(device=f"cuda:{idx}")
        memory_used.append((mem[1] - mem[0]) / 1024**2)

    free_gpu = int(np.argmin(memory_used))

    if strict and memory_used[free_gpu] > 100:
        raise RuntimeError("All GPUs are busy!")

    return free_gpu


def get_total_gpu_memory(gpu_index: int) -> float:  # in GB
    import torch

    return torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3


if __name__ == "__main__":
    import torch

    data = []
    for strict in [True, False]:
        for i in range(get_gpu_count()):
            gpu_idx = get_freer_gpu(strict=strict)
            device = f"cuda:{gpu_idx}"
            data.append(torch.FloatTensor([1]).to(device))
            print(f"[strict={strict}] change device {device}")
