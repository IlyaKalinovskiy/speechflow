import sys
import subprocess

import numpy as np

__all__ = ["get_gpu_count", "get_freer_gpu", "get_total_gpu_memory"]


def get_gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def get_freer_gpu(strict: bool = True) -> int:
    if sys.platform == "win32":
        return 0

    arch = subprocess.check_output(
        "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used", shell=True
    )

    memory_used = [int(x.split()[2]) for x in arch.decode("utf-8").split("\n") if x]
    free_gpu = int(np.argmin(memory_used))

    if strict and memory_used[free_gpu] > 50:
        raise RuntimeError("All GPUs are busy!")

    return free_gpu


def get_total_gpu_memory(gpu_index: int) -> float:
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
