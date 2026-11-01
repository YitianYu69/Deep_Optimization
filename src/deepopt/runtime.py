"""Device selection and reproducibility settings."""

import os
import random

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; set runtime.device = 'cpu'")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable")
        device = torch.device("cuda", index)
    return device


def seed_everything(seed: int, deterministic: bool) -> None:
    # Set before the first CUDA matrix multiplication.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
