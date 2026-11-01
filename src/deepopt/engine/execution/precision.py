"""Optional bf16 autocast, independent of epoch orchestration."""

from contextlib import nullcontext

import torch


class PrecisionPolicy:
    def __init__(self, device: torch.device, precision: str):
        if device.type not in ("cpu", "cuda"):
            raise ValueError("The training core supports cpu and cuda devices")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be fp32 or bf16; fp16 is unsupported")
        if device.type == "cpu" and precision != "fp32":
            raise ValueError("CPU training currently requires fp32")
        if device.type == "cuda" and precision == "bf16":
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("Selected CUDA device does not support bf16")
        self.device, self.precision = device, precision

    def autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

