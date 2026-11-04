"""Combined AMP and operation-level FP8 policy, independent of epochs."""

from contextlib import nullcontext

import torch


class PrecisionPolicy:
    def __init__(self, device: torch.device, precision: str, *, fp8=None):
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
        self.fp8 = fp8

    def autocast(self):
        if self.precision == "fp32" and self.fp8 is None:
            return nullcontext()
        from deepopt.opt._autocast import autocast

        return autocast(self.device.type, dtype=torch.bfloat16,
                        enabled=self.precision == "bf16", fp8=self.fp8)

