"""
FP8 linear with explicit forward and backward computations.
"""

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from ..backends import gemm_nt
from ..quantization.quantizer import quantize_fp8
from ..quantization.tensor import QuantizedTensor
from ..scaling_strategy.amax import AmaxScaling





class DOLinear(nn.Module):
    def __init__(self, original: nn.Linear, *, plan: dict):
        super().__init__()

        if not isinstance(original, nn.Linear):
            raise ValueError("original must be an nn.Linear")
        
        fp8_trainig_plan = plan["fp8_training"]
        fp8_dtype = fp8_trainig_plan["fp8_dtype"]
        block_shape = plan["block_shape"]
        backend = fp8_trainig_plan.get("backend", "reference")
        num_stages = fp8_trainig_plan.get("num_stages", 3)

        if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Unsupported FP8 dtype")
        if backend not in ("reference", "triton"):
            raise ValueError("backend must be 'reference' or 'triton'")
        if type(num_stages) is not int or num_stages not in (1, 2, 3, 4):
            raise ValueError("num_stages must be 1, 2, 3, or 4")
        if backend == "triton" and block_shape is not None:
            if any(size % 32 != 0 for size in block_shape):
                raise ValueError(
                    "Triton DOLinear training currently requires both "
                    "block dimensions to be multiples of 32, because "
                    "backward transposes the quantized operands"
                )
        
        # Validate
        AmaxScaling(block_shape=block_shape)

        self.in_features = original.in_features
        self.out_features = original.out_features

        self.weight = original.weight
        self.bias = original.bias

        self.fp8_dtype = fp8_dtype
        self.block_shape = block_shape
        self.plan = plan
        
        self.train(original.training)

    @classmethod
    def from_linear(cls, original: nn.Linear, **kwargs) -> "DOLinear":
        return cls(original, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input must have last dimension {self.in_features}"
            )
        if x.numel() == 0:
            raise ValueError("DOLinear currenly requires non-empty input")
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Input must be FP32, BF16, or FP16")
        if x.device != self.weight.device:
            raise ValueError("Input and weight must be on the same device")
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("DOLinear currently supports CPU and CUDA")
        
        out_dtype = x.dtype
        if torch.is_autocast_enabled(x.device.type):
            out_dtype = torch.get_autocast_dtype(x.device.type)

        # An explicit plan setting overrides that default
        out_dtype = self.plan["fp8_training"].get("out_dtype", out_dtype)
        
        return _DOLinearFunction.apply(
            x, self.weight, self.bias, self.plan, out_dtype
        )





class _DOLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        plan: dict,
        out_dtype: torch.dtype
    ):
        fp8_trainig_plan = plan["fp8_training"]
        fp8_dtype = fp8_trainig_plan["fp8_dtype"]
        block_shape = plan["block_shape"]
        backend = fp8_trainig_plan.get("backend", "reference")
        num_stages = fp8_trainig_plan.get("num_stages", 3)

        x_matrix = x.reshape(-1, weight.shape[1])

        qx = quantize_fp8(
            x_matrix, fp8_dtype=fp8_dtype,
            block_shape=block_shape
        )
        qw = quantize_fp8(
            weight, fp8_dtype=fp8_dtype,
            block_shape=block_shape
        )

        output = gemm_nt(qx, qw, backend=backend, bias=bias, out_dtype=out_dtype, num_stages=num_stages)
        need_dx, need_dw = ctx.needs_input_grad[:2]

        ctx.save_for_backward(
            qx.data if need_dw else None,
            qx.scale_inv if need_dw else None,
            qw.data if need_dx else None,
            qw.scale_inv if need_dx else None
        )

        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.out_features = weight.shape[0]
        ctx.fp8_dtype = fp8_dtype
        ctx.block_shape = block_shape
        ctx.backend = backend
        ctx.num_stages = num_stages

        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        x_data, x_scale_inv, w_data, w_scale_inv = ctx.saved_tensors

        grad_matrix = grad_output.reshape(-1, ctx.out_features)

        dx = None
        dw = None
        db = None
        if need_dx or need_dw:
            qg = quantize_fp8(
                grad_matrix, fp8_dtype=ctx.fp8_dtype, block_shape=ctx.block_shape
            )

            if need_dx:
                qw = QuantizedTensor(
                    data=w_data, scale_inv=w_scale_inv, block_shape=ctx.block_shape
                )

                dx = gemm_nt(
                    qg, qw.transpose(),
                    backend=ctx.backend,
                    out_dtype=ctx.input_dtype,
                    num_stages=ctx.num_stages
                ).reshape(ctx.input_shape)
            
            if need_dw:
                qx = QuantizedTensor(
                    data=x_data, scale_inv=x_scale_inv, block_shape=ctx.block_shape
                )

                dw = gemm_nt(
                    qg.transpose(),
                    qx.transpose(),
                    backend=ctx.backend,
                    out_dtype=ctx.weight_dtype,
                    num_stages=ctx.num_stages
                )
        
        if need_db:
            db = grad_matrix.sum(dim=0, dtype=torch.float32).to(dtype=ctx.bias_dtype)
        
        return dx, dw, db, None, None