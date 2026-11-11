"""
Reference GEMM for quantized FP8 operands.
"""

import torch

from deepopt.opt.quantization.tensor import QuantizedTensor



@torch.no_grad()
def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None
):
    """
    Compute dequantize(a) @ dequantize(b).T + optional bias

    Shapes:
        a.data: [M, K]
        b.data: [N, K]
        bias: [N]
        output: [M, N]
    """

    if a.data.device != b.data.device:
        raise ValueError("GEMM operands must be on the same device")
    if a.data.shape[1] != b.data.shape[1]:
        raise ValueError("GEMM operands must have the same reduction dimension")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be FP32, BF16, or FP16")
    
    device_type = a.data.device.type
    if device_type not in ("cpu", "cuda"):
        raise ValueError("Reference GEMM only supported CPU and CUDA")
    
    if bias is not None:
        if bias.device != a.data.device:
            raise ValueError("Bias must be on the operands' device")

        if tuple(bias.shape) != (b.data.shape[0],):
            raise ValueError("Bias must have the shape [N]")
        
        if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Bias must be FP32, BF16, or FP16")
        
    with torch.autocast(device_type=device_type, enabled=False):
        a_fp32 = a.dequantize()
        b_fp32 = b.dequantize()

        output = torch.matmul(a_fp32, b_fp32.T)
        if bias is not None:
            output = output + bias.float()
        return output.to(out_dtype)

