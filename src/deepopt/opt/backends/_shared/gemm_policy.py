"""Explicit Linear dW experiments, captured by the forward autograd owner."""

from contextlib import contextmanager
from contextvars import ContextVar


_linear = ContextVar("deepopt_linear_backward_policy", default=("inherit", "preserve"))
_gradient_reduction = ContextVar("deepopt_linear_gradient_reduction", default="separate")
_forward_backend = ContextVar("deepopt_linear_forward_backend", default="inherit")


def linear_forward_backend():
    return _forward_backend.get()


def gradient_reduction_policy():
    return _gradient_reduction.get()


def gradient_statistics_geometry(rows, cols, budget=8*2**20):
    """Bound every reduction plane, independently of the dense FP8 outputs."""
    chunk=256
    while chunk <= 4096:
        parts=(rows+chunk-1)//chunk
        count=parts
        planes=parts
        while count>1:
            count=(count+127)//128
            planes+=count
        amax=parts*((cols+63)//64)
        count=amax
        amax_planes=0
        while count>1024:
            count=(count+1023)//1024
            amax_planes+=count
        size=4*(planes*cols+2*amax+amax_planes+2)
        if size<=budget:
            return chunk,parts,size
        chunk*=2
    return None


def linear_backward_policy():
    return _linear.get()


@contextmanager
def linear_backward_scope(*, dw_backend="inherit", native_layout="preserve", gradient_reduction="separate", forward_backend="inherit"):
    if dw_backend not in ("inherit", "native_precise") or native_layout not in ("preserve", "prepare"):
        raise ValueError("dw_backend must be inherit/native_precise and native_layout preserve/prepare")
    token = _linear.set((dw_backend, native_layout))
    if gradient_reduction not in ("separate","shared"):
        _linear.reset(token)
        raise ValueError("gradient_reduction must be separate or shared")
    reduction_token=_gradient_reduction.set(gradient_reduction)
    if forward_backend not in ("inherit","native_fast","native_precise"):
        _gradient_reduction.reset(reduction_token)
        _linear.reset(token)
        raise ValueError("forward_backend must be inherit, native_fast or native_precise")
    forward_token=_forward_backend.set(forward_backend)
    try:
        yield
    finally:
        _forward_backend.reset(forward_token)
        _gradient_reduction.reset(reduction_token)
        _linear.reset(token)


def prepare_native_operands(a, b):
    from deepopt.opt.backends.native.fp8 import unsupported_reason, row_major_eligible
    if a.data.device.type != "cuda" or unsupported_reason(a, b) != "native GEMM requires aligned row-major operand storage":
        return a, b, None
    from deepopt.opt.quantization.tensor import QuantizedTensor
    from torch._subclasses.fake_tensor import FakeTensor
    original = {"a_strides": tuple(a.data.stride()), "b_strides": tuple(b.data.stride())}

    def prepared(q):
        if row_major_eligible(q.data):
            return q
        if isinstance(q.data, FakeTensor):
            import torch
            data = torch.empty(tuple(q.data.shape),device=q.data.device,dtype=q.data.dtype)
        else:
            from deepopt.opt.backends.dispatch import backend_module
            _pack_fp8_rows = backend_module("quantization", "gemm.fp8")._pack_fp8_rows
            data = _pack_fp8_rows(q.data, 16)
        return QuantizedTensor(data, q.scale_inv, q.block_shape)

    pa, pb = prepared(a), prepared(b)
    return pa, pb, {"kind": "row_major_copy", "copied_a": pa is not a, "copied_b": pb is not b,
                    "copied_bytes": sum(q.data.numel() * q.data.element_size()
                                        for q, p in ((a,pa),(b,pb)) if q is not p), **original}
