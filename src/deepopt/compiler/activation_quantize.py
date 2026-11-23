"""Reference/fake contracts for generated activation-to-FP8 operand preparation."""

import torch
from torch._subclasses.fake_tensor import FakeTensor


def preparation_backend(*, compression=None, policy=None):
    """Validate an atomic producer against explicitly pinned auxiliary routes."""
    from deepopt.kernel_policy import resolve_kernel_policy

    selected = resolve_kernel_policy(policy)
    backend = selected.route("compiler_preparation", "compiler.preparation")
    families = ("quantization", "scaling") + (("activation_codec",) if compression is not None else ())
    for family in families:
        if selected.pinned(family, "linear.prepare_input") and selected.route(family, "linear.prepare_input") != backend:
            raise ValueError(
                f"Conflicting Mixed routes: fused {backend} compiler preparation requires "
                f"{family}/linear.prepare_input={backend}; use an unfused schedule"
            )
    return backend


def prepare(x, *, activation, fp8_dtype, paired, compression):
    from deepopt.opt.quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair
    from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation

    if x.device.type == "cuda" and not isinstance(x, FakeTensor):
        from importlib import import_module
        backend = preparation_backend(compression=compression)
        implementation = import_module(
            f"deepopt.compiler.codegen.{backend}.activation_quantize"
        ).prepare

        return implementation(
            x, activation=activation, fp8_dtype=fp8_dtype,
            paired=paired, compression=compression,
        )
    kind, approximate = activation
    values = (
        torch.nn.functional.gelu(x, approximate=approximate)
        if kind == "gelu" else torch.nn.functional.silu(x)
    )
    if compression is not None:
        normal, packed = quantize_pack_fp8_activation(
            values, fp8_dtype=fp8_dtype, group_size=compression["group_size"],
            bits=compression["bits"],
        )
        return normal, None, packed
    if paired:
        normal, transposed = _quantize_tensorwise_pair(values, fp8_dtype=fp8_dtype)
        return normal, transposed, None
    return quantize_fp8(values, fp8_dtype=fp8_dtype), None, None
