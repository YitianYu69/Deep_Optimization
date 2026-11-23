"""BF16 matrix/codec execution shared by compiler launch programs.

The native implementation is a control, not an implicit fallback from a
requested generated kernel. Optional GPU compilers are loaded only on demand.
"""

from importlib import import_module

import torch
from torch.nn import functional as F

from ._shared.bf16_gemm import BF16GemmConfig
from ._shared.gemm_diagnostics import gemm_launch_is_recording, observe_bf16_gemm, record_gemm_launch


def activate(value, activation):
    if activation is None:
        return value
    kind, approximate = activation
    if kind == "gelu":
        return F.gelu(value, approximate=approximate)
    if kind == "silu":
        return F.silu(value)
    raise ValueError(f"Unsupported BF16 activation: {activation!r}")


def derivative(gradient, value, activation):
    if activation is None:
        return gradient
    kind, approximate = activation
    if kind == "gelu":
        return torch.ops.aten.gelu_backward.default(gradient, value, approximate=approximate)
    if kind == "silu":
        return torch.ops.aten.silu_backward.default(gradient, value)
    raise ValueError(f"Unsupported BF16 activation: {activation!r}")


@record_gemm_launch
def product(a, b, *, backend="native", bias=None, activation=None,
            derivative_input=None, derivative_activation=None, packed_b=None,
            tile=64, num_stages=3, return_preactivation=False, residual=None, derivative_packed=None,
            kernel_config=None, role=None, packed_route="direct", panel_size=1024, bf16_policy=None,
            operand_route=None):
    """Return BF16 ``a @ b.T``; optionally decode a transposed saved B tile.

    A packed B describes X[M,K], and is consumed as X.T[K,M] for dW.
    All multiplicands and bias round to BF16 before arithmetic. Epilogues
    round the product before applying activation/activation backward.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    from deepopt._bf16_policy import resolve_bf16_policy
    from ._shared.bf16_routes import BF16Route

    policy = resolve_bf16_policy(bf16_policy)
    if policy.storage_enabled and backend == "tilelang":
        raise ValueError("bf16_storage TileLang arithmetic is not qualified")

    if packed_route not in ("direct", "panel"):
        raise ValueError("packed_route must be direct or panel")
    if type(panel_size) is not int or panel_size <= 0:
        raise ValueError("panel_size must be a positive integer")
    if packed_route == "panel" and (packed_b is None or backend != "triton"):
        raise ValueError("Panel decoding requires packed_b and the Triton backend")
    if packed_route == "panel":
        if kernel_config is not None and not isinstance(kernel_config, BF16GemmConfig):
            raise TypeError("BF16 kernel_config must be an immutable BF16GemmConfig")
        from ._shared.bf16_gemm import legacy_bf16_config
        config = kernel_config or legacy_bf16_config(tile, num_stages)
        if panel_size % config.block_k:
            raise ValueError("panel_size must be a multiple of GEMM block_k")
    if backend not in ("native", "triton", "tilelang"):
        raise ValueError("BF16 product backend must be native, triton, or tilelang")
    if kernel_config is not None and not isinstance(kernel_config, BF16GemmConfig):
        raise TypeError("BF16 kernel_config must be an immutable BF16GemmConfig")
    if kernel_config is not None and backend != "triton":
        raise ValueError("Explicit BF16 kernel_config requires the Triton backend")
    if role is not None and role not in ("forward", "dx", "dw", "unspecified"):
        raise ValueError("BF16 product role must be forward, dx, dw, or unspecified")
    if role is None and gemm_launch_is_recording():
        from deepopt.kernel_policy import current_kernel_role

        ambient = (current_kernel_role() or "unspecified").rsplit(".", 1)[-1]
        role = ambient if ambient in ("forward", "dx", "dw") else "unspecified"
    role = role or "unspecified"
    representation = ("direct" if operand_route == "tiled" else operand_route) or (
        packed_route if packed_b is not None else "dense")
    if packed_b is None and representation in ("direct", "panel"):
        raise ValueError("Direct/panel operand_route requires a packed operand")
    if packed_b is not None and backend != "native" and representation != packed_route:
        raise ValueError("operand_route must match the actual packed decoding route")
    # Native packed execution decodes a complete operand before its library
    # call; it must never claim a direct packed library GEMM.
    if backend == "native" and packed_b is not None:
        representation = "materialized"
    route = BF16Route(role, backend, representation, policy)
    from ._shared.gemm_diagnostics import note_gemm_launch
    if gemm_launch_is_recording():
        note_gemm_launch(route_contract=route.as_dict(), bf16_policy=policy.as_dict())
    if a.ndim != 2 or a.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("BF16 product requires a BF16/FP32 matrix")
    if packed_b is None:
        if b is None or b.ndim != 2 or b.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("BF16 product requires two BF16/FP32 matrices")
        n, reduction = b.shape
        if b.device != a.device:
            raise ValueError("BF16 operands must share a device")
    else:
        reduction, n = packed_b.shape
        if packed_b.block_shape is not None:
            raise ValueError("Packed-read BF16 GEMM requires flat affine groups")
        if packed_b.data.device != a.device:
            raise ValueError("BF16 packed operand must share the input device")
    if a.shape[1] != reduction:
        raise ValueError("BF16 operands have different reduction dimensions")
    if bias is not None and (tuple(bias.shape) != (n,) or bias.device != a.device):
        raise ValueError("BF16 bias must have shape [N] on the operand device")
    if bias is not None and bias.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("BF16 bias must have BF16 or FP32 storage")
    if derivative_input is not None and (
        tuple(derivative_input.shape) != (a.shape[0], n) or derivative_input.device != a.device
    ):
        raise ValueError("Activation backward input must match the BF16 product")
    if derivative_input is not None and derivative_input.dtype != torch.bfloat16:
        raise TypeError("BF16 activation backward requires a BF16 preactivation")
    if residual is not None and (residual.shape != (a.shape[0], n) or residual.device != a.device
                                 or residual.dtype != torch.bfloat16):
        raise ValueError("BF16 residual must match the product shape, device and BF16 dtype")
    if derivative_packed is not None and (derivative_packed.shape != (a.shape[0], n)
            or derivative_packed.block_shape is not None or derivative_packed.data.device != a.device):
        raise ValueError("Packed derivative must have flat groups and match the BF16 product")
    if a.device.type == "cuda" and not isinstance(a, FakeTensor) and backend != "native":
        if torch.version.hip is not None or torch.cuda.get_device_capability(a.device) < (8, 0):
            raise ValueError("Generated BF16 GEMM requires NVIDIA compute capability >= 8")
        from ._shared.bf16_resources import BF16ResourceError
        try:
            result = import_module(f"deepopt.opt.backends.{backend}.gemm.bf16").product(
                a, b, bias=bias, activation=activation, derivative_input=derivative_input,
                derivative_activation=derivative_activation, packed_b=packed_b,
                tile=tile, num_stages=num_stages, return_preactivation=return_preactivation, residual=residual,
                derivative_packed=derivative_packed,
                **({"kernel_config": kernel_config, "role": role, "packed_route": packed_route,
                    "panel_size": panel_size, "bf16_policy": policy} if backend == "triton" else {}),
            )
        except BF16ResourceError as error:
            error.evidence.update(bf16_policy=policy.as_dict(), route_contract=route.as_dict())
            raise
        observe_bf16_gemm(a, b, backend, result[0] if return_preactivation else result, role,
                          packed_b=packed_b, execution="generated")
        return result
    # CPU/FakeTensor is a semantic reference, never evidence of a GPU launch.
    if packed_b is not None:
        from deepopt.opt.quantization.activation import dequantize_unpack_activation

        b = dequantize_unpack_activation(packed_b, dtype=torch.bfloat16, transpose=True)
    with torch.autocast(a.device.type, enabled=False):
        value = F.linear(a.to(torch.bfloat16), b.to(torch.bfloat16),
                         bias.to(torch.bfloat16) if bias is not None else None)
        if residual is not None:
            value = value + residual
        preactivation = value
        value = activate(value, activation)
        if derivative_input is not None:
            value = derivative(value, derivative_input, derivative_activation)
        if derivative_packed is not None:
            from deepopt.opt.quantization.activation import dequantize_unpack_activation
            if a.device.type == "cuda" and not isinstance(a, FakeTensor):
                from deepopt.kernel_policy import kernel_policy_scope
                from deepopt.opt.backends.activation import executor
                from deepopt.opt.nn.activation.state import SavedActivation
                from deepopt.opt.nn.activation.geometry import ReLUGeometry

                saved = SavedActivation(derivative_packed.data, derivative_packed.step, derivative_packed.minimum,
                    derivative_packed.group_size, bits=derivative_packed.bits, packed=True)
                with kernel_policy_scope(role="activation.backward"):
                    value = executor(value, "gelu").backward(value, saved, ReLUGeometry(tuple(value.shape), "last"), dtype=torch.bfloat16)
            else:
                coefficient = dequantize_unpack_activation(derivative_packed, dtype=torch.float32)
                value = (value.float() * coefficient).to(torch.bfloat16)
    observe_bf16_gemm(a, b, backend, value, role, packed_b=packed_b,
                      execution="native" if a.device.type == "cuda" and not isinstance(a, FakeTensor)
                      else "semantic_reference")
    return (value, preactivation) if return_preactivation else value


def prepare(value, *, activation=None, compression=None, backend="native", derivative_compression=None, bf16_policy=None):
    """Produce the dense BF16 forward operand and its optional packed save."""
    from deepopt._bf16_policy import resolve_bf16_policy
    policy = resolve_bf16_policy(bf16_policy)
    if policy.storage_enabled and backend == "tilelang":
        raise ValueError("bf16_storage TileLang preparation is not qualified")
    from torch._subclasses.fake_tensor import FakeTensor
    from deepopt.opt.quantization.activation import quantize_pack_activation

    if derivative_compression is not None and (activation is None or compression is None or any(
            derivative_compression[key] != compression[key] for key in ("group_size", "bits"))):
        raise ValueError("Joint BF16 preparation requires matching output/derivative codec groups and bits")
    if (backend != "native" and value.device.type == "cuda" and not isinstance(value, FakeTensor)
            and compression is not None and value.is_contiguous()
            and compression["group_size"] <= 16384):
        return import_module(f"deepopt.opt.backends.{backend}.gemm.bf16").prepare(
            value, activation=activation, compression=compression, derivative_compression=derivative_compression,
        )
    dense = activate(value, activation).to(torch.bfloat16)
    packed = (quantize_pack_activation(dense, compression["group_size"], bits=compression["bits"])
              if compression is not None else None)
    if derivative_compression is not None:
        from deepopt.opt.backends.reference.activation.math import evaluate
        from deepopt.opt._execution import finite_checks_enabled

        values, coefficient = evaluate(value.float(), *activation)
        if finite_checks_enabled() and not isinstance(value, FakeTensor) and not torch.isfinite(coefficient).all():
            raise ValueError("Activation derivative compression requires finite coefficients")
        dense = values.to(torch.bfloat16)
        packed = quantize_pack_activation(dense, compression["group_size"], bits=compression["bits"])
        saved = quantize_pack_activation(coefficient, compression["group_size"], bits=compression["bits"])
        return dense, packed, saved
    return dense, packed
