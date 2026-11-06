import torch
from torch.nn import functional as F

from .._autocast import _get_fp8_plan
from ..scaling_strategy.amax import AmaxScaling
from .AOTAutograd_Func.linear_autograd_func import _DOLinearFunction

_FLOAT_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_OPERANDS = (
    "activation",
    "weight",
    "grad"
)


def validate_fp8_plan(plan: dict) -> dict:
    """
    Return a validated, independently owned plan with explicit defaults.
    """

    if not isinstance(plan, dict):
        raise TypeError("plan must be a dirtionarh")
    if set(plan) != {"fp8_training", "block_shapes"}:
        raise ValueError("plan must contain exactly 'fp8_training' and 'block_shapes'")
    
    training = plan["fp8_training"]
    block_shapes = plan["block_shapes"]

    if not isinstance(training, dict):
        raise TypeError("fp8_training must be a dictionary")
    
    allowed_fields = {
        "fp8_dtypes",
        "backend",
        "num_stages",
        "out_dtype"
    }

    if "fp8_dtypes" not in training:
        raise ValueError("fp8_training requires 'fp8_dtypes'")
    if set(training) - allowed_fields:
        raise ValueError("Unknown fields in fp8_training")

    fp8_dtypes = training["fp8_dtypes"]
    if not isinstance(fp8_dtypes, dict):
        raise TypeError("fp8_dtypes must be a dictionary")
    if set(fp8_dtypes) != set(_OPERANDS):
        raise ValueError(
            "fp8_dtypes must define activation, weight, and grad"
        )

    for operand in _OPERANDS:
        dtype = fp8_dtypes[operand]

        if (not isinstance(dtype, torch.dtype) or dtype not in _FP8_DTYPES):
            raise ValueError(
                f"Unsupported FP8 dtype for {operand!r}"
            )

    if not isinstance(block_shapes, dict):
        raise TypeError("block_shapes must be a dictionary")
    if set(block_shapes) != set(_OPERANDS):
        raise ValueError(
            "block_shapes must define activation, weight, and grad"
        )

    for operand in _OPERANDS:
        AmaxScaling(block_shape=block_shapes[operand])

    backend = training.get("backend", "reference")
    if backend not in ("reference", "triton"):
        raise ValueError(
            "backend must be 'reference' or 'triton'"
        )

    num_stages = training.get("num_stages", 3)

    if (type(num_stages) is not int or num_stages not in (1, 2, 3, 4)):
        raise ValueError("num_stages must be 1, 2, 3, or 4")

    out_dtype = training.get("out_dtype")
    if out_dtype is not None:
        if (not isinstance(out_dtype, torch.dtype) or out_dtype not in _FLOAT_DTYPES):
            raise ValueError(
                "out_dtype must be FP32, BF16, FP16, or None"
            )

    return {
        "fp8_training": {
            "fp8_dtypes": dict(fp8_dtypes),
            "backend": backend,
            "num_stages": num_stages,
            "out_dtype": out_dtype,
        },
        "block_shapes": dict(block_shapes),
    }




@torch.compiler.disable
def do_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    *,
    plan: dict
) -> torch.Tensor:
    """Run FP8 projection/autograd eagerly inside an otherwise compiled model.

    Plan selection and quantized-tensor metadata use Python validation and
    data-dependent checks. Keep this boundary out of Dynamo tracing, including
    when a TorchFunctionMode is active; surrounding model operations can still
    compile with fullgraph=False. Triton kernels retain their own compilation.
    """

    active_plan = _get_fp8_plan(x.device.type) if isinstance(x, torch.Tensor) else None
    if active_plan is False:
        return F.linear(x, weight, bias)
    if active_plan is not None:
        plan = active_plan
    checked_plan = validate_fp8_plan(plan)
    training = checked_plan["fp8_training"]

    tensors = {
        "input" : x,
        "weight" : weight
    }

    if bias is not None:
        tensors["bias"] = bias
    
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise TypeError(f"{name} must have strided layout")
        if tensor.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"{name} must have FP32, BF16, or FP16")
        
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("FP8 linear supports CPU and CUDA")
    
    for name, tensor in tensors.items():
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same device as input")
    if x.ndim < 1:
        raise ValueError("Input must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("Weight must have shape [out_features, in_features]")
    if x.numel() == 0 or weight.numel() == 0:
        raise ValueError("FP8 linear currently requires non-empty input and weight")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("Input's last dimension must match weights.shape[1]")
    if bias is not None:
        if tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("Bias must have shape [out_features]")
    if training["backend"] == "triton" and x.device.type != "cuda":
        raise ValueError("The Triton backend require CUDA")
    
    out_dtype = training["out_dtype"]
    if out_dtype is None:
        if torch.is_autocast_enabled(x.device.type):
            out_dtype = torch.get_autocast_dtype(x.device.type)
        else:
            out_dtype = x.dtype
    
    return _DOLinearFunction.apply(
        x, weight, bias,
        checked_plan, out_dtype
    )
