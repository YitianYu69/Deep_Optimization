import torch
from contextlib import nullcontext
from torch.nn import functional as F

from .._autocast import _get_fp8_plan, _suspend_fp8_interception
from ..scaling_strategy.amax import AmaxScaling
from .AOTAutograd_Func.linear_autograd_func import _DOLinearFunction
from deepopt.config import delayed_scaling_config_from_dict
from dataclasses import asdict

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


class _FrozenPlan(dict):
    """Owned immutable dictionaries retain the public dict/equality contract."""

    def __init__(self, values):
        dict.__init__(self, ((name, _FrozenPlan(value) if isinstance(value, dict) else value)
                            for name, value in values.items()))

    def _immutable(self, *args, **kwargs):
        raise TypeError("Validated FP8 plans are immutable; validate a new plan to change policy")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


class _ValidatedFP8Plan(_FrozenPlan):
    def __init__(self, values):
        super().__init__(values)
        training = self["fp8_training"]
        self.delayed_signature = (tuple(training["fp8_dtypes"][role] for role in _OPERANDS),
                                  tuple(training["delayed_scaling"][name] for name in
                                        ("history_length", "amax_algorithm", "margin", "reduce_amax")))


def _prepare_fp8_plan(plan):
    """Keep the public mutable-plan API; freeze only an internal execution copy."""
    if isinstance(plan, _ValidatedFP8Plan):
        return plan
    return _ValidatedFP8Plan(validate_fp8_plan(plan))


def validate_fp8_plan(plan: dict) -> dict:
    """
    Return a validated, independently owned mutable plan with explicit defaults.
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
        "out_dtype", "scaling_recipe", "delayed_scaling", "backward_backend"
    }

    if "fp8_dtypes" not in training:
        raise ValueError("fp8_training requires 'fp8_dtypes'")
    if "max_num_imprecise_acc" in training:
        raise ValueError("max_num_imprecise_acc was removed; create a new plan using native Triton accumulation")
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

    num_stages = training.get("num_stages", "auto")

    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")

    out_dtype = training.get("out_dtype")
    recipe = training.get("scaling_recipe", "current")
    if recipe not in ("current", "delayed"):
        raise ValueError("scaling_recipe must be current or delayed")
    delayed = delayed_scaling_config_from_dict(training.get("delayed_scaling", {}))
    if recipe == "delayed" and any(shape is not None for shape in block_shapes.values()):
        raise ValueError("Delayed scaling requires tensorwise block_shapes")
    backward_backend = training.get("backward_backend", "inherit")
    if backward_backend not in ("inherit", "native_fast", "native_precise"):
        raise ValueError("backward_backend must be inherit, native_fast, or native_precise")
    if backward_backend != "inherit" and (backend != "triton" or any(shape is not None for shape in block_shapes.values())):
        raise ValueError("Native FP8 backward requires backend=triton and tensorwise block_shapes")
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
            "scaling_recipe": recipe,
            "delayed_scaling": asdict(delayed),
            "backward_backend": backward_backend,
        },
        "block_shapes": dict(block_shapes),
    }




@torch.compiler.disable
def do_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    *,
    plan: dict, _weight_cache=None, _scaling_state=None, _state_training=True,
) -> torch.Tensor:
    """Run FP8 projection/autograd eagerly inside an otherwise compiled model.

    Plan selection and quantized-tensor metadata use Python validation and
    data-dependent checks. Keep this boundary out of Dynamo tracing, including
    when a TorchFunctionMode is active; surrounding model operations can still
    compile with fullgraph=False. Triton kernels retain their own compilation.
    """

    with _suspend_fp8_interception():
        return _do_linear(x, weight, bias, plan=plan, weight_cache=_weight_cache,
                          scaling_state=_scaling_state, state_training=_state_training)


def _do_linear(x, weight, bias, *, plan, weight_cache, scaling_state=None, state_training=True):
    active_plan = _get_fp8_plan(x.device.type) if isinstance(x, torch.Tensor) else None
    if active_plan is False:
        return F.linear(x, weight, bias)
    if active_plan is not None:
        plan = active_plan
    checked_plan = _prepare_fp8_plan(plan)
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
    
    if training["scaling_recipe"] == "delayed":
        if scaling_state is None:
            raise ValueError("Delayed scaling requires a prepared adapter or registered FX call site; bare F.linear is stateless")
        if scaling_state.recipe_signature != checked_plan.delayed_signature:
            raise ValueError("Delayed-scaling state does not match the active recipe; prepare a matching adapter")
    else:
        scaling_state = None
    scope = nullcontext()
    if scaling_state is not None:
        from ..scaling_strategy.delayed import fp8_scaling_scope

        scope = fp8_scaling_scope()
    with scope:
        return _DOLinearFunction.apply(
            x, weight, bias,
            checked_plan, out_dtype,
            torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values()),
            weight_cache, scaling_state, state_training and torch.is_grad_enabled(),
        )
