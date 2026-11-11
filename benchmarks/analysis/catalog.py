"""Public operation inventory and editable representative workloads; no Torch import."""

from copy import deepcopy


ACTIVATIONS = ("relu", "sigmoid", "silu", "gelu", "gelu_tanh", "tanh", "relu6")
POOL_OPS = tuple(f"{family}{rank}d" for rank in (1, 2, 3) for family in (
    "max_pool", "avg_pool", "adaptive_max_pool", "adaptive_avg_pool", "lp_pool", "max_unpool")) + (
    "fractional_max_pool2d", "fractional_max_pool3d")
INDEX_ALIASES = tuple(name + "_with_indices" for name in POOL_OPS if (
    name.startswith(("max_pool", "adaptive_max_pool", "fractional_max_pool"))))
OPS = ("linear", "hf_conv1d", *(f"conv{n}d" for n in (1, 2, 3)),
       *(f"conv_transpose{n}d" for n in (1, 2, 3)), "batch_norm", "group_norm", "layer_norm",
       *ACTIVATIONS, *POOL_OPS, *INDEX_ALIASES, "scaled_dot_product_attention", "multihead_attention",
       "linear_pipeline", "conv_pipeline", "attention_pipeline", "model",
       "activation_codec", "fp8_activation_codec", "boolean_codec", "index_codec")


def family(op):
    if op in ("linear", "hf_conv1d", "linear_pipeline"):
        return "linear"
    if op.startswith("conv"):
        return "convolution"
    if op in ACTIVATIONS:
        return "activation"
    if op.endswith("norm"):
        return "normalization"
    if op in POOL_OPS or op in INDEX_ALIASES:
        return "pooling"
    if "attention" in op:
        return "attention"
    return "model" if op == "model" else "codec"


def default_workload(op, small=False):
    if op not in OPS:
        raise ValueError(f"Unknown operator {op}")
    options = {}
    if op in ("linear", "hf_conv1d", "linear_pipeline", "activation_codec", "fp8_activation_codec",
              "boolean_codec", "index_codec"):
        shape = [16, 32] if small else [4096, 768]
        if op in ("linear", "hf_conv1d", "linear_pipeline"):
            options["out_features"] = 64 if small else 3072
    elif op.startswith("conv") or op in POOL_OPS or op in INDEX_ALIASES:
        rank = int(op.removesuffix("_with_indices")[-2]) if op != "conv_pipeline" else 2
        shape = [2, 8] + [8 if small else (16 if rank == 3 else 56)] * rank
        if op.startswith("conv"):
            options = {"out_channels": 8 if small else 64, "kernel_size": 3, "padding": 1}
        elif "adaptive" in op:
            options = {"output_size": [2] * rank}
        elif "fractional" in op:
            options = {"kernel_size": 2, "output_size": [3] * rank}
        elif "lp_pool" in op:
            options = {"norm_type": 2, "kernel_size": 2, "stride": 2}
        else:
            options = {"kernel_size": 2, "stride": 2}
    elif op in ("batch_norm", "group_norm"):
        shape = [2, 8, 4, 4] if small else [8, 64, 56, 56]
        if op == "group_norm":
            options["num_groups"] = 4 if small else 32
    elif "attention" in op:
        shape = ([2, 2, 8, 16] if small else [8, 12, 197, 64]) if op == "scaled_dot_product_attention" else (
            [2, 8, 32] if small else [8, 197, 768])
        if op != "scaled_dot_product_attention":
            options["num_heads"] = 2 if small else 12
    else:
        shape = [4, 32] if small else [4096, 768]
    return {"name": op, "op": op, "shape": shape, "kwargs": options, "layout": "contiguous",
            "training": True, "input_requires_grad": True, "parameter_requires_grad": True,
            "calls_per_step": None, "gradient_mode": "all"}


def workload_list(ops=None, small=False):
    return [default_workload(op, small) for op in (ops or OPS) if op != "model"]


def normalize_workload(value):
    if not isinstance(value, dict) or "op" not in value:
        raise ValueError("Each workload must be an object with an op")
    result = default_workload(value["op"], small=True)
    allowed = set(result) | {"factory", "factory_kwargs", "steps_per_epoch"}
    if set(value) - allowed:
        raise ValueError(f"Unknown workload fields: {sorted(set(value) - allowed)}")
    result.update(deepcopy(value))
    if not isinstance(result["name"], str) or not result["name"]:
        raise ValueError("Workload name must be nonempty")
    if not isinstance(result["shape"], list) or not result["shape"] or any(
            type(n) is not int or n < 1 for n in result["shape"]):
        raise ValueError("Workload shape must contain positive integers")
    if not isinstance(result["kwargs"], dict):
        raise ValueError("Workload kwargs must be an object")
    for key in ("training", "input_requires_grad", "parameter_requires_grad"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("calls_per_step", "steps_per_epoch"):
        if result.get(key) is not None and (not isinstance(result[key], (float, int)) or result[key] <= 0):
            raise ValueError(f"{key} must be positive or null")
    if result["layout"] not in ("contiguous", "strided", "channels_last"):
        raise ValueError("layout must be contiguous, strided, or channels_last")
    if result["gradient_mode"] not in ("all", "dx", "dw", "db", "none"):
        raise ValueError("gradient_mode must be all, dx, dw, db, or none")
    if result["op"] == "model" and not result.get("factory"):
        raise ValueError("model workloads require factory='importable_module:function'")
    return result


def inventory():
    return [{"op": op, "family": family(op), "torch": (
        "torch.nn.MultiheadAttention" if op == "multihead_attention" else
        "transformers.pytorch_utils.Conv1D" if op == "hf_conv1d" else
        f"torch.nn.functional.{op.replace('_tanh', '')}" if op not in (
            "linear_pipeline", "conv_pipeline", "attention_pipeline", "model") and not op.endswith("codec")
        else "matched native pipeline / storage-copy control"),
        "kind": "diagnostic" if op.endswith("codec") else "operator_or_pipeline"} for op in OPS]
