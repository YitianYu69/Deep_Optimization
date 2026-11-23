"""Dependency-free experiment expansion and validation (also used by --dry-run)."""

from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math


CASES = {
    "torch_bf16": (False, False),
    "do_bf16": (False, False),
    "compression": (False, True),
    "fp8": (True, False),
    "fp8_compression": (True, True),
}

DEFAULT_SETTINGS = {
    "fp8": {"backend": "triton", "trace_mode": "modules", "activation_dtype": "e4m3fn",
            "intercept_linear": False, "intercept_convolution": False,
            "intercept_normalization": False, "intercept_pooling": False, "exclude_modules": [],
            "weight_dtype": "e4m3fn", "grad_dtype": "e4m3fn", "num_stages": "auto",
            "out_dtype": None, "scaling_recipe": "current", "backward_backend": "inherit",
            "backward_precision": "native", "convolution_policy": "strict", "block_shapes": {
                "activation": None, "weight": None, "grad": None},
            "delayed_scaling": {"history_length": 1024, "amax_algorithm": "max",
                                "margin": 0, "reduce_amax": True}},
    "compression": {"bits": 2, "group_size": 1024, "attention_bits": None,
                    "norm_xhat_recenter": False},
    "runtime": {"execution": "eager", "binding": "functional", "scope_lifetime": "fresh", "numerical_probe": "loss",
                "bf16_policy": "torch_amp",
                "qualification_case": "group1024-slot-0",
                "norm_kernel_variant": "auto", "norm_row_chunk": 4, "norm_max_programs": 512, "cache_enabled": False,
                "codec_pack_variant": "auto", "codec_groups_per_program": 4, "spatial_activation_layout": "auto", "activation_layout": "auto",
                "codec_num_warps": 4, "activation_pointwise_tile": 512,
                "producer_pack": "off", "gelu_saved_state": "derivative", "silu_saved_state": "derivative",
                "strided_indexing": "wide", "convolution_codec": "auto", "convolution_kernel_variant": "auto", "pool_kernel_variant": "auto",
                "attention_backend": "native", "attention_pack": "separate", "attention_grad_layout": "auto",
                "linear_dw_backend": "inherit", "native_operand_layout": "preserve",
                "linear_forward_backend": "inherit", "linear_gradient_reduction": "separate",
                "timing_fill_uninitialized_memory": None, "optimizer_warmup_steps": 0,
                "compression_bits_by_family": None, "compression_exclude_modules": [],
                "check_finite": False, "input_dtype": "bf16", "output_dtype": "auto",
                "input_scaling": "included", "output_scaling": "current",
                "sdpa_backend": "auto", "optimizer": "none", "optimizer_options": {}, "lr": 0.001,
                "accumulation_steps": 1, "checkpoint": False, "compression_families": None,
                "compile": {"backend": "inductor", "fullgraph": False, "dynamic": False, "mode": None, "options": {}}},
}

# A finite sensitivity grid, not a claim to enumerate all possible tensor/block sizes.
EXTENDED_GRID = {
    "compression.bits": [1, 2, 4, 8],
    "compression.group_size": [1024],
    # Keep five independently executed, named cells under the required recipe.
    # These are repetitions, not claims of coverage of five different codecs.
    "runtime.qualification_case": [f"group1024-slot-{index}" for index in range(5)],
    "compression.attention_bits": [None, 1, 2, 4, 8],
    "compression.norm_xhat_recenter": [False, True],
    "fp8.activation_dtype": ["e4m3fn", "e5m2"],
    "fp8.weight_dtype": ["e4m3fn", "e5m2"],
    "fp8.grad_dtype": ["e4m3fn", "e5m2"],
    "fp8.num_stages": ["auto", 1, 2, 3, 4, 5],
    "fp8.out_dtype": [None, "bf16", "fp32", "fp16"],
    "fp8.trace_mode": ["modules", "full", "auto"],
    "fp8.intercept_linear": [False, True],
    "fp8.intercept_convolution": [None, False, True],
    "fp8.intercept_normalization": [None, False, True],
    "fp8.intercept_pooling": [None, False, True],
    "fp8.scaling_recipe": ["current", "delayed"],
    "fp8.block_shapes.activation": [None, [1, 32], [32, 32]],
    "fp8.block_shapes.weight": [None, [1, 32], [32, 32]],
    "fp8.block_shapes.grad": [None, [1, 32], [32, 32]],
    "fp8.backward_backend": ["inherit", "native_fast", "native_precise"],
    "fp8.backward_precision": ["native", "auto"],
    "fp8.convolution_policy": ["strict", "depthwise", "native_spatial"],
    "fp8.delayed_scaling.history_length": [16, 1024],
    "fp8.delayed_scaling.amax_algorithm": ["max", "most_recent"],
    "fp8.delayed_scaling.margin": [0, 1],
    "fp8.delayed_scaling.reduce_amax": [False, True],
    "runtime.execution": ["eager", "graph", "compile"],
    "runtime.binding": ["functional", "prepared", "intercept"],
    "runtime.scope_lifetime": ["fresh", "reused"],
    "runtime.norm_kernel_variant": ["baseline", "auto", "candidate", "batch_shared", "small_forward", "layer_fused", "layer_tiled"],
    "runtime.norm_row_chunk": [1, 4, 8],
    "runtime.norm_max_programs": [0, 264, 528, 1056],
    "runtime.codec_pack_variant": ["auto", "grouped", "batched", "serial"],
    "runtime.codec_groups_per_program": [1, 2, 4, 8],
    "runtime.spatial_activation_layout": ["auto", "aligned", "compact"],
    "runtime.codec_num_warps": ["auto", 1, 2, 4, 8],
    "runtime.producer_pack": ["off", "gelu", "layer_norm", "all"],
    "runtime.gelu_saved_state": ["auto", "derivative", "recompute"],
    "runtime.silu_saved_state": ["derivative", "recompute"],
    "runtime.linear_forward_backend": ["inherit", "native_fast", "native_precise"],
    "runtime.linear_gradient_reduction": ["separate", "shared"],
    "runtime.convolution_codec": ["auto","materialized"],
    "runtime.pool_kernel_variant": ["auto","generic"],
    "runtime.strided_indexing": ["wide", "bounded"],
    "runtime.linear_dw_backend": ["inherit", "native_precise"],
    "runtime.native_operand_layout": ["preserve", "prepare"],
    "runtime.activation_pointwise_tile": [512, 1024, 2048, 4096],
    "runtime.timing_fill_uninitialized_memory": [None, False, True],
    "runtime.optimizer_warmup_steps": [0, 10],
    "runtime.compression_bits_by_family": [None, {}, {"linear": 8, "normalization": None, "attention": 8}],
    "runtime.compression_exclude_modules": [[], ["head", "patch_embed*"]],
    "runtime.cache_enabled": [False, True],
    "runtime.check_finite": [False, True],
    "runtime.input_dtype": ["bf16", "fp8"],
    "runtime.output_dtype": ["auto", "bf16", "fp8"],
    "runtime.input_scaling": ["included", "precomputed"],
    "runtime.output_scaling": ["current", "provided"],
    "runtime.sdpa_backend": ["auto", "math", "flash", "efficient"],
    "runtime.optimizer": ["none", "sgd", "adamw"],
    "runtime.accumulation_steps": [1, 4],
    "runtime.checkpoint": [False, True],
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown setting: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            result[key] = deepcopy(value) if key in ("options", "optimizer_options", "compression_bits_by_family") else merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def assign(settings, dotted, value):
    target = settings
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"Unknown grid setting: {dotted}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"Unknown grid setting: {dotted}")
    target[parts[-1]] = deepcopy(value)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def bf16_policy_identity(settings, case):
    """Keep raw Torch controls on AMP while identifying the selected DO recipe."""
    selected = settings["runtime"].get("bf16_policy", "torch_amp")
    if selected not in ("torch_amp", "bf16_storage"):
        raise ValueError("runtime.bf16_policy must be torch_amp or bf16_storage")
    return {"name": "torch_amp" if case == "torch_bf16" else selected, "version": 1}


def numerical_recipe_identity(settings, case):
    """Cold, dependency-free identity beyond the unchanged public BF16 v1.

    Subrecipes prevent old partial storage/preparation results from becoming
    evidence for the expanded managed-model behavior.
    """
    policy = bf16_policy_identity(settings, case)
    runtime, compression = settings.get("runtime", {}), settings.get("compression", {})
    compressed = CASES.get(case, (False, False))[1]
    attention = compression.get("attention_bits")
    effective = (max(4, compression.get("bits", 2)) if attention is None else attention) if compressed else 0
    return dict(bf16_policy=policy, primitive_recipe_version=2 if policy["name"] == "bf16_storage" else None,
        prepared_attention_recipe_version=2 if runtime.get("binding") == "prepared" and
            (compressed or policy["name"] == "bf16_storage") else None,
        attention_backend=runtime.get("attention_backend", "native"), effective_attention_bits=effective,
        activation_bits=compression.get("bits", 2) if compressed else 0,
        group_size=compression.get("group_size", 1024) if compressed else None,
        norm_xhat_recenter=compression.get("norm_xhat_recenter", False) if compressed else False)


def validate_settings(settings):
    f, c, r = (settings[k] for k in ("fp8", "compression", "runtime"))
    if not isinstance(r["qualification_case"], str) or not r["qualification_case"]:
        raise ValueError("qualification_case must be a nonempty provenance label")
    if f["convolution_policy"] not in ("strict", "native_spatial", "depthwise"):
        raise ValueError("fp8.convolution_policy must be strict, native_spatial or depthwise")
    choices = {"execution": ("eager", "graph", "compile"),
               "bf16_policy": ("torch_amp", "bf16_storage"),
               "binding": ("functional", "prepared", "intercept"),
               "scope_lifetime": ("fresh", "reused"), "numerical_probe": ("loss", "random_upstream"),
               "norm_kernel_variant": ("auto", "baseline", "candidate", "batch_shared", "small_forward", "layer_fused", "layer_tiled", "channels_last"),
               "codec_pack_variant": ("auto", "grouped", "batched", "serial"),
               "spatial_activation_layout": ("auto", "aligned", "compact"),
               "activation_layout": ("auto", "baseline", "channels_last"),
               "producer_pack": ("off", "gelu", "layer_norm", "all"), "gelu_saved_state": ("auto", "derivative", "recompute"),
               "silu_saved_state": ("derivative", "recompute"),
               "strided_indexing": ("wide", "bounded"),
               "convolution_codec": ("auto","direct","materialized"),
               "convolution_kernel_variant": ("auto","baseline","tuned"),
               "attention_backend": ("native", "flash_attn_2"), "attention_pack": ("auto", "separate", "joint"),
               "attention_grad_layout": ("auto", "preserve", "bshd", "bhsd", "interleaved"),
               "pool_kernel_variant": ("auto","generic"),
               "linear_dw_backend": ("inherit", "native_precise"), "native_operand_layout": ("preserve", "prepare"),
               "linear_forward_backend": ("inherit", "native_fast", "native_precise"),
               "linear_gradient_reduction": ("separate", "shared"),
               "input_dtype": ("bf16", "fp8"), "output_dtype": ("auto", "bf16", "fp8"),
               "input_scaling": ("included", "precomputed"),
               "output_scaling": ("current", "provided"),
               "sdpa_backend": ("auto", "math", "flash", "efficient"),
               "optimizer": ("none", "sgd", "adamw")}
    for key, values in choices.items():
        if r[key] not in values:
            raise ValueError(f"runtime.{key} must be one of {values}")
    for key in ("cache_enabled", "check_finite", "checkpoint"):
        if type(r[key]) is not bool:
            raise ValueError(f"runtime.{key} must be boolean")
    if type(r["accumulation_steps"]) is not int or r["accumulation_steps"] < 1:
        raise ValueError("accumulation_steps must be a positive integer")
    if type(r["norm_row_chunk"]) is not int or r["norm_row_chunk"] not in (1, 2, 4, 8, 16) or type(r["norm_max_programs"]) is not int or not 0 <= r["norm_max_programs"] <= 4096:
        raise ValueError("norm_row_chunk must be 1/2/4/8/16 and norm_max_programs must be 0..4096")
    if type(r["codec_groups_per_program"]) is not int or r["codec_groups_per_program"] not in (1, 2, 4, 8):
        raise ValueError("codec_groups_per_program must be 1/2/4/8")
    if r["codec_num_warps"] != "auto" and (type(r["codec_num_warps"]) is not int or r["codec_num_warps"] not in (1, 2, 4, 8)):
        raise ValueError("codec_num_warps must be auto/1/2/4/8")
    if type(r["activation_pointwise_tile"]) is not int or r["activation_pointwise_tile"] not in (512, 1024, 2048, 4096):
        raise ValueError("activation_pointwise_tile must be 512/1024/2048/4096")
    if r["timing_fill_uninitialized_memory"] is not None and type(r["timing_fill_uninitialized_memory"]) is not bool:
        raise ValueError("timing_fill_uninitialized_memory must be boolean or null")
    if type(r["optimizer_warmup_steps"]) is not int or r["optimizer_warmup_steps"] < 0:
        raise ValueError("optimizer_warmup_steps must be a nonnegative integer")
    if not isinstance(r["lr"], (int, float)) or not math.isfinite(r["lr"]) or r["lr"] <= 0:
        raise ValueError("lr must be positive and finite")
    if not isinstance(r["optimizer_options"], dict) or {"params", "lr"} & r["optimizer_options"].keys():
        raise ValueError("optimizer_options must be an object without params/lr (use runtime.lr)")
    if (type(c["bits"]) is not int or c["bits"] not in (1, 2, 4, 8)
            or c["attention_bits"] is not None and (type(c["attention_bits"]) is not int or c["attention_bits"] not in (1, 2, 4, 8))):
        raise ValueError("compression bits must be 1, 2, 4, or 8")
    if type(c["group_size"]) is not int or c["group_size"] < 1:
        raise ValueError("compression.group_size must be a positive integer")
    if type(c["norm_xhat_recenter"]) is not bool:
        raise ValueError("norm_xhat_recenter must be boolean")
    if r["compression_families"] is not None and (
            not isinstance(r["compression_families"], list)
            or any(x not in ("linear", "convolution", "normalization", "activation", "pooling", "attention", "binary", "padding")
                   for x in r["compression_families"])):
        raise ValueError("compression_families must be null or a list of operator families")
    families = ("linear", "convolution", "normalization", "activation", "pooling", "attention", "binary", "padding")
    if r["compression_bits_by_family"] is not None and (
            not isinstance(r["compression_bits_by_family"], dict)
            or any(key not in families or value is not None and (type(value) is not int or value not in (1, 2, 4, 8))
                   for key, value in r["compression_bits_by_family"].items())):
        raise ValueError("compression_bits_by_family must map families to 1/2/4/8 or null (disabled)")
    if not isinstance(r["compression_exclude_modules"], list) or any(
            not isinstance(value, str) or not value for value in r["compression_exclude_modules"]):
        raise ValueError("compression_exclude_modules must be a list of nonempty module globs")
    if f["backend"] not in ("triton", "reference") or f["scaling_recipe"] not in ("current", "delayed"):
        raise ValueError("Unsupported FP8 backend or scaling recipe")
    for key in ("activation_dtype", "weight_dtype", "grad_dtype"):
        if f[key] not in ("e4m3fn", "e5m2"):
            raise ValueError(f"Unsupported fp8.{key}")
    if f["num_stages"] != "auto" and (type(f["num_stages"]) is not int or f["num_stages"] not in range(1, 6)):
        raise ValueError("num_stages must be auto or an integer 1..5")
    if f["trace_mode"] not in ("modules", "full", "auto"):
        raise ValueError("Unsupported trace_mode")
    for key in ("intercept_linear", "intercept_convolution", "intercept_normalization", "intercept_pooling"):
        if type(f[key]) is not bool and (key == "intercept_linear" or f[key] is not None):
            raise ValueError(f"fp8.{key} must be boolean (or null for inherited non-linear interception)")
    if not isinstance(f["exclude_modules"], list) or any(not isinstance(x, str) or not x.strip() for x in f["exclude_modules"]):
        raise ValueError("fp8.exclude_modules must be a list of nonempty strings")
    if f["out_dtype"] not in (None, "bf16", "fp16", "fp32"):
        raise ValueError("Unsupported out_dtype")
    if f["backward_backend"] not in ("inherit", "native_fast", "native_precise"):
        raise ValueError("Unsupported backward_backend")
    if f["backward_precision"] not in ("native", "auto"):
        raise ValueError("Unsupported backward_precision")
    for block in f["block_shapes"].values():
        if block is not None and (not isinstance(block, (list, tuple)) or len(block) != 2
                                  or any(type(x) is not int or x < 1 for x in block)):
            raise ValueError("FP8 blocks must be null or two positive integers")
    d = f["delayed_scaling"]
    if type(d["history_length"]) is not int or d["history_length"] < 1:
        raise ValueError("history_length must be positive")
    if d["amax_algorithm"] not in ("max", "most_recent") or type(d["margin"]) is not int or not -32 <= d["margin"] <= 32:
        raise ValueError("Invalid delayed scaling algorithm/margin")
    if type(d["reduce_amax"]) is not bool:
        raise ValueError("reduce_amax must be boolean")
    compiled = r["compile"]
    if type(compiled["fullgraph"]) is not bool or compiled["dynamic"] is not None and type(compiled["dynamic"]) is not bool:
        raise ValueError("compile fullgraph/dynamic must be boolean (dynamic may be null)")
    if not isinstance(compiled["backend"], str) or compiled["mode"] is not None and not isinstance(compiled["mode"], str):
        raise ValueError("compile backend/mode must be strings (mode may be null)")


def variants(settings, grid, sweep):
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid {key} must be a nonempty list")
        probe = deepcopy(settings)
        assign(probe, key, values[0])
    changes = [()]
    if sweep == "cartesian":
        changes = product(*(tuple((key, value) for value in values) for key, values in grid.items()))
    elif sweep == "one-at-a-time":
        changes += [((key, value),) for key, values in grid.items() for value in values]
    else:
        raise ValueError("sweep must be one-at-a-time or cartesian")
    seen = set()
    for updates in changes:
        value = deepcopy(settings)
        for key, item in updates:
            assign(value, key, item)
        validate_settings(value)
        key = fingerprint(value)
        if key not in seen:
            seen.add(key)
            yield key, value


def incompatibility(settings, case, workload, device):
    fp8, compressed = CASES[case]
    f, r = settings["fp8"], settings["runtime"]
    if case != "torch_bf16" and r.get("bf16_policy", "torch_amp") == "bf16_storage" and (
            fp8 or r["input_dtype"] == "fp8" or r["output_dtype"] == "fp8"):
        return "bf16_storage requires FP8 disabled and native BF16 input/output"
    if r["optimizer_warmup_steps"] and r["optimizer"] == "none":
        return "Shared optimizer warmup requires an optimizer"
    blocks = any(x is not None for x in f["block_shapes"].values())
    if fp8 and f["exclude_modules"] and (r["binding"] == "intercept" or any(
            f[key] for key in ("intercept_linear", "intercept_convolution", "intercept_normalization", "intercept_pooling"))):
        return "Module exclusions require interception to be disabled"
    if fp8 and f["scaling_recipe"] == "delayed" and blocks:
        return "Delayed scaling requires tensorwise blocks"
    if fp8 and f["backward_backend"] != "inherit" and (blocks or f["backend"] != "triton"):
        return "Native FP8 backward requires tensorwise Triton execution"
    if fp8 and f["backend"] == "triton" and not device.startswith("cuda"):
        return "Triton FP8 requires CUDA; select backend=reference for CPU validation"
    if r["execution"] == "graph" and not device.startswith("cuda"):
        return "CUDA Graph requires CUDA"
    selective = r["compression_bits_by_family"] is not None or r["compression_exclude_modules"]
    if r["execution"] == "graph" and (r["compression_families"] is not None or selective):
        return "Family policy hooks require eager execution; compare with the all-family hook control"
    if r["execution"] == "graph" and fp8 and f["scaling_recipe"] == "delayed":
        return "Delayed scaling updates require eager processing"
    if r["execution"] == "graph" and fp8 and r["cache_enabled"]:
        return "FP8 graph updates require cache_enabled=False so changed weights are requantized on every replay"
    if r["execution"] == "graph" and r["check_finite"] and (fp8 or compressed):
        return "Host-visible finite diagnostics are incompatible with CUDA capture"
    if r["binding"] == "intercept" and fp8 and f["scaling_recipe"] == "delayed":
        return "Stateless interception cannot own delayed scaling histories"
    if r["binding"] != "functional" and r["output_dtype"] != "auto":
        return "Explicit output precision requires functional binding; prepared pipelines use the planner"
    if r["output_scaling"] == "provided" and r["output_dtype"] != "fp8":
        return "Provided output scale requires explicit FP8 output"
    if (r["compression_families"] is not None or selective) and r["binding"] == "intercept":
        return "Family ablations require prepared or functional binding"
    if fp8 and device.startswith("cuda"):
        for block in f["block_shapes"].values():
            if block is not None and (list(block) == [1, 1] or any(x & (x - 1) for x in block)):
                return "CUDA FP8 blocks require power-of-two dimensions excluding [1,1]"
    if r["input_dtype"] == "fp8" and workload["op"] in ("model", "linear_pipeline", "conv_pipeline", "attention_pipeline"):
        return "Model/pipeline inputs are native; FP8 conversions are owned by the pipeline"
    return None
