"""Bounded, pointer-free GEMM layouts and direct launches of measured winners."""

from dataclasses import dataclass
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from deepopt.opt.backends.triton.gemm.autotune import AUTOTUNE_KEY, MAX_BLOCK_K


@dataclass(frozen=True)
class GemmLaunchPlan:
    output_stride: int
    arguments: Mapping[str, int | str | bool]
    cache_key: tuple


@lru_cache(maxsize=256)
def gemm_launch_plan(m, n, k, a_block, b_block, a_format, b_format, output_format,
                     device, capability, sms, a_stride, b_stride, stages, bias_format):
    physical_format = output_format.split(";")[0]
    alignment = 16 if "float8" in physical_format else 4 if physical_format == "torch.float32" else 8
    stride = ((n + alignment - 1) // alignment) * alignment
    scale_k = ((k + MAX_BLOCK_K - 1) // MAX_BLOCK_K) * MAX_BLOCK_K
    ar, ac = a_block or (m, scale_k)
    br, bc = b_block or (n, scale_k)
    args = dict(A_BLOCK_ROWS=ar, A_BLOCK_COLS=ac, B_BLOCK_ROWS=br, B_BLOCK_COLS=bc,
                A_SCALE_COLS=(k + ac - 1) // ac, B_SCALE_COLS=(k + bc - 1) // bc,
                TENSORWISE=a_block is None and b_block is None,
                NATIVE_FP8=all(block is None or block[1] % 32 == 0 for block in (a_block, b_block)),
                HAS_BIAS=bias_format is not None, OUTPUT_FORMAT=output_format,
                A_FP8_FORMAT=a_format, B_FP8_FORMAT=b_format, DEVICE_INDEX=device,
                DEVICE_CAPABILITY=capability, NUM_SMS=sms, OUTPUT_STRIDE=stride,
                PIPELINE_STAGES=0 if stages == "auto" else stages, A_ROW_STRIDE=a_stride, B_ROW_STRIDE=b_stride)
    explicit = {"M": m, "N": n, "K": k, **args}
    dtypes = ("torch.float32", "torch.float32", bias_format or physical_format, "torch.float32", physical_format)
    key = tuple(explicit[name] for name in AUTOTUNE_KEY) + dtypes
    return GemmLaunchPlan(stride, MappingProxyType(args), key)


def selected_config(kernel, key):
    return kernel.configs[0] if len(kernel.configs) == 1 else kernel.cache.get(key)


def launch_selected(kernel, config, args, kwargs, grid):
    """Preserve Config hooks/options; keep post-processing at the call site.

    Inputs/descriptors are always supplied by this invocation. Nothing here
    saves their pointers or closes over activation/output storage.
    """
    if config.pre_hook is not None:
        config.pre_hook({**dict(zip(kernel.arg_names, args)), **kwargs, **config.all_kwargs(), "grid": grid, "warmup": False})
    result = kernel.fn.run(*args, grid=grid, warmup=False, **kwargs, **config.all_kwargs())
    kernel.best_config = config  # Diagnostic compatibility, never used for dispatch.
    return result
