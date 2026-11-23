"""Per-invocation codec experiments; policies never retain tensor storage."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_packing = ContextVar("deepopt_activation_packing", default=("auto", 4))
_spatial_layout = ContextVar("deepopt_spatial_activation_layout", default="auto")
_pack_warps = ContextVar("deepopt_activation_pack_warps", default=4)
_pointwise_tile = ContextVar("deepopt_activation_pointwise_tile", default=512)
_producer_pack = ContextVar("deepopt_producer_pack", default="off")
_gelu_state = ContextVar("deepopt_gelu_saved_state", default="auto")
_silu_state = ContextVar("deepopt_silu_saved_state", default="derivative")
_strided_index = ContextVar("deepopt_strided_indexing", default="wide")
_convolution_codec = ContextVar("deepopt_convolution_codec",default="auto")
_attention_pack = ContextVar("deepopt_attention_packing", default="auto")
_attention_grad_layout = ContextVar("deepopt_attention_grad_layout", default="auto")
_activation_layout = ContextVar("deepopt_activation_layout", default="auto")


def activation_layout_policy():
    return _activation_layout.get()


def saved_activation_layout(function):
    @wraps(function)
    def call(ctx, *args):
        policy = ctx.activation_kernel_policy
        if policy == activation_layout_policy():
            return function(ctx, *args)
        token = _activation_layout.set(policy)
        try:
            return function(ctx, *args)
        finally:
            _activation_layout.reset(token)
    return call


def channels_last_activation_plan(shape, strides, group, device):
    """Coalesce channels sharing a codec phase, without changing logical groups."""
    from math import gcd, prod
    policy = activation_layout_policy()
    if (policy == "baseline" or len(shape) != 4 or group < 32 or group > 2048 or group & (group - 1)
            or strides != (prod(shape[1:]), 1, shape[3] * shape[1], shape[1])):
        return None
    spatial = prod(shape[2:])
    if not spatial:
        return None
    period = group // gcd(spatial, group)
    if period > 4:
        # Small spatial maps need many channel phases (49 and 1024 are
        # coprime). Own the complete original flat group in a bounded 2-D
        # tile instead; masking and local code reordering preserve the codec.
        channel_span = (group + spatial - 1) // spatial + 1
        padded_spatial = 1 << (spatial - 1).bit_length()
        padded_channels = 1 << (channel_span - 1).bit_length()
        if policy != "channels_last" or spatial > 64 or padded_spatial * padded_channels > 8192:
            return None
        return {"variant": "group_owned", "spatial_tile": padded_spatial,
                "channel_tile": padded_channels, "period": period}
    # Automatic rollout needs a measured complete-pipeline win. The explicit
    # candidate remains available to the GPU screen in the meantime.
    if policy != "channels_last":
        return None
    return {"variant": "phased", "period": period, "channel_tile": 4}


def attention_packing():
    return _attention_pack.get()


def attention_gradient_layout():
    return _attention_grad_layout.get()


def attention_gradient_plan(shape, strides, dtype, device, compression=None):
    requested = attention_gradient_layout()
    if requested == "preserve":
        return "auto"
    if requested != "auto":
        return requested
    if (tuple(shape) == (512,197,12,64) and all(tuple(s)==(151296,64,12608,1) for s in strides)
            and str(dtype)=="torch.bfloat16" and compression is not None and compression.get("enabled",False)
            and (compression.get("attention_bits") or max(4,compression.get("bits",2)))==4
            and compression["group_size"]==1024):
        from .normalization import measured_norm_device
        if measured_norm_device(device):
            return "bshd"
    return "auto"


def attention_pack_plan(shape, strides, dtype, bits, group_size, device):
    """Select only measured H100 domains; this never reads tensor values.

    Forced separate/joint settings retain the caller's warp/index controls.
    Auto owns those controls only for the measured B512/T197/H12/D64 domain.
    Dense BSHD keeps the substantially faster ordinary contiguous packer.
    """
    requested = attention_packing()
    result = {"implementation": "separate" if requested == "auto" else requested,
              "num_warps": None, "indexing": None, "reason": "explicit" if requested != "auto" else "unmeasured"}
    if requested != "auto" or tuple(shape) != (512, 197, 12, 64) or str(dtype) != "torch.bfloat16":
        return result
    if bits not in (4, 8) or group_size != 1024 or getattr(device, "type", None) != "cuda":
        return result
    layouts = {(453888, 2304, 64, 1): "interleaved", (151296, 64, 12608, 1): "bhsd",
               (151296, 768, 64, 1): "bshd"}
    layout = layouts.get(tuple(strides))
    if layout is None:
        return result
    from .normalization import measured_norm_device

    if not measured_norm_device(device):
        return result
    result.update(implementation="separate" if layout == "bshd" else "joint",
                  num_warps=2 if layout == "bshd" or bits == 8 else 4,
                  indexing="bounded", reason="h100_recovery_measured_" + layout)
    return result


@contextmanager
def attention_pack_dispatch(selection):
    """Temporarily apply the selected launch settings to QKV packing only."""
    warp = _pack_warps.set(selection["num_warps"]) if selection["num_warps"] is not None else None
    index = _strided_index.set(selection["indexing"]) if selection["indexing"] is not None else None
    try:
        yield
    finally:
        if index is not None:
            _strided_index.reset(index)
        if warp is not None:
            _pack_warps.reset(warp)


def convolution_codec():
    return _convolution_codec.get()
_dispatches = ContextVar("deepopt_codec_dispatches", default=None)


@contextmanager
def record_codec_dispatches():
    records = []
    token = _dispatches.set(records)
    try:
        yield records
    finally:
        _dispatches.reset(token)


def observe_pack(x, shape, group, bits, block_shape, implementation, batch, warps, source):
    records = _dispatches.get()
    if records is not None:
        records.append({"operation":"pack", "shape":tuple(shape), "source_shape":tuple(x.shape),
                        "source_strides":tuple(x.stride()), "dtype":str(x.dtype), "bits":bits,
                        "group_size":group, "block_shape":block_shape, "implementation":implementation,
                        "groups_per_program":batch, "num_warps":warps,
                        "source_narrow":source.get("SOURCE_NARROW",False)})
        records[-1]["rounded_source_dtype"]=str(source["SOURCE_DTYPE"]) if "SOURCE_DTYPE" in source else None


def observe_producer_pack(x, packed, group, bits, block_shape):
    records = _dispatches.get()
    if records is not None:
        from .gemm_diagnostics import _module
        records.append({"operation":"producer_pack_consume", "module":_module.get(),
                        "shape":tuple(x._data.shape), "dtype":str(x._data.dtype),
                        "group_size":group,"bits":bits,"block_shape":block_shape,"reused":packed is not None})


def strided_indexing():
    return _strided_index.get()


def narrow_source_indices(shape, strides):
    from math import prod
    return (strided_indexing() == "bounded" and prod(shape) < 2**31
            and all(s >= 0 for s in strides) and sum((d-1)*s for d,s in zip(shape,strides)) < 2**31)


def producer_packing():
    return _producer_pack.get()


def gelu_saved_state():
    return _gelu_state.get()


def silu_saved_state():
    return _silu_state.get()


def spatial_row_alignment(geometry, dtype, block_shape, compression, device, *, delayed=False):
    policy = spatial_activation_layout()
    if policy == "compact":
        return 1 if not geometry.pointwise else 16
    if policy == "aligned" or delayed or not compression.get("enabled", False):
        return 16
    # V4 measured identical operands at these exact regular-convolution domains.
    eligible = (tuple(geometry.input_shape) in tuple((512, c, 224, 224) for c in (1, 3, 4))
                and geometry.out_channels == 768 and tuple(geometry.kernel_size) == (16, 16)
                and tuple(geometry.stride) == (16, 16) and tuple(geometry.padding) == (0, 0)
                and tuple(geometry.dilation) == (1, 1) and geometry.groups == 1
                and not geometry.synthetic_padding_right and str(dtype) == "torch.float8_e4m3fn"
                and block_shape is None and compression.get("bits", 2) == 2
                and compression.get("group_size") == 1024)
    if eligible:
        from .normalization import measured_norm_device
        if measured_norm_device(device):
            return 1
    return 16


def select_pack_warps(requested, shape, dtype, bits, group, block_shape, device, *, contiguous, strided=False, grouped=True):
    if requested != "auto":
        if type(requested) is not int or requested not in (1, 2, 4, 8):
            raise ValueError("num_warps must be auto, 1, 2, 4, or 8")
        return requested
    if (grouped and not strided and contiguous and block_shape is None and bits == 2 and group in (512, 1024)
            and tuple(shape) in ((100864, 768), (100864, 3072))
            and str(dtype) in ("torch.bfloat16", "torch.float8_e4m3fn")):
        from .normalization import measured_norm_device
        if measured_norm_device(device):
            return 1 if group == 512 else 2
    return 4


def activation_packing():
    return _packing.get()


def spatial_activation_layout():
    return _spatial_layout.get()


def activation_pack_warps():
    return _pack_warps.get()


def activation_pointwise_tile():
    return _pointwise_tile.get()


@contextmanager
def activation_codec_scope(pack="auto", *, groups_per_program=4, spatial_layout="auto",
                           num_warps=4, pointwise_tile=512, producer_pack="off", gelu_state="auto",
                           strided_index="wide",convolution_codec="auto", attention_pack="auto",
                           attention_grad_layout="auto", activation_layout="auto", silu_state="derivative"):
    """Batched packing is opt-in until complete-update GPU acceptance."""
    if pack not in ("auto", "grouped", "batched", "serial"):
        raise ValueError("pack must be auto, grouped, batched, or serial")
    if type(groups_per_program) is not int or groups_per_program not in (1, 2, 4, 8):
        raise ValueError("groups_per_program must be 1, 2, 4, or 8")
    if spatial_layout not in ("auto", "aligned", "compact"):
        raise ValueError("spatial_layout must be auto, aligned or compact")
    if num_warps != "auto" and (type(num_warps) is not int or num_warps not in (1, 2, 4, 8)):
        raise ValueError("num_warps must be auto, 1, 2, 4, or 8")
    if producer_pack not in ("off", "gelu", "layer_norm", "all") or gelu_state not in ("auto", "derivative", "recompute"):
        raise ValueError("producer_pack must be off/gelu/layer_norm/all and gelu_state auto/derivative/recompute")
    if silu_state not in ("derivative", "recompute"):
        raise ValueError("silu_state must be derivative or recompute")
    if strided_index not in ("wide", "bounded"):
        raise ValueError("strided_index must be wide or bounded")
    if convolution_codec not in ("auto","direct","materialized"):
        raise ValueError("convolution_codec must be auto, direct or materialized")
    if attention_pack not in ("auto", "separate", "joint"):
        raise ValueError("attention_pack must be auto, separate or joint")
    if attention_grad_layout not in ("auto", "preserve", "bshd", "bhsd", "interleaved"):
        raise ValueError("attention_grad_layout must be auto, preserve, bshd, bhsd or interleaved")
    if activation_layout not in ("auto", "baseline", "channels_last"):
        raise ValueError("activation_layout must be auto, baseline or channels_last")
    if type(pointwise_tile) is not int or pointwise_tile not in (512, 1024, 2048, 4096):
        raise ValueError("pointwise_tile must be 512, 1024, 2048, or 4096")
    gradient_token = _attention_grad_layout.set(attention_grad_layout)
    attention_token = _attention_pack.set(attention_pack)
    token = _packing.set((pack, groups_per_program))
    spatial_token = _spatial_layout.set(spatial_layout)
    warp_token = _pack_warps.set(num_warps)
    tile_token = _pointwise_tile.set(pointwise_tile)
    producer_token = _producer_pack.set(producer_pack)
    state_token = _gelu_state.set(gelu_state)
    silu_token = _silu_state.set(silu_state)
    index_token = _strided_index.set(strided_index)
    convolution_token=_convolution_codec.set(convolution_codec)
    activation_layout_token = _activation_layout.set(activation_layout)
    try:
        yield
    finally:
        _activation_layout.reset(activation_layout_token)
        _attention_grad_layout.reset(gradient_token)
        _attention_pack.reset(attention_token)
        _convolution_codec.reset(convolution_token)
        _strided_index.reset(index_token)
        _gelu_state.reset(state_token)
        _silu_state.reset(silu_token)
        _producer_pack.reset(producer_token)
        _packing.reset(token)
        _spatial_layout.reset(spatial_token)
        _pack_warps.reset(warp_token)
        _pointwise_tile.reset(tile_token)
