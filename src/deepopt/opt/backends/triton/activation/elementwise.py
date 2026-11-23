"""Bounded launch orchestration for fused nonlinear activation kernels."""

from math import prod

import torch
import triton

from .elementwise_forward import _activation_forward, _statistics, _reduce_ranges
from .elementwise_backward import _activation_backward
from ..quantization.fp8 import _reduce_amax, _block_scales
from ..quantization.activation.affine import _finish_min_max
from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.backends.reference.activation.elementwise import allocate_saved
from deepopt.opt.nn.activation.state import SavedActivation


def forward(x, geometry, **options):
    with torch.cuda.device(x.device):
        return _forward(x, geometry, **options)


def _forward(x, geometry, *, kind, approximate, dtype, block_shape, packed, bits, group_size,
             save_mask, inplace, reuse=False, scale=None, observe=False, check_finite=False, capture_scale=False,
             recompute=False, output_pack=None, recompute_input=None):
    data, inverse, meta = tensor_arguments(x)
    rows, cols = geometry.matrix_shape
    count = rows * cols
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    maximum = torch.finfo(dtype).max if fp8 else 0.0
    obr, obc = block_shape or (max(1, rows), max(1, cols))
    sr, sc = (1, 1) if block_shape is None else (triton.cdiv(rows, obr), triton.cdiv(cols, obc))
    output = data if inplace else torch.empty((rows, cols), dtype=dtype, device=x.device)
    out_inverse = (inverse if reuse else torch.empty((sr, sc), dtype=torch.float32, device=x.device)) if fp8 else None
    # Recompute addresses saved inputs in the original logical order. An
    # explicit SiLU experiment may retain a BF16 contiguous copy of NHWC
    # input; count that copy rather than silently saving strided storage.
    recompute_data = (geometry.matrix(recompute_input) if recompute_input is not None else data.contiguous()) if recompute else None
    saved = (SavedActivation(recompute_data, recompute=(kind, approximate)) if recompute else
             allocate_saved(geometry, x.device, packed=packed, bits=bits, group_size=group_size,
                            block_shape=block_shape, mask=kind == "relu6")) if save_mask else None
    info = {"backend": "triton", "activation": kind, "forward_launches": int(count > 0),
            "scale_launches": 0, "coefficient_launches": 0,
            "backward_launches": int(save_mask and count > 0),
            "scale_mode": "reuse" if reuse else "provided" if scale is not None else "current" if fp8 else "none",
            "saved_state": ("mask" if kind == "relu6" else "derivative") if save_mask else "none"}
    info["producer_pack_fused"] = output_pack is not None
    if recompute:
        info["saved_state"] = "input_recompute"
        info["recompute_input_copy_bytes"] = (data.numel() * data.element_size()
                                               if recompute_data is not data else 0)
    if not count:
        return output, out_inverse, saved, None, None, info

    # ReLU6 exact mask words own flat aligned ranges. Affine saved coefficients
    # instead own the configured flat groups or activation scaling rectangles.
    codec = save_mask and packed and kind != "relu6"
    from deepopt.opt.backends._shared.activation_codec import activation_pointwise_tile
    tiled = block_shape is not None and (kind != "relu6" or not (save_mask and packed))
    br, bc = block_shape if tiled else (0, 0)
    group = br * bc if br else group_size if codec else activation_pointwise_tile()
    groups = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if br else triton.cdiv(count, group)
    block = min(4096 if group > 16384 else 16384, max(32, triton.next_power_of_2(group)))
    parts = triton.cdiv(group, block)
    precomputed = codec and parts > 1
    if precomputed:
        lo = torch.empty((groups, parts), device=x.device, dtype=torch.float32)
        hi = torch.empty_like(lo)
        _statistics[(groups * parts,)](data, inverse, lo, hi, rows, cols, geometry.shape,
                                    geometry.layout == "channels", meta, kind, approximate,
                                    group, br, bc, parts, True, block, FLAT_GRID=True, enable_fp_fusion=False)
        info["coefficient_launches"] += 1
        n = parts
        while n > 1024:
            nxt = triton.cdiv(n, 1024)
            nlo = torch.empty((groups, nxt), dtype=torch.float32, device=x.device)
            nhi = torch.empty_like(nlo)
            _reduce_ranges[(groups * nxt,)](lo, hi, nlo, nhi, n, nxt, 1024, FLAT_GRID=True)
            lo, hi, n = nlo, nhi, nxt
            info["coefficient_launches"] += 1
        _finish_min_max[(groups,)](lo, hi, saved.step, saved.minimum, n, bits,
                                  triton.next_power_of_2(n), enable_fp_fusion=False)
        info["coefficient_launches"] += 1

    # Fuse a current scale only when this program contains its entire domain.
    current = (fp8 and not reuse and scale is None and parts == 1 and
               (block_shape is None and groups == 1 or block_shape is not None and (br, bc) == block_shape))
    if fp8 and not reuse and scale is None and not current:
        n = triton.cdiv(obr * obc, 4096)
        partial = torch.empty((sr * sc, n), device=x.device, dtype=torch.float32)
        # A tensorwise ViT MLP activation has 75,648 partials. CUDA limits
        # grid.y to 65,535; keep logical groups/parts on the primary axis.
        _statistics[(sr * sc * n,)](data, inverse, partial, partial, rows, cols, geometry.shape,
                                  geometry.layout == "channels", meta, kind, approximate,
                                  obr * obc, obr, obc, n, False, 4096, FLAT_GRID=True, enable_fp_fusion=False)
        info["scale_launches"] += 1
        while n > 1024:
            nxt = triton.cdiv(n, 1024)
            reduced = torch.empty((sr * sc, nxt), device=x.device, dtype=torch.float32)
            _reduce_amax[(sr * sc * nxt,)](partial, reduced, n, nxt, BLOCK=1024)
            partial, n = reduced, nxt
            info["scale_launches"] += 1
        scale = torch.empty((sr, sc), device=x.device, dtype=torch.float32)
        _block_scales[(sr * sc,)](partial, scale, out_inverse, n,
                                 BLOCK=triton.next_power_of_2(n), FP8_MAX=maximum)
        info["scale_launches"] += 1
    observations = torch.empty(groups * parts, device=x.device, dtype=torch.float32) if observe else None
    snapshot = torch.empty(2, device=x.device, dtype=torch.float32) if capture_scale else None
    errors = torch.empty(groups * parts, device=x.device, dtype=torch.int32) if check_finite else output
    words = saved.data.shape[1] if codec else 0
    channel_input = (meta[0] == 0 or meta[0] == 1 and x._block_shape is None and data.is_contiguous())
    if (channel_input and not inplace and geometry.layout == "last" and block_shape is None
            and kind != "relu6" and not observe and not capture_scale and not recompute and output_pack is None
            and (not fp8 or scale is not None)):
        from deepopt.opt.backends._shared.activation_codec import channels_last_activation_plan
        physical_strides = ((geometry.shape[1] * prod(geometry.shape[2:]), 1,
                             geometry.shape[3] * geometry.shape[1], geometry.shape[1])
                            if meta[0] == 1 and len(geometry.shape) == 4 else tuple(x.stride()))
        selection = channels_last_activation_plan(tuple(x.shape), physical_strides, group, x.device)
        if selection is not None:
            from . import channels_last as channel_kernels
            if not fp8:
                output = output.reshape(geometry.shape[0], *geometry.shape[2:], geometry.shape[1]).movedim(-1, 1)
            spatial = prod(geometry.shape[2:])
            period, channels = selection["period"], selection["channel_tile"]
            precision = dict(input_inverse=inverse, scale=scale if fp8 else output,
                             output_inverse=out_inverse if fp8 else output,
                             INPUT_FP8=meta[0] == 1, OUTPUT_FP8=fp8, MAXIMUM=maximum)
            if selection["variant"] == "group_owned":
                channel_kernels._channels_last_group_activation[(groups,)](
                    data, output, saved.data if saved else output, saved.step if codec else output,
                    saved.minimum if codec else output, errors, geometry.shape[1], spatial, count,
                    group, words, kind, approximate, save_mask, codec, bits, check_finite,
                    selection["spatial_tile"], channels, block,
                    **precision, num_warps=4, enable_fp_fusion=False)
            else:
                channel_programs = triton.cdiv(geometry.shape[1], period * channels)
                channel_kernels._channels_last_activation[(geometry.shape[0] * period * channel_programs, triton.cdiv(spatial, group))](
                    data, output, saved.data if saved else output, saved.step if codec else output,
                    saved.minimum if codec else output, errors, geometry.shape[0], geometry.shape[1], spatial,
                    group, groups, words, period, channel_programs, kind, approximate, save_mask, codec, bits,
                    check_finite, block, channels,
                    **precision, num_warps=4, enable_fp_fusion=False)
            if check_finite and errors.any().item():
                raise ValueError("Activation compression requires finite coefficients")
            info.update(forward_variant="channels_last_group_owned" if selection["variant"] == "group_owned"
                        else "channels_last_grouped", channel_phase=period, channel_tile=channels,
                        channel_input_fp8=meta[0] == 1)
            return output, out_inverse, saved, None, None, info
    _activation_forward[(groups * parts,)](
        data, inverse, output, saved.data if saved else output,
        saved.step if codec else output, saved.minimum if codec else output,
        scale if scale is not None else output, out_inverse if fp8 else output,
        observations if observe else output, snapshot if capture_scale else output, errors,
        rows, cols, geometry.shape, geometry.layout == "channels", meta, kind, approximate,
        group, br, bc, words, save_mask, packed, bits, precomputed, fp8, reuse, current, obr, obc, sc,
        maximum, inplace, observe, capture_scale, check_finite, parts, block,
        output_words=output_pack.data if output_pack is not None else output,
        output_steps=output_pack.step if output_pack is not None else output,
        output_minima=output_pack.minimum if output_pack is not None else output,
        OUTPUT_PACK=output_pack is not None,
        num_warps=8 if block >= 8192 else 4, FLAT_GRID=True, SAVE_INPUT=recompute, enable_fp_fusion=False)
    if check_finite and errors.any().item():
        raise ValueError("Activation casting/compression requires finite values and positive finite scales")
    if inplace and meta[0] > 0:
        torch.autograd.graph.increment_version(data)
    return output, out_inverse, saved, snapshot, observations, info


def backward(gradient, saved, geometry, *, dtype, gradient_output=None, pointwise_tile=512):
    with torch.cuda.device(gradient.device):
        out = gradient_output
        if not gradient.numel():
            empty = torch.empty(geometry.matrix_shape, device=gradient.device, dtype=dtype)
            return out.convert(geometry.restore(empty)) if out is not None else empty
        fused = out is not None and not out.checked and (
            out.scale is not None or gradient.numel() <= 4096
            or out.block_shape is not None and out.block_shape[0] * out.block_shape[1] <= 16384)
        if out is not None and not fused:
            dense = backward(gradient, saved, geometry, dtype=torch.float32, pointwise_tile=pointwise_tile)
            return out.convert(geometry.restore(dense))
        target = ReLUGeometry(geometry.shape, out.layout) if fused else geometry
        rows, cols = target.matrix_shape
        obr, obc = (out.block_shape or (rows, cols)) if fused else (1, 1)
        current = fused and out.scale is None
        block = max(32, triton.next_power_of_2(obr * obc)) if current else pointwise_tile
        programs = triton.cdiv(rows, obr) * triton.cdiv(cols, obc) if current else triton.cdiv(rows * cols, block)
        if fused:
            out.allocate(rows, cols, gradient.device, programs)
        output = torch.empty((rows, cols), device=gradient.device, dtype=out.dtype if fused else dtype)
        data, inverse, meta = tensor_arguments(gradient)
        c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
        br, bc = saved.block_shape or (0, 0)
        words = saved.data.shape[1] if saved.packed and not saved.mask else 0
        if (not fused and geometry.layout == "last" and saved.block_shape is None and saved.recompute is None
                and len(geometry.shape) == 4 and meta[0] == 0):
            from deepopt.opt.backends._shared.activation_codec import channels_last_activation_plan
            selection = channels_last_activation_plan(tuple(gradient.shape), tuple(gradient.stride()),
                                                       saved.group_size, gradient.device)
            if selection is not None:
                from .channels_last import _channels_last_activation_dx
                output = output.reshape(geometry.shape[0], *geometry.shape[2:], geometry.shape[1]).movedim(-1, 1)
                _channels_last_activation_dx[(triton.cdiv(gradient.numel(), 1024),)](data, saved.data,
                    saved.step if saved.step is not None else output, saved.minimum if saved.minimum is not None else output,
                    output, c, s, gradient.numel(), geometry.matrix_shape[1], saved.group_size, words,
                    saved.bits, saved.mask, saved.packed, 1024, num_warps=4, enable_fp_fusion=False)
                return output
        _activation_backward[(programs,)](
            data, inverse, saved.data, saved.step if saved.step is not None else output,
            saved.minimum if saved.minimum is not None else output, output,
            out.scale if fused and out.scale is not None else inverse,
            out.inverse if fused else output, out.observations if fused and out.observations is not None else output,
            rows, cols, geometry.shape, meta, target.layout == "channels", geometry.layout == "channels",
            c, s, geometry.matrix_shape[1], saved.group_size, br, bc, words, saved.bits, saved.mask, saved.packed,
            bool(fused), bool(current), obr, obc, bool(fused and out.observations is not None),
            torch.finfo(out.dtype).max if fused else 0.0, block,
            RECOMPUTE=saved.recompute is not None, KIND=saved.recompute[0] if saved.recompute else "gelu",
            APPROXIMATE=saved.recompute[1] if saved.recompute else "none",
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
        return out.wrap(output) if fused else output
