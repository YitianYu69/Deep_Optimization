"""TileLang LayerNorm, including compressed pre-affine state correction."""

import torch
from deepopt.opt.nn.normalization.layer_state import LayerNormState
from ...activation.common import dense, cast, convert, unpack, check_finite as ensure_finite, output_pack as save_output_pack
from ..common import statistics, xhat, compress, affine, input_gradient, affine_gradient
from . import fast


def _cooperative_forward(x, geometry, *, weight, bias, dtype, block_shape, compression, check_finite,
                         scale=None, observe=False, capture_scale=False, center_shift=None,
                         output_pack=None, moment_gain=None):
    native = dtype in fast.NATIVE
    values, mean, inverse, h = fast.forward(x, weight, bias, geometry,
        dtype if native else torch.float32, save_h=compression["enabled"])
    if h is not None and check_finite:
        ensure_finite(h, "LayerNorm xhat compression requires finite normalized values")
    packed = compress(h, compression, center_shift, moment_gain,
        min(compression["group_size"], geometry.elements)) if h is not None else None
    values = geometry.transport.matrix(geometry.restore(values))
    if native:
        output, out_inverse, snapshot, observations = values, None, None, None
    else:
        # FP8 preparation stays a separate selected-family helper. In Mixed,
        # an explicit quantization/scaling pin must not disappear into fusion.
        output, out_inverse, snapshot, observations = convert(values, dtype, block_shape, scale,
            observe, capture_scale, check_finite)
    save_output_pack(output, output_pack)
    decoded_input_bytes = x.numel() * 4 if hasattr(x, "_data") else 0
    info = dict(backend="tilelang", forward_variant="cooperative", forward_launches=1,
        statistics_launches=0, scale_launches=int(out_inverse is not None),
        compression_launches=int(packed is not None), workspace_dtype="torch.float32" if h is not None or not native or decoded_input_bytes else None,
        producer_pack_fused=False, producer_pack_ready=output_pack is not None, launch_counts_complete=False,
        scale_mode="provided" if scale is not None else "current" if out_inverse is not None else "none",
        recenter_launches=int(center_shift is not None), recenter_scratch_bytes=0, recenter_partial_launches=0,
        recenter_partial_tiles=0, recenter_finish_launches=int(center_shift is not None),
        recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4,
        moment_launches=int(moment_gain is not None), moment_scratch_bytes=0,
        decoded_input_workspace_bytes=decoded_input_bytes,
        full_size_xhat_workspace_bytes=0 if h is None else h.numel() * h.element_size())
    return output, out_inverse, LayerNormState(mean, inverse, packed), snapshot, observations, info


def forward(x, geometry, *, weight, bias, dtype, block_shape, compression, check_finite,
            scale=None, observe=False, capture_scale=False, center_shift=None, output_pack=None, moment_gain=None):
    if x.device.type == "cuda" and fast.supported(x, geometry) and dtype in (*fast.NATIVE, torch.float8_e4m3fn, torch.float8_e5m2):
        return _cooperative_forward(x, geometry, weight=weight, bias=bias, dtype=dtype,
            block_shape=block_shape, compression=compression, check_finite=check_finite, scale=scale,
            observe=observe, capture_scale=capture_scale, center_shift=center_shift,
            output_pack=output_pack, moment_gain=moment_gain)
    values = geometry.matrix(dense(x)).contiguous()
    mean, inverse = statistics(values, geometry.eps)
    h = xhat(values, mean, inverse)
    if compression["enabled"] and check_finite:
        ensure_finite(h, "LayerNorm xhat compression requires finite normalized values")
    packed = compress(h, compression, center_shift, moment_gain, min(compression["group_size"], max(geometry.elements, 1)))
    values = affine(h, weight, bias, max(geometry.elements, 1), 1)
    values = geometry.transport.matrix(geometry.restore(values))
    output, out_inverse, snapshot, observations = convert(values, dtype, block_shape, scale, observe,
        capture_scale, check_finite)
    save_output_pack(output, output_pack)
    info = dict(backend="tilelang", forward_variant="generic", forward_launches=int(values.numel() > 0),
        statistics_launches=int(values.numel() > 0), scale_launches=int(out_inverse is not None),
        compression_launches=int(packed is not None), workspace_dtype="torch.float32", producer_pack_fused=False,
        producer_pack_ready=output_pack is not None, launch_counts_complete=False,
        scale_mode="provided" if scale is not None else "current" if out_inverse is not None else "none",
        recenter_launches=int(center_shift is not None), recenter_scratch_bytes=0, recenter_partial_launches=0,
        recenter_partial_tiles=0, recenter_finish_launches=int(center_shift is not None),
        recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4,
        moment_launches=int(moment_gain is not None), moment_scratch_bytes=0)
    return output, out_inverse, LayerNormState(mean, inverse, packed), snapshot, observations, info


def backward(gradient, weight, geometry, *, packed, x, mean, inverse, need_dx, need_dw, need_db,
             input_dtype, gradient_output=None, execution_info=None, center_shift=None, moment_gain=None):
    if (gradient.device.type == "cuda" and fast.supported(gradient, geometry) and input_dtype in fast.NATIVE
            and (not (need_dx or need_dw) or packed is not None or fast.supported(x, geometry))):
        dx, dw, db, workspace = fast.backward(gradient, weight, geometry, packed=packed, x=x, mean=mean,
            inverse=inverse, need_dx=need_dx, need_dw=need_dw, need_db=need_db, input_dtype=input_dtype,
            gradient_output=gradient_output, center_shift=center_shift, moment_gain=moment_gain)
        if execution_info is not None:
            execution_info.update(backend="tilelang", backward_variant="cooperative",
                input_gradient_launches=int(need_dx), affine_gradient_launches=2 * int(need_dw or need_db),
                backward_reduction_launches=int(need_dx), backward_workspace_dtype="torch.float32",
                affine_partial_workspace_bytes=workspace, full_size_weighted_gradient_workspace_bytes=0)
        return dx, dw, db
    dy = geometry.matrix(dense(gradient)).contiguous()
    if not dy.numel():
        dx = torch.empty(geometry.shape, device=gradient.device, dtype=input_dtype) if need_dx else None
        if dx is not None and gradient_output is not None:
            dx = gradient_output.convert(dx)
        dw = torch.zeros(geometry.normalized_shape, device=gradient.device, dtype=torch.float32) if need_dw else None
        db = torch.zeros(geometry.normalized_shape, device=gradient.device, dtype=torch.float32) if need_db else None
        return dx, dw, db
    if need_dx or need_dw:
        h = xhat(unpack(packed) if packed is not None else geometry.matrix(dense(x)), mean, inverse,
            reconstruct=packed is None, center_shift=center_shift, moment_gain=moment_gain)
    else:
        h = torch.empty_like(dy)
    dw, db = affine_gradient(dy, h, max(geometry.elements, 1), 1, need_dw, need_db)
    dw = dw.reshape(geometry.normalized_shape) if dw is not None else None
    db = db.reshape(geometry.normalized_shape) if db is not None else None
    dx = None
    if need_dx:
        u = affine(dy, weight, None, max(geometry.elements, 1), 1)
        dx = geometry.restore(input_gradient(u, h, inverse))
        dx = gradient_output.convert(dx) if gradient_output is not None else cast(dx, input_dtype)
    if execution_info is not None:
        execution_info.update(backend="tilelang", backward_variant="generic", input_gradient_launches=int(need_dx),
            affine_gradient_launches=int(need_dw or need_db), backward_reduction_launches=int(need_dx),
            backward_workspace_dtype="torch.float32")
    return dx, dw, db
