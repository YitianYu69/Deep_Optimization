"""TileLang BatchNorm2d and GroupNorm semantic executors."""

import torch
from deepopt.opt.nn.normalization.geometry import output_strides
from ..activation.common import dense, cast, unpack
from .common import statistics, xhat, compress, affine, input_gradient, affine_gradient


def forward(x, weight, bias, geometry, compression, *, running_mean=None, running_var=None, momentum=None,
            out_dtype=None, output_layout=None, center_shift=None, execution_info=None):
    values = geometry.matrix(dense(x)).contiguous()
    mean, inverse = statistics(values, geometry.eps, batch_stats=geometry.batch_stats,
        running_mean=running_mean, running_var=running_var, momentum=momentum)
    h = xhat(values, mean, inverse)
    packed = compress(h, compression, center_shift)
    output = affine(geometry.restore(h), weight, bias, geometry.channels, geometry.spatial)
    dtype = out_dtype or x.dtype
    output = cast(output, dtype)
    result = torch.empty_strided(geometry.shape, output_strides(x, output_layout), dtype=dtype, device=x.device)
    result.copy_(output)
    if execution_info is not None:
        execution_info.update(backend="tilelang", forward_variant="generic", forward_launches=3,
            launch_counts_complete=False,
            statistics_launches=1, compression_launches=int(packed is not None), recenter_launches=int(center_shift is not None),
            recenter_scratch_bytes=0, recenter_partial_launches=0, recenter_partial_tiles=0,
            recenter_finish_launches=int(center_shift is not None), recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4)
    return result, mean, inverse, packed


def backward(gradient, weight, geometry, *, packed=None, x=None, mean=None, inverse=None,
             need_dx=True, need_dw=True, need_db=True, input_dtype=None, gradient_output=None,
             center_shift=None, execution_info=None):
    grad = dense(gradient)
    if packed is not None:
        h = xhat(unpack(packed), mean, inverse, reconstruct=False, center_shift=center_shift)
    elif x is not None:
        h = xhat(geometry.matrix(dense(x)), mean, inverse)
    else:
        h = torch.empty(geometry.matrix_shape, device=gradient.device)
    dw, db = affine_gradient(grad, geometry.restore(h), geometry.channels, geometry.spatial, need_dw, need_db)
    dx = None
    if need_dx:
        u = affine(grad, weight, None, geometry.channels, geometry.spatial)
        dx = geometry.restore(input_gradient(geometry.matrix(u), h, inverse, geometry.batch_stats))
        dx = gradient_output.convert(dx) if gradient_output is not None else cast(dx, input_dtype or gradient.dtype)
    if execution_info is not None:
        execution_info.update(backend="tilelang", backward_variant="generic", input_gradient_launches=int(need_dx),
            affine_gradient_launches=int(need_dw or need_db), backward_reduction_launches=int(need_dx), backward_scratch_bytes=0)
    return dx, dw, db
