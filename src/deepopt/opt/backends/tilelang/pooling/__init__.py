"""All spatial pooling families using TileLang-owned windows and gradients.

Every backward output owns one input element and gathers its contributions;
overlapping pooling windows therefore require no floating-point atomics.
"""

from functools import lru_cache
from math import isinf, prod
import warnings

import torch

from .._runtime import runtime, compile_kernel
from ..activation.common import dense, cast, convert, pack, unpack, auxiliary_module, check_finite as ensure_finite
from deepopt.opt.nn.activation.geometry import ReLUGeometry
from deepopt.opt.nn.pooling.state import PoolState
from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt.quantization.indices import PackedIndices, empty_indices


def _regular_windows(g):
    """Static max/average windows need neither samples nor a window table."""
    return g.family in ("max", "average")


def _backward_variant(g):
    if not _regular_windows(g):
        return "generic"
    # A dilated window's bounding interval can contain holes, but disjoint
    # bounding intervals still give each input at most one output candidate.
    return ("regular_direct" if all(s >= (k - 1) * d + 1
            for k, s, d in zip(g.kernel, g.stride, g.dilation)) else "regular_bounded")


@lru_cache(maxsize=256)
def _windows_kernel(nc, inputs, outputs, kernel_size, strides, padding, rank, family, device):
    _, T = runtime()
    ov = prod(outputs)

    @T.prim_func
    def execute(Meta: T.Tensor((3, 5), "int64"), Samples: T.Tensor((nc, rank), "float32"),
                Windows: T.Tensor((nc * ov, 6), "int64")):
        with T.Kernel(T.ceildiv(nc * ov, 128), threads=128) as bx:
            for t in T.Parallel(128):
                p = T.cast(bx, "int64") * 128 + t
                if p < nc * ov:
                    for axis in T.serial(3):
                        length, out, size = Meta[axis, 0], Meta[axis, 1], Meta[axis, 2]
                        stride, pad = Meta[axis, 3], Meta[axis, 4]
                        coordinate = T.alloc_var("int64")
                        start = T.alloc_var("int64")
                        extent = T.alloc_var("int64")
                        if axis == 0:
                            coordinate = p % ov // (outputs[1] * outputs[2])
                        elif axis == 1:
                            coordinate = p % ov // outputs[2] % outputs[1]
                        else:
                            coordinate = p % outputs[2]
                        if family == "adaptive_max" or family == "adaptive_average":
                            start = coordinate * length // out
                            extent = ((coordinate + 1) * length + out - 1) // out - start
                        elif family == "fractional_max":
                            if axis < 3 - rank:
                                start = 0
                            elif (out == 1) | (coordinate == out - 1):
                                start = length - size
                            else:
                                sample_axis = 2 - axis if rank == 2 else axis
                                sample = Samples[p // ov, sample_axis]
                                alpha = T.cast(length - size, "float32") / T.cast(out - 1, "float32")
                                start = T.cast((T.cast(coordinate, "float32") + sample) * alpha, "int64") - T.cast(sample * alpha, "int64")
                            extent = size
                        else:
                            start = coordinate * stride - pad
                            extent = size
                        Windows[p, axis] = start
                        Windows[p, axis + 3] = extent

    return compile_kernel(execute, device)


def _windows(g, samples, device):
    nc = g.batch * g.channels
    result = torch.empty((nc * g.output_elements, 6), dtype=torch.int64, device=device)
    meta = torch.tensor(tuple(zip(g.input_spatial, g.output_spatial, g.kernel, g.stride, g.padding)),
        dtype=torch.int64, device=device)
    sample_data = cast(samples, torch.float32).reshape(nc, g.rank).contiguous() if samples is not None else torch.empty(
        (nc, g.rank), dtype=torch.float32, device=device)
    if result.numel():
        _windows_kernel(nc, g.input_spatial, g.output_spatial, g.kernel, g.stride, g.padding, g.rank,
            g.family, str(device))(meta, sample_data, result)
    return result


@lru_cache(maxsize=256)
def _forward_kernel(nc, inputs, outputs, dilation, padding, family, power, include_pad, divisor, window_volume, device,
                    regular_kernel=None, regular_stride=None, write_public=True, write_route=True, write_factor=True):
    _, T = runtime()
    iz, iy, ix = inputs
    iv, ov = prod(inputs), prod(outputs)
    dz, dy, dx = dilation
    pz, py, px = padding
    regular = regular_kernel is not None
    k_depth, k_height, k_width = regular_kernel or (1, 1, 1)
    sz, sy, sx = regular_stride or (1, 1, 1)
    finite_power = family == "power" and not isinf(power)
    infinite_power = family == "power" and isinf(power)
    negative_infinite = infinite_power and power < 0

    @T.prim_func
    def execute(X: T.Tensor((nc * iv,), "float32"), W: T.Tensor((1, 1) if regular else (nc * ov, 6), "int64"),
                Y: T.Tensor((nc * ov,), "float32"), Public: T.Tensor((nc * ov if write_public else 1,), "int64"),
                Route: T.Tensor((nc * ov if write_route else 1,), "int64"),
                Factor: T.Tensor((nc * ov if write_factor else 1,), "float32")):
        with T.Kernel(T.ceildiv(nc * ov, 32), threads=32) as block:
            for lane in T.Parallel(32):
                p = T.cast(block, "int64") * 32 + lane
                total = T.alloc_var("float32")
                best = T.alloc_var("float32")
                code = T.alloc_var("int64")
                selected = T.alloc_var("int64")
                sign = T.alloc_var("int64")
                count = T.alloc_var("int64")
                value = T.alloc_var("float32")
                if p < nc * ov:
                    if regular:
                        z0 = p % ov // (outputs[1] * outputs[2]) * sz - pz
                        y0 = p % ov // outputs[2] % outputs[1] * sy - py
                        x0 = p % outputs[2] * sx - px
                        ez, ey, ex = k_depth, k_height, k_width
                    else:
                        z0, y0, x0 = W[p, 0], W[p, 1], W[p, 2]
                        ez, ey, ex = W[p, 3], W[p, 4], W[p, 5]
                    total = 0.0
                    best = T.infinity("float32") if negative_infinite else -T.infinity("float32")
                    code = 0
                    selected = -1
                    sign = 0
                    count = 0
                    for q in T.serial(ez * ey * ex):
                        zp = z0 + q // (ey * ex) * dz
                        yp = y0 + q // ex % ey * dy
                        xp = x0 + q % ex * dx
                        if (zp >= 0) & (zp < iz) & (yp >= 0) & (yp < iy) & (xp >= 0) & (xp < ix):
                            index = (zp * iy + yp) * ix + xp
                            raw = X[p // ov * iv + index]
                            count += 1
                            if family == "average" or family == "adaptive_average":
                                total += raw
                            elif finite_power:
                                total += T.pow(raw, power)
                            else:
                                value = T.abs(raw) if family == "power" else raw
                                if (selected < 0) | (value != value) | (value < best if negative_infinite else value > best):
                                    best = value
                                    selected = index
                                    code = q
                                    sign = T.if_then_else(raw > 0, 1, T.if_then_else(raw < 0, 2, 0))
                    if family == "average" or family == "adaptive_average":
                        if divisor is not None:
                            Y[p] = total / divisor
                        elif include_pad and family == "average":
                            padded = T.max(0, T.min(z0 + ez, iz + pz) - z0) * T.max(0, T.min(y0 + ey, iy + py) - y0) * T.max(0, T.min(x0 + ex, ix + px) - x0)
                            Y[p] = total / padded
                        else:
                            Y[p] = total / count
                        if write_factor:
                            Factor[p] = 0.0
                    elif finite_power:
                        z = total / count
                        base = z * window_volume
                        Y[p] = T.pow(base, 1.0 / power)
                        if write_factor:
                            Factor[p] = T.if_then_else(T.abs(z) <= 0, 0.0,
                                T.pow(base, 1.0 / power - 1.0) * (window_volume / power) / count)
                    else:
                        Y[p] = best
                        if write_factor:
                            Factor[p] = 0.0
                    if write_public:
                        Public[p] = T.max(selected, 0)
                    if write_route:
                        Route[p] = code * 3 + sign if infinite_power else code

    return compile_kernel(execute, device)


@lru_cache(maxsize=256)
def _coefficient_kernel(n, power, device):
    _, T = runtime()

    @T.prim_func
    def execute(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < n:
                    Y[p] = power * T.pow(X[p], power - 1.0)

    return compile_kernel(execute, device)


@lru_cache(maxsize=256)
def _unpool_kernel(nc, iv, ov, backward, device):
    _, T = runtime()
    count = nc * iv if backward else nc * ov

    @T.prim_func
    def execute(X: T.Tensor((nc * ov if backward else nc * iv,), "float32"),
                Indices: T.Tensor((nc * iv,), "int64"), Y: T.Tensor((count,), "float32"),
                Errors: T.Tensor((nc * iv,), "int32")):
        with T.Kernel(T.ceildiv(count, 32), threads=32) as block:
            for lane in T.Parallel(32):
                p = T.cast(block, "int64") * 32 + lane
                value = T.alloc_var("float32")
                if p < count:
                    if backward:
                        index = Indices[p]
                        if (index >= 0) & (index < ov):
                            Y[p] = X[p // iv * ov + index]
                        else:
                            Y[p] = 0.0
                    else:
                        value = 0.0
                        for q in T.serial(iv):
                            index = Indices[p // ov * iv + q]
                            if (index >= 0) & (index < ov):
                                if index == p % ov:
                                    value = X[p // ov * iv + q]
                            if p % ov == 0:
                                Errors[p // ov * iv + q] = T.cast((index < 0) | (index >= ov), "int32")
                        Y[p] = value

    return compile_kernel(execute, device)


def _pack_routes(matrix, value_count, compressed):
    if not compressed:
        return matrix.contiguous()
    backend_pack = auxiliary_module("activation_codec", "quantization.activation.indices").pack
    result = empty_indices(tuple(matrix.shape), value_count, matrix.device)
    backend_pack(matrix, result)
    return result


def _unpack_routes(routing):
    if not isinstance(routing, PackedIndices):
        return routing
    backend_unpack = auxiliary_module("activation_codec", "quantization.activation.indices").unpack
    result = torch.empty(routing.shape, dtype=torch.int64, device=routing.data.device)
    backend_unpack(routing, result)
    return result


def forward(x, g, *, options, dtype, block_shape, compression, check_finite, need_state,
            input_block_shape=None, samples=None, scale=None, observe=False, capture_scale=False):
    source = dense(x).contiguous()
    nc, iv, ov = g.batch * g.channels, g.input_elements, g.output_elements
    values = torch.empty(g.output_shape, device=x.device, dtype=torch.float32)
    need_public = bool(options.get("return_indices", False))
    need_route = need_state and g.routes and g.family != "unpool"
    need_factor = need_state and g.family == "power" and not isinf(g.power)
    route = torch.empty(g.output_shape if need_route else (1,), device=x.device, dtype=torch.int64)
    public = (torch.empty(g.output_shape, device=x.device, dtype=torch.int64)
              if need_public else route.view(-1)[:1])
    # Unwritten ABI arguments use correctly shaped views of existing storage.
    # No full-size public-index/factor allocation is needed for ordinary max.
    factor = torch.empty(g.output_shape, device=x.device, dtype=torch.float32) if need_factor else values.view(-1)[:1]
    routing = coefficient = None
    coefficient_layout = getattr(x, "_activation_layout", "last")
    if g.family == "unpool":
        if torch.are_deterministic_algorithms_enabled():
            if torch.is_deterministic_algorithms_warn_only_enabled():
                warnings.warn("CUDA max_unpool may be nondeterministic for repeated indices", UserWarning, stacklevel=2)
            else:
                raise RuntimeError("CUDA max_unpool does not have a deterministic implementation for repeated indices")
        indices = options["indices"].contiguous()
        errors = torch.empty(nc * iv, dtype=torch.int32, device=x.device)
        if values.numel():
            _unpool_kernel(nc, iv, ov, False, str(x.device))(source.view(-1), indices.view(-1), values.view(-1), errors)
            if errors.any().item():
                raise RuntimeError("unpool indices outside output bounds")
        if need_state:
            routing = _pack_routes(ReLUGeometry(g.shape, g.layout).matrix(indices), g.route_values, compression["enabled"])
    elif values.numel():
        regular = _regular_windows(g)
        # The analytic specialization never reads W. Reuse an existing scalar
        # view for its ABI rather than allocating any window metadata.
        windows = route.view(-1)[:1].view(1, 1) if regular else _windows(g, samples, x.device)
        dilation = (1, 1, 1) if g.family.startswith("adaptive") or g.family == "fractional_max" else g.dilation
        _forward_kernel(nc, g.input_spatial, g.output_spatial, dilation, g.padding, g.family,
            g.power, g.include_pad, g.divisor, prod(g.kernel), str(x.device),
            g.kernel if regular else None, g.stride if regular else None, need_public, need_route, need_factor)(
                source.view(-1), windows, values.view(-1), public.view(-1), route.view(-1), factor.view(-1))
        if need_state and g.routes:
            routing = _pack_routes(g.transport.matrix(route), g.route_values, compression["enabled"])
        if need_state and g.family == "power" and not isinf(g.power):
            coefficients = torch.empty_like(source)
            _coefficient_kernel(source.numel(), g.power, str(x.device))(source.view(-1), coefficients.view(-1))
            transport = ReLUGeometry(g.shape, coefficient_layout)
            coefficients = transport.matrix(coefficients).contiguous()
            if check_finite and compression["enabled"]:
                ensure_finite(coefficients, "LP derivative compression requires finite preprocessed coefficients")
            blocks = getattr(x, "_block_shape", input_block_shape)
            coefficient = pack(coefficients, compression["group_size"], blocks, compression["bits"]) if compression["enabled"] else coefficients
    output, inverse, snapshot, observations = convert(g.transport.matrix(values), dtype, block_shape, scale,
        observe, capture_scale, check_finite)
    public = public if need_public else None
    factor = g.transport.matrix(factor).contiguous() if need_factor else None
    state = PoolState(routing, coefficient, factor, public, samples if need_state else None, coefficient_layout)
    info = dict(backend="tilelang", forward_variant="regular_analytic" if _regular_windows(g) else "generic",
        forward_launches=int(values.numel() > 0),
        launch_counts_complete=False,
        window_launches=int(values.numel() > 0 and g.family != "unpool" and not _regular_windows(g)),
        window_metadata_bytes=0 if _regular_windows(g) or g.family == "unpool" else nc * ov * 6 * 8,
        public_indices_buffer_bytes=nc * ov * 8 if need_public else 0,
        routing_buffer_bytes=nc * ov * 8 if need_route else 0,
        factor_buffer_bytes=nc * ov * 4 if need_factor else 0,
        packing_launches=int(compression["enabled"] and need_state), output_cast_launches=1, scale_launches=int(inverse is not None),
        scratch_bytes=0, workspace_dtype="torch.float32", deterministic_strategy="input_owned_gather",
        scale_mode="provided" if scale is not None else "current" if inverse is not None else "none")
    return output, inverse, state, snapshot, observations, info


@lru_cache(maxsize=256)
def _backward_kernel(nc, inputs, outputs, dilation, padding, family, power, include_pad, divisor, device,
                     regular_kernel=None, regular_stride=None, compact_buffers=False):
    _, T = runtime()
    iz, iy, ix = inputs
    iv, ov = prod(inputs), prod(outputs)
    dz, dy, dx = dilation
    pz, py, px = padding
    regular = regular_kernel is not None
    k_depth, k_height, k_width = regular_kernel or (1, 1, 1)
    sz, sy, sx = regular_stride or (1, 1, 1)
    finite_power = family == "power" and not isinf(power)
    has_routes = family in ("max", "adaptive_max", "fractional_max") or family == "power" and isinf(power)
    route_count = nc * ov if has_routes or not compact_buffers else 1
    coefficient_count = nc * iv if finite_power or not compact_buffers else 1
    factor_count = nc * ov if finite_power or not compact_buffers else 1

    @T.prim_func
    def execute(D: T.Tensor((nc * ov,), "float32"), W: T.Tensor((1, 1) if regular else (nc * ov, 6), "int64"),
                Route: T.Tensor((route_count,), "int64"), A: T.Tensor((coefficient_count,), "float32"),
                Factor: T.Tensor((factor_count,), "float32"), Y: T.Tensor((nc * iv,), "float32")):
        with T.Kernel(T.ceildiv(nc * iv, 32), threads=32) as block:
            for lane in T.Parallel(32):
                p = T.cast(block, "int64") * 32 + lane
                total = T.alloc_var("float32")
                count = T.alloc_var("int64")
                if p < nc * iv:
                    zp, yp, xp = p % iv // (iy * ix), p % iv // ix % iy, p % ix
                    total = 0.0
                    if regular:
                        # Clamp before ceil-division so no negative division
                        # semantics enter the inverse window bounds.
                        lz = T.min(outputs[0], (T.max(0, zp + pz - (k_depth - 1) * dz) + sz - 1) // sz)
                        ly = T.min(outputs[1], (T.max(0, yp + py - (k_height - 1) * dy) + sy - 1) // sy)
                        lx = T.min(outputs[2], (T.max(0, xp + px - (k_width - 1) * dx) + sx - 1) // sx)
                        hz = T.min(outputs[0], (zp + pz) // sz + 1)
                        hy = T.min(outputs[1], (yp + py) // sy + 1)
                        hx = T.min(outputs[2], (xp + px) // sx + 1)
                    else:
                        lz, ly, lx = 0, 0, 0
                        hz, hy, hx = outputs
                    cz, cy, cx = hz - lz, hy - ly, hx - lx
                    # Empty candidate ranges occur in stride gaps/tails. Keep
                    # their address expressions defined even if hoisted by a
                    # lowering pass out of the zero-trip loop below.
                    cy_safe, cx_safe = T.max(cy, 1), T.max(cx, 1)
                    # Z/Y/X ascending order is the same order as the generic
                    # full-output scan, preserving deterministic accumulation.
                    for q in T.serial(cz * cy * cx):
                        oz, oy, ox = lz + q // (cy_safe * cx_safe), ly + q // cx_safe % cy_safe, lx + q % cx_safe
                        o = p // iv * ov + (oz * outputs[1] + oy) * outputs[2] + ox
                        if regular:
                            z0, y0, x0 = oz * sz - pz, oy * sy - py, ox * sx - px
                            ez, ey, ex = k_depth, k_height, k_width
                        else:
                            z0, y0, x0 = W[o, 0], W[o, 1], W[o, 2]
                            ez, ey, ex = W[o, 3], W[o, 4], W[o, 5]
                        rz, ry, rx = zp - z0, yp - y0, xp - x0
                        valid = (rz >= 0) & (ry >= 0) & (rx >= 0) & (rz % dz == 0) & (ry % dy == 0) & (rx % dx == 0) & (rz // dz < ez) & (ry // dy < ey) & (rx // dx < ex)
                        if valid:
                            if family == "average" or family == "adaptive_average":
                                count = 0
                                for k in T.serial(ez * ey * ex):
                                    kz = z0 + k // (ey * ex) * dz
                                    ky = y0 + k // ex % ey * dy
                                    kx = x0 + k % ex * dx
                                    if (kz >= 0) & (kz < iz) & (ky >= 0) & (ky < iy) & (kx >= 0) & (kx < ix):
                                        count += 1
                                if divisor is not None:
                                    total += D[o] / divisor
                                elif include_pad and family == "average":
                                    padded = T.max(0, T.min(z0 + ez, iz + pz) - z0) * T.max(0, T.min(y0 + ey, iy + py) - y0) * T.max(0, T.min(x0 + ex, ix + px) - x0)
                                    total += D[o] / padded
                                else:
                                    total += D[o] / count
                            elif finite_power:
                                total += D[o] * Factor[o]
                            else:
                                code = Route[o] // 3 if family == "power" else Route[o]
                                local = ((rz // dz) * ey + ry // dy) * ex + rx // dx
                                if local == code:
                                    if family == "power":
                                        sign = T.if_then_else(Route[o] % 3 == 1, 1.0, T.if_then_else(Route[o] % 3 == 2, -1.0, 0.0))
                                        total += D[o] * sign
                                    else:
                                        total += D[o]
                    Y[p] = total * A[p] if finite_power else total

    return compile_kernel(execute, device)


def backward(gradient, g, *, routing, coefficient, factor, samples, coefficient_layout,
             input_dtype, gradient_output=None, execution_info=None):
    dy = dense(gradient).contiguous()
    nc, iv, ov = g.batch * g.channels, g.input_elements, g.output_elements
    result = torch.empty(g.shape, dtype=torch.float32, device=gradient.device)
    if result.numel():
        if g.family == "unpool":
            codes = ReLUGeometry(g.shape, g.layout).restore(_unpack_routes(routing)).contiguous()
            errors = torch.empty(nc * iv, dtype=torch.int32, device=gradient.device)
            _unpool_kernel(nc, iv, ov, True, str(gradient.device))(dy.view(-1), codes.view(-1), result.view(-1), errors)
        else:
            regular = _regular_windows(g)
            finite_power = g.family == "power" and not isinf(g.power)
            codes = (g.transport.restore(_unpack_routes(routing)).contiguous() if routing is not None else
                     torch.empty(g.output_shape if g.routes else (1,), dtype=torch.int64, device=gradient.device))
            windows = codes.view(-1)[:1].view(1, 1) if regular else _windows(g, samples, gradient.device)
            coefficients = unpack(coefficient) if isinstance(coefficient, PackedActivation) else coefficient
            a = (ReLUGeometry(g.shape, coefficient_layout).restore(coefficients).contiguous() if coefficients is not None else
                 torch.empty_like(result) if finite_power else result.view(-1)[:1])
            f = (g.transport.restore(factor).contiguous() if factor is not None else
                 torch.empty(g.output_shape, device=gradient.device) if finite_power else result.view(-1)[:1])
            dilation = (1, 1, 1) if g.family.startswith("adaptive") or g.family == "fractional_max" else g.dilation
            _backward_kernel(nc, g.input_spatial, g.output_spatial, dilation, g.padding, g.family,
                g.power, g.include_pad, g.divisor, str(gradient.device),
                g.kernel if regular else None, g.stride if regular else None, True)(
                    dy.view(-1), windows, codes.view(-1), a.view(-1), f.view(-1), result.view(-1))
    if execution_info is not None:
        execution_info.update(backend="tilelang", backward_launches=int(result.numel() > 0),
            backward_workspace_dtype="torch.float32", backward_variant=_backward_variant(g),
            backward_window_launches=int(result.numel() > 0 and g.family != "unpool" and not _regular_windows(g)),
            deterministic_strategy="input_owned_gather")
    return gradient_output.convert(result) if gradient_output is not None else cast(result, input_dtype)


__all__ = ["forward", "backward"]
