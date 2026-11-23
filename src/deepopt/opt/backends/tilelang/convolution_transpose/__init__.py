"""TileLang transposed convolution reuses adjoint implicit FP8 kernels."""


def _pointwise(geometry, *operands):
    return geometry.pointwise and all(q.block_shape is None for q in operands)


def _gemm(a, b, backend, out_dtype, num_stages, output=None, bias=None, autotune=True):
    from deepopt.opt.backends import gemm_nt
    fused = output if backend in ("tilelang", "triton") else None
    value = gemm_nt(a, b, backend=backend, out_dtype=out_dtype, bias=bias, num_stages=num_stages,
        autotune=autotune, gradient_output=fused)
    return output.convert(value, matrix=True) if output is not None and fused is None else value


def forward(qx, qw, geometry, *, bias=None, out_dtype, num_stages="auto", output=None,
            gemm_backend="tilelang", autotune=True):
    if _pointwise(geometry, qx, qw):
        return _gemm(qx, qw.transpose(contiguous=False), gemm_backend, out_dtype, num_stages, output, bias, autotune)
    from ..convolution import input_gradient
    return input_gradient(qx, qw, geometry.adjoint, bias=bias, out_dtype=out_dtype,
        num_stages=num_stages, gradient_output=output, autotune=autotune)


def input_gradient(qg, qw, geometry, *, out_dtype, num_stages="auto", gradient_output=None,
                   gemm_backend="tilelang", autotune=True):
    if _pointwise(geometry, qg, qw):
        return _gemm(qg, qw, gemm_backend, out_dtype, num_stages, gradient_output, autotune=autotune)
    from ..convolution import forward as convolution
    return convolution(qg, qw, geometry.adjoint, out_dtype=out_dtype, num_stages=num_stages,
        gradient_output=gradient_output, autotune=autotune)


def weight_gradient(qx, qg, geometry, *, out_dtype, num_stages="auto", gemm_backend="tilelang", autotune=True):
    if _pointwise(geometry, qx, qg):
        return _gemm(qx.transpose(contiguous=False), qg.transpose(contiguous=False), gemm_backend,
            out_dtype, num_stages, autotune=autotune)
    from ..convolution import weight_gradient as convolution
    return convolution(qg, qx, geometry.adjoint, out_dtype=out_dtype, num_stages=num_stages, autotune=autotune)
