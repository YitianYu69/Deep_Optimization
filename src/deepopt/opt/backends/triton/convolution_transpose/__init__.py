"""Adjoint FP8 execution without expanded-zero or unfolded activations."""

from ... import gemm_nt


def _pointwise(geometry, *operands):
    return geometry.pointwise and all(q.block_shape is None for q in operands)


def _gemm(a, b, backend, out_dtype, num_stages, output=None, bias=None):
    # Native GEMM has no FP8-output epilogue. Preserve its computation and
    # explicitly convert rather than reporting a fused cast.
    fused = output if backend == "triton" else None
    value = gemm_nt(a, b, backend=backend, out_dtype=out_dtype, bias=bias,
                    num_stages=num_stages, gradient_output=fused)
    return output.convert(value, matrix=True) if output is not None and fused is None else value


def forward(qx, qw, geometry, *, bias=None, out_dtype, num_stages="auto", output=None, gemm_backend="triton"):
    if _pointwise(geometry, qx, qw):
        return _gemm(qx, qw.transpose(contiguous=False), gemm_backend, out_dtype, num_stages, output, bias)
    from ..convolution.backward import input_gradient

    return input_gradient(qx, qw, geometry.adjoint, bias=bias, out_dtype=out_dtype,
                          num_stages=num_stages, gradient_output=output)


def input_gradient(qg, qw, geometry, *, out_dtype, num_stages="auto", gradient_output=None, gemm_backend="triton"):
    if _pointwise(geometry, qg, qw):
        return _gemm(qg, qw, gemm_backend, out_dtype, num_stages, gradient_output)
    from ..convolution.forward import forward as convolution

    return convolution(qg, qw, geometry.adjoint, out_dtype=out_dtype, num_stages=num_stages,
                       gradient_output=gradient_output)


def weight_gradient(qx, qg, geometry, *, out_dtype, num_stages="auto", gemm_backend="triton"):
    if _pointwise(geometry, qx, qg):
        return _gemm(qx.transpose(contiguous=False), qg.transpose(contiguous=False),
                     gemm_backend, out_dtype, num_stages)
    from ..convolution.backward import weight_gradient as convolution

    return convolution(qg, qx, geometry.adjoint, out_dtype=out_dtype, num_stages=num_stages)
