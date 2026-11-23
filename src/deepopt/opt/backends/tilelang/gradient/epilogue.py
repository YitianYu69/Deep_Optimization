"""Explicit FP8 gradient output conversion using TileLang arithmetic."""

def output(values, gradient_output):
    owner = gradient_output
    from .._runtime import auxiliary_module
    if owner.scale is None:
        conversion = auxiliary_module("gradient", "quantization.fp8").quantize
        data, inverse = conversion(values, owner.dtype, owner.block_shape, owner.checked)
        owner.inverse = inverse
        owner.observations = None
        owner.observation_count = 0
        # Keep the same physical FP8 gradient row layout as the fused producer.
        packing = auxiliary_module("gradient", "gemm.fp8")._pack_fp8_rows
        data = packing(data, 16)[:, :values.shape[1]]
    else:
        conversion = auxiliary_module("gradient", "quantization.delayed").convert
        data, _, snapshot, observations, _ = conversion(
            values, owner.scale, owner.dtype, normal=True, transposed=False, bias=False,
            observe=owner.training, check_finite=owner.checked, autotune=False)
        owner.inverse = snapshot[1:2].reshape(1, 1)
        owner.observations = observations
        owner.observation_count = observations.numel() if observations is not None else 0
    result = owner.wrap(data)
    cast_backend = "tilelang" if ".tilelang." in conversion.__module__ else "triton"
    owner.info.update(fused=False, output_cast_launches=1, output_cast_stages=1,
                      workspace_dtype=str(values.dtype), kernel_backend=cast_backend)
    if cast_backend == "tilelang":
        owner.info.update(scale_reduction_launches=int(owner.scale is None),
                          observation_reduction_launches=int(owner.scale is not None and owner.training),
                          scale_snapshot_launches=int(owner.scale is not None),
                          row_pack_launches=int(owner.scale is None and values.shape[1] % 16 != 0))
    return result
