"""Pointwise launch ABI shared by target-specific compiler emitters.

Importing the compiler never imports either optional GPU compiler. Artifacts
own code and static metadata; runtime tensors are bound afresh for each call.
"""


def launch_pointwise(artifact, inputs, output, block_size):
    count = output.numel()
    if not count:
        return
    if getattr(artifact, "backend", "triton") == "tilelang":
        artifact.kernel(*inputs, output)
        return
    import triton

    # A generated artifact bakes its block into the source, so an unmasked
    # store is only safe with exactly that geometry. Legacy artifacts that
    # still take BLOCK_SIZE as a constexpr keep the caller-supplied value.
    baked = bool(getattr(artifact, "baked_block_size", False))
    block = getattr(artifact, "block_size", block_size) if baked else block_size
    if baked:
        # An unmasked store is only safe when the grid divides the element
        # count. Refuse to guess: the artifact must prove its own geometry.
        config = getattr(artifact, "config", None)
        if config is None:
            raise ValueError(
                "baked pointwise artifact has no recorded launch geometry"
            )
        if not config.masked and count % block:
            raise ValueError(
                f"baked pointwise artifact block {block} does not divide {count} elements"
            )
    options = {"enable_fp_fusion": False}
    warps = getattr(artifact, "num_warps", None)
    if warps:
        options["num_warps"] = warps
    if not baked:
        options["BLOCK_SIZE"] = block
    artifact.kernel[(triton.cdiv(count, block),)](
        *inputs, output, count, **options,
    )
