"""Scale-boundary arithmetic shared by host tests and JIT reductions."""


def scale_chunk_length(a_start, a_step, a_block, b_start, b_step, b_block, limit):
    """Count consecutive samples sharing both operands' reduction scales.

    Starts are nonnegative logical matrix coordinates, steps and blocks are
    positive, and limit bounds the remaining work and hardware dot tile.
    Scale values are never read on the host. This function is also JIT-compiled
    by the CUDA kernels, so nonaligned groups and strided gathers use exactly
    the same boundary rule.
    """
    a_length = (a_block - 1 - a_start % a_block) // a_step + 1
    b_length = (b_block - 1 - b_start % b_block) // b_step + 1
    return min(limit, min(a_length, b_length))
