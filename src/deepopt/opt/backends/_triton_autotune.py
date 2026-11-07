"""Host-side TMA launch policies, importable without Triton or CUDA."""

from math import gcd


def baseline_gemm_tile(a_block_shape, b_block_shape):
    """Historical launch retained as a measured candidate, never a fallback."""
    blocks = (a_block_shape, b_block_shape)
    if any(block is not None and block[1] % 32 for block in blocks):
        # Fused FP32 dequantization/TF32x3 uses the smaller resource footprint.
        return 64, 64, 32, 4
    block_k = 64
    for block in blocks:
        if block is not None:
            block_k = gcd(block_k, block[1])
    return 128, 128, block_k, 8


# Order broad shapes across K before adding alternate warp counts. The
# shortlist keeps the historical launch plus at most seven other candidates.
# Narrow matrices also need narrow tiles: all of the former candidates spent
# at least 64 rows/columns of work on a ten-class classifier dimension.
MAX_TUNING_CONFIGS = 8
AUTOTUNE_TILES = (
    (64, 64, 32, 4), (64, 64, 64, 4), (128, 128, 128, 8),
    (64, 128, 32, 4), (128, 64, 32, 4),
    (64, 128, 64, 4), (128, 64, 64, 4),
    (128, 128, 32, 8), (128, 128, 64, 8),
    (64, 64, 32, 8),
) + tuple(
    (bm, bn, bk, 4)
    for bm, bn in (
        (64, 16), (16, 64), (128, 16), (16, 128),
        (64, 32), (32, 64), (128, 32), (32, 128),
        (16, 16), (16, 32), (32, 16), (32, 32),
    )
    for bk in (32, 64, 128)
)
MAX_BLOCK_M = max(tile[0] for tile in AUTOTUNE_TILES)
MAX_BLOCK_N = max(tile[1] for tile in AUTOTUNE_TILES)
MAX_BLOCK_K = max(tile[2] for tile in AUTOTUNE_TILES)

# TensorDescriptor has no .dtype, so Triton's implicit tensor dtype keys do
# not distinguish the input formats. Device and policy keys also prevent a
# cached winner from bypassing pruning for a different scale layout/stage.
AUTOTUNE_KEY = (
    "M", "N", "K", "A_FP8_FORMAT", "B_FP8_FORMAT", "OUTPUT_DTYPE",
    "A_BLOCK_ROWS", "A_BLOCK_COLS", "B_BLOCK_ROWS", "B_BLOCK_COLS",
    "A_SCALE_COLS", "B_SCALE_COLS", "TENSORWISE", "NATIVE_FP8", "HAS_BIAS",
    "PIPELINE_STAGES", "DEVICE_INDEX", "DEVICE_CAPABILITY",
)


def set_descriptor_block_shapes(args):
    """Run for every benchmark and for the final/cached winning launch."""
    bm, bn, bk = args["BLOCK_M"], args["BLOCK_N"], args["BLOCK_K"]
    args["a_desc"].block_shape = [bm, bk]
    args["b_desc"].block_shape = [bn, bk]
    args["c_desc"].block_shape = [bm, bn]


def prune_gemm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    compatible = []
    for config in configs:
        if config.num_stages != args["PIPELINE_STAGES"]:
            continue
        bm, bn, bk = (config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        if args["NATIVE_FP8"]:
            if args["A_BLOCK_COLS"] % bk or args["B_BLOCK_COLS"] % bk:
                continue
        elif bm > 64 or bn > 64 or bk != 32:
            # Fused FP32 dequantization/TF32x3 needs more resources than FP8.
            continue
        compatible.append(config)
    if not compatible:
        raise ValueError("No Triton autotune configuration matches the scale layout and pipeline stages")

    # Keep two useful tile widths per output axis, including both 16 and 32
    # for tiny heads. Large dimensions retain the original 64/128 search.
    def axis_tiles(size):
        if size <= 32:
            return (16, 32)
        if size <= 64:
            return (32, 64)
        return (64, 128)

    baseline = baseline_gemm_tile(
        (args["A_BLOCK_ROWS"], args["A_BLOCK_COLS"]),
        (args["B_BLOCK_ROWS"], args["B_BLOCK_COLS"]),
    )
    m_tiles, n_tiles = axis_tiles(args["M"]), axis_tiles(args["N"])
    # Keep intermediate 64-wide launches when the budget permits: hardware
    # efficiency can outweigh edge waste, even on a ten-class dimension.
    extended_m = (*m_tiles, 64) if args["M"] <= 32 else m_tiles
    extended_n = (*n_tiles, 64) if args["N"] <= 32 else n_tiles
    # K=10 does not need 64/128-wide reductions. Keep the historical candidate
    # regardless, so every measured winner competes with the previous policy.
    k_limit = max(32, 1 << (args["K"] - 1).bit_length())

    def tile(config):
        return (*(config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K")),
                config.num_warps)

    selected = [config for config in compatible if tile(config) == baseline]
    preferred, neighboring = [], []
    for config in compatible:
        bm, bn, bk, _ = tile(config)
        if tile(config) == baseline or bk > k_limit:
            continue
        if bm in m_tiles and bn in n_tiles:
            preferred.append(config)
        elif bm in extended_m and bn in extended_n:
            neighboring.append(config)
    for config in preferred + neighboring:
        selected.append(config)
        if len(selected) == MAX_TUNING_CONFIGS:
            break
    if not selected:
        raise ValueError("No Triton autotune configuration matches the matrix shape")
    return selected
