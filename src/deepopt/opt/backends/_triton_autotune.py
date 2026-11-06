"""Host-side TMA autotuning policy, importable without Triton or CUDA."""


# A bounded search keeps first-use tuning practical. The 32-wide candidates
# remain valid for every native FP8 scale width accepted by the backend.
AUTOTUNE_TILES = (
    (64, 64, 32, 4), (64, 64, 32, 8),
    (64, 128, 32, 4), (128, 64, 32, 4), (128, 128, 32, 8),
    (64, 64, 64, 4), (64, 128, 64, 4), (128, 64, 64, 4),
    (128, 128, 64, 8), (128, 128, 128, 8),
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
    selected = []
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
        selected.append(config)
    if not selected:
        raise ValueError("No Triton autotune configuration matches the scale layout and pipeline stages")
    return selected
