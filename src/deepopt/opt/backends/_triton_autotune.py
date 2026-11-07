"""Host-side TMA launch policies, importable without Triton or CUDA."""

from collections import defaultdict
from math import ceil, gcd


def baseline_gemm_tile(a_block_shape, b_block_shape):
    """Historical launch retained as a measured candidate, never a fallback."""
    blocks = (a_block_shape, b_block_shape)
    if any(block is not None and block[1] % 32 for block in blocks):
        return 64, 64, 32, 4
    block_k = 64
    for block in blocks:
        if block is not None:
            block_k = gcd(block_k, block[1])
    return 128, 128, block_k, 8


# Reserve room for independent tile, warp and pipeline choices. Selection
# below is independent of this tuple's ordering.
MAX_TUNING_CONFIGS = 48
MAX_TILE_FAMILIES = 12
SPLIT_K_CHOICES = (1, 2, 4, 8)
AUTOTUNE_TILES = tuple(
    (bm, bn, bk, warps)
    for bm in (16, 32, 64, 128, 256)
    for bn in (16, 32, 64, 128, 256)
    for bk in (32, 64, 128)
    for warps in (2, 4, 8)
    if bm * bn <= 32768
    and (max(bm, bn) < 256 or min(bm, bn) >= 64)
    and (warps != 2 or bm * bn <= 4096)
    and (warps != 8 or bm * bn >= 4096)
)
MAX_BLOCK_M = max(tile[0] for tile in AUTOTUNE_TILES)
MAX_BLOCK_N = max(tile[1] for tile in AUTOTUNE_TILES)
MAX_BLOCK_K = max(tile[2] for tile in AUTOTUNE_TILES)

# Descriptors have no .dtype; their FP8 formats must be explicit cache keys.
# PIPELINE_STAGES=0 means automatic; NUM_STAGES is the candidate's loop depth.
AUTOTUNE_KEY = (
    "M", "N", "K", "A_FP8_FORMAT", "B_FP8_FORMAT", "OUTPUT_DTYPE",
    "A_BLOCK_ROWS", "A_BLOCK_COLS", "B_BLOCK_ROWS", "B_BLOCK_COLS",
    "A_SCALE_COLS", "B_SCALE_COLS", "TENSORWISE", "NATIVE_FP8", "HAS_BIAS",
    "PIPELINE_STAGES", "DEVICE_INDEX", "DEVICE_CAPABILITY", "NUM_SMS",
)


def autotune_cache_key(named_args):
    """Match Triton's explicit keys followed by tensor dtype keys, in arg order."""
    return (tuple(named_args[name] for name in AUTOTUNE_KEY)
            + tuple(str(value.dtype) for value in named_args.values() if hasattr(value, "dtype")))


def split_k_enabled(m, n, k):
    """Limit partial-buffer allocation and search to skinny, long reductions."""
    return min(m, n) <= 32 and k >= 512 and m * n <= 262144


def set_descriptor_block_shapes(args):
    """Run for every benchmark and for the final/cached winning launch."""
    bm, bn, bk = args["BLOCK_M"], args["BLOCK_N"], args["BLOCK_K"]
    args["a_desc"].block_shape = [bm, bk]
    args["b_desc"].block_shape = [bn, bk]
    args["c_desc"].block_shape = [bm, bn]


def prune_gemm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    m, n, k = (args[name] for name in ("M", "N", "K"))
    sms = args.get("NUM_SMS", 132)
    requested_stage = args["PIPELINE_STAGES"]
    baseline = baseline_gemm_tile(
        (args["A_BLOCK_ROWS"], args["A_BLOCK_COLS"]),
        (args["B_BLOCK_ROWS"], args["B_BLOCK_COLS"]),
    )
    k_limit = max(32, 1 << (k - 1).bit_length())

    def signature(config):
        return (*(config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K")),
                config.num_warps, config.num_stages, config.kwargs.get("SPLIT_K", 1))

    def axis_tiles(size):
        if size <= 64:
            return (16, 32, 64)
        # Small tiles can be necessary to populate the GPU even on a long axis.
        return (16, 32, 64, 128) if min(m, n) <= 32 else (32, 64, 128, 256)

    families = defaultdict(list)
    historical = None
    for config in configs:
        bm, bn, bk, warps, stages, split = signature(config)
        if stages not in (1, 2, 3, 4) or (requested_stage and stages != requested_stage):
            continue
        if args["NATIVE_FP8"]:
            if args["A_BLOCK_COLS"] % bk or args["B_BLOCK_COLS"] % bk:
                continue
        elif bm > 64 or bn > 64 or bk != 32:
            continue
        is_baseline = (bm, bn, bk, warps) == baseline and split == 1
        if is_baseline and stages == (requested_stage or 3):
            historical = config
        if not is_baseline and (bm not in axis_tiles(m) or bn not in axis_tiles(n) or bk > k_limit):
            continue
        if split not in SPLIT_K_CHOICES:
            continue
        if split > 1:
            ctas = ceil(m / bm) * ceil(n / bn)
            if (not split_k_enabled(m, n, k) or ceil(k / bk) < split * 2
                    or ctas >= sms or ctas * split > sms * 4):
                continue
        families[bm, bn, bk, split].append(config)
    if not families:
        raise ValueError("No Triton autotune configuration matches the scale layout and pipeline stages")

    def family_cost(family):
        bm, bn, bk, split = family
        ctas = ceil(m / bm) * ceil(n / bn)
        waves = ceil(ctas * split / sms)
        # Approximate padded compute per wave, with a small penalty for tiny
        # tensor-core tiles and the extra FP32 partial/reduction traffic.
        compute = waves * bm * bn * ceil(ceil(k / bk) / split) * bk
        compute *= max(1, 4096 / (bm * bn)) ** 0.25
        traffic = ctas * (bm + bn) * ceil(k / bk) * bk
        reduction = (split + 1) * m * n * 4 if split > 1 else 0
        return compute + (traffic + reduction) / sms

    # Round-robin across K widths and split factors before adding runner-up
    # spatial tiles. This prevents one reduction width from consuming the cap.
    buckets = defaultdict(list)
    for family in families:
        buckets[family[2:]].append(family)
    for bucket in buckets.values():
        bucket.sort(key=lambda family: (family_cost(family), family))
    bucket_order = sorted(buckets, key=lambda key: (family_cost(buckets[key][0]), key))
    chosen_families = []
    if historical is not None:
        bm, bn, bk, _, _, split = signature(historical)
        chosen_families.append((bm, bn, bk, split))
    # Reserve unsplit narrow-axis alternatives before split/K buckets consume
    # the budget: padding waste alone cannot predict tensor-core efficiency.
    for axis, size in enumerate((m, n)):
        if size <= 32:
            for width in (16, 32, 64):
                alternatives = [family for family in families if family[axis] == width and family[3] == 1]
                if alternatives:
                    family = min(alternatives, key=lambda item: (family_cost(item), item))
                    if family not in chosen_families:
                        chosen_families.append(family)
    for rank in range(max(map(len, buckets.values()))):
        for key in bucket_order:
            bucket = buckets[key]
            if rank < len(bucket) and bucket[rank] not in chosen_families:
                chosen_families.append(bucket[rank])
                if len(chosen_families) == MAX_TILE_FAMILIES:
                    break
        if len(chosen_families) == MAX_TILE_FAMILIES:
            break

    def variant_order(config):
        bm, bn, bk, warps, stages, split = signature(config)
        ideal_stage = min(3, ceil(ceil(k / bk) / split))
        stage_order = sorted((1, 2, 3, 4), key=lambda stage: (abs(stage - ideal_stage), stage))
        ideal_warps = 2 if bm * bn <= 2048 else 4 if bm * bn <= 16384 else 8
        warp_order = sorted((2, 4, 8), key=lambda warp: (abs(warp - ideal_warps), warp))
        si, wi = stage_order.index(stages), warp_order.index(warps)
        return si + wi, si, wi, signature(config)

    variants = [sorted(families[family], key=variant_order) for family in chosen_families]
    selected = [historical] if historical is not None else []
    seen = {signature(config) for config in selected}
    for rank in range(max(map(len, variants))):
        for family_variants in variants:
            if rank < len(family_variants):
                config = family_variants[rank]
                if signature(config) not in seen:
                    selected.append(config)
                    seen.add(signature(config))
                    if len(selected) == MAX_TUNING_CONFIGS:
                        return selected
    return selected
