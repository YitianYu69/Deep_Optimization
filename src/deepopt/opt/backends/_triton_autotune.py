"""Host-side TMA launch policies, importable without Triton or CUDA."""

from collections import defaultdict
from math import ceil, gcd


def baseline_gemm_tile(a_block_shape, b_block_shape):
    """Historical tile/warp choice retained as a measured candidate."""
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
AUTOTUNE_STAGES = (1, 2, 3, 4, 5)
GROUP_SIZE_M_CHOICES = (1, 8)
SPLIT_K_CHOICES = (1, 2, 4, 8)
MAX_SPLIT_K_SCRATCH_BYTES = 64 * 1024 * 1024
# CUDA seeds from Triton's matrix-multiplication tutorial (M, N, K, warps, stages).
# Reserve these measured choices instead of letting a cost estimate remove them.
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
TUTORIAL_GEMM_CONFIGS = (
    (128, 256, 64, 8, 3), (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4), (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4), (128, 32, 32, 4, 4),
    (64, 32, 32, 2, 5), (32, 64, 32, 2, 5),
    (128, 256, 128, 8, 3), (256, 128, 128, 8, 3),
    (256, 64, 128, 4, 4), (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4), (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4), (128, 32, 64, 4, 4),
)
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
    "MAX_NUM_IMPRECISE_ACC",
    "A_ROW_STRIDE", "B_ROW_STRIDE",
)


def autotune_cache_key(named_args):
    """Match Triton's explicit keys followed by tensor dtype keys, in arg order."""
    return (tuple(named_args[name] for name in AUTOTUNE_KEY)
            + tuple(str(value.dtype) for value in named_args.values() if hasattr(value, "dtype")))


def split_k_limit(m, n, k):
    """Bound FP32 scratch while admitting long, dense weight-gradient products."""
    skinny = min(m, n) <= 32 and k >= 512 and m * n <= 262144
    if not skinny and k < 4096:
        return 1
    # Eight-element alignment bounds either FP32 or 16-bit output row strides.
    stride = ((n + 7) // 8) * 8
    return max((split for split in SPLIT_K_CHOICES
                if split == 1 or split * m * stride * 4 <= MAX_SPLIT_K_SCRATCH_BYTES), default=1)


def split_k_enabled(m, n, k):
    return split_k_limit(m, n, k) > 1


class DescriptorKernel:
    """Set descriptor tiles for every launch without serializing config hooks.

    Triton's disk-result cache cannot serialize Config.pre_hook. This adapter
    wraps the JIT function, including benchmark, cached and warmup launches.
    """

    def __init__(self, fn):
        self.fn, self.arg_names = fn, fn.arg_names

    def _prepare(self, args, kwargs):
        set_descriptor_block_shapes({**dict(zip(self.arg_names, args)), **kwargs})

    def run(self, *args, **kwargs):
        self._prepare(args, kwargs)
        return self.fn.run(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        self._prepare(args, kwargs)
        return self.fn.warmup(*args, **kwargs)


def grouped_tile_coordinates(pid, num_m, num_n, group_m):
    """Map a flat grid to grouped M tiles; also compiled by Triton at use."""
    if group_m == 1:
        return pid // num_n, pid % num_n
    per_group = group_m * num_n
    first_m = (pid // per_group) * group_m
    rows = min(num_m - first_m, group_m)
    in_group = pid % per_group
    return first_m + in_group % rows, in_group // rows


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
                config.num_warps, config.num_stages, config.kwargs.get("SPLIT_K", 1),
                config.kwargs.get("GROUP_SIZE_M", 1))

    def axis_tiles(size):
        if size <= 64:
            return (16, 32, 64)
        # Small tiles can be necessary to populate the GPU even on a long axis.
        return (16, 32, 64, 128) if min(m, n) <= 32 else (32, 64, 128, 256)

    families = defaultdict(list)
    compatible = {}
    historical = None
    for config in configs:
        bm, bn, bk, warps, stages, split, group = signature(config)
        if stages not in AUTOTUNE_STAGES or (requested_stage and stages != requested_stage):
            continue
        if group not in GROUP_SIZE_M_CHOICES:
            continue
        if group != 1 and (ceil(m / bm) == 1 or ceil(n / bn) == 1):
            # Grouping changes no coordinates when either axis has only one tile.
            continue
        if args["NATIVE_FP8"]:
            if args["A_BLOCK_COLS"] % bk or args["B_BLOCK_COLS"] % bk:
                continue
        elif bm > 64 or bn > 64 or bk != 32:
            continue
        is_baseline = (bm, bn, bk, warps) == baseline and split == 1
        if is_baseline and stages == (requested_stage or 3) and group == 1:
            historical = config
        if not is_baseline and (bm not in axis_tiles(m) or bn not in axis_tiles(n) or bk > k_limit):
            continue
        if split not in SPLIT_K_CHOICES:
            continue
        if split > 1:
            ctas = ceil(m / bm) * ceil(n / bn)
            long_reduction = k >= 4096
            if (split > split_k_limit(m, n, k) or ceil(k / bk) < split * 2
                    or ctas >= sms * (2 if long_reduction else 1)
                    or ctas * split > sms * (8 if long_reduction else 4)):
                continue
        families[bm, bn, bk, split].append(config)
        compatible[signature(config)] = config
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
        bm, bn, bk, _, _, split, _ = signature(historical)
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
        bm, bn, bk, warps, stages, split, group = signature(config)
        ideal_stage = min(3, ceil(ceil(k / bk) / split))
        stage_order = sorted(AUTOTUNE_STAGES, key=lambda stage: (abs(stage - ideal_stage), stage))
        ideal_warps = 2 if bm * bn <= 2048 else 4 if bm * bn <= 16384 else 8
        warp_order = sorted((2, 4, 8), key=lambda warp: (abs(warp - ideal_warps), warp))
        si, wi = stage_order.index(stages), warp_order.index(warps)
        return si + wi, si, wi, group != 8, signature(config)

    def variant_cost(config):
        bm, bn, bk, _, _, split, _ = signature(config)
        return family_cost((bm, bn, bk, split)), variant_order(config)

    def other_order(sig):
        return (*sig[:-1], 8 if sig[-1] == 1 else 1)

    variants = [sorted(families[family], key=variant_order) for family in chosen_families]
    selected, seen = [], set()

    def add(config):
        if config is not None and signature(config) not in seen and len(selected) < MAX_TUNING_CONFIGS:
            selected.append(config)
            seen.add(signature(config))

    add(historical)
    for bm, bn, bk, warps, stages in TUTORIAL_GEMM_CONFIGS:
        group = 8 if ceil(m / bm) > 1 and ceil(n / bn) > 1 else 1
        add(compatible.get((bm, bn, bk, warps, requested_stage or stages, 1, group)))
    # Retain shape/scale-aware skinny and split-K alternatives beside the seeds.
    for family_variants in variants:
        add(family_variants[0])
    # Ensure stage depth is actually measured, rather than lost to the cap.
    for stage in AUTOTUNE_STAGES:
        if not any(config.num_stages == stage for config in selected):
            choices = [config for group in variants for config in group if config.num_stages == stage]
            if choices:
                add(min(choices, key=variant_cost))
    # Compare launch order with the same math tile, stages, warps and split.
    # Bound these pairs so scheduling cannot consume the whole search budget.
    pairs = sum(other_order(sig) in seen for sig in seen) // 2
    for config in tuple(selected):
        if pairs >= 8:
            break
        other = compatible.get(other_order(signature(config)))
        if other is not None and signature(other) not in seen:
            add(other)
            pairs += 1
    for rank in range(max(map(len, variants))):
        for family_variants in variants:
            if rank < len(family_variants):
                config = family_variants[rank]
                if other_order(signature(config)) in seen:
                    continue
                add(config)
                if len(selected) == MAX_TUNING_CONFIGS:
                    return selected
    return selected
