"""Bounded, immutable BF16 GEMM launch metadata, independent of GPU imports.

Candidates are hypotheses for separately timed qualification, never automatic
winners. Ordinary execution retains the declared legacy schedule unless the
caller explicitly supplies a configuration. No tuning or tensors live here.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path


BF16_TUNING_VERSION = "bf16-role-shape-v1"
MAX_BF16_CANDIDATES = 32
MAX_BF16_METADATA_ENTRIES = 256


@lru_cache(maxsize=1)
def bf16_source_version():
    """Identify this process's source specialization without GPU imports."""
    root = Path(__file__).resolve().parents[1]
    digest = sha256(BF16_TUNING_VERSION.encode())
    for path in (Path(__file__), root / "triton/gemm/bf16.py", root / "bf16.py",
                 root / "triton/gemm/panel.py", root / "_shared/bf16_resources.py",
                 root / "_shared/operand_device.py", root / "_shared/bf16_routes.py", root / "triton/activation/math.py",
                 root / "triton/quantization/activation/affine.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _metadata(value):
    return value is None or type(value) in (str, int, bool) or (
        type(value) is tuple and all(_metadata(item) for item in value))


@dataclass(frozen=True)
class BF16GemmConfig:
    block_m: int = 64
    block_n: int = 64
    block_k: int = 32
    num_warps: int = 4
    num_stages: int = 3
    ordering: str = "column_major"
    group_m: int = 1

    def __post_init__(self):
        for name in ("block_m", "block_n", "block_k", "num_warps", "num_stages", "group_m"):
            _positive_int(getattr(self, name), name)
        if self.block_m not in (16, 32, 64, 128) or self.block_n not in (16, 32, 64, 128):
            raise ValueError("BF16 BM/BN must be 16, 32, 64, or 128")
        if self.block_k not in (32, 64, 128):
            raise ValueError("BF16 BK must be 32, 64, or 128")
        if self.num_warps not in (4, 8) or self.num_stages not in (1, 2, 3, 4, 5):
            raise ValueError("BF16 configurations require 4/8 warps and 1 through 5 stages")
        if self.ordering not in ("column_major", "row_major", "grouped_m"):
            raise ValueError("BF16 ordering must be column_major, row_major, or grouped_m")
        if self.group_m not in (1, 4, 8, 16) or self.ordering != "grouped_m" and self.group_m != 1:
            raise ValueError("BF16 group_m requires grouped_m ordering and must be 1, 4, 8, or 16")

    def as_dict(self):
        return asdict(self)

    @property
    def kernel_arguments(self):
        return {"BM": self.block_m, "BN": self.block_n, "BK": self.block_k,
                "ORDER": self.ordering, "GROUP_M": self.group_m,
                "num_warps": self.num_warps, "num_stages": self.num_stages,
                "enable_fp_fusion": False}


@dataclass(frozen=True)
class BF16GemmKey:
    role: str
    m: int
    n: int
    k: int
    device: str
    capability: tuple[int, int]
    a_dtype: str
    b_dtype: str
    a_stride: tuple[int, int]
    b_stride: tuple[int, int]
    # Physical codec metadata includes format, orientation, bits, group and
    # payload/metadata strides; epilogue includes every optional operand layout.
    compression: tuple
    epilogue: tuple
    deterministic: bool
    precision_policy: tuple
    kernel_policy: str
    runtime_version: tuple
    source_version: str
    a_storage_offset: int = 0
    b_storage_offset: int = 0
    tuning_version: str = BF16_TUNING_VERSION
    decode_route: str = "direct"
    panel_size: int = 0
    bf16_policy: tuple = (("name", "torch_amp"), ("version", 1))

    def __post_init__(self):
        if any(not _metadata(getattr(self, name)) for name in self.__dataclass_fields__):
            raise TypeError("BF16 GEMM keys must contain immutable scalar/tuple metadata only")
        if self.role not in ("forward", "dx", "dw", "unspecified"):
            raise ValueError("BF16 GEMM role must be forward, dx, dw, or unspecified")
        from deepopt._bf16_policy import resolve_bf16_policy
        resolve_bf16_policy(dict(self.bf16_policy))
        if self.decode_route not in ("direct", "panel") or type(self.panel_size) is not int or self.panel_size < 0:
            raise ValueError("BF16 decode route/panel size is invalid")
        if self.decode_route == "panel" and self.panel_size == 0:
            raise ValueError("Panel decoding requires a positive panel size")
        for name in ("m", "n", "k"):
            _positive_int(getattr(self, name), name)
        for name in ("a_storage_offset", "b_storage_offset"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"BF16 GEMM {name} must be a nonnegative integer")
        for name in ("capability", "a_stride", "b_stride"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 2 or any(type(v) is not int or v < 0 for v in values):
                raise ValueError(f"BF16 GEMM {name} must be a nonnegative integer pair")
        for name in ("compression", "epilogue", "precision_policy", "runtime_version"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"BF16 GEMM {name} must be an immutable tuple")
        if type(self.deterministic) is not bool:
            raise TypeError("BF16 deterministic policy must be boolean")
        for name in ("device", "a_dtype", "b_dtype", "kernel_policy", "source_version", "tuning_version"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"BF16 GEMM {name} must be a nonempty string")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BF16GemmCandidate:
    config: BF16GemmConfig
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BF16GemmSelection:
    key: BF16GemmKey
    config: BF16GemmConfig
    reason: str
    # CPU eligibility and a generated launch do not qualify speed/correctness.
    qualification: str = "unmeasured"
    packed_read_status: str = "disabled"

    def as_dict(self):
        return asdict(self)


@lru_cache(maxsize=32, typed=True)
def legacy_bf16_config(tile=64, num_stages=3):
    """Preserve the original square schedule and first-axis-fast program order."""
    if type(tile) is not int or tile not in (32, 64, 128):
        raise ValueError("BF16 legacy tile must be 32, 64, or 128")
    return BF16GemmConfig(tile, tile, 32, 4, num_stages)


def _unsupported_reason(key):
    if not key.device.startswith("cuda"):
        return "generated BF16 requires a CUDA device"
    if key.capability < (8, 0):
        return "generated BF16 requires NVIDIA capability >= 8.0"
    packed = bool(key.compression and key.compression[0] == "affine_flat")
    if key.a_dtype not in ("torch.bfloat16", "torch.float32") or key.b_dtype not in (
            ("torch.int32",) if packed else ("torch.bfloat16", "torch.float32")):
        return "unsupported BF16 operand storage format"
    if not key.compression or key.compression[0] not in ("dense", "affine_flat"):
        return "unsupported BF16 compression format"
    if packed and (len(key.compression) < 4 or key.compression[1] != "source_KN_right_KN"
                   or key.compression[2] not in (1, 2, 4, 8)
                   or type(key.compression[3]) is not int or key.compression[3] < 1):
        return "unsupported BF16 packed format/orientation"
    return None


@lru_cache(maxsize=MAX_BF16_METADATA_ENTRIES)
def bf16_gemm_candidates(key):
    """Return a small reproducible per-role/shape table for offline measurement.

    Single-K accumulation remains the only supported reduction contract. This
    table does not create split-K partial buffers or change rounding policy.
    """
    if not isinstance(key, BF16GemmKey):
        raise TypeError("BF16 candidates require a BF16GemmKey")
    seeds = [BF16GemmConfig()]
    if key.role == "dw":
        tiles = ((32, 64, 64), (64, 32, 64), (64, 64, 128), (64, 128, 64), (128, 64, 64))
    elif key.m <= 32:
        tiles = ((16, 64, 32), (32, 128, 32), (32, 64, 64), (64, 64, 32))
    elif key.role == "dx":
        tiles = ((64, 32, 32), (64, 128, 64), (128, 64, 32), (64, 64, 64))
    else:
        tiles = ((32, 64, 32), (64, 128, 32), (128, 64, 64), (64, 64, 64))
    for bm, bn, bk in tiles:
        # Vary one launch feature at a time around simple tiling hypotheses.
        seeds.extend((BF16GemmConfig(bm, bn, bk),
                      BF16GemmConfig(bm, bn, bk, num_stages=1),
                      BF16GemmConfig(bm, bn, bk, num_warps=8, num_stages=4),
                      BF16GemmConfig(bm, bn, bk, ordering="row_major"),
                      BF16GemmConfig(bm, bn, bk, ordering="grouped_m", group_m=8)))
    candidates = []
    unsupported = _unsupported_reason(key)
    for config in dict.fromkeys(seeds):
        reason = unsupported
        if reason is None and config.ordering == "grouped_m" and (key.m <= config.block_m or key.n <= config.block_n):
            reason = "grouping is redundant for a single output tile axis"
        elif reason is None and config.block_k > 32 and key.k <= 32:
            reason = "larger BK is redundant for a reduction no longer than 32"
        candidates.append(BF16GemmCandidate(config, reason))
    return tuple(candidates[:MAX_BF16_CANDIDATES])


@lru_cache(maxsize=MAX_BF16_METADATA_ENTRIES, typed=True)
def bf16_gemm_selection(key, config=None, *, tile=64, num_stages=3):
    """Describe explicit execution; never select an unmeasured candidate."""
    if not isinstance(key, BF16GemmKey):
        raise TypeError("BF16 selection requires a BF16GemmKey")
    if config is not None and not isinstance(config, BF16GemmConfig):
        raise TypeError("BF16 kernel_config must be an immutable BF16GemmConfig")
    reason = _unsupported_reason(key)
    if reason is not None:
        raise ValueError(reason)
    selected = config if config is not None else legacy_bf16_config(tile, num_stages)
    packed = bool(key.compression and key.compression[0] != "dense")
    return BF16GemmSelection(key, selected,
        "explicit_configuration" if config is not None else "declared_legacy_baseline_no_qualified_measurement",
        packed_read_status="experimental_explicit_control" if packed else "disabled")


def bf16_tile_coordinates(pid, rows, columns, config):
    """CPU oracle for generated program ordering, including partial M groups."""
    for name, value in (("rows", rows), ("columns", columns)):
        _positive_int(value, name)
    if type(pid) is not int or not 0 <= pid < rows * columns:
        raise ValueError("BF16 program id lies outside the launch grid")
    if config.ordering == "column_major":
        return pid % rows, pid // rows
    if config.ordering == "row_major":
        return pid // columns, pid % columns
    span = config.group_m * columns
    group = pid // span
    first_row = group * config.group_m
    group_rows = min(rows - first_row, config.group_m)
    local = pid % span
    return first_row + local % group_rows, local // group_rows
