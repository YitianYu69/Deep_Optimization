"""Torch-free SI-GB aliases for public packed-offload configuration tables."""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation


_GB_ALIASES = {
    "host_pinned_gb": "host_pinned_bytes",
    "device_staging_gb": "device_staging_bytes",
    "d2h_inflight_gb": "d2h_inflight_bytes",
    "host_pageable_gb": "host_pageable_bytes",
    "peak_budget_gb": "peak_budget_bytes",
    "copy_chunk_gb": "copy_chunk_bytes",
}
_AUTO_BUDGETS = ("device_staging_bytes", "d2h_inflight_bytes")
_MAX_ALIAS_BYTES = (1 << 63) - 1


def _gb_bytes(value, name):
    if type(value) not in (str, int, float, Decimal):
        raise ValueError(f"packed_activation_offload.{name} must be a finite nonnegative GB amount")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"packed_activation_offload.{name} must be a finite nonnegative GB amount") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"packed_activation_offload.{name} must be a finite nonnegative GB amount")
    if amount.is_zero():
        return 0
    _, digits, exponent = amount.as_tuple()
    exponent += 9  # 1 GB is exactly 10**9 bytes, independent of Decimal context.
    if exponent < 0:
        fraction = -exponent
        if fraction > len(digits) or any(digits[-fraction:]):
            raise ValueError(f"packed_activation_offload.{name} must represent an exact integer number of bytes")
        digits, exponent = digits[:-fraction], 0
    # Inspect the decimal exponent before constructing a potentially huge int.
    if len(digits) + exponent > 19:
        raise ValueError(f"packed_activation_offload.{name} exceeds signed 64-bit byte capacity")
    result = int("".join(str(digit) for digit in digits)) * 10**exponent
    if result > _MAX_ALIAS_BYTES:
        raise ValueError(f"packed_activation_offload.{name} exceeds signed 64-bit byte capacity")
    return result


def normalize_packed_offload_units(raw: Mapping) -> dict:
    """Return canonical byte fields without mutating the incoming user table.

    GB means SI 10**9 bytes. GB aliases require exact integer-byte values at
    most signed-64-bit capacity. Numeric byte values are left untouched.
    Either transport limit can also be "auto", encoded as zero bytes.
    Supplying both units for one quantity is always an error. An enabled GB
    route missing its pinned quantity defaults to 80 GB per rank. Normalize
    each incoming table before merging inherited canonical byte defaults.
    """
    if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
        raise ValueError("packed_activation_offload must be a table with string keys")
    result = dict(raw)
    aliases = set(raw) & _GB_ALIASES.keys()
    for name in sorted(aliases):
        target = _GB_ALIASES[name]
        if target in raw:
            raise ValueError(f"packed_activation_offload cannot supply both {name} and {target}")
        value = result.pop(name)
        if target not in _AUTO_BUDGETS or type(value) is not str or value != "auto":
            value = _gb_bytes(value, name)
        result[target] = value
    for name in _AUTO_BUDGETS:
        if type(result.get(name)) is str and result[name] == "auto":
            result[name] = 0
    if aliases and raw.get("enabled") is True and "host_pinned_bytes" not in result:
        result["host_pinned_bytes"] = 80_000_000_000
    return result
