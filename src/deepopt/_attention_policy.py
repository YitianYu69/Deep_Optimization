"""Shared, dependency-free effective attention codec recipe."""


def effective_attention_bits(compression):
    if not compression or not compression.get("enabled", False):
        return 0
    override = compression.get("attention_bits")
    bits = max(4, compression.get("bits", 2)) if override is None else override
    if type(bits) is not int or bits not in (1, 2, 4, 8):
        raise ValueError("Attention compression bits must be 1, 2, 4 or 8")
    return bits
