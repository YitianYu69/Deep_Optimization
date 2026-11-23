"""Placeholder for a dedicated Triton backward lowering pass.

Backward ops are currently represented by ordinary IR nodes and use the
existing native fallback in the wrapper. This file is retained as the
extension point for a future backward-specific Triton backend.
"""
