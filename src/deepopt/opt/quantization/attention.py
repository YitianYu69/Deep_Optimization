"""Compress only Q/K/V saved by native attention, including math-backend views.

Attention kernels also save probabilities, masks, outputs and normalization
state. Those tensors remain exact. Tracking data-only Q/K/V transformations
keeps native SDPA backend selection and forward computation unchanged.
"""

from dataclasses import dataclass

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from .activation import PackedActivation, quantize_pack_activation, dequantize_unpack_activation


@dataclass(frozen=True)
class _SavedQKV:
    packed: PackedActivation
    shape: tuple
    dtype: torch.dtype


class AttentionCompression(TorchDispatchMode):
    """Track Q/K/V aliases/casts and replace only their autograd saved values.

    Weak storage references prevent allocator-address reuse from mistagging a
    later attention probability/output, without retaining dense Q/K/V storage.
    No tensor payload is inspected on the host.
    """

    _views = frozenset({
        "alias", "as_strided", "clone", "contiguous", "detach", "expand",
        "flatten", "permute", "repeat", "repeat_interleave", "reshape",
        "select", "slice", "squeeze", "t", "to", "transpose", "unsqueeze",
        "view", "_reshape_alias", "_to_copy", "_unsafe_view",
    })

    def __init__(self, query, key, value, *, group_size, bits, attn_mask=None):
        super().__init__()
        self.group_size, self.bits = group_size, bits
        self._tracked = {}
        self._excluded = {}
        self._in_codec = False
        self._hooks = None
        for tensor in (query, key, value):
            self._tag(tensor, self._tracked)
        if isinstance(attn_mask, torch.Tensor):
            self._tag(attn_mask, self._excluded)

    @staticmethod
    def _key(tensor):
        return (tensor.device, tensor.dtype, tensor.untyped_storage()._cdata,
                tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))

    def _tag(self, tensor, registry):
        if (isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                and not tensor.is_nested and tensor.is_floating_point()):
            registry[self._key(tensor)] = StorageWeakRef(tensor.untyped_storage())

    def _contains(self, tensor, registry):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.is_nested or not tensor.is_floating_point()):
            return False
        reference = registry.get(self._key(tensor))
        return reference is not None and not reference.expired()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if self._in_codec:
            return func(*args, **kwargs)
        name = func._schema.name.split("::")[-1]
        source = args[0] if args else None
        propagate = name in self._views
        if name in ("mul", "div") and len(args) >= 2:
            other = args[1]
            propagate = not isinstance(other, torch.Tensor) or other.numel() == 1
            if name == "mul" and not self._contains(source, self._tracked):
                if not isinstance(source, torch.Tensor) or source.numel() == 1:
                    source, propagate = other, True
        tracked = propagate and self._contains(source, self._tracked)
        excluded = propagate and self._contains(source, self._excluded)
        result = func(*args, **kwargs)
        if tracked or excluded:
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for tensor in outputs:
                if tracked:
                    self._tag(tensor, self._tracked)
                if excluded:
                    self._tag(tensor, self._excluded)
        return result

    def _quantize_pack(self, tensor):
        self._in_codec = True
        try:
            if (tensor.ndim >= 2 and tensor.numel() and self._contains(tensor, self._tracked)
                    and not self._contains(tensor, self._excluded)):
                matrix = tensor.reshape(-1, tensor.shape[-1])
                packed = quantize_pack_activation(matrix, group_size=self.group_size, bits=self.bits)
                return _SavedQKV(packed, tuple(tensor.shape), tensor.dtype)
            # Saved outputs must not retain their own grad_fn through a hook.
            return tensor.detach()
        finally:
            self._in_codec = False

    @staticmethod
    def _dequantize_unpack(saved):
        if isinstance(saved, _SavedQKV):
            return dequantize_unpack_activation(saved.packed, dtype=saved.dtype).reshape(saved.shape)
        return saved

    def __enter__(self):
        super().__enter__()
        try:
            self._hooks = torch.autograd.graph.saved_tensors_hooks(self._quantize_pack, self._dequantize_unpack)
            self._hooks.__enter__()
        except BaseException:
            super().__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._hooks.__exit__(exc_type, exc_value, traceback)
        finally:
            super().__exit__(exc_type, exc_value, traceback)
