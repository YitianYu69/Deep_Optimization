"""Compress only Q/K/V saved by native attention, including math-backend views.

Attention kernels also save probabilities, masks, outputs and normalization
state. Those tensors remain exact. Tracking data-only Q/K/V transformations
keeps native SDPA backend selection and forward computation unchanged.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from math import copysign

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from .activation import PackedActivation, _quantize_pack_strided_activation, dequantize_unpack_activation
from deepopt.kernel_policy import current_kernel_policy, kernel_policy_scope


_codec_observations = ContextVar("deepopt_attention_codec_observations", default=None)


@contextmanager
def observe_attention_codec():
    """Untimed metadata only, including saves hidden by SDPA's inner hooks."""
    events = []
    token = _codec_observations.set(events)
    try:
        yield events
    finally:
        _codec_observations.reset(token)


@dataclass(frozen=True)
class _QKVPadding:
    logical_shape: tuple
    right: int


@dataclass(frozen=True)
class _SavedQKV:
    packed: PackedActivation
    shape: tuple
    dtype: torch.dtype
    kernel_policy: object = None
    padding: _QKVPadding | None = None


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
        self._padded = {}
        self._in_codec = False
        self._hooks = None
        self.kernel_policy = current_kernel_policy()
        for tensor in (query, key, value):
            self._tag(tensor, self._tracked)
        if isinstance(attn_mask, torch.Tensor):
            self._tag(attn_mask, self._excluded)

    @staticmethod
    def _key(tensor):
        return (tensor.device, tensor.dtype, tensor.untyped_storage()._cdata,
                tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))

    def _tag(self, tensor, registry, padding=None):
        if (isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                and not tensor.is_nested and tensor.is_floating_point()):
            key = self._key(tensor)
            registry[key] = StorageWeakRef(tensor.untyped_storage())
            if registry is self._tracked:
                if padding is None:
                    self._padded.pop(key, None)
                else:
                    self._padded[key] = padding

    def _contains(self, tensor, registry):
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.is_nested or not tensor.is_floating_point()):
            return False
        reference = registry.get(self._key(tensor))
        return reference is not None and not reference.expired()

    def _padding(self, tensor):
        # The tracked weak reference, rather than an allocator address alone,
        # owns the descriptor's validity. Neither registry retains a tensor.
        return self._padded.get(self._key(tensor)) if self._contains(tensor, self._tracked) else None

    def _invalidate_padded_writes(self, tree):
        storages = set()
        def visit(value):
            if isinstance(value, torch.Tensor) and value.layout == torch.strided and not value.is_nested:
                storages.add(value.untyped_storage()._cdata)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
        visit(tree)
        affected = {key[2] for key in self._padded if key[2] in storages}
        for key in tuple(self._tracked):
            if key[2] in affected:
                self._tracked.pop(key)
                self._padded.pop(key, None)

    @staticmethod
    def _preserves_padded_axis(name, source, tensor, args, kwargs):
        """Prove the final feature axis still contains the same zero suffix."""
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim < 2
                or tensor.shape[-1] != source.shape[-1]):
            return False
        def argument(index, key, default=None):
            return args[index] if len(args) > index else kwargs.get(key, default)
        if name in {"alias", "clone", "contiguous", "detach", "to", "_to_copy"}:
            return tensor.shape == source.shape
        if name in {"reshape", "view", "_unsafe_view", "flatten"}:
            return (not isinstance(argument(1, "dtype"), torch.dtype)
                    and tensor.numel() == source.numel())
        if name in {"as_strided", "_reshape_alias"}:
            return (tensor.shape == source.shape and tensor.stride() == source.stride()
                    and tensor.storage_offset() == source.storage_offset())
        if name == "permute":
            return argument(1, "dims")[-1] % source.ndim == source.ndim - 1
        if name == "transpose":
            return all(argument(index, key) % source.ndim != source.ndim - 1
                       for index, key in ((1, "dim0"), (2, "dim1")))
        if name in {"slice", "select"}:
            dim = argument(1, "dim", 0) % source.ndim
            return (dim != source.ndim - 1 or name == "slice"
                    and tensor.stride(-1) == source.stride(-1)
                    and tensor.storage_offset() == source.storage_offset())
        if name == "unsqueeze":
            return argument(1, "dim") % tensor.ndim != tensor.ndim - 1
        if name in {"squeeze", "expand", "repeat"}:
            return True
        if name == "repeat_interleave":
            dim, repeats = argument(2, "dim"), argument(1, "repeats")
            return dim is not None and (dim % source.ndim != source.ndim - 1 or type(repeats) is int and repeats == 1)
        return False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if self._in_codec:
            return func(*args, **kwargs)
        name = func._schema.name.split("::")[-1]
        mutable = func._schema.is_mutable
        if mutable:
            # A write through any alias can destroy a known-zero suffix. Do
            # not retain a dense source solely for a version check, or retag
            # an out= result as a fresh zero-padded coefficient.
            self._invalidate_padded_writes((args, kwargs))
        source = args[0] if args else None
        propagate = name in self._views and not mutable
        if name in ("mul", "div") and len(args) >= 2:
            other = args[1]
            propagate = not mutable and (not isinstance(other, torch.Tensor) or other.numel() == 1)
            if name == "mul" and not self._contains(source, self._tracked):
                if not isinstance(source, torch.Tensor) or source.numel() == 1:
                    source, propagate = other, not mutable
        padding = self._padding(source)
        zero_right_pad = False
        if name == "constant_pad_nd" and isinstance(source, torch.Tensor) and source.ndim >= 2:
            pads = args[1] if len(args) > 1 else kwargs.get("pad", ())
            value = args[2] if len(args) > 2 else kwargs.get("value", 0)
            zero_right_pad = (not mutable and len(pads) == 2 and pads[0] == 0 and type(pads[1]) is int
                              and pads[1] >= 0 and isinstance(value, (int, float)) and value == 0
                              and copysign(1., value) > 0)
            propagate = zero_right_pad
        tracked = propagate and self._contains(source, self._tracked)
        excluded = propagate and self._contains(source, self._excluded)
        result = func(*args, **kwargs)
        if tracked or excluded:
            outputs = result if isinstance(result, (tuple, list)) else (result,)
            for tensor in outputs:
                if tracked:
                    descriptor = None
                    if zero_right_pad or padding is not None:
                        if zero_right_pad or self._preserves_padded_axis(name, source, tensor, args, kwargs):
                            logical_width = padding.logical_shape[-1] if padding is not None else source.shape[-1]
                            descriptor = _QKVPadding((*tensor.shape[:-1], logical_width), tensor.shape[-1] - logical_width)
                        else:
                            # Unproven transforms keep exact state; packing a
                            # former zero suffix as data would change the ABI.
                            continue
                    self._tag(tensor, self._tracked, descriptor)
                if excluded:
                    self._tag(tensor, self._excluded)
        return result

    def _quantize_pack(self, tensor):
        self._in_codec = True
        try:
            if (tensor.ndim >= 2 and tensor.numel() and self._contains(tensor, self._tracked)
                    and not self._contains(tensor, self._excluded)):
                padding = self._padding(tensor)
                logical = tensor if padding is None else tensor[..., :padding.logical_shape[-1]]
                if logical.numel():
                    packed = _quantize_pack_strided_activation(logical, group_size=self.group_size, bits=self.bits)
                    observations = _codec_observations.get()
                    if observations is not None and not isinstance(logical, FakeTensor):
                        if len(observations) >= 10000:
                            raise MemoryError("Attention codec diagnostics exceeded their metadata bound")
                        observations.append(dict(shape=tuple(logical.shape), dtype=str(logical.dtype),
                            bits=packed.bits, group_size=packed.group_size,
                            payload_shape=tuple(packed.data.shape), metadata_shape=tuple(packed.step.shape),
                            payload_bytes=packed.data.numel() * packed.data.element_size(),
                            metadata_bytes=(packed.step.numel() + packed.minimum.numel()) * 4))
                    return _SavedQKV(packed, tuple(tensor.shape), tensor.dtype, self.kernel_policy, padding)
            # Saved outputs must not retain their own grad_fn through a hook.
            return tensor.detach()
        finally:
            self._in_codec = False

    @staticmethod
    def _dequantize_unpack(saved):
        if isinstance(saved, _SavedQKV):
            with kernel_policy_scope(saved.kernel_policy):
                shape = saved.padding.logical_shape if saved.padding is not None else saved.shape
                restored = dequantize_unpack_activation(saved.packed, dtype=saved.dtype).reshape(shape)
                return (torch.nn.functional.pad(restored, (0, saved.padding.right))
                        if saved.padding is not None and saved.padding.right else restored)
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
