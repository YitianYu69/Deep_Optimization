"""Bounded, non-checkpointed reuse of versioned parameter quantization."""

from collections import OrderedDict
from threading import Lock
from weakref import ref

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .._execution import finite_checks_enabled


class WeightCache:
    def __init__(self, capacity=4):
        self.capacity = capacity
        self._entries = OrderedDict()
        self._lock = Lock()

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __reduce__(self):
        # Deep copies and whole-module checkpoints start with an empty cache.
        return type(self), (self.capacity,)

    def quantize(self, weight, *, fp8_dtype, block_shape, quantizer, paired=False, recipe=None):
        if weight.is_inference() or not torch.is_autocast_cache_enabled():
            self.clear()
            return quantizer(weight, fp8_dtype=fp8_dtype, block_shape=block_shape)
        owner = weight._base if weight._base is not None else weight
        version = weight._version
        fake = isinstance(weight, FakeTensor)
        # Fake execution has no data address or real CUDA stream. Base identity
        # plus storage offset distinguishes views without reading real payloads.
        address = weight.storage_offset() if fake else weight.data_ptr()
        stream = (torch.cuda.current_stream(weight.device).cuda_stream
                  if not fake and weight.device.type == "cuda" else None)
        key = (id(owner), version, address, tuple(weight.shape), tuple(weight.stride()),
               weight.dtype, weight.device, fp8_dtype, block_shape, finite_checks_enabled(), stream, paired, recipe)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0]() is owner:
                self._entries.move_to_end(key)
                return entry[1]
            # Optimizer/load/copy updates invalidate all layouts of an owner.
            def changed_generation(item):
                previous = item[-1]
                return (recipe is not None and previous is not None
                        and previous[0][:2] == recipe[0][:2] and previous != recipe)

            stale = [item for item, (reference, _) in self._entries.items()
                     if reference() is None or (reference() is owner and
                                                (item[1] != version or changed_generation(item)))]
            for item in stale:
                del self._entries[item]
        # A cache populated during inference_mode must remain usable by a later
        # training forward, whose autograd context saves these tensors.
        with torch.inference_mode(False):
            quantized = quantizer(weight, fp8_dtype=fp8_dtype, block_shape=block_shape)
        with self._lock:
            self._entries[key] = ref(owner), quantized
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return quantized
