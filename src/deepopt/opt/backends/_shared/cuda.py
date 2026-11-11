"""Pointer-free metadata shared by launches on the same CUDA device."""

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def device_metadata(index):
    properties = torch.cuda.get_device_properties(index)
    return properties.major * 10 + properties.minor, properties.multi_processor_count
