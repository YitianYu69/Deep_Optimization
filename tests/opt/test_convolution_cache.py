"""Pointwise convolution weight views retain cache ownership and versions."""

import unittest

import torch
from torch import nn

from deepopt import FP8Config
from deepopt.opt.nn.convolution.modules import DOConv2d, DOConv3d
from deepopt.opt.policy import conv_plan
from deepopt.opt.quantization.cache import WeightCache


class _CountingWeightCache(WeightCache):
    def __init__(self):
        super().__init__()
        self.misses, self.owners = 0, []

    def quantize(self, weight, *, quantizer, **options):
        self.owners.append(weight._base if weight._base is not None else weight)

        def count(value, **quantization_options):
            self.misses += 1
            return quantizer(value, **quantization_options)

        return super().quantize(weight, quantizer=count, **options)


class PointwiseConvolutionCacheTests(unittest.TestCase):
    def test_matrix_view_reuses_original_parameter_cache_and_invalidates_on_update(self):
        for rank, source_type, adapter_type in ((2, nn.Conv2d, DOConv2d), (3, nn.Conv3d, DOConv3d)):
            for layout in ("contiguous", "channels_last", "strided"):
                with self.subTest(rank=rank, layout=layout):
                    source = source_type(4, 8, 1)
                    if layout == "channels_last":
                        source = source.to(memory_format=torch.channels_last if rank == 2
                                           else torch.channels_last_3d)
                    elif layout == "strided":
                        source.weight = nn.Parameter(torch.randn(8, 8, *([1] * rank))[:, ::2])
                    plan = conv_plan(FP8Config(enabled=True, backend="reference"))
                    model = adapter_type.from_conv(source, plan=plan)
                    cache = _CountingWeightCache()
                    model._fp8_weight_cache = cache
                    x = torch.randn(2, 4, *([3] * rank))
                    with torch.no_grad(), torch.autocast("cpu", enabled=False, cache_enabled=True):
                        first = model(x)
                        second = model(x)
                        self.assertEqual(cache.misses, 1)
                        self.assertTrue(all(owner is source.weight for owner in cache.owners))
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                        source.weight.add_(0.25)
                        model(x)
                        model(x)
                        self.assertEqual(cache.misses, 2)


if __name__ == "__main__":
    unittest.main()
