"""Compile and replay byte-preserving FP8 layout conversions on a real GPU."""

import unittest

import torch

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class FP8LayoutCUDATests(unittest.TestCase):
    def test_transposed_tails_padding_formats_and_changing_value_graph_replay(self):
        from deepopt.opt.backends.triton.gemm.fp8 import _pack_fp8_rows

        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = torch.randn(65, 37, device="cuda").to(dtype).T
            source[0, 0] = -0.0
            for implementation in ("flat", "tiled", "auto"):
                actual = _pack_fp8_rows(source, 16, implementation=implementation)
                torch.testing.assert_close(actual[:, :source.shape[1]].float(), source.float(), rtol=0, atol=0)
                self.assertTrue(torch.equal(actual[:, :source.shape[1]].contiguous().view(torch.uint8),
                                            source.contiguous().view(torch.uint8)))
                self.assertTrue(bool((actual[:, source.shape[1]:].float() == 0).all()))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _pack_fp8_rows(source, 16, implementation="tiled")
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = _pack_fp8_rows(source, 16, implementation="tiled")
            for _ in range(2):
                source.copy_(torch.randn(source.shape, device="cuda").to(dtype))
                graph.replay()
                torch.testing.assert_close(output[:, :source.shape[1]].float(), source.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
