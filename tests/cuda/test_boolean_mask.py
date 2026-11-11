"""Real CUDA boolean packing, strided input and graph replay correctness."""

import importlib.util
import os
import unittest

import torch

from deepopt.opt.quantization.boolean import PackedBooleanMask, pack_boolean_mask, unpack_boolean_mask


REASON = ("NVIDIA CUDA hardware is required" if not torch.cuda.is_available() or torch.version.hip else
          "Triton is not installed" if importlib.util.find_spec("triton") is None else "")
if REASON and os.environ.get("DEEPOPT_REQUIRE_CUDA_BOOLEAN") == "1":
    raise RuntimeError(REASON)


@unittest.skipIf(REASON, REASON)
class BooleanMaskCUDATests(unittest.TestCase):
    def check_mask(self, mask):
        expected = pack_boolean_mask(mask.cpu())
        packed = pack_boolean_mask(mask)
        torch.testing.assert_close(packed.data.cpu(), expected.data, rtol=0, atol=0)
        restored = unpack_boolean_mask(packed)
        self.assertEqual(restored.dtype, torch.bool)
        self.assertEqual(restored.device, mask.device)
        self.assertTrue(restored.is_contiguous())
        torch.testing.assert_close(restored, mask, rtol=0, atol=0)

    def test_exact_bits_and_shapes_across_tails_scalar_and_large_masks(self):
        for count in (0, 1, 31, 32, 33, 4095, 4096, 4097, 1048583):
            for kind in ("random", "false", "true"):
                with self.subTest(count=count, kind=kind):
                    self.check_mask(torch.randint(2, (count,), device="cuda", dtype=torch.bool) if kind == "random"
                                    else torch.full((count,), kind == "true", device="cuda", dtype=torch.bool))
        self.check_mask(torch.tensor(True, device="cuda"))
        self.check_mask(torch.tensor(False, device="cuda"))
        self.check_mask(torch.empty(2, 0, 7, dtype=torch.bool, device="cuda"))
        packed = PackedBooleanMask(torch.tensor([-2147483648, -1], device="cuda", dtype=torch.int32), (33,))
        torch.testing.assert_close(unpack_boolean_mask(packed).cpu(),
                                   unpack_boolean_mask(PackedBooleanMask(packed.data.cpu(), (33,))), rtol=0, atol=0)

    def test_noncontiguous_offset_slice_broadcast_and_channels_last(self):
        source = torch.randint(2, (3, 9, 17), device="cuda", dtype=torch.bool)
        for mask in (source.transpose(0, 2), source[1:, 1::2, 2::3],
                     source[:, :1, :1].expand(3, 11, 17), source[0, 0, 0].expand(3, 5, 7),
                     source.unsqueeze(0).to(memory_format=torch.channels_last)):
            with self.subTest(shape=mask.shape, stride=mask.stride()):
                self.check_mask(mask)
        # Profiling warmed launches ensures packing never materializes a dense input.
        mask = source.transpose(0, 2)
        pack_boolean_mask(mask)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            pack_boolean_mask(mask)
        names = {event.key for event in profile.key_averages()}
        self.assertFalse(names & {"aten::clone", "aten::contiguous", "aten::copy_", "aten::reshape"})

    def test_capture_replay_reads_new_bits_on_a_nondefault_stream(self):
        source = torch.zeros(5, 33, dtype=torch.bool, device="cuda").T
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                unpack_boolean_mask(pack_boolean_mask(source))
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            packed = pack_boolean_mask(source)
            restored = unpack_boolean_mask(packed)
        pointers = packed.data.data_ptr(), restored.data_ptr()
        for value in (True, False, True):
            source.fill_(value)
            graph.replay()
            torch.testing.assert_close(restored, source, rtol=0, atol=0)
            self.assertEqual((packed.data.data_ptr(), restored.data_ptr()), pointers)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Two CUDA devices are required")
    def test_output_uses_input_device_when_another_device_is_current(self):
        with torch.cuda.device(0):
            mask = torch.randint(2, (33,), device="cuda:1", dtype=torch.bool)
            self.check_mask(mask)
            self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == "__main__":
    unittest.main()
