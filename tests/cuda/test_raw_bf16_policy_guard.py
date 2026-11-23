"""Raw replay scope guards; mocked CPU tests do not qualify CUDA execution."""

from contextlib import ExitStack, contextmanager, nullcontext
import unittest
from unittest.mock import Mock, patch

import torch

from deepopt._bf16_policy import bf16_policy_scope
from deepopt._config_resolution import ResolvedConfig
from deepopt.cuda_graph import CapturedGraph
from deepopt.cuda_graph._validation import operation_policy_spec
from deepopt.kernel_policy import kernel_policy_scope
from deepopt.opt._autocast import _ActivationCompressionScope, fp8_context


def compression_scope(settings):
    return _ActivationCompressionScope("cuda", settings, config=ResolvedConfig({})).enter()


class RawBF16PolicyGuardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.selected_device = None
        self.stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        self.stack.enter_context(patch("torch.cuda.current_device", return_value=0))
        self.stack.enter_context(patch("torch.cuda.device", side_effect=self.device_scope))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch("torch.cuda.Stream", return_value=Mock(device=torch.device("cuda:1"))))
        self.stack.enter_context(patch("torch.cuda.stream", side_effect=lambda *_: nullcontext()))
        self.native = self.stack.enter_context(patch("torch.cuda.CUDAGraph"))
        self.stack.enter_context(patch("torch.cuda.graph", side_effect=lambda *_, **__: nullcontext()))

    @contextmanager
    def device_scope(self, device):
        previous = self.selected_device
        self.selected_device = torch.device(device)
        try:
            yield
        finally:
            self.selected_device = previous

    def capture(self):
        handle = CapturedGraph(device="cuda:1", config={})
        with handle:
            pass
        self.addCleanup(handle.close)
        return handle

    def test_snapshot_and_matching_replay_use_explicit_graph_device(self):
        observed_devices = []

        def snapshot():
            observed_devices.append(self.selected_device)
            return operation_policy_spec()

        with patch("deepopt.cuda_graph.capture.operation_policy_spec", side_effect=snapshot):
            with bf16_policy_scope("bf16_storage", "cuda"):
                handle = self.capture()
                handle.replay()
        self.assertEqual(observed_devices, [torch.device("cuda:1")] * 2)
        self.native.return_value.replay.assert_called_once()
        self.assertTrue(handle.captured)

    def test_changed_bf16_scope_rejects_before_replay_then_matching_scope_recovers(self):
        handle = self.capture()
        with bf16_policy_scope("bf16_storage", "cuda"):
            with self.assertRaisesRegex(ValueError, "Operation policy"):
                handle.replay()
        self.native.return_value.replay.assert_not_called()
        self.assertTrue(handle.captured)
        handle.replay()
        self.native.return_value.replay.assert_called_once()

    def test_missing_retained_policy_rejects_before_native_replay(self):
        handle = self.capture()
        handle._operation_policy = None
        with self.assertRaisesRegex(ValueError, "missing"):
            handle.replay()
        self.native.return_value.replay.assert_not_called()

    def test_fp8_compression_and_kernel_scope_changes_reject_before_replay(self):
        handle = self.capture()
        contexts = (
            fp8_context("cuda", {"enabled": True, "intercept_linear": False}, config=ResolvedConfig({})),
            compression_scope({"enabled": True, "bits": 2, "group_size": 1024}),
            kernel_policy_scope("tilelang"),
        )
        for context in contexts:
            with self.subTest(context=type(context).__name__), context:
                with self.assertRaisesRegex(ValueError, "Operation policy"):
                    handle.replay()
        self.native.return_value.replay.assert_not_called()

    def test_explicit_disable_or_codec_changes_cannot_reuse_compressed_capture(self):
        with compression_scope({"enabled": True, "bits": 2, "group_size": 1024}):
            handle = self.capture()
            for changed in (False, {"enabled": True, "bits": 4, "group_size": 1024},
                            {"enabled": True, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}):
                with self.subTest(changed=changed), compression_scope(changed):
                    with self.assertRaisesRegex(ValueError, "Operation policy"):
                        handle.replay()
            handle.replay()
        self.native.return_value.replay.assert_called_once()

    def test_policy_guard_preserves_failed_capture_and_close_state_checks(self):
        handle = CapturedGraph(device="cuda:1", config={})
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with handle:
                raise RuntimeError("body failed")
        with self.assertRaisesRegex(RuntimeError, "successful"):
            handle.replay()
        handle.close()
        handle.close()
        self.native.return_value.reset.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "successful"):
            handle.replay()
        self.native.return_value.replay.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class RawBF16PolicyGuardCUDATests(unittest.TestCase):
    def test_raw_bf16_replay_rejects_scope_change_without_launching_captured_work(self):
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.ones(2048, device=device, dtype=torch.bfloat16)
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                x + 2
        current.wait_stream(stream)
        handle = CapturedGraph(device=device, stream=stream, config={})
        try:
            with bf16_policy_scope("bf16_storage", "cuda"), compression_scope(
                    {"enabled": True, "bits": 2, "group_size": 1024}):
                with handle:
                    output = x + 2
                x.fill_(3)
                handle.replay()
                torch.testing.assert_close(output, torch.full_like(x, 5))
            x.fill_(7)
            with self.assertRaisesRegex(ValueError, "Operation policy"):
                handle.replay()
            torch.testing.assert_close(output, torch.full_like(x, 5))
        finally:
            handle.close()


if __name__ == "__main__":
    unittest.main()
