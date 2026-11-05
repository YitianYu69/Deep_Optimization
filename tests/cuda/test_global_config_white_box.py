"""CUDA configuration precedence and snapshots; mocks do not prove GPU replay."""

from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

import torch

from deepopt import CUDAGraphCore, cuda_graph_core
from deepopt.config import get_config, load_config, reset_config
from deepopt.cuda_graph import CapturedCallable, CapturedFunction, CapturedGraph


def graph_config(**options):
    return {"name": "capture-defaults", "seed": 17,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01},
            "runtime": {"device": "cpu", "cuda_graph": options}}


class CUDAGlobalConfigWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_loading_defaults_constructing_cores_and_decorators_never_initializes_cuda(self):
        with patch("torch.cuda.init", side_effect=AssertionError("initialized CUDA")), \
                patch("torch.cuda.is_available", side_effect=AssertionError("queried CUDA")), \
                patch("torch.cuda.current_device", side_effect=AssertionError("selected CUDA")):
            load_config(graph_config(device="cuda:1", warmup=5, capture_error_mode="thread_local"))
            core = CUDAGraphCore()
            raw = core.capture()
            decorated = core.capture(lambda value: value + 1)
        self.assertEqual(raw.device, "cuda:1")
        self.assertEqual(raw.warmup, 5)
        self.assertEqual(raw.capture_error_mode, "thread_local")
        self.assertIsNone(raw.graph)
        self.assertFalse(decorated.captured)
        self.assertEqual(decorated._capture.warmup, 5)

    def test_ordinary_cores_snapshot_but_singleton_uses_defaults_for_each_new_handle(self):
        load_config(graph_config(device="cuda:1", warmup=4))
        core = CUDAGraphCore()
        existing = cuda_graph_core.capture()
        load_config(graph_config(device="cuda:2", warmup=7))
        self.assertEqual(core.capture().warmup, 4)
        self.assertEqual(core.capture().device, "cuda:1")
        self.assertEqual(existing.warmup, 4)
        self.assertEqual(existing.device, "cuda:1")
        latest = cuda_graph_core.capture()
        self.assertEqual(latest.warmup, 7)
        self.assertEqual(latest.device, "cuda:2")
        changed = core.config
        changed["runtime"]["cuda_graph"]["warmup"] = 99
        self.assertEqual(core.capture().warmup, 4)

    def test_per_call_config_and_explicit_keywords_override_without_mutating_defaults(self):
        load_config(graph_config(device="cuda:1", warmup=5, autograd=True, allow_unused_input=True))
        before = get_config()
        core = CUDAGraphCore("cuda:0", config={"runtime": {"cuda_graph": {"warmup": 6}}})
        function = lambda value: value
        inputs = (object(),)
        with patch("deepopt.cuda_graph.core.CapturedCallable") as capture:
            core.capture_callable(
                function, inputs,
                config={"runtime": {"cuda_graph": {"device": "cuda:3", "warmup": 8}}},
                warmup=2, autograd=False, allow_unused_input=False,
            )
        self.assertEqual(capture.call_args.args, (function, inputs))
        options = capture.call_args.kwargs
        self.assertEqual(options["device"], "cuda:0")
        self.assertEqual(options["warmup"], 2)
        self.assertFalse(options["autograd"])
        self.assertFalse(options["allow_unused_input"])
        self.assertEqual(options["config"]["runtime"]["cuda_graph"]["warmup"], 2)
        self.assertEqual(get_config(), before)

    def test_device_defaults_do_not_turn_cpu_or_auto_into_a_cpu_capture(self):
        for runtime_device, expected in (("cpu", None), ("auto", None), ("cuda:2", "cuda:2")):
            with self.subTest(device=runtime_device):
                config = graph_config()
                config["runtime"]["device"] = runtime_device
                load_config(config)
                self.assertEqual(CUDAGraphCore().capture().device, expected)
        load_config(graph_config(device="cuda:3"))
        self.assertIsNone(CUDAGraphCore(None).capture().device)

    def test_direct_lazy_handles_keep_configuration_from_their_construction(self):
        load_config(graph_config(device="cuda:1", warmup=4))
        graph = CapturedGraph(config={"runtime": {"cuda_graph": {"capture_error_mode": "relaxed"}}})
        function = CapturedFunction(lambda value: value, config={"runtime": {"cuda_graph": {"warmup": 6}}})
        load_config(graph_config(device="cuda:2", warmup=9))
        self.assertEqual((graph.device, graph.warmup, graph.capture_error_mode), ("cuda:1", 4, "relaxed"))
        self.assertEqual((function._capture.device, function._capture.warmup), ("cuda:1", 6))
        self.assertFalse(function.captured)

    def test_direct_model_handle_resolves_config_before_native_dispatch(self):
        load_config(graph_config(warmup=5, allow_unused_input=True))
        function = lambda value: value * 2
        with patch("deepopt.cuda_graph.model.cuda_device", return_value=torch.device("cpu")), \
                patch("deepopt.cuda_graph.model.capture_guard", return_value=nullcontext()), \
                patch("torch.cuda.device", return_value=nullcontext()), \
                patch("torch.random.fork_rng", return_value=nullcontext()), \
                patch("torch.cuda.current_stream"), \
                patch("torch.cuda.stream", return_value=nullcontext()), \
                patch("torch.cuda.make_graphed_callables", return_value=function) as native:
            handle = CapturedCallable(function, (torch.ones(2, requires_grad=True),),
                                      config={"runtime": {"cuda_graph": {"warmup": 2}}})
        self.assertEqual(native.call_args.kwargs["num_warmup_iters"], 2)
        self.assertTrue(native.call_args.kwargs["allow_unused_input"])
        self.assertTrue(handle.autograd)
        load_config(graph_config(autograd=False, warmup=9))
        self.assertTrue(handle.autograd)
        self.assertEqual(handle.config["runtime"]["cuda_graph"]["warmup"], 2)
        handle.close()

    def test_warmup_uses_configured_count_and_explicit_iterations_win(self):
        load_config(graph_config(warmup=4))
        core = CUDAGraphCore("cuda:0")
        current = Mock()
        stream = Mock(device=torch.device("cuda:0"))
        with patch("deepopt.cuda_graph.core.cuda_device", return_value=torch.device("cuda:0")), \
                patch("torch.cuda.device", return_value=nullcontext()), \
                patch("torch.cuda.current_stream", return_value=current), \
                patch("torch.cuda.stream", side_effect=lambda *args: nullcontext()):
            for kwargs, expected in (({}, 4), ({"config": {"runtime": {"cuda_graph": {"warmup": 2}}}}, 2),
                                     ({"iterations": 1, "config": {"runtime": {"cuda_graph": {"warmup": 2}}}}, 1)):
                with self.subTest(kwargs=kwargs):
                    function = Mock()
                    core.warmup(function, stream=stream, **kwargs)
                    self.assertEqual(function.call_count, expected)
        self.assertEqual(current.wait_stream.call_count, 3)
        self.assertEqual(stream.wait_stream.call_count, 3)


if __name__ == "__main__":
    unittest.main()
