"""Capture ownership and delegation tests; mocks do not prove GPU correctness."""

from contextlib import ExitStack, nullcontext
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt import CUDAGraphCore, TrainingRuntime, cuda_graph_core
from deepopt._config_resolution import ResolvedConfig
from deepopt.cuda_graph import CapturedFunction
from deepopt.cuda_graph._validation import (arguments, capture_guard, check_arguments,
                                            operation_policy_spec, static_arguments, tensor_spec)
from deepopt.cuda_graph.model import _own_backward_gradients
from deepopt.opt._autocast import _ActivationCompressionScope, fp8_context


class GraphLifecycleWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        self.stack.enter_context(patch("torch.cuda.current_device", return_value=0))
        self.stack.enter_context(patch("torch.cuda.device", side_effect=lambda *a: nullcontext()))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch("torch.cuda.Stream", return_value=Mock(device=torch.device("cuda:0"))))
        self.stack.enter_context(patch("torch.cuda.stream", side_effect=lambda *a: nullcontext()))
        self.native = self.stack.enter_context(patch("torch.cuda.CUDAGraph"))
        self.context = self.stack.enter_context(patch("torch.cuda.graph", side_effect=lambda *a, **k: nullcontext()))

    def test_context_records_then_replays_and_releases_native_graph(self):
        core = CUDAGraphCore("cuda:0")
        with core.capture(pool=(1, 2), capture_error_mode="thread_local") as handle:
            self.assertFalse(handle.captured)
            with self.assertRaisesRegex(RuntimeError, "successful"):
                handle.replay()
        self.context.assert_called_once_with(handle.graph, pool=(1, 2), stream=handle.stream,
                                             capture_error_mode="thread_local")
        self.assertTrue(handle.captured)
        handle.replay()
        self.native.return_value.replay.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            with handle:
                pass
        handle.close()
        handle.close()
        self.native.return_value.reset.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "successful"):
            handle.replay()

    def test_body_failure_never_becomes_replayable_and_releases_guard(self):
        handle = cuda_graph_core.capture()
        with self.assertRaisesRegex(ValueError, "body"):
            with handle:
                raise ValueError("body")
        self.assertFalse(handle.captured)
        with self.assertRaises(RuntimeError):
            handle.replay()
        with cuda_graph_core.capture():
            pass

    def test_enter_and_exit_failures_release_capture_guard(self):
        self.context.side_effect = RuntimeError("enter failed")
        with self.assertRaisesRegex(RuntimeError, "enter failed"):
            with cuda_graph_core.capture():
                pass
        self.context.side_effect = None
        self.context.return_value.__exit__.side_effect = RuntimeError("exit failed")
        handle = cuda_graph_core.capture()
        with self.assertRaisesRegex(RuntimeError, "exit failed"):
            with handle:
                pass
        self.assertFalse(handle.captured)
        with capture_guard():
            pass

    def test_nested_capture_and_close_during_capture_are_rejected(self):
        with cuda_graph_core.capture() as handle:
            with self.assertRaisesRegex(RuntimeError, "one DeepOpt"):
                with cuda_graph_core.capture():
                    pass
            with self.assertRaisesRegex(RuntimeError, "during capture"):
                handle.close()

    def test_external_capture_is_rejected_without_leaking_guard(self):
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with cuda_graph_core.capture():
                    pass
        with cuda_graph_core.capture():
            pass

    def test_stream_device_mismatch_rejected_before_native_capture(self):
        stream = Mock(device=torch.device("cuda:1"))
        with self.assertRaisesRegex(ValueError, "stream"):
            with cuda_graph_core.capture(stream=stream):
                pass
        self.native.assert_not_called()

    def test_warmup_joins_stream_even_on_failure(self):
        current, side = Mock(), Mock(device=torch.device("cuda:0"))
        with patch("torch.cuda.current_stream", return_value=current), \
                patch("torch.cuda.stream", return_value=nullcontext()):
            with self.assertRaisesRegex(ValueError, "warmup"):
                cuda_graph_core.warmup(Mock(side_effect=ValueError("warmup")), stream=side)
        side.wait_stream.assert_called_once_with(current)
        current.wait_stream.assert_called_once_with(side)


class CaptureContractsWhiteBoxTests(unittest.TestCase):
    def test_static_backward_buffers_do_not_corrupt_accumulation_or_grad_results(self):
        scratch = torch.zeros(1)
        class ReusableBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight, factor):
                ctx.save_for_backward(weight)
                ctx.factor = factor
                return weight.square() * factor

            @staticmethod
            def backward(ctx, grad):
                weight, = ctx.saved_tensors
                scratch.copy_(2 * weight * ctx.factor * grad)
                return scratch.detach(), None

        weight = nn.Parameter(torch.tensor([2.]))
        for factor in (1., 3.):
            output = _own_backward_gradients(ReusableBackward.apply(weight, factor))
            output.sum().backward()
        torch.testing.assert_close(weight.grad, torch.tensor([16.]))
        self.assertNotEqual(weight.grad.data_ptr(), scratch.data_ptr())
        first, = torch.autograd.grad(_own_backward_gradients(ReusableBackward.apply(weight, 1.)).sum(), weight)
        second, = torch.autograd.grad(_own_backward_gradients(ReusableBackward.apply(weight, 3.)).sum(), weight)
        torch.testing.assert_close(first, torch.tensor([4.]))
        torch.testing.assert_close(second, torch.tensor([12.]))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_constructing_core_and_decorators_does_not_initialize_cuda(self):
        with patch("torch.cuda.init", side_effect=AssertionError("initialized")), \
                patch("torch.cuda.is_available", side_effect=AssertionError("queried")):
            core = CUDAGraphCore()
            @core.capture
            def first(x):
                return x + 1
            @core.capture(warmup=2)
            def second(x):
                return x * 2
            self.assertIsInstance(first, CapturedFunction)
            self.assertEqual(second.__name__, "second")
            self.assertFalse(first.captured)

    def test_bad_options(self):
        with self.assertRaises(ValueError):
            CUDAGraphCore("cpu")
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cuda_graph_core.capture(warmup=value)
        with self.assertRaises(ValueError):
            cuda_graph_core.capture(capture_error_mode="invalid")
        with self.assertRaises(TypeError):
            cuda_graph_core.capture(123)

    def test_unavailable_cuda_has_actionable_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA-enabled"):
                with cuda_graph_core.capture():
                    pass

    def test_shape_stride_dtype_device_and_grad_signature(self):
        x = torch.zeros(2, 3)
        specs = (tensor_spec(x),)
        check_arguments((x.clone(),), {}, specs)
        for value in (torch.zeros(3, 2), x.double(), x.requires_grad_(), torch.zeros(3, 2).t()):
            with self.subTest(spec=tensor_spec(value)), self.assertRaisesRegex(ValueError, "must match"):
                check_arguments((value,), {}, specs)
        with self.assertRaises(TypeError):
            check_arguments((), {"x": x}, specs)
        with self.assertRaises(ValueError):
            arguments((torch.ones(2),), torch.device("cuda:0"))
        with self.assertRaises(TypeError):
            arguments([x], torch.device("cuda:0"))

    def test_static_inputs_own_storage_and_preserve_sliced_stride(self):
        x = torch.arange(12.).reshape(3, 4)[:, ::2].requires_grad_()
        copied, = static_arguments((x,))
        self.assertEqual(copied.stride(), x.stride())
        self.assertNotEqual(copied.data_ptr(), x.data_ptr())
        self.assertTrue(copied.is_leaf)
        self.assertTrue(copied.requires_grad)
        torch.testing.assert_close(copied, x)

    def test_decorator_replays_without_python_and_checks_signature_first(self):
        function = Mock()
        captured = CapturedFunction(function)
        captured._capture._state = "ready"
        captured._capture.device = "cuda:0"
        captured._inputs = (torch.zeros(2),)
        captured._specs = (tensor_spec(captured._inputs[0]),)
        captured._autocast = (False, None)
        captured._operation_policy = operation_policy_spec()
        captured._outputs = torch.ones(2)
        captured._capture.replay = Mock()
        with patch("torch.cuda.device", return_value=nullcontext()):
            self.assertIs(captured(torch.full((2,), 7.)), captured._outputs)
            with self.assertRaises(ValueError):
                captured(torch.ones(3))
        captured._capture.replay.assert_called_once()
        function.assert_not_called()
        torch.testing.assert_close(captured._inputs[0], torch.full((2,), 7.))

    def test_operation_policy_retains_explicit_disable_and_inherited_scope_semantics(self):
        original = operation_policy_spec()
        with _ActivationCompressionScope("cuda", {"enabled": True, "group_size": 512}, config=ResolvedConfig()).enter():
            enabled = operation_policy_spec()
            self.assertEqual(enabled[1], (True, 512, 2, None, False))
            with _ActivationCompressionScope("cuda", None, config=ResolvedConfig()).enter():
                self.assertEqual(operation_policy_spec(), enabled)
            with _ActivationCompressionScope("cuda", False, config=ResolvedConfig()).enter():
                self.assertNotEqual(operation_policy_spec(), enabled)
                self.assertEqual(operation_policy_spec()[1][0], False)
            with _ActivationCompressionScope("cuda", {"enabled": True, "norm_xhat_recenter": True}, config=ResolvedConfig()).enter():
                self.assertNotEqual(operation_policy_spec(), enabled)
                self.assertTrue(operation_policy_spec()[1][-1])
        self.assertEqual(operation_policy_spec(), original)
        with fp8_context("cuda", {"enabled": True, "intercept_linear": False}, config=ResolvedConfig()):
            no_interception = operation_policy_spec()
        with fp8_context("cuda", {"enabled": True, "intercept_linear": True}, config=ResolvedConfig()):
            intercepted = operation_policy_spec()
        self.assertEqual(no_interception[:2], intercepted[:2])
        self.assertNotEqual(no_interception[2], intercepted[2])

    def test_decorator_rejects_changed_compression_policy_before_replay(self):
        captured = CapturedFunction(Mock())
        captured._capture._state = "ready"
        captured._capture.device = "cuda:0"
        captured._inputs = (torch.zeros(2),)
        captured._specs = (tensor_spec(captured._inputs[0]),)
        captured._autocast = (False, None)
        captured._outputs = torch.ones(2)
        captured._capture.replay = Mock()
        with _ActivationCompressionScope("cuda", {"enabled": True, "group_size": 512}, config=ResolvedConfig()).enter():
            captured._operation_policy = operation_policy_spec()
            for policy in (False, {"enabled": True, "group_size": 129},
                           {"enabled": True, "bits": 4}, {"enabled": True, "attention_bits": 2}):
                with _ActivationCompressionScope("cuda", policy, config=ResolvedConfig()).enter():
                    with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "policy changed"):
                        captured(torch.full((2,), 7.))
        captured._capture.replay.assert_not_called()
        torch.testing.assert_close(captured._inputs[0], torch.zeros(2))

    def test_runtime_rejects_raw_capture_before_python_or_optimizer_mutation(self):
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.1), checks=False)
        runtime.device = torch.device("cuda:0")
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            for call in (lambda: runtime.compute(None, Mock()), lambda: runtime.backward(torch.tensor(1.)),
                         runtime.step, runtime.zero_grad, lambda: runtime.set_execution_mode("analyze")):
                with self.assertRaisesRegex(RuntimeError, "capture_model"):
                    call()
        self.assertEqual(runtime.mode_controller.completed_updates, 0)
        self.assertFalse(runtime._pending_gradients)


class ModelCaptureWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("deepopt.cuda_graph.model.cuda_device", return_value=torch.device("cpu")))
        self.stack.enter_context(patch("deepopt.cuda_graph.model.capture_guard", side_effect=lambda: nullcontext()))
        self.stack.enter_context(patch("torch.cuda.device", side_effect=lambda *a: nullcontext()))
        self.stack.enter_context(patch("torch.random.fork_rng", side_effect=lambda **k: nullcontext()))
        self.stack.enter_context(patch("torch.cuda.current_stream"))
        self.stack.enter_context(patch("torch.cuda.stream", side_effect=lambda *a: nullcontext()))
        self.graph_call = Mock()

        def make(model, args, **kwargs):
            original = model.forward if isinstance(model, nn.Module) else model
            def graphed(*args):
                self.graph_call()
                return original(*args)
            if isinstance(model, nn.Module):
                model.forward = graphed
                return model
            return graphed

        self.native = self.stack.enter_context(patch("torch.cuda.make_graphed_callables", side_effect=make))

    def test_statistics_distinguish_replay_from_mode_and_autocast_fallback(self):
        model = nn.Linear(3, 2)
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        self.assertEqual(handle.statistics, {"forward_replays": 0, "eager_mode_fallbacks": 0,
                                             "eager_autocast_fallbacks": 0})
        model(torch.ones(2, 3))
        model.eval()
        model(torch.ones(1, 3))
        model.train()
        with patch("deepopt.cuda_graph.model.autocast_spec", return_value=(True, torch.bfloat16)):
            model(torch.ones(1, 3))
        model(torch.ones(2, 3))
        expected = {"forward_replays": 2, "eager_mode_fallbacks": 1, "eager_autocast_fallbacks": 1}
        self.assertEqual(handle.statistics, expected)
        self.assertEqual(self.graph_call.call_count, 2)
        snapshot = handle.statistics
        snapshot["forward_replays"] = 999
        self.assertEqual(handle.statistics, expected)
        handle.close()
        self.assertEqual(handle.statistics, expected)

    def test_statistics_do_not_count_rejected_or_failed_forward_calls(self):
        model = nn.Linear(3, 2)
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 3))
        self.graph_call.side_effect = RuntimeError("replay failed")
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            model(torch.ones(2, 3))
        self.graph_call.side_effect = None
        model.eval()
        with self.assertRaises(RuntimeError):
            model(torch.ones(1, 2))
        self.assertEqual(handle.statistics, {"forward_replays": 0, "eager_mode_fallbacks": 0,
                                             "eager_autocast_fallbacks": 0})
        handle.close()

    def test_original_identity_optimizer_and_state_keys_survive_capture_and_close(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        original_parameters, keys = list(model.parameters()), list(model.state_dict())
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),), warmup=4)
        self.assertIs(handle.model, model)
        self.assertEqual(list(model.state_dict()), keys)
        self.assertTrue(all(a is b for a, b in zip(original_parameters, optimizer.param_groups[0]["params"])))
        handle(torch.ones(2, 3)).sum().backward()
        self.graph_call.assert_called_once()
        self.assertEqual(self.native.call_args.kwargs["num_warmup_iters"], 4)
        handle.close()
        self.assertNotIn("forward", model.__dict__)
        model(torch.ones(1, 3))
        handle.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            handle(torch.ones(2, 3))

    def test_replaced_storage_and_changed_input_fail_before_replay(self):
        model = nn.Linear(3, 2)
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        with self.assertRaisesRegex(ValueError, "must match"):
            model(torch.ones(1, 3))
        model.weight = nn.Parameter(model.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "storage"):
            handle(torch.ones(2, 3))
        self.graph_call.assert_not_called()

    def test_eval_or_child_mode_change_uses_original_forward(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.Dropout(.1))
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        model.eval()
        handle(torch.ones(1, 3))  # Eager path permits different validation sizes.
        model.train()
        model[1].eval()
        handle(torch.ones(1, 3))
        self.graph_call.assert_not_called()
        model.train()
        handle(torch.ones(2, 3))
        self.graph_call.assert_called_once()

    def test_failed_capture_restores_forward_and_buffers(self):
        model = nn.BatchNorm1d(3)
        original = {name: value.clone() for name, value in model.named_buffers()}
        def fail(model, *args, **kwargs):
            model.num_batches_tracked.add_(3)
            model.forward = Mock()
            raise ValueError("capture failed")
        self.native.side_effect = fail
        with self.assertRaisesRegex(ValueError, "capture failed"):
            cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        self.assertNotIn("forward", model.__dict__)
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, original[name])

    def test_reject_pending_gradients_hooks_and_double_capture(self):
        model = nn.Linear(3, 2)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(ValueError, "gradients"):
            cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        model.zero_grad(set_to_none=True)
        hook = model.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "hooks"):
            cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        hook.remove()
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        with self.assertRaisesRegex(ValueError, "already captured"):
            cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        handle.close()

    def test_close_never_overwrites_a_later_user_forward(self):
        model = nn.Linear(3, 2)
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        replacement = model.forward = Mock()
        with self.assertRaisesRegex(RuntimeError, "refusing"):
            handle.close()
        self.assertIs(model.forward, replacement)

    def test_high_level_callable_capture_keeps_function_and_input_gradients(self):
        def function(x, scale):
            return x * scale
        x, scale = torch.ones(2, requires_grad=True), torch.tensor(3., requires_grad=True)
        for method in (cuda_graph_core.capture_model, cuda_graph_core.capture_callable):
            with self.subTest(method=method.__name__):
                handle = method(function, (x, scale))
                self.assertIsNone(handle.model)
                self.assertIs(handle.function, function)
                actual = handle(x, scale)
                torch.testing.assert_close(actual, function(x, scale))
                gradients = torch.autograd.grad(actual.sum(), (x, scale))
                torch.testing.assert_close(gradients[0], torch.full((2,), 3.))
                torch.testing.assert_close(gradients[1], torch.tensor(2.))
                handle.close()

    def test_autograd_requires_explicit_differentiable_surface(self):
        with self.assertRaisesRegex(ValueError, "autograd=False"):
            cuda_graph_core.capture_callable(lambda x: x + 1, (torch.ones(2),))
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "inference_mode"):
            cuda_graph_core.capture_model(nn.Linear(2, 1), (torch.ones(2),))

    def test_inference_capture_installs_native_handle_and_restores_model(self):
        model = nn.Linear(3, 2).requires_grad_(False)
        # The fixture uses CPU tensors as stand-ins. Mock the native inference
        # boundary as a whole; the real nested constructor correctly rejects CPU.
        class InferenceCapture:
            def __init__(self, function, **options):
                self.graph = Mock()
                self.call = Mock(return_value=torch.ones(2, 2))

            def __call__(self, *args):
                return self.call(*args)

            def close(self):
                self.graph.reset()

        with patch("deepopt.cuda_graph.model.CapturedFunction", InferenceCapture):
            handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),), autograd=False)
            captured = handle._graphed
            native = captured.graph
            self.assertIs(handle.graph, native)
            self.assertIs(handle.model, model)
            model(torch.ones(2, 3))
            self.assertEqual(captured.call.call_count, 2)  # Capture setup and explicit model call.
            handle.close()
            native.reset.assert_called_once()
            self.assertNotIn("forward", model.__dict__)

    def test_autocast_change_falls_back_without_replaying_a_different_precision(self):
        model = nn.Linear(3, 2)
        handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
        with patch("deepopt.cuda_graph.model.autocast_spec", return_value=(True, torch.bfloat16)):
            model(torch.ones(1, 3))
        self.graph_call.assert_not_called()
        handle.close()

    def test_compression_scope_changes_fall_back_and_matching_scope_replays(self):
        model = nn.Linear(3, 2)
        with _ActivationCompressionScope("cuda", {"enabled": True, "group_size": 512}, config=ResolvedConfig()).enter():
            handle = cuda_graph_core.capture_model(model, (torch.ones(2, 3),))
            model(torch.ones(2, 3))
            for policy in (False, {"enabled": True, "group_size": 129},
                           {"enabled": True, "bits": 4}, {"enabled": True, "attention_bits": 2}):
                with _ActivationCompressionScope("cuda", policy, config=ResolvedConfig()).enter():
                    model(torch.ones(1, 3))
            with _ActivationCompressionScope("cuda", None, config=ResolvedConfig()).enter():
                model(torch.ones(2, 3))
        self.assertEqual(self.graph_call.call_count, 2)
        self.assertEqual(handle.statistics, {"forward_replays": 2, "eager_mode_fallbacks": 0,
                                             "eager_autocast_fallbacks": 4})
        handle.close()


if __name__ == "__main__":
    unittest.main()
