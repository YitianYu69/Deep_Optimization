"""Whole-model BF16 VAE ownership, numerical and saved-state contracts."""

from copy import deepcopy
import unittest

import torch

from deepopt import autocast
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model
from examples.vae.mbconv_vae import MBConvVAE_V2, SwiGLU
from examples.vae.train_vae import vae_loss


COMPRESSION = dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)


class NativeVAETests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(5201)

    def pair(self, compressed=True, *, share=False, full_width=False, policy="torch_amp"):
        compression = dict(COMPRESSION, enabled=compressed)
        config = (dict(runtime=dict(precision="bf16", bf16_policy=policy, fp8=dict(enabled=False),
                                    activation_compression=compression)) if policy == "bf16_storage" else {})
        model = MBConvVAE_V2(base_channels=32 if full_width else 4, max_channels=512 if full_width else 16,
                            blocks_per_stage=2 if full_width else 1, latent_channels=4 if full_width else 2).to(self.device)
        model = prepare_fp8_model(model, False, activation_compression=compression, config=config)[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True, "use_native_autodiff": True,
                     "share_compatible_packs": share}), config=config)
        return model, reference, compression

    def forward_loss(self, model, x, compression, seed=721, policy="torch_amp"):
        # Restore the RNG just before each paired forward, including latent z.
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        with autocast(self.device, dtype=torch.bfloat16, fp8=False, bf16_policy=policy,
                      activation_compression=compression, config={}):
            outputs = model(x)
        loss, terms = vae_loss(outputs[0], x, outputs[1], outputs[2], beta=1e-6)
        return outputs, loss, terms

    def assert_native(self, model):
        report = compilation_report(model)
        self.assertTrue(report.graphs)
        self.assertGreater(sum(graph.native_autodiff_operations for graph in report.graphs), 0)
        self.assertEqual(report.as_dict()["runtime_fallbacks"], 0)
        for graph in report.graphs:
            self.assertFalse(graph.eager_autograd_operations)
            self.assertFalse(graph.legacy_backward_operations)
            self.assertFalse(graph.unsupported_deepopt_operations)
        return report

    def check_pair(self, compressed, shape):
        model, reference, compression = self.pair(compressed)
        x = torch.randn(shape, device=self.device, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        expected, expected_loss, expected_terms = self.forward_loss(reference, rx, compression)
        actual, loss, terms = self.forward_loss(model, x, compression)
        for value, control in zip(actual, expected):
            torch.testing.assert_close(value, control, rtol=0, atol=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        for key in terms:
            torch.testing.assert_close(terms[key], expected_terms[key], rtol=0, atol=0)
        # Backward deliberately runs after the compression scope has exited.
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        for (name, value), (other_name, control) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            torch.testing.assert_close(value.grad, control.grad, rtol=0, atol=0, msg=name)
        self.assert_native(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
        optimizer.step()
        reference_optimizer.step()
        for value, control in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(value, control, rtol=0, atol=0)

    def test_whole_vae_matches_eager_for_even_and_odd_images(self):
        for compressed in (False, True):
            for shape in ((1, 3, 16, 16), (1, 3, 17, 19)):
                with self.subTest(compressed=compressed, shape=shape):
                    torch.compiler.reset()
                    self.check_pair(compressed, shape)

    def test_swiglu_training_then_no_grad_validation(self):
        class SwiGLUWithProjection(SwiGLU):
            def forward(self, x):
                gate = torch.nn.functional.silu(self.gate_proj(x))
                up = self.up_proj(x)
                return self.down_proj(gate * up), up

        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                torch.compiler.reset()
                compression = dict(COMPRESSION, enabled=compressed)
                source = SwiGLUWithProjection(8, 8).to(self.device)
                model, _ = prepare_fp8_model(source, False, config={}, activation_compression=compression)
                eager = deepcopy(model).eval()
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True}), config={})
                images = torch.randn(2, 8, 8, 8, device=self.device)

                def forward(current, value):
                    with autocast(self.device, dtype=torch.bfloat16, fp8=False,
                                  activation_compression=compression, config={}):
                        return current(value)

                training_output, _ = forward(model, images)
                training_output.float().sum().backward()
                self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

                model.eval()
                with torch.no_grad():
                    for batch in (images, images[:1]):
                        output, up = forward(model, batch)
                        expected, expected_up = forward(eager, batch)
                        for value, control in ((up, expected_up), (output, expected)):
                            self.assertFalse(value.requires_grad)
                            self.assertIsNone(value.grad_fn)
                            torch.testing.assert_close(value, control, rtol=0, atol=0, check_stride=True)
                report = self.assert_native(model)
                if self.device == "cuda":
                    self.assertGreater(report.graphs[-1].triton_regions, 0)

    def test_complete_saved_state_is_typed_and_compression_reduces_physical_storage(self):
        from benchmarks.analysis.measurement import SavedStorageLifetimes

        saved_bytes = {}
        for compressed in (False, True):
            torch.compiler.reset()
            model, _, compression = self.pair(compressed, share=True)
            x = torch.randn(1, 3, 16, 16, device=self.device, requires_grad=True)
            tracker = SavedStorageLifetimes((*model.parameters(), *model.buffers()))
            with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack):
                _, loss, _ = self.forward_loss(model, x, compression)
            saved_bytes[compressed] = tracker.snapshot()["live_nonpersistent_bytes"]
            loss.backward()
            self.assertEqual(tracker.snapshot()["live_unique_bytes"], 0)
            report = self.assert_native(model)
            if not compressed:
                continue
            shared = [decision for graph in report.graphs for decision in graph.pack_sharing_decisions
                      if decision["selected"]]
            self.assertTrue(shared)
            from examples.vae.qualification import saved_state_evidence

            self.assertTrue(saved_state_evidence(report.as_dict(), True)["passed"])
        self.assertLess(saved_bytes[True], saved_bytes[False])

    def test_full_width_cpu_saved_state_gate_and_exact_training_for_both_policies(self):
        from benchmarks.analysis.measurement import SavedStorageLifetimes
        from examples.vae.qualification import saved_state_evidence

        if self.device != "cpu":
            self.skipTest("Full-width CPU diagnostic; original224 CUDA qualification is separate")
        for policy in ("torch_amp", "bf16_storage"):
            with self.subTest(policy=policy):
                torch.compiler.reset()
                model, reference, compression = self.pair(share=True, full_width=True)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 32317999)
                x = torch.randn(1, 3, 16, 16, requires_grad=True)
                rx = x.detach().clone().requires_grad_()
                expected, expected_loss, expected_terms = self.forward_loss(reference, rx, compression, policy=policy)
                tracker = SavedStorageLifetimes((*model.parameters(), *model.buffers()))
                abi = []
                def save(value):
                    abi.append((tuple(value.shape), str(value.dtype)))
                    return tracker.pack(value)
                with torch.autograd.graph.saved_tensors_hooks(save, tracker.unpack):
                    actual, loss, terms = self.forward_loss(model, x, compression, policy=policy)
                report = self.assert_native(model)
                region_values = [value for graph in report.graphs for region in graph.native_autodiff_graphs
                                 for value in region["saved_state"]]
                # The native report describes model saves; the eager loss can
                # append its own small saves after that ABI.
                self.assertEqual(abi[:len(region_values)],
                                 [(tuple(v["shape"]), v["dtype"]) for v in region_values])
                before_classification = tracker.snapshot()
                evidence = saved_state_evidence(report.as_dict(), True)
                self.assertEqual(tracker.snapshot(), before_classification)
                aliases = [row for row in evidence["values"] if row["category"] == "persistent_model_state_reference"
                           and row["bytes"] >= 4096 and row["name"].endswith("__weight")]
                self.assertEqual(len(aliases), 20)
                for value, control in zip(actual, expected):
                    torch.testing.assert_close(value, control, rtol=0, atol=0)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                for name in terms:
                    torch.testing.assert_close(terms[name], expected_terms[name], rtol=0, atol=0)
                live = tracker.snapshot()["live_nonpersistent_bytes"]
                loss.backward(retain_graph=True)
                self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], live)
                model.zero_grad(set_to_none=True)
                x.grad = None
                loss.backward()
                expected_loss.backward()
                self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], 0)
                torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
                for (name, parameter), (_, control) in zip(model.named_parameters(), reference.named_parameters()):
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    torch.testing.assert_close(parameter.grad, control.grad, rtol=0, atol=0, msg=name)
                torch.optim.SGD(model.parameters(), lr=.001).step()
                torch.optim.SGD(reference.parameters(), lr=.001).step()
                for parameter, control in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter, control, rtol=0, atol=0)

    def test_selective_frozen_vae_paths_preserve_named_gradient_masks(self):
        for compressed in (False, True):
            for need_input, train_parameters in ((True, False), (False, True), (True, True), (False, False)):
                with self.subTest(compressed=compressed, input=need_input, parameters=train_parameters):
                    torch.compiler.reset()
                    model, reference, compression = self.pair(compressed)
                    for index, (parameter, control) in enumerate(zip(model.parameters(), reference.parameters())):
                        parameter.requires_grad_(train_parameters and index % 2 == 0)
                        control.requires_grad_(parameter.requires_grad)
                    x = torch.randn(1, 3, 17, 19, device=self.device, requires_grad=need_input)
                    rx = x.detach().clone().requires_grad_(need_input)
                    outputs, loss, _ = self.forward_loss(model, x, compression)
                    expected, expected_loss, _ = self.forward_loss(reference, rx, compression)
                    for actual, control in zip(outputs, expected):
                        torch.testing.assert_close(actual, control, rtol=0, atol=0)
                    self.assertEqual(loss.requires_grad, expected_loss.requires_grad)
                    if not loss.requires_grad:
                        continue
                    loss.backward()
                    expected_loss.backward()
                    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
                    for (name, parameter), (_, control) in zip(model.named_parameters(), reference.named_parameters()):
                        torch.testing.assert_close(parameter.grad, control.grad, rtol=0, atol=0, msg=name)
                    self.assert_native(model)

    def test_metadata_audit_uses_the_real_multiplication_compression_contract(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        saved = []
        with FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=False):
            x = torch.empty(2, 512, dtype=torch.bfloat16, requires_grad=True)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                          activation_compression=COMPRESSION, config={}):
                with torch.autograd.graph.saved_tensors_hooks(lambda v: saved.append(v) or v, lambda v: v):
                    output = x * x
            output.sum().backward()
        self.assertEqual([value.dtype for value in saved], [torch.int32, torch.float32, torch.float32])
        self.assertEqual(saved[0].shape, (1, 64))
        saved.clear()
        with FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=False):
            x = torch.empty(2, 4, 8, 8, requires_grad=True)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                          activation_compression=COMPRESSION, config={}):
                with torch.autograd.graph.saved_tensors_hooks(lambda v: saved.append(v) or v, lambda v: v):
                    output = torch.nn.functional.pad(x, (1, 2, 1, 2), mode="reflect")
            output.sum().backward()
        self.assertEqual(saved, [])


@unittest.skipUnless(torch.cuda.is_available(), "Whole VAE BF16 qualification requires CUDA")
class NativeVAECudaTests(NativeVAETests):
    device = "cuda"

    def test_full_width_storage_vae_small_batch_update(self):
        # Small spatial correctness gate for the complete original-width
        # architecture. Batch128/image224 capacity remains a separate H200 job.
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                torch.compiler.reset()
                native, eager, compression = self.pair(compressed, full_width=True, policy="bf16_storage")
                self.assertEqual(sum(parameter.numel() for parameter in native.parameters()), 32317999)
                x = torch.randn(1, 3, 32, 32, device="cuda", requires_grad=True)
                reference = x.detach().clone().requires_grad_()
                expected, expected_loss, _ = self.forward_loss(eager, reference, compression, policy="bf16_storage")
                actual, loss, _ = self.forward_loss(native, x, compression, policy="bf16_storage")
                for value, oracle in zip(actual, expected):
                    torch.testing.assert_close(value, oracle, rtol=0, atol=0)
                self.assertEqual(actual[0].dtype, torch.bfloat16)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                # Backward and updates occur after the forward policy scope.
                loss.backward()
                expected_loss.backward()
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                for (name, parameter), (_, oracle) in zip(native.named_parameters(), eager.named_parameters()):
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0, msg=name)
                self.assert_native(native)
                torch.optim.SGD(native.parameters(), lr=.001).step()
                torch.optim.SGD(eager.parameters(), lr=.001).step()
                for parameter, oracle in zip(native.parameters(), eager.parameters()):
                    torch.testing.assert_close(parameter, oracle, rtol=0, atol=0)

    # Explicit selectors prove these are device tests; the inherited FakeTensor
    # audit is deliberately absent from the strict CUDA coordinator stage.
    def test_whole_vae_matches_eager_for_even_and_odd_images(self):
        super().test_whole_vae_matches_eager_for_even_and_odd_images()

    def test_complete_saved_state_is_typed_and_compression_reduces_physical_storage(self):
        super().test_complete_saved_state_is_typed_and_compression_reduces_physical_storage()

    def test_selective_frozen_vae_paths_preserve_named_gradient_masks(self):
        super().test_selective_frozen_vae_paths_preserve_named_gradient_masks()

    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")


if __name__ == "__main__":
    unittest.main()
