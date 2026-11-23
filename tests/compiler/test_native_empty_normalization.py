"""Empty normalization retains native derivatives and mutable-buffer semantics."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.opt.nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOLayerNorm


class EmptyNormalizationTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def check(self, kind, shape, *, compressed, affine, needs, training=True,
              dtype=torch.bfloat16, momentum=.1, track=True):
        plan = dict(fp8=False, activation_compression=dict(enabled=compressed, bits=2, group_size=1024))
        if kind == "layer":
            original = nn.LayerNorm(shape[-1], elementwise_affine=affine)
            module = DOLayerNorm(original, plan=plan)
        elif kind == "group":
            module = DOGroupNorm(nn.GroupNorm(2, shape[1], affine=affine), plan=plan)
        else:
            module = DOBatchNorm2d(nn.BatchNorm2d(shape[1], affine=affine, momentum=momentum,
                                                track_running_stats=track), plan=plan)
        module.to(device=self.device, dtype=dtype).train(training)
        if affine:
            module.weight.requires_grad_(needs[1])
            module.bias.requires_grad_(needs[2])
        root = nn.Module()
        root.add_module("norm", module)
        graph = torch.fx.Graph()
        value = graph.placeholder("x")
        graph.output(graph.call_module("norm", (value,)))
        model = torch.fx.GraphModule(root, graph)
        x = torch.empty(shape, device=self.device, dtype=dtype, requires_grad=needs[0])
        ShapeProp(model).propagate(x)
        reference = deepcopy(model)
        report = CompilationReport()
        compiled = compile_fx(model, [x], config=BackendConfig(require_full_aot=True), report=report)
        persistent = {value.untyped_storage()._cdata for value in (*model.parameters(), *model.buffers())}
        before = {name: value.detach().clone() for name, value in model.named_buffers()}
        for invocation in range(3):
            ref = x.detach().clone().requires_grad_(needs[0])
            expected = reference(ref)
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                actual = compiled(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, expected.dtype)
            if actual.requires_grad:
                actual.sum().backward()
                expected.sum().backward()
                if needs[0]:
                    torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
                for a, b in zip(model.parameters(), reference.parameters()):
                    if a.requires_grad:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            for value in saved:
                self.assertNotEqual(value.dtype, torch.int32, "Empty input must not create a codec payload")
                if value.is_floating_point() and value.numel():
                    self.assertIn(value.untyped_storage()._cdata, persistent,
                                  "Empty adjoint must not retain nonpersistent dense state")
            expected_buffers = dict(reference.named_buffers())
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, expected_buffers[name], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(value).all())
                if name.endswith("num_batches_tracked") and training:
                    torch.testing.assert_close(value, before[name] + invocation + 1, rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for graph in report.graphs:
            self.assertFalse(graph.eager_autograd_operations)
            self.assertFalse(graph.legacy_backward_operations)

    def test_empty_layer_group_batch_match_native_and_elide_state(self):
        cases = (("layer", (0, 1024)), ("layer", (2, 0)), ("layer", (0, 0)),
                 ("group", (0, 4, 16, 32)), ("batch", (0, 4, 16, 32)),
                 ("batch", (2, 4, 0, 8)), ("batch", (2, 4, 8, 0)))
        for kind, shape in cases:
            for dtype in (torch.bfloat16, torch.float32):
                for compressed in (False, True):
                    for affine, masks in ((True, ((True, True, True), (False, True, True),
                                                  (True, False, False), (False, False, True),
                                                  (False, False, False))),
                                          (False, ((True, False, False), (False, False, False)))):
                        for needs in masks:
                            with self.subTest(kind=kind, shape=shape, dtype=dtype, compressed=compressed,
                                              affine=affine, needs=needs):
                                self.check(kind, shape, compressed=compressed, affine=affine, needs=needs, dtype=dtype)

    def test_batch_training_eval_tracking_and_cumulative_momentum(self):
        for compressed in (False, True):
            for training in (False, True):
                for momentum in (None, .1):
                    for track in (False, True):
                        with self.subTest(compressed=compressed, training=training, momentum=momentum, track=track):
                            self.check("batch", (0, 4, 16, 32), compressed=compressed, affine=True,
                                       needs=(True, True, True), training=training, momentum=momentum, track=track)


@unittest.skipUnless(torch.cuda.is_available(), "Empty normalization CUDA qualification requires CUDA")
class EmptyNormalizationCUDATests(EmptyNormalizationTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
