"""Trainer accumulation, recomputation and exact restore for decoder layers."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report


class ConvTransposeIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1701)

    def test_trainer_accumulation_recompute_and_checkpoint_resume(self):
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                cfg = deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules", scaling_recipe="delayed",
                                        backward_precision="auto" if automatic else "native", intercept_linear=False)

                def build():
                    model = nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4),
                                          nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(4, 2, 1))
                    parameters = tuple(model.parameters())
                    trainer = deepopt.Trainer(model, torch.optim.SGD(parameters, lr=0.001, momentum=0.9), device="cpu",
                                              loss_fn=F.mse_loss, fp8=cfg, checks=False, accumulation_steps=2,
                                              config={"runtime": {"activation_compression": {"enabled": True, "group_size": 17}}})
                    self.assertEqual(tuple(trainer.model.parameters()), parameters)
                    return trainer

                trainer = build()
                for _ in range(2):
                    loss = trainer.runtime.compute(torch.randn(2, 4, 3, 3),
                                                   lambda x: deepopt.fp8_checkpoint(trainer.model, x).float().square().mean())
                    trainer.runtime.backward(loss)
                trainer.runtime.step()
                # GroupNorm already supplies FP8: do not update a second input
                # quantizer. Weight/output histories update once per optimizer step.
                self.assertEqual(trainer.model[2].fp8_scaling.updates[0].item(), 0)
                self.assertEqual(trainer.model[2].fp8_scaling.updates[1].item(), 1)
                self.assertEqual(trainer.model[2].output_scaling.scaling.updates[2].item(), 1)
                trainer = build()
                batches = [(torch.randn(2, 4, 3, 3), torch.randn(2, 2, 6, 6)) for _ in range(2)]
                trainer.fit(batches, epochs=1, progress_bar=False)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "transpose.pt"
                    save_checkpoint(path, trainer, generators={}, history=[{}])
                    restored = build()
                    restore_checkpoint(read_checkpoint(path), restored, generators={}, batch_size_mode="manual")
                rng = torch.get_rng_state().clone()
                trainer.fit(batches, epochs=2, progress_bar=False)
                expected = deepcopy(trainer.model.state_dict())
                torch.set_rng_state(rng)
                restored.fit(batches, epochs=2, progress_bar=False)
                for name, tensor in restored.model.state_dict().items():
                    if isinstance(tensor, torch.Tensor):
                        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0, msg=name)
                    else:
                        self.assertEqual(tensor, expected[name])
                self.assertEqual(precision_report(trainer.model)["analyses"], 1)
                self.assertEqual(precision_report(trainer.model)["fallbacks"], ())

    def test_delayed_graph_rejection_is_explicit(self):
        model, _ = prepare_fp8_model(nn.Sequential(nn.ConvTranspose2d(4, 4, 4, stride=2)),
                                     deepopt.FP8Config(enabled=True, backend="reference", scaling_recipe="delayed"))
        with self.assertRaisesRegex(ValueError, "Delayed scaling requires eager preprocessing"):
            deepopt.CUDAGraphCore("cuda").capture_model(model, (torch.randn(1, 4, 3, 3),), autograd=True)

    def test_nontraceable_model_preserves_eager_behavior_and_reports_failure(self):
        class ShapeGuard(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1)

            def forward(self, x):
                if x.ndim != 4:
                    raise ValueError("Expected batched image")
                return self.up(x)

        model, report = prepare_fp8_model(ShapeGuard(), deepopt.FP8Config(enabled=True, backend="reference", trace_mode="auto"))
        self.assertEqual(report.trace_mode, "modules")
        self.assertEqual(len(precision_report(model)["fallbacks"]), 2)
        self.assertEqual(tuple(model(torch.randn(1, 4, 3, 3)).shape), (1, 4, 6, 6))

    def test_fake_cuda_requires_no_kernel_import(self):
        from types import SimpleNamespace
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.nn.autograd.convolution_transpose.fp8 import _FP8ConvTransposeFunction
        from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry
        from deepopt.opt.policy import conv_plan

        for compressed in (False, True):
            with FakeTensorMode(), torch.no_grad():
                x = torch.empty(2, 4, 3, 5, device="cuda")
                w = torch.empty(4, 6, 4, 4, device="cuda")
                plan = conv_plan(deepopt.FP8Config(enabled=True, backend="triton"), activation_compression=compressed)
                geometry = ConvTransposeGeometry.from_tensors(x, w, stride=2, padding=1)
                ctx = SimpleNamespace(needs_input_grad=(True, True, False))
                ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
                y = _FP8ConvTransposeFunction.forward(ctx, x, w, None, geometry, plan, torch.float32, True, None, None, True, None)
                self.assertEqual(y.shape, (2, 6, 6, 10))
                dx, dw, *_ = _FP8ConvTransposeFunction.backward(ctx, torch.ones_like(y))
                self.assertEqual(dx.shape, x.shape)
                self.assertEqual(dw.shape, w.shape)


if __name__ == "__main__":
    unittest.main()
