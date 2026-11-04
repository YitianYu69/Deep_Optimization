"""Real ViT and Hugging Face training/checkpoint behavior on the reference backend."""

from copy import deepcopy
import io
import unittest

import torch

from deepopt.opt import transform_model
from tests.opt.full_model_support import (
    assert_relative_error, fp8_plan, logits_and_loss, make_gpt2, make_llama,
    make_timm_vit, make_vit,
)


class FullModelBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def _exercise(self, factory, mode):
        original, inputs, labels = factory()
        reference = deepcopy(original).train()
        model, report = transform_model(original.train(), plan=fp8_plan(), trace_mode=mode)
        self.assertTrue(report.converted_modules or report.function_calls)
        actual, loss = logits_and_loss(model, inputs, labels)
        expected, expected_loss = logits_and_loss(reference, inputs, labels)
        self.assertEqual(actual.shape, expected.shape)
        assert_relative_error(self, actual, expected, 0.15, "model logits vs FP32")
        assert_relative_error(self, loss, expected_loss, 0.05, "model loss vs FP32")

        loss.backward()
        expected_loss.backward()
        actual_parameters = dict(model.named_parameters())
        reference_parameters = dict(reference.named_parameters())
        self.assertEqual(actual_parameters.keys(), reference_parameters.keys())
        actual_grads, expected_grads = [], []
        for name, parameter in actual_parameters.items():
            baseline = reference_parameters[name]
            self.assertEqual(parameter.grad is None, baseline.grad is None, name)
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                actual_grads.append(parameter.grad.flatten())
                expected_grads.append(baseline.grad.flatten())
        self.assertTrue(actual_grads)
        assert_relative_error(
            self, torch.cat(actual_grads), torch.cat(expected_grads), 0.25,
            "all parameter gradients vs FP32",
        )

        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in model.named_parameters()))

        # The transformed model retains the original checkpoint key contract.
        checkpoint = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        restored_original, _, _ = factory()
        restored_original.load_state_dict(saved["model"], strict=True)
        restored, _ = transform_model(restored_original, plan=fp8_plan(), trace_mode=mode)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01, momentum=0.9)
        restored_optimizer.load_state_dict(saved["optimizer"])

        for candidate, opt in ((model, optimizer), (restored, restored_optimizer)):
            _, next_loss = logits_and_loss(candidate, inputs, labels)
            next_loss.backward()
            opt.step()
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)

    def test_torchvision_vit_full_graph_forward_backward_and_checkpoint_resume(self):
        self._exercise(make_vit, "full")

    def test_timm_vit_module_conversion_forward_backward_and_checkpoint_resume(self):
        self._exercise(make_timm_vit, "modules")

    def test_hf_gpt2_module_conversion_training_and_checkpoint_resume(self):
        self._exercise(make_gpt2, "modules")

    def test_hf_llama_auto_fallback_training_and_checkpoint_resume(self):
        self._exercise(make_llama, "auto")


if __name__ == "__main__":
    unittest.main()
