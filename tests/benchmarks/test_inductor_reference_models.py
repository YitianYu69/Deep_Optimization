"""Adapter contracts and explicit CPU probes, not full-model CUDA evidence."""

from contextlib import nullcontext
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from benchmarks.compiler import inductor_reference_models as adapter


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 1)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(4, 1000)

    def forward(self, images):
        return self.head(self.dropout(self.features(images).mean((2, 3))))


class InductorReferenceModelTests(unittest.TestCase):
    def setUp(self):
        state = adapter.capture_rng()
        self.addCleanup(adapter.restore_rng, state)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_planning_is_standard_library_only_with_all_full_model_counts(self):
        source = Path(adapter.__file__).resolve()
        code = f'''import runpy, json, sys
module = runpy.run_path({str(source)!r})
assert not any(name in sys.modules for name in ("torch", "torchvision", "deepopt"))
print(json.dumps([module["describe_reference_model"](name) for name in module["MODEL_NAMES"]]))
'''
        result = subprocess.run([sys.executable, "-S", "-c", code], text=True,
                                capture_output=True, timeout=20, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        descriptions = json.loads(result.stdout)
        self.assertEqual([row["expected_parameter_count"] for row in descriptions],
                         [11689512, 86567656, 5288548, 32317999])
        for row in descriptions:
            self.assertEqual(row["input_shape"], [1, 3, 224, 224])
            self.assertTrue(row["full_model_reference"])
            self.assertIsNone(row["weights"])
            self.assertFalse(row["pretrained"])
            self.assertFalse(row["deepopt_native_autodiff"])
            self.assertFalse(row["cuda_verified"])
            self.assertFalse(row["compression"]["enabled"])
            self.assertEqual(row["operator_policy"], "native_torch_ops_only")
            self.assertTrue(row["native_fx_provenance_required"])

    def test_full_classifier_factories_receive_weights_none_without_real_allocations(self):
        # These are allocation-free factory contract mocks, not full forward
        # executions or a substitute corpus entry for the canonical model.
        class SizedMock(nn.Module):
            def __init__(self, count):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(count))

            def _apply(self, function, recurse=True):
                # Count/factory mock only: FakeTensor's weakrefs cannot be
                # swapped by Module.to in this Torch build. Real probes below
                # exercise actual device transfers and complete updates.
                return self

        for name, factory_name, count in (("resnet18", "resnet18", 11689512),
                                          ("vit-base", "vit_b_16", 86567656),
                                          ("efficientnet-b0", "efficientnet_b0", 5288548)):
            with self.subTest(name=name), FakeTensorMode():
                factory = Mock(side_effect=lambda weights, count=count: SizedMock(count))
                models = SimpleNamespace(**{factory_name: factory})

                def load(module):
                    return models if module == "torchvision.models" else importlib.import_module(module)

                with patch.object(adapter, "import_module", side_effect=load):
                    built = adapter.build_reference_model(name, device="cpu")
                factory.assert_called_once_with(weights=None)
                self.assertEqual(built["metadata"]["parameter_count"], count)
                self.assertEqual(built["metadata"]["parameter_bytes"], 4 * count)
                self.assertEqual(built["metadata"]["autodiff_owner"], "pytorch_aotautograd")
                self.assertEqual(built["inputs"][1].dtype, torch.int64)
                self.assertTrue(built["model"].training)

    def test_full_model_rejects_a_silent_tiny_substitute_and_missing_torchvision(self):
        with patch.object(adapter, "_full_network", return_value=TinyClassifier()):
            with self.assertRaisesRegex(ValueError, "identity differs"):
                adapter.build_reference_model("resnet18", device="cpu")

        def unavailable(module):
            if module == "torchvision.models":
                raise ImportError("unavailable torchvision")
            return importlib.import_module(module)

        with patch.object(adapter, "import_module", side_effect=unavailable):
            with self.assertRaisesRegex(RuntimeError, "no model substitute or download"):
                adapter.build_reference_model("vit-base", device="cpu")

    def test_probe_requires_an_explicit_label_and_preserves_distinct_identity(self):
        with self.assertRaisesRegex(ValueError, "probe_label"):
            adapter.build_reference_model("resnet18", device="cpu", probe_factory=TinyClassifier)
        with self.assertRaisesRegex(ValueError, "image_size=224"):
            adapter.describe_reference_model("resnet18", image_size=19)
        with self.assertRaisesRegex(ValueError, "group_size=1024"):
            adapter.describe_reference_model("resnet18", group_size=1024.)
        result = adapter.build_model("resnet18", "cpu", image_size=19,
                                     probe_factory=TinyClassifier, probe_label="tiny-classifier-cpu")
        metadata = result["metadata"]
        self.assertEqual(metadata["model_scope"], "diagnostic_probe")
        self.assertFalse(metadata["full_model_reference"])
        self.assertNotEqual(metadata["parameter_count"], metadata["expected_parameter_count"])
        self.assertEqual(metadata["compression"]["group_size"], 1024)

    def test_classifier_probe_complete_update_uses_bf16_fp32_ce_and_rng_hooks(self):
        built = adapter.build_reference_model("efficientnet-b0", device="cpu", image_size=19,
            probe_factory=TinyClassifier, probe_label="tiny-stochastic-classifier-cpu", seed=31)
        self.assertIs(built["execution_context"], nullcontext)
        self.assertFalse(built["metadata"]["forward_context_required"])
        self.assertFalse(built["metadata"]["prepared_custom_autograd"])
        model, inputs = built["model"], built["inputs"]
        control = deepcopy(model)
        control_inputs = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in inputs)
        expected_optimizer = built["optimizer_factory"](control.parameters())
        rng = built["capture_rng"]()
        with built["execution_context"]():
            expected = control(*control_inputs)
        built["restore_rng"](rng)
        with built["execution_context"]():
            actual = model(*inputs)
        self.assertEqual(actual["loss"].dtype, torch.float32)
        self.assertEqual(actual["logits"].dtype, torch.bfloat16)
        self.assertEqual(actual["logits"].shape, (1, 1000))
        torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
        built["loss_fn"](actual).backward()
        expected["loss"].backward()
        self.assertEqual(inputs[0].grad.dtype, torch.float32)
        before = model.network.head.weight.detach().clone()
        built["optimizer"].step()
        expected_optimizer.step()
        self.assertFalse(torch.equal(before, model.network.head.weight))
        for parameter, other in zip(model.parameters(), control.parameters()):
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            torch.testing.assert_close(parameter, other, rtol=0, atol=0)

    def test_real_tiny_vae_probe_keeps_original_stochastic_loss_but_never_claims_full_recipe(self):
        from examples.vae.mbconv_vae import MBConvVAE_V2

        built = adapter.build_reference_model("vae-v2", device="cpu", image_size=19, beta=2e-6,
            probe_factory=lambda: MBConvVAE_V2(base_channels=2, max_channels=8, blocks_per_stage=1, latent_channels=1),
            probe_label="actual-tiny-vae-cpu", seed=11)
        model, images = built["model"], built["inputs"][0]
        state = built["capture_rng"]()
        first = model(images)
        built["restore_rng"](state)
        second = model(images)
        self.assertEqual(set(first), {"loss", "reconstruction", "mu", "logvar", "l1", "kl"})
        self.assertEqual(first["reconstruction"].shape, images.shape)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertEqual(first["loss"].dtype, torch.float32)
        torch.testing.assert_close(first["loss"], first["l1"] + 2e-6 * first["kl"], rtol=0, atol=0)
        first["loss"].backward()
        before = model.network.encoder[0].weight.detach().clone()
        built["optimizer"].step()
        self.assertFalse(torch.equal(before, model.network.encoder[0].weight))
        self.assertFalse(built["metadata"]["full_model_reference"])
        self.assertFalse(built["metadata"]["original_vae_optimizer_used"])
        self.assertEqual(built["metadata"]["optimizer_scope"], "diagnostic_sgd_not_original_recipe")

    def test_original_vae_full_factory_options_and_count_without_allocating_real_parameters(self):
        class SizedMock(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(32317999))

            def _apply(self, function, recurse=True):
                return self

        factory = Mock(side_effect=SizedMock)

        def load(module):
            if module == "examples.vae.mbconv_vae":
                return SimpleNamespace(MBConvVAE_V2=factory)
            return importlib.import_module(module)

        with FakeTensorMode(), patch.object(adapter, "import_module", side_effect=load), \
                patch.object(adapter, "_installed_identity", return_value={"test": "mocked_installed_identity"}) as identity:
            built = adapter.build_reference_model("vae-v2", device="cpu")
        identity.assert_called_once_with()
        factory.assert_called_once_with(base_channels=32, max_channels=512, blocks_per_stage=2, latent_channels=4)
        self.assertEqual(built["metadata"]["parameter_count"], 32317999)
        self.assertEqual(built["metadata"]["vae_blur_pool_policy"], "existing_transparent_primitive_decomposition")

    def test_compressed_recipe_is_rejected_before_dependency_or_model_loading(self):
        self.assertEqual(adapter.RECIPES, ("torch-bf16",))
        factory = Mock(side_effect=AssertionError("must not construct a model"))
        with patch.object(adapter, "_installed_identity") as identity, \
                patch.object(adapter, "import_module") as load:
            for name in adapter.MODEL_NAMES:
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "Unknown recipe"):
                        adapter.describe_reference_model(name, "deepopt-bf16-compressed", group_size=1024)
                    with self.assertRaisesRegex(ValueError, "Unknown recipe"):
                        adapter.build_model(name, "cpu", recipe="deepopt-bf16-compressed", group_size=1024,
                                            probe_factory=factory, probe_label="rejected-compression-probe")
        factory.assert_not_called()
        identity.assert_not_called()
        load.assert_not_called()



if __name__ == "__main__":
    unittest.main()
