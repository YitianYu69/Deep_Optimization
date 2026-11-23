"""Graph-wide QuEST toy behavior across vision module families."""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from activation_compression import toy_quest_saved_linear as toy


def arguments(dtype="fp32", methods=None, bits=None):
    return SimpleNamespace(seed=14, dtype=dtype, methods=methods or list(toy.METHODS), bits=bits or [2],
                           group_size=32, hadamard_size=8, classes=3)


class QuestModuleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_all_catalog_modules_match_forward_and_have_finite_gradients_in_both_dtypes(self):
        catalog = toy.module_cases()
        required = {"linear", "layernorm", "groupnorm", "multihead-attention", "transformer-encoder",
                    "relu", "gelu", "silu", "sigmoid", "tanh", "relu6", "depthwise-conv2d", "grouped-conv2d"}
        for rank in (1, 2, 3):
            required.update(f"{prefix}{rank}d" for prefix in (
                "conv", "conv-transpose", "batchnorm", "instancenorm", "maxpool", "avgpool",
                "adaptive-avgpool", "adaptive-maxpool", "max-unpool"))
        self.assertTrue(required <= catalog.keys())
        for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for name, (factory, shape) in catalog.items():
                with self.subTest(dtype=label, module=name):
                    torch.manual_seed(14)
                    model = factory()
                    source = torch.randn(2, *shape)
                    parameter_versions = {key: parameter._version for key, parameter in model.named_parameters()}
                    buffers = {key: value.clone() for key, value in model.named_buffers()}
                    rows = toy.graph_probe(model, source, arguments(label), torch.device("cpu"), dtype)
                    self.assertEqual(len(rows), 5)
                    self.assertTrue(rows[0]["saved_activations"]["module_calls"])
                    for row in rows:
                        self.assertEqual(row["status"], "ok", row.get("error", name))
                        self.assertTrue(row["forward_matches_native"])
                        self.assertEqual(row["errors_vs_native"]["output"]["max_abs"], 0.)
                        self.assertTrue(row["errors_vs_native"]["dx"]["finite"])
                        for cosine in row["gradient_cosine_similarity"]["per_parameter"].values():
                            self.assertTrue(cosine is None or -1. <= cosine <= 1.)
                    for key, parameter in model.named_parameters():
                        self.assertEqual(parameter._version, parameter_versions[key], key)
                    for key, value in model.named_buffers():
                        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
                    self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))
                    json.dumps(rows, allow_nan=False)

    def test_noisy_inputs_and_bit_widths_cover_each_non_linear_family(self):
        catalog = toy.module_cases()
        for name in ("conv2d", "layernorm", "relu", "multihead-attention", "avgpool2d"):
            factory, shape = catalog[name]
            for distribution in toy.DISTRIBUTIONS:
                with self.subTest(module=name, distribution=distribution):
                    source = toy.samples(2, math.prod(shape), distribution, torch.Generator().manual_seed(14), .25)
                    source = source.reshape(2, *shape)
                    rows = toy.graph_probe(factory(), source, arguments(bits=[1, 2, 4, 8]),
                                           torch.device("cpu"), torch.float32)
                    self.assertEqual(len(rows), 17)
                    for row in rows:
                        self.assertEqual(row["status"], "ok", row.get("error", name))
                        self.assertTrue(row["forward_matches_native"])
                    self.assertTrue(any(row["saved_activations"]["compressed_tensors"] > 0 for row in rows))

    def test_parameters_and_amp_casts_stay_exact_while_linear_input_is_compressed(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = nn.Linear(32, 16)
                reference = deepcopy(model)
                source, upstream = torch.randn(3, 32), torch.randn(3, 16).to(dtype)
                x, baseline = source.clone().requires_grad_(), source.clone().requires_grad_()
                codec = toy.SavedActivations(model, "quest", 2, 16, 8)
                with codec.capture(), toy.amp(torch.device("cpu"), dtype):
                    actual = model(x)
                with toy.amp(torch.device("cpu"), dtype):
                    expected = reference(baseline)
                actual.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, baseline.grad, rtol=0, atol=0)
                self.assertFalse(torch.equal(model.weight.grad, reference.weight.grad))
                self.assertEqual(codec.summary()["compressed_tensors"], 1)
                self.assertEqual(codec.summary()["exact_reasons"]["parameter_or_buffer"], 1)
                torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)

    def test_padded_noncontiguous_activations_restore_shape_dtype_and_domain(self):
        for method in toy.METHODS:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(method=method, dtype=dtype):
                    source = (torch.rand(7, 11, dtype=dtype) + 1).T.requires_grad_()
                    self.assertFalse(source.is_contiguous())
                    codec = toy.SavedActivations(nn.Identity(), method, 2, 32, 8)
                    saved = codec.pack(source)
                    self.assertIsInstance(saved, toy.SavedActivation)
                    self.assertEqual(saved.packed.spec.shape, (1, 96))
                    decoded = codec.unpack(saved)
                    self.assertEqual(decoded.shape, source.shape)
                    self.assertEqual(decoded.dtype, dtype)
                    self.assertTrue(bool(torch.isfinite(decoded).all()))
                    self.assertGreaterEqual(float(decoded.min()), float(source.detach().min()))
                    self.assertLessEqual(float(decoded.max()), float(source.detach().max()))
                    self.assertLess(codec.summary()["activation_bytes_after"], source.numel() * source.element_size())
                    self.assertEqual(codec.summary()["activation_bytes_after"], saved.packed.saved_bytes + 8)

    def test_convolution_weights_stay_exact_including_amp_and_transposed_variants(self):
        catalog = toy.module_cases()
        names = [f"{prefix}{rank}d" for prefix in ("conv", "conv-transpose") for rank in (1, 2, 3)]
        names += ["depthwise-conv2d", "grouped-conv2d"]
        for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            for name in names:
                with self.subTest(dtype=label, module=name):
                    factory, shape = catalog[name]
                    rows = toy.graph_probe(factory(), torch.randn(2, *shape), arguments(label, methods=["quest"]),
                                           torch.device("cpu"), dtype)
                    self.assertEqual(rows[1]["status"], "ok")
                    self.assertEqual(rows[1]["errors_vs_native"]["dx"]["max_abs"], 0.)
                    self.assertGreater(rows[1]["saved_activations"]["exact_reasons"]["parameter_or_buffer"], 0)
                    self.assertGreater(rows[1]["saved_activations"]["compressed_tensors"], 0)

    def test_small_tensors_stay_native_when_metadata_would_increase_payload(self):
        codec = toy.SavedActivations(nn.Identity(), "quest", 2, 256, 128)
        source = torch.randn(2, 2, requires_grad=True)
        saved = codec.pack(source)
        self.assertIsInstance(saved, torch.Tensor)
        torch.testing.assert_close(saved, source, rtol=0, atol=0)
        summary = codec.summary()
        self.assertEqual(summary["compressed_tensors"], 0)
        self.assertEqual(summary["native_fallback_reasons"], {"metadata_would_expand_tensor": 1})
        self.assertEqual(summary["activation_bytes_before"], summary["activation_bytes_after"])

    def test_dropout_indices_and_statistics_are_exact_and_parameterless_cosine_is_explicit(self):
        for model, shape in ((nn.Dropout(.25), (2, 8, 32)), (toy.PoolUnpool(2), (2, 8, 8, 8)),
                             (nn.BatchNorm2d(8), (2, 8, 8, 8)), (nn.GroupNorm(4, 8), (2, 8, 8, 8))):
            with self.subTest(module=type(model).__name__):
                codec = toy.SavedActivations(model, "quest-centered", 2, 32, 8)
                source = torch.randn(*shape, requires_grad=True)
                with codec.capture():
                    result = model(source)
                result.square().mean().backward()
                exact = codec.summary()["exact_reasons"]
                if not isinstance(model, nn.Dropout):
                    self.assertTrue(exact)
                if isinstance(model, toy.PoolUnpool):
                    self.assertIn("indices_or_integer_mask", exact)
                if isinstance(model, nn.Dropout):
                    self.assertEqual(codec.summary()["exact_mask_tensors"], 1)
                    self.assertEqual(codec.summary()["lossy_value_tensors"], 0)
        self.assertEqual(toy.gradient_cosine_similarity({}, {}), {"global": None, "per_parameter": {}})

    def test_hook_cleanup_after_forward_failure_and_errors_are_reported(self):
        class Broken(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def forward(self, x):
                self.linear(x)
                raise RuntimeError("intentional forward failure")
        model = Broken()
        rows = toy.graph_probe(model, torch.randn(2, 32), arguments(methods=["quest"]),
                               torch.device("cpu"), torch.float32)
        self.assertTrue(all(row["status"] == "error" for row in rows))
        self.assertTrue(all("intentional forward failure" in row["error"] for row in rows))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))

    def test_chunked_global_cosine_matches_concatenation(self):
        actual = {"first": torch.randn(13, 17), "second": torch.randn(23)}
        reference = {key: value + torch.randn_like(value) * .2 for key, value in actual.items()}
        expected = toy.error(torch.cat([value.flatten() for value in actual.values()]),
                             torch.cat([reference[key].flatten() for key in actual]))["cosine"]
        original = toy.MAX_CODEC_VALUES
        try:
            toy.MAX_CODEC_VALUES = 7
            self.assertAlmostEqual(toy.gradient_cosine_similarity(actual, reference)["global"], expected, places=12)
        finally:
            toy.MAX_CODEC_VALUES = original

    def test_cli_serializes_every_module_and_preserves_no_module_option(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "modules.json"
            command = [sys.executable, toy.__file__, "--device", "cpu", "--dtype", "bf16", "--steps", "0",
                       "--rows", "2", "--in-features", "32", "--out-features", "16", "--group-size", "32",
                       "--hadamard-size", "8", "--bits", "2", "--methods", "quest-centered",
                       "--distributions", "student-t", "--noise-std", ".25", "--threads", "1", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual({row["module"] for row in report["module_probes"]}, set(toy.module_cases()))
            self.assertEqual(len(report["module_probes"]), len(toy.module_cases()) * 2)
            self.assertEqual(report["model_probes"], [])
            self.assertTrue(all(row["status"] == "ok" for row in report["module_probes"]))
            result = subprocess.run([*command, "--modules", "none"], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(output.read_text())["module_probes"], [])


if __name__ == "__main__":
    unittest.main()
