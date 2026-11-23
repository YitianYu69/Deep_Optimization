"""Full torchvision vision backbones, random weights and synthetic images only."""

from importlib.util import find_spec
import json
import os
from types import SimpleNamespace
import unittest

import torch

from activation_compression import toy_quest_saved_linear as toy


class QuestFullModelTests(unittest.TestCase):
    def setUp(self):
        if find_spec("torchvision") is None:
            message = "Full QuEST model tests require optional torchvision"
            if os.environ.get("DEEPOPT_REQUIRE_MODEL_TESTS") == "1":
                self.fail(message)
            self.skipTest(message)
        torch.manual_seed(14)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def exercise(self, name, dtype=torch.float32, device=torch.device("cpu"), image_size=32):
        args = SimpleNamespace(seed=14, dtype="bf16" if dtype == torch.bfloat16 else "fp32",
                               methods=["minmax", "quest-centered"], bits=[4], group_size=256,
                               hadamard_size=128, classes=3)
        model = toy.make_vision_model(name, classes=3, image_size=image_size)
        parameters = sum(value.numel() for value in model.parameters())
        expected_minimum = {"vit-b-16": 85_000_000, "efficientnet-b0": 4_000_000, "resnet18": 11_000_000}
        self.assertGreater(parameters, expected_minimum[name])
        if name == "vit-b-16":
            self.assertEqual(len(model.encoder.layers), 12)
            self.assertEqual(model.hidden_dim, 768)
            self.assertGreater(float(model.heads.head.weight.detach().norm()), 0.)
        source = toy.samples(2, 3 * image_size ** 2, "outliers", torch.Generator().manual_seed(14), .25)
        source = source.reshape(2, 3, image_size, image_size)
        rows = toy.graph_probe(model, source, args, device, dtype, classification=True)
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(rows[0]["gradient_cosine_similarity"]["global"], 1.)
        for row in rows:
            self.assertEqual(row["status"], "ok", row.get("error", name))
            self.assertTrue(row["forward_matches_native"])
            self.assertGreater(row["parameter_gradients"], 20)
            self.assertEqual(row["missing_parameter_gradients"], [])
            self.assertEqual(row["unexpected_parameter_gradients"], [])
            self.assertTrue(-1. <= row["gradient_cosine_similarity"]["global"] <= 1.)
            self.assertTrue(row["errors_vs_native"]["dx"]["finite"])
            self.assertGreater(row["errors_vs_native"]["dx"]["reference_l2"], 0.)
            self.assertGreater(row["errors_vs_native"]["dx"]["actual_l2"], 0.)
            calls = row["saved_activations"]["module_calls"]
            self.assertIn("Conv2d", calls)
            if name == "vit-b-16":
                self.assertIn("MultiheadAttention", calls)
                self.assertIn("LayerNorm", calls)
                self.assertIn("GELU", calls)
            else:
                self.assertIn("BatchNorm2d", calls)
                self.assertIn("SiLU" if name == "efficientnet-b0" else "ReLU", calls)
            if row["method"] != "native":
                self.assertGreater(row["saved_activations"]["compressed_tensors"], 10)
                compressed_types = {key for key, value in row["saved_activations"]["by_module"].items()
                                    if value["compressed_tensors"]}
                self.assertIn("Conv2d", compressed_types)
                self.assertIn("LayerNorm" if name == "vit-b-16" else "BatchNorm2d", compressed_types)
        json.dumps(rows, allow_nan=False)
        return rows

    def test_vit_base_patch16_full_backbone_forward_backward(self):
        self.exercise("vit-b-16")

    def test_efficientnet_b0_full_backbone_forward_backward(self):
        self.exercise("efficientnet-b0")

    def test_resnet18_full_backbone_forward_backward(self):
        self.exercise("resnet18")

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         "224-pixel CUDA model checks require BF16 support")
    def test_all_full_models_at_224_pixels_with_cuda_bf16(self):
        for name in toy.VISION_MODELS:
            with self.subTest(model=name):
                self.exercise(name, torch.bfloat16, torch.device("cuda"), image_size=224)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
