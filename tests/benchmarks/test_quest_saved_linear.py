"""The quantizer-only toy preserves forward/dx and really packs saved inputs."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from activation_compression import toy_quest_saved_linear as toy


class QuestToyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_hadamard_matches_matrix_and_preserves_rows_norms_and_inverse(self):
        matrix = torch.ones(1, 1, dtype=torch.float64)
        for _ in range(3):
            matrix = torch.kron(torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64), matrix)
        x = torch.randn(7, 24, dtype=torch.float64)
        actual = toy.hadamard(x, 8)
        expected = (x.reshape(-1, 8) @ matrix / 8 ** .5).reshape_as(x)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(toy.hadamard(actual, 8), x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1), rtol=1e-12, atol=1e-12)

    def test_real_packing_roundtrip_includes_negative_words_and_partial_words(self):
        for bits in (1, 2, 4, 8):
            codes = torch.randint(1 << bits, (7, 65))
            codes[0].fill_((1 << bits) - 1)
            packed = toy.pack_codes(codes, bits)
            self.assertEqual(packed.dtype, torch.int32)
            self.assertLess(int(packed[0, 0]), 0)
            torch.testing.assert_close(toy.unpack_codes(packed, bits, 65), codes, rtol=0, atol=0)

    def test_zero_blocks_and_saved_byte_accounting(self):
        for method in toy.METHODS:
            for bits in (1, 2, 4, 8):
                x = torch.zeros(5, 32, dtype=torch.bfloat16)
                packed = toy.encode(x, method, bits, 16, 8)
                torch.testing.assert_close(toy.decode(packed), x, rtol=0, atol=0)
                expected = packed.data.numel() * 4 + packed.scale.numel() * 4
                if method in ("minmax", "quest-centered"):
                    expected += packed.offset.numel() * 4
                self.assertEqual(packed.saved_bytes, expected)

    def test_gaussian_grid_preserves_reference_half_step_tie_phase(self):
        x = torch.tensor([[-1., 0., 1., 0.]])
        packed = toy.encode(x, "gaussian", 2, 4, 4)
        expected_step = 2 * toy.GAUSSIAN_CLIP[2] * (x.square().mean().sqrt()) / torch.tensor(3.)
        torch.testing.assert_close(packed.scale[0, 0], expected_step, rtol=0, atol=0)
        decoded = toy.decode(packed, torch.float32)
        self.assertEqual(float(decoded[0, 1]), float(-expected_step / 2))

    def test_chunked_codec_matches_unchunked_values_and_payloads(self):
        x = torch.randn(11, 64).bfloat16()
        for method in toy.METHODS:
            expected = toy.encode(x, method, 2, 16, 8)
            original = toy.MAX_CODEC_VALUES
            try:
                toy.MAX_CODEC_VALUES = 33
                actual = toy.encode(x, method, 2, 16, 8)
                torch.testing.assert_close(actual.data, expected.data, rtol=0, atol=0)
                torch.testing.assert_close(actual.scale, expected.scale, rtol=0, atol=0)
                if expected.offset is not None:
                    torch.testing.assert_close(actual.offset, expected.offset, rtol=0, atol=0)
                torch.testing.assert_close(toy.decode(actual), toy.decode(expected), rtol=0, atol=0)
            finally:
                toy.MAX_CODEC_VALUES = original

    def test_native_bit_label_matches_the_requested_compute_dtype(self):
        self.assertEqual(toy.cases(SimpleNamespace(dtype="fp32", bits=[], methods=[])), [("native", 32)])
        self.assertEqual(toy.cases(SimpleNamespace(dtype="bf16", bits=[], methods=[])), [("native", 16)])

    def test_input_distributions_are_finite_reproducible_and_use_only_the_given_generator(self):
        global_state = torch.random.get_rng_state().clone()
        for distribution in toy.DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                first = toy.samples(128, 32, distribution, torch.Generator().manual_seed(14), noise_std=.25)
                second = toy.samples(128, 32, distribution, torch.Generator().manual_seed(14), noise_std=.25)
                other = toy.samples(128, 32, distribution, torch.Generator().manual_seed(15), noise_std=.25)
                self.assertEqual(first.shape, (128, 32))
                self.assertEqual(first.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(first).all()))
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                self.assertFalse(torch.equal(first, other))
        torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)

    def test_extra_noise_has_the_requested_standard_deviation(self):
        for distribution in toy.DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                clean = toy.samples(256, 64, distribution, torch.Generator().manual_seed(14))
                unchanged = toy.samples(256, 64, distribution, torch.Generator().manual_seed(14), noise_std=0.)
                noisy = toy.samples(256, 64, distribution, torch.Generator().manual_seed(14), noise_std=.4)
                torch.testing.assert_close(clean, unchanged, rtol=0, atol=0)
                difference = noisy - clean
                self.assertAlmostEqual(float(difference.mean()), 0., delta=.015)
                self.assertAlmostEqual(float(difference.std()), .4, delta=.015)

    def test_new_inputs_have_the_documented_distribution_structure(self):
        def sample(distribution):
            return toy.samples(16384, 16, distribution, torch.Generator().manual_seed(14))
        uniform = sample("uniform")
        self.assertLessEqual(float(uniform.abs().max()), math.sqrt(3.))
        self.assertAlmostEqual(float(uniform.var()), 1., delta=.02)
        for distribution in ("laplace", "student-t", "sparse-spikes"):
            with self.subTest(distribution=distribution):
                value = sample(distribution)
                self.assertGreater(float((value.abs() > 3).float().mean()), .005)
        lognormal = sample("lognormal")
        lower_bound = -math.exp(.5) / math.sqrt(math.e * (math.e - 1))
        self.assertGreaterEqual(float(lognormal.min()), lower_bound)
        self.assertGreater(float(lognormal.max()), 10.)
        mixture = sample("gaussian-mixture")
        self.assertLess(float((mixture.abs() < .5).float().mean()), .05)
        self.assertAlmostEqual(float(mixture.var()), 1., delta=.02)
        correlated = sample("correlated")
        self.assertAlmostEqual(float(torch.corrcoef(correlated[:, :2].T)[0, 1]), .9, delta=.02)
        row_rms = sample("heteroscedastic").square().mean(-1).sqrt()
        self.assertGreater(float(row_rms.quantile(.9) / row_rms.quantile(.1)), 20.)

    def test_invalid_distributions_and_noise_levels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown input distribution"):
            toy.samples(4, 8, "unknown", torch.Generator())
        for noise_std in (-1., float("nan"), float("inf"), -float("inf")):
            with self.subTest(noise_std=noise_std):
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    toy.samples(4, 8, "gaussian", torch.Generator(), noise_std)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    toy.main(["--noise-std=" + str(noise_std)])
                self.assertEqual(raised.exception.code, 2)

    def test_noisy_distributions_support_matched_training_in_both_dtypes(self):
        args = SimpleNamespace(steps=2, seed=14, train_samples=32, valid_samples=16,
                               in_features=32, hidden=32, classes=3, batch_size=8,
                               group_size=16, hadamard_size=8, learning_rate=.01, noise_std=.25,
                               methods=list(toy.METHODS), bits=[2])
        for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            args.dtype = label
            for distribution in toy.DISTRIBUTIONS:
                with self.subTest(dtype=label, distribution=distribution):
                    args.train_distribution = distribution
                    with redirect_stdout(io.StringIO()):
                        rows = toy.train(args, torch.device("cpu"), dtype)
                    self.assertEqual(len(rows), 5)
                    for row in rows:
                        self.assertEqual(row["status"], "ok")
                        self.assertEqual(row["initial_validation"], rows[0]["initial_validation"])
                        self.assertEqual(len(row["curve"]), 2)
                        for point in row["curve"]:
                            self.assertTrue(-1. <= point["gradient_cosine_similarity"]["global"] <= 1.)
                    json.dumps(rows, allow_nan=False)

    def test_gradient_cosines_match_names_and_weight_global_by_gradient_norm(self):
        actual = {"weight": torch.tensor([3., 0.]), "bias": torch.tensor([4.])}
        reference = {"bias": torch.tensor([4.]), "weight": torch.tensor([0., 3.])}
        similarity = toy.gradient_cosine_similarity(actual, reference)
        self.assertEqual(similarity["per_parameter"], {"weight": 0., "bias": 1.})
        self.assertAlmostEqual(similarity["global"], 16 / 25)

    def test_gradient_cosines_handle_zero_and_nonfinite_vectors(self):
        zero = {"weight": torch.zeros(2)}
        self.assertEqual(toy.gradient_cosine_similarity(zero, zero)["global"], 1.)
        for value in (torch.ones(2), torch.tensor([float("nan"), 1.]), torch.tensor([float("inf"), 1.])):
            for actual, reference in ((zero, {"weight": value}), ({"weight": value}, zero)):
                similarity = toy.gradient_cosine_similarity(actual, reference)
                self.assertIsNone(similarity["global"])
                self.assertIsNone(similarity["per_parameter"]["weight"])
                json.dumps(similarity, allow_nan=False)
                self.assertEqual(toy.format_metric(similarity["global"]), "n/a")

    def test_centered_extension_preserves_group_means_and_constant_blocks(self):
        x = torch.randn(5, 32) + 3.
        packed = toy.encode(x, "quest-centered", 2, 16, 8)
        decoded = toy.decode(packed, torch.float32)
        torch.testing.assert_close(decoded.reshape(-1, 16).mean(-1), x.reshape(-1, 16).mean(-1), rtol=0, atol=5e-7)
        constant = torch.full((5, 32), .375, dtype=torch.bfloat16)
        for method in ("minmax", "quest-centered"):
            torch.testing.assert_close(toy.decode(toy.encode(constant, method, 2, 16, 8)), constant, rtol=0, atol=0)

    def test_matched_linear_forward_dx_db_and_native_dw_reconstruction(self):
        for dtype in (torch.float32, torch.bfloat16):
            original_x, original_w, original_b = torch.randn(7, 32), torch.randn(16, 32), torch.randn(16)
            upstream = torch.randn(7, 16).to(dtype)
            reference = None
            for method in ("native", *toy.METHODS):
                x = original_x.clone().requires_grad_()
                model = toy.ToyLinear(32, 16, method, 2, 16, 8)
                with torch.no_grad():
                    model.weight.copy_(original_w); model.bias.copy_(original_b)
                with toy.amp(torch.device("cpu"), dtype):
                    y = model(x)
                y.backward(upstream)
                values = (y.detach(), x.grad, model.weight.grad, model.bias.grad)
                if method == "native":
                    reference = values
                    continue
                for index in (0, 1, 3):
                    torch.testing.assert_close(values[index], reference[index], rtol=0, atol=0)
                decoded = toy.decode(toy.encode(original_x.to(dtype), method, 2, 16, 8), dtype)
                expected_dw = (upstream.T @ decoded).float()
                torch.testing.assert_close(model.weight.grad, expected_dw, rtol=0, atol=0)
                self.assertGreater(toy.error(model.weight.grad, reference[2])["relative_l2"], 0.)

    def test_saved_state_contains_packed_payload_without_dense_input(self):
        observed = []
        x = torch.randn(7, 32, requires_grad=True)
        model = toy.ToyLinear(32, 16, "quest-centered", 2, 16, 8)
        with torch.autograd.graph.saved_tensors_hooks(lambda value: observed.append(value) or value, lambda value: value):
            model(x).sum().backward()
        self.assertTrue(any(value.dtype == torch.int32 for value in observed))
        self.assertFalse(any(tuple(value.shape) == tuple(x.shape) for value in observed))
        self.assertTrue(all(value.dtype != torch.bool for value in observed))

    def test_frozen_weight_and_biasless_backward(self):
        model = toy.ToyLinear(32, 16, "quest", 2, 16, 8, bias=False)
        model.weight.requires_grad_(False)
        x = torch.randn(7, 32, requires_grad=True)
        dy = torch.randn(7, 16)
        y = model(x)
        y.backward(dy)
        torch.testing.assert_close(x.grad, dy @ model.weight, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_bad_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            toy.encode(torch.randn(3, 33), "minmax", 2, 16, 8)
        with self.assertRaisesRegex(ValueError, "Hadamard"):
            toy.encode(torch.randn(3, 32), "quest", 2, 16, 3)

    def test_copied_single_file_runs_full_comparisons_and_matched_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy_quest_saved_linear.py"
            shutil.copyfile(toy.__file__, path)
            command = [sys.executable, str(path), "--device", "cpu", "--rows", "16",
                "--in-features", "32", "--out-features", "16", "--hidden", "32",
                "--group-size", "16", "--hadamard-size", "8", "--bits", "2",
                "--steps", "4", "--train-samples", "32", "--valid-samples", "16",
                "--batch-size", "8", "--classes", "3", "--threads", "1", "--noise-std", "0.25",
                "--train-distribution", "student-t", "--modules", "none", "--output", "result.json"]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(len(report["probe"]), len(toy.DISTRIBUTIONS) * 4)
            self.assertEqual({row["distribution"] for row in report["probe"]}, set(toy.DISTRIBUTIONS))
            self.assertEqual(report["distribution_definitions"], toy.DISTRIBUTIONS)
            self.assertEqual(report["configuration"]["noise_std"], .25)
            self.assertEqual(report["configuration"]["train_distribution"], "student-t")
            self.assertEqual(len(report["training"]), 4)
            self.assertTrue(all(row["status"] == "ok" for row in report["training"]))
            initial = [row["initial_validation"] for row in report["training"]]
            self.assertTrue(all(row == initial[0] for row in initial))
            for label in ("dx_cos=", "dw_cos=", "db_cos=", "grad_cos="):
                self.assertIn(label, result.stdout)
            for row in report["probe"]:
                similarity = row["gradient_cosine_similarity"]
                self.assertAlmostEqual(similarity["per_parameter"]["weight"], row["errors_vs_native"]["dw"]["cosine"])
                self.assertAlmostEqual(similarity["per_parameter"]["bias"], 1.)
                self.assertAlmostEqual(row["errors_vs_native"]["dx"]["cosine"], 1.)
                self.assertTrue(-1. <= similarity["global"] <= 1.)
            for row in report["training"]:
                for point in row["curve"]:
                    similarity = point["gradient_cosine_similarity"]
                    self.assertEqual(set(similarity["per_parameter"]), {"0.weight", "0.bias", "2.weight", "2.bias"})
                    self.assertTrue(-1. <= similarity["global"] <= 1.)
                    for name, cosine in similarity["per_parameter"].items():
                        self.assertTrue(-1. <= cosine <= 1.)
                        if row["method"] == "native" or name.endswith(".bias"):
                            self.assertAlmostEqual(cosine, 1.)
            self.assertTrue(any(point["gradient_cosine_similarity"]["global"] < .999
                                for row in report["training"] if row["method"] != "native" for point in row["curve"]))


if __name__ == "__main__":
    unittest.main()
