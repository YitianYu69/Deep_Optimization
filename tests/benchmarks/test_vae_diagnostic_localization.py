"""Real CPU localization replays; CUDA gates remain separately executable."""

from collections import OrderedDict
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from examples.vae.diagnostic_localization import (
    BoundaryTrace, cancellation_cotangent, chain_replay_diagnostics,
    decoder_geometry_metadata, nearest_replay_diagnostics, _ordered_2x_adjoint, _prepared_pair,
)


class VAEDiagnosticLocalizationTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_decoder_geometry_receipt_never_claims_unexecuted_shapes(self):
        rows = decoder_geometry_metadata()
        self.assertEqual([row["module"] for row in rows], ["decoder.0", "decoder.4", "decoder.8"])
        self.assertEqual([row["output_shape"] for row in rows],
                         [[128, 512, 56, 56], [128, 352, 112, 112], [128, 192, 224, 224]])
        self.assertFalse(any(row["execution_observed"] for row in rows))
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            nearest_replay_diagnostics(large=True)

    def test_cancellation_cotangent_requires_fp32_accumulation_before_store(self):
        gradient = cancellation_cotangent((1, 2, 6, 10), layout="channels_last")
        oracle = _ordered_2x_adjoint(gradient, (1, 2, 3, 5), 7)
        self.assertTrue(torch.equal(oracle, torch.full_like(oracle, 1. / 512)))
        rounded_each_add = gradient[..., 0::2, 0::2] + gradient[..., 0::2, 1::2]
        rounded_each_add = rounded_each_add + gradient[..., 1::2, 0::2]
        rounded_each_add = rounded_each_add + gradient[..., 1::2, 1::2]
        self.assertFalse(torch.equal(rounded_each_add, oracle))

    def test_nearest_replays_real_cpu_forms_layouts_and_cotangents(self):
        rng = torch.get_rng_state().clone()
        report = nearest_replay_diagnostics(geometry_indices=(0,), chunk_elements=37)
        self.assertEqual(len(report["cases"]), 12)
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["cuda_execution_observed"])
        self.assertFalse(report["global_first_divergence_verified"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for row in report["cases"]:
            self.assertTrue(row["identical_inputs"])
            self.assertTrue(row["identical_cotangents"])
            self.assertTrue(row["native_ownership_verified"])
            self.assertFalse(row["target_geometry_executed"])
            self.assertTrue(row["candidate_boundaries"]["complete"], row["candidate_boundaries"])
        json.dumps(report, allow_nan=False)

    def test_chain_replays_match_cpu_parameter_gradients_and_boundaries(self):
        report = chain_replay_diagnostics(geometry_indices=(0,), chunk_elements=43)
        self.assertEqual(len(report["cases"]), 6)
        self.assertTrue(report["passed"], report)
        for row in report["cases"]:
            self.assertTrue(row["candidate_boundaries"]["complete"], row["candidate_boundaries"])
            self.assertTrue(row["candidate_boundaries"]["passed"], row["candidate_boundaries"])
            self.assertTrue(any(name.startswith("parameter_gradient/conv.") for name in row["comparisons"]))
        json.dumps(report, allow_nan=False)

    def test_matched_boundary_difference_is_labeled_and_inputs_are_not_retained(self):
        class ChangedNearest(nn.Upsample):
            def forward(self, value):
                return super().forward(value) + .125

        reference = BoundaryTrace(modules=("resize",), chunk_elements=11)
        candidate = BoundaryTrace(reference=reference, modules=("resize",), chunk_elements=13)
        source = torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True)
        source_ref = weakref.ref(source)
        model = nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2)))
        changed = nn.Sequential(OrderedDict(resize=ChangedNearest(scale_factor=2)))
        with reference.eager(model):
            output = model(source)
            output.backward(cancellation_cotangent(output.shape))
        output_ref = weakref.ref(output)
        with candidate.eager(changed):
            other = changed(source.detach().clone().requires_grad_())
            other.backward(cancellation_cotangent(other.shape))
        report = candidate.report()
        self.assertEqual(report["first_differing_boundary"]["module"], "resize")
        self.assertEqual(report["first_differing_boundary"]["role"], "output")
        self.assertFalse(report["passed"])
        self.assertFalse(report["global_first_divergence_verified"])
        self.assertFalse(model.resize._forward_hooks)
        del source, output, other
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(output_ref())

    def test_oversized_reference_is_explicitly_incomplete(self):
        reference = BoundaryTrace(modules=("resize",), max_bytes=4)
        model = nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2)))
        with reference.eager(model):
            model(torch.ones(1, 2, 3, 5))
        report = reference.report()
        self.assertFalse(report["complete"])
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(report["context_succeeded"])
        self.assertEqual(report["reference_bytes"], 0)
        self.assertEqual(report["boundaries"][0]["status"], "unavailable_reference_byte_budget")
        self.assertIsNone(report["first_differing_boundary"])

    def test_later_oracle_failure_preserves_complete_boundary_coverage(self):
        from examples.vae.qualification import NumericalComparisonError

        model = nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2)))
        reference = BoundaryTrace(modules=("resize",))
        candidate = BoundaryTrace(reference=reference, modules=("resize",))
        with reference.eager(model):
            output = model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
            cotangent = cancellation_cotangent(output.shape)
            output.backward(cotangent)
        with self.assertRaises(NumericalComparisonError), candidate.eager(model):
            output = model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
            output.backward(cotangent)
            raise NumericalComparisonError({"name": "gradient/input", "passed": False}, {})
        report = candidate.report()
        self.assertTrue(report["coverage_complete"], report)
        self.assertFalse(report["context_succeeded"])
        self.assertFalse(report["complete"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_reference_boundaries"], [])
        self.assertTrue(all(row["bitwise_equal"] for row in report["boundaries"]))
        self.assertEqual(report["errors"], [{"phase": "eager_context", "exception": "NumericalComparisonError"}])
        json.dumps(report, allow_nan=False)

    def test_bitwise_metric_distinguishes_signed_zero_without_failing_numerical_budget(self):
        model = nn.Sequential(OrderedDict(leaf=nn.Identity()))
        reference = BoundaryTrace(modules=("leaf",))
        candidate = BoundaryTrace(reference=reference, modules=("leaf",))
        with reference.eager(model):
            model(torch.tensor([-0., 1.], dtype=torch.bfloat16))
        with candidate.eager(model):
            model(torch.tensor([0., 1.], dtype=torch.bfloat16))
        report = candidate.report()
        self.assertTrue(report["passed"])
        self.assertTrue(report["coverage_complete"])
        self.assertTrue(report["context_succeeded"])
        self.assertFalse(report["boundaries"][0]["bitwise_equal"])
        self.assertEqual(report["boundaries"][0]["relative_l2"], 0.)

    def test_native_failure_retains_generated_code_before_any_module_match(self):
        from deepopt import autocast
        from deepopt.compiler.runtime import current_native_boundary_observer
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry

        compression = dict(enabled=False, bits=2, group_size=1024, norm_xhat_recenter=True)
        _, model = _prepared_pair(nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2))),
                                  compression, device=torch.device("cpu"))
        original = Nearest2dGeometry.forward
        def broken(specification, value):
            return original(specification, value).float()
        with tempfile.TemporaryDirectory() as directory:
            trace = BoundaryTrace(modules=("missing_module",), artifact_dir=directory)
            with patch.object(Nearest2dGeometry, "forward", broken), trace.native(), \
                    autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                             activation_compression=compression, config={}):
                with self.assertRaisesRegex(TypeError, "violates captured dtype"):
                    model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
            report = trace.report()
            self.assertFalse(report["complete"])
            self.assertIsNone(current_native_boundary_observer())
            files = report["artifacts"]["files"]
            self.assertTrue(any(row["path"].endswith(".py") for row in files))
            self.assertTrue(any(row["path"].endswith(".json") for row in files))
            self.assertFalse(report["artifacts"]["omissions"], report["artifacts"])
            for row in files:
                path = Path(row["path"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(report["tensor_checkpoints_written"], 0)
            json.dumps(report, allow_nan=False)

    def test_native_backward_keeps_forward_observer_after_context_exit(self):
        from deepopt import autocast
        from deepopt.compiler.runtime import current_native_boundary_observer

        compression = dict(enabled=False, bits=2, group_size=1024, norm_xhat_recenter=True)
        eager, native = _prepared_pair(nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2))),
                                       compression, device=torch.device("cpu"))
        reference = BoundaryTrace(modules=("resize",))
        candidate = BoundaryTrace(reference=reference, modules=("resize",))
        options = dict(dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                       activation_compression=compression, config={})
        with reference.eager(eager):
            with autocast("cpu", **options):
                output = eager(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
            cotangent = cancellation_cotangent(output.shape)
            output.backward(cotangent)
        with candidate.native(), autocast("cpu", **options):
            output = native(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
        self.assertIsNone(current_native_boundary_observer())
        output.backward(cotangent)
        report = candidate.report()
        self.assertTrue(report["complete"], report)
        self.assertTrue(report["passed"], report)
        self.assertTrue(any(row["phase"] == "backward" for row in report["boundaries"]))

    def test_native_repeat_pairs_adjoint_results_and_caps_artifacts(self):
        from deepopt import autocast

        compression = dict(enabled=False, bits=2, group_size=1024, norm_xhat_recenter=True)
        _, first = _prepared_pair(nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2))),
                                  compression, device=torch.device("cpu"))
        _, second = _prepared_pair(nn.Sequential(OrderedDict(resize=nn.Upsample(scale_factor=2))),
                                   compression, device=torch.device("cpu"))
        reference = BoundaryTrace(modules=("resize",))
        options = dict(dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                       activation_compression=compression, config={})
        with tempfile.TemporaryDirectory() as directory:
            candidate = BoundaryTrace(reference=reference, modules=("resize",), artifact_dir=directory,
                                       max_artifact_bytes=1, max_artifact_files=1)
            for trace, model in ((reference, first), (candidate, second)):
                with trace.native():
                    with autocast("cpu", **options):
                        output = model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16, requires_grad=True))
                    output.backward(cancellation_cotangent(output.shape))
            report = candidate.report()
            self.assertTrue(report["passed"], report)
            adjoint, = (row for row in report["boundaries"] if row["role"] == "input_adjoint")
            self.assertTrue(adjoint["paired"])
            self.assertTrue(adjoint["bitwise_equal"])
            self.assertEqual(report["artifacts"]["bytes"], 0)
            self.assertTrue(report["artifacts"]["omissions"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_operator_failure_retains_partial_localization_metadata(self):
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry

        original = Nearest2dGeometry.forward
        def broken(specification, value):
            return original(specification, value).float()
        with patch.object(Nearest2dGeometry, "forward", broken), self.assertRaises(TypeError) as caught:
            nearest_replay_diagnostics(geometry_indices=(0,), layouts=("nchw",),
                                       compressed=(False,), include_noninteger=False)
        partial = caught.exception.boundary_localization
        self.assertTrue(partial["reference"]["boundaries"])
        self.assertTrue(partial["candidate"]["errors"])
        json.dumps(partial, allow_nan=False)


@unittest.skipUnless(torch.cuda.is_available(), "Localization CUDA replays require an actual GPU")
class VAEDiagnosticLocalizationCudaTests(unittest.TestCase):
    def test_cuda_nearest_cancellation_geometry_and_native_owner(self):
        report = nearest_replay_diagnostics(device="cuda", geometry_indices=(0,))
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["cuda_execution_observed"])

    def test_cuda_conv_nearest_depthwise_matches_shared_cotangent(self):
        report = chain_replay_diagnostics(device="cuda", geometry_indices=(0,))
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["cuda_execution_observed"])


if __name__ == "__main__":
    unittest.main()
