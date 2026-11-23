"""CPU implementation contracts; these do not qualify CUDA or the target optimizer."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from examples.vae import benchmark_vae as benchmark
from examples.vae.qualification import (NumericalComparisonError, TensorOracle, named_tree, paired_trajectory, restore_rng,
                                       rng_state, saved_state_evidence, preserve_rng, resume_roundtrip)
from examples.vae.qualification import assert_bf16_forward_domains
from benchmarks.compiler import qualify_backend as coordinator


class VAEMemoryBudgetTests(unittest.TestCase):
    @staticmethod
    def original_geometry_fields():
        """Recorded H200 state-role geometry; synthetic CPU payloads only."""
        import math
        from examples.vae.mbconv_vae import MBConvVAE_V2
        from examples.vae.qualification import post_update_reference_fields

        with torch.device("meta"):
            model = MBConvVAE_V2(base_channels=32, max_channels=512, blocks_per_stage=2, latent_channels=4)
            state = {}
            for index, parameter in enumerate(model.parameters()):
                values = {"EMA_momentum": torch.empty_like(parameter), "HB_momentum": torch.empty_like(parameter)}
                if parameter.ndim > 1:
                    values.update(row_sq=torch.empty((parameter.shape[0], 1)),
                                  col_sq=torch.empty((1, math.prod(parameter.shape[1:]))))
                else:
                    values["exp_avg_sq"] = torch.empty_like(parameter)
                state[index] = values
        assert sum(parameter.numel() for parameter in model.parameters()) == 32317999
        assert len(list(model.parameters())) == 332
        groups = {"eps": 0., "heavy_tailed_noise": False, "last_ns5": False, "lr": 0.,
                  "momentum": 0., "nesterov": False, "overshoot": 0., "params": list(range(332)),
                  "rms_beta": 0., "weight_decay": 0.}
        random_state = rng_state()
        random_state["python"] = torch.tensor(list(repr(random_state["python"]).encode()), dtype=torch.uint8)
        fields = []

        def append(name, shape, dtype):
            item_size = torch.empty((), dtype=dtype).element_size()
            fields.append({"name": name, "shape": list(shape), "dtype": str(dtype),
                           "elements": math.prod(shape), "element_size": item_size,
                           "bytes": math.prod(shape) * item_size})

        for name, shape in (("reconstruction", (1, 3, 224, 224)), ("mu", (1, 4, 28, 28)),
                            ("logvar", (1, 4, 28, 28))):
            append("output/" + name, shape, torch.bfloat16)
        for name in ("loss", "l1", "kl"):
            append("output/" + name, (), torch.float32)
        append("recipe/bf16_policy/name", (14,), torch.uint8)
        append("recipe/bf16_policy/version", (), torch.int64)
        append("recipe/beta", (), torch.float64)
        append("gradient/input", (1, 3, 224, 224), torch.float32)
        for name, parameter in model.named_parameters():
            append("gradient/" + name, parameter.shape, parameter.dtype)
        fields.extend(post_update_reference_fields(model, {"state": state, "param_groups": [groups]}, random_state))
        return fields

    def test_pending_record_batches_flush_before_cap_and_comparison_reclaims_references(self):
        from examples.vae.qualification import ChunkTransport
        from benchmarks.analysis import numerical_ipc

        oracle = TensorOracle(max_bytes=4096)
        prefix = torch.zeros(3600, dtype=torch.uint8)
        oracle.record("prefix", prefix)
        with patch.object(numerical_ipc, "send_snapshot", wraps=numerical_ipc.send_snapshot) as send:
            with ChunkTransport(oracle.record, {"probe": "adaptive-flush"}) as transport:
                transport("small/a", torch.ones(160, dtype=torch.uint8))
                transport("small/b", torch.ones(100, dtype=torch.uint8))
            self.assertEqual(send.call_count, 2)
        self.assertEqual(oracle.bytes, 3860)
        oracle.compare("prefix", prefix)
        self.assertEqual(oracle.retained_bytes, 260)
        with ChunkTransport(oracle.compare, {"probe": "reclaimed-reference"}) as transport:
            transport("small/a", torch.ones(160, dtype=torch.uint8))
            transport("small/b", torch.ones(100, dtype=torch.uint8))
        self.assertTrue(oracle.finish()["passed"])
        self.assertEqual(oracle.retained_bytes, 0)

    def test_reference_preflight_rejects_original_state_at_512_mib_with_exact_inventory(self):
        from examples.vae.qualification import ChunkTransport

        fields = self.original_geometry_fields()
        without_rng = sum(item["bytes"] for item in fields if not item["name"].startswith("rng/"))
        self.assertEqual(without_rng, 648411375)
        oracle = TensorOracle(max_bytes=512 * 2**20)
        with self.assertRaisesRegex(MemoryError, "Complete VAE reference cannot fit") as caught:
            with ChunkTransport(oracle.record, {"probe": "original-reference-preflight"}) as transport:
                transport.preflight_reference(fields)
        report = caught.exception.oracle_report
        plan = report["reference_memory_plan"]
        self.assertEqual(plan["required_reference_bytes"], sum(item["bytes"] for item in fields))
        self.assertEqual(plan["optimizer_state_bytes_by_role"], {
            "EMA_momentum": 129271996, "HB_momentum": 129271996, "row_sq": 334860,
            "col_sq": 280676, "exp_avg_sq": 445244})
        self.assertGreater(plan["required_reference_bytes"], 512 * 2**20)
        self.assertEqual(report["recorded_tensor_count"], 0)
        self.assertFalse(plan["reference_fits_budget"])
        VAEFailureEvidenceTests.assert_metadata(self, report)
        json.dumps(report, allow_nan=False)

    def test_original_parameter_and_custom_state_geometry_roundtrip_with_explicit_one_gib(self):
        from examples.vae.qualification import ChunkTransport

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        fields = self.original_geometry_fields()
        oracle = TensorOracle(max_bytes=2**30)
        with ChunkTransport(oracle.record, {"probe": "original-geometry-record"}) as transport:
            transport.preflight_reference(fields)
            for item in fields:
                value = torch.zeros(item["shape"], dtype=getattr(torch, item["dtype"].removeprefix("torch.")))
                transport(item["name"], value)
        self.assertEqual(oracle.bytes, sum(item["bytes"] for item in fields))
        self.assertLess(oracle.memory_plan["reference_and_transport_upper_bound_bytes"], 2**30)
        with ChunkTransport(oracle.compare, {"probe": "original-geometry-compare"}) as transport:
            for item in fields:
                value = torch.zeros(item["shape"], dtype=getattr(torch, item["dtype"].removeprefix("torch.")))
                transport(item["name"], value)
        report = oracle.finish()
        self.assertTrue(report["passed"])
        self.assertEqual(report["completed_comparison_count"], len(fields))
        self.assertEqual(oracle.retained_bytes, 0)

    def test_staging_capacity_error_keeps_partial_oracle_and_actionable_byte_counts(self):
        from examples.vae.qualification import ChunkTransport

        oracle = TensorOracle(max_bytes=4096, context={"route": "eager", "phase": "state_comparison", "update": 1})
        oracle.record("retained", torch.zeros(4000, dtype=torch.uint8))
        with self.assertRaisesRegex(MemoryError, "required_bytes") as caught:
            with ChunkTransport(oracle.record, {"probe": "staging-evidence"}) as transport:
                transport("too-large", torch.zeros(100, dtype=torch.uint8))
        report = caught.exception.oracle_report
        self.assertEqual(report["unconsumed_names"], ["retained"])
        self.assertEqual(report["memory_error"]["budget_bytes"], 4096)
        self.assertEqual(report["memory_error"]["retained_bytes"], 4000)
        self.assertEqual(report["memory_error"]["required_bytes"], 4300)
        self.assertEqual(report["memory_error"]["name"], "too-large")
        self.assertEqual(report["context"]["phase"], "state_comparison")


class VAEFailureEvidenceTests(unittest.TestCase):
    @staticmethod
    def assert_metadata(test, value):
        test.assertNotIsInstance(value, torch.Tensor)
        if isinstance(value, dict):
            for item in value.values():
                VAEFailureEvidenceTests.assert_metadata(test, item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                VAEFailureEvidenceTests.assert_metadata(test, item)
        else:
            test.assertTrue(value is None or type(value) in (str, bool, int, float), type(value))

    def test_fail_fast_snapshot_retains_comparisons_context_and_unconsumed_names(self):
        oracle = TensorOracle(context={"route": "native", "update": 3, "phase": "gradient_comparison"})
        for name in ("output/loss", "gradient/input", "gradient/weight"):
            oracle.record(name, torch.ones(4))
        oracle.compare("output/loss", torch.ones(4))
        with self.assertRaises(NumericalComparisonError) as caught:
            oracle.compare("gradient/input", torch.full((4,), 2.))
        report = caught.exception.oracle_report
        self.assertEqual([row["name"] for row in report["comparisons"]], ["output/loss", "gradient/input"])
        self.assertEqual(report["first_failure"]["name"], "gradient/input")
        self.assertEqual(report["unconsumed_names"], ["gradient/weight"])
        self.assertEqual(report["unconsumed_count"], 1)
        self.assertEqual(report["context"], {"route": "native", "update": 3, "phase": "gradient_comparison"})
        self.assertEqual(caught.exception.comparison["expected"],
                         {"present": True, "shape": [4], "dtype": "torch.float32"})
        self.assertEqual(caught.exception.comparison["actual"], caught.exception.comparison["expected"])
        self.assertEqual(caught.exception.comparison["rtol"], 8 * torch.finfo(torch.float32).eps)
        self.assertEqual(caught.exception.comparison["atol"], 2 * torch.finfo(torch.float32).eps)
        self.assert_metadata(self, caught.exception.qualification_evidence)
        json.dumps(caught.exception.qualification_evidence, allow_nan=False)
        report["comparisons"].clear()
        self.assertEqual(len(oracle.errors), 2)

    def test_structural_and_nonfinite_failures_include_expected_and_actual_metadata(self):
        for expected, actual, failure in (
                (None, torch.ones(1), "presence"),
                (torch.ones(1), None, "presence"),
                (torch.ones(2), torch.ones(1), "shape_or_dtype"),
                (torch.ones(1), torch.ones(1, dtype=torch.bfloat16), "shape_or_dtype"),
                (torch.tensor([float("inf")]), torch.tensor([float("inf")]), "nonfinite")):
            with self.subTest(failure=failure, expected=expected, actual=actual):
                oracle = TensorOracle()
                oracle.record("gradient/x", expected)
                with self.assertRaises(NumericalComparisonError) as caught:
                    oracle.compare("gradient/x", actual)
                row = caught.exception.comparison
                self.assertIn(failure, row["failure"])
                self.assertEqual(row["expected"]["present"], expected is not None)
                self.assertEqual(row["actual"]["present"], actual is not None)
                self.assertEqual(caught.exception.oracle_report["completed_comparison_count"], 1)
                json.dumps(caught.exception.oracle_report, allow_nan=False)

    def test_diagnostic_oracle_collects_every_named_failure_and_still_rejects(self):
        oracle = TensorOracle(diagnostic=True)
        oracle.record("gradient/input", torch.ones(3))
        oracle.record("gradient/weight", torch.ones(4))
        oracle.record("gradient/bias", None)
        oracle.compare("gradient/input", torch.zeros(3))
        oracle.compare("gradient/weight", torch.zeros(4))
        oracle.compare("gradient/bias", torch.zeros(1))
        with self.assertRaises(NumericalComparisonError) as caught:
            oracle.raise_if_failed()
        report = caught.exception.oracle_report
        self.assertEqual(report["failure_count"], 3)
        self.assertEqual(report["unconsumed_count"], 0)
        self.assertEqual(report["completed_comparison_count"], 3)
        self.assertEqual(report["first_failure"]["name"], "gradient/input")
        self.assertFalse(report["qualification_eligible"])
        self.assertFalse(report["passed"])
        with self.assertRaises(NumericalComparisonError):
            oracle.finish()

    def test_metadata_count_limit_fails_explicitly_without_silent_truncation(self):
        oracle = TensorOracle(diagnostic=True, max_comparisons=2)
        for name in ("first", "second"):
            oracle.record(name, None)
        with self.assertRaisesRegex(MemoryError, "metadata limit"):
            oracle.record("third", None)
        oracle.compare("first", None)
        oracle.compare("second", None)
        with self.assertRaises(NumericalComparisonError) as caught:
            oracle.compare("unexpected", None)
        self.assertEqual(caught.exception.comparison["failure"], "comparison_metadata_limit_exceeded")
        self.assertTrue(caught.exception.oracle_report["metadata_limit_exceeded"])
        self.assertEqual(caught.exception.oracle_report["completed_comparison_count"], 2)
        self.assertFalse(caught.exception.oracle_report["exhaustive"])

    def test_ipc_callback_preserves_typed_error_metadata_and_original_compare_traceback(self):
        import traceback
        from examples.vae.qualification import ChunkTransport

        oracle = TensorOracle(context={"route": "native", "update": 1, "phase": "gradient_comparison"})
        oracle.record("gradient/input", torch.ones(73))
        oracle.record("gradient/weight", torch.ones(2))
        with self.assertRaises(NumericalComparisonError) as caught:
            with ChunkTransport(oracle.compare, {"probe": "typed-callback"}, chunk_elements=7) as transport:
                transport("gradient/input", torch.zeros(73))
        error = caught.exception
        self.assertEqual(error.oracle_report["unconsumed_names"], ["gradient/weight"])
        self.assertEqual(error.comparison["context"]["route"], "native")
        # unittest clears the caught traceback; check the retained diagnostic
        # attributes here and test a live traceback below.
        self.assert_metadata(self, error.qualification_evidence)
        try:
            with ChunkTransport(TensorOracle().compare, {"probe": "unexpected-callback"}) as transport:
                transport("gradient/missing", torch.zeros(3))
        except NumericalComparisonError as live:
            self.assertIn("compare", [frame.name for frame in traceback.extract_tb(live.__traceback__)])
        else:
            self.fail("Missing reference must preserve a typed callback failure")

    @staticmethod
    def pair(gradient_scale=2.):
        class ScaleAdjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value, scale):
                ctx.scale = scale
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                return gradient * ctx.scale, None

        class Tiny(torch.nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.25))
                self.bias = torch.nn.Parameter(torch.tensor(.125))
                self.scale = scale

            def forward(self, value):
                result = ScaleAdjoint.apply(value * self.weight + self.bias, self.scale)
                return result, result * .1, result * .01

        class CountingSGD(torch.optim.SGD):
            def __init__(self, parameters):
                super().__init__(parameters, lr=.001)
                self.steps = 0

            def step(self, *args, **kwargs):
                self.steps += 1
                return super().step(*args, **kwargs)

        eager, candidate = Tiny(1.), Tiny(gradient_scale)
        return eager, candidate, CountingSGD(eager.parameters()), CountingSGD(candidate.parameters())

    def test_paired_diagnostic_collects_all_gradients_before_candidate_optimizer_and_cleans_context(self):
        from contextlib import contextmanager

        eager, candidate, a, b = self.pair()
        evidence, contexts = [], []

        @contextmanager
        def observing(route, model):
            contexts.append(("enter", route))
            try:
                yield
            finally:
                contexts.append(("exit", route))

        with self.assertRaises(NumericalComparisonError) as caught:
            paired_trajectory(eager, candidate, a, b, torch.ones(1, 3, 4, 4), [0.],
                {"enabled": False}, device="cpu", managed=False, diagnostic=True,
                evidence_callback=evidence.append, route_context=observing)
        error = caught.exception
        rows = error.oracle_report["comparisons"]
        names = {row["name"] for row in rows}
        self.assertTrue({"gradient/input", "gradient/weight", "gradient/bias"} <= names)
        self.assertGreaterEqual(error.oracle_report["failure_count"], 3)
        self.assertEqual(a.steps, 1)
        self.assertEqual(b.steps, 0)
        self.assertFalse(error.qualification_evidence["partial_update"]["native"]["optimizer_attempted"])
        self.assertTrue(error.qualification_evidence["partial_update"]["native"]["backward_completed"])
        self.assertTrue(error.qualification_evidence["partial_update"]["native"]["gradient_comparisons_complete"])
        self.assertTrue(error.qualification_evidence["partial_update"]["eager"]["completed"])
        self.assertEqual(error.qualification_evidence["context"]["phase"], "gradient_comparison")
        self.assertEqual(error.qualification_evidence["context"]["route"], "native")
        self.assertEqual(contexts, [("enter", "eager"), ("exit", "eager"), ("enter", "native"), ("exit", "native")])
        self.assertEqual(len(evidence), 1)
        self.assert_metadata(self, evidence[0])
        json.dumps(evidence[0], allow_nan=False)
        self.assertTrue(any(name.startswith("parameter/") for name in error.oracle_report["unconsumed_names"]))

    def test_default_paired_failure_stops_at_first_gradient_and_observer_cannot_replace_it(self):
        import traceback

        eager, candidate, a, b = self.pair()
        def broken_observer(report):
            raise RuntimeError("diagnostic callback failed")
        try:
            paired_trajectory(eager, candidate, a, b, torch.ones(1, 3, 4, 4), [0.],
                {"enabled": False}, device="cpu", managed=False, evidence_callback=broken_observer)
        except NumericalComparisonError as error:
            gradient_rows = [row for row in error.oracle_report["comparisons"] if row["name"].startswith("gradient/")]
            self.assertEqual([row["name"] for row in gradient_rows], ["gradient/input"])
            self.assertEqual(error.qualification_evidence["context"]["route"], "native")
            self.assertIn("compare", [frame.name for frame in traceback.extract_tb(error.__traceback__)])
        else:
            self.fail("The original numerical failure must propagate")
        self.assertEqual(b.steps, 0)

    def test_broken_snapshot_diagnostics_cannot_replace_a_numerical_failure(self):
        oracle = TensorOracle()
        oracle.record("gradient/x", torch.ones(2))
        with patch.object(oracle, "snapshot", side_effect=RuntimeError("collector failed")):
            with self.assertRaises(NumericalComparisonError) as caught:
                oracle.compare("gradient/x", torch.zeros(2))
        self.assertEqual(caught.exception.comparison["name"], "gradient/x")
        self.assertEqual(caught.exception.oracle_report["diagnostic_collection_error"], "RuntimeError")

    def test_route_observer_cannot_suppress_or_replace_a_comparison_failure(self):
        for cleanup in ("raise", "suppress"):
            class Observer:
                def __enter__(self):
                    return self

                def __exit__(self, kind, error, traceback):
                    if kind is not None and cleanup == "raise":
                        raise RuntimeError("diagnostic cleanup failed")
                    return kind is not None

            eager, candidate, a, b = self.pair()
            with self.subTest(cleanup=cleanup), self.assertRaises(NumericalComparisonError) as caught:
                paired_trajectory(eager, candidate, a, b, torch.ones(1, 3, 4, 4), [0.],
                    {"enabled": False}, device="cpu", managed=False, diagnostic=True,
                    route_context=lambda route, model: Observer())
            self.assertEqual(caught.exception.comparison["name"], "gradient/input")
            self.assertEqual(b.steps, 0)
            if cleanup == "raise":
                self.assertEqual(caught.exception.qualification_evidence["diagnostic_context_error"]["type"], "RuntimeError")

    def test_successful_diagnostic_comparison_is_explicitly_ineligible_to_qualify(self):
        eager, candidate, a, b = self.pair(gradient_scale=1.)
        report = paired_trajectory(eager, candidate, a, b, torch.ones(1, 3, 4, 4), [0.],
            {"enabled": False}, device="cpu", managed=False, diagnostic=True)
        self.assertFalse(report["passed"])
        self.assertTrue(report["numerical_comparisons_passed"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["qualification_eligible"])

    def test_execution_error_is_preserved_when_partial_snapshot_collection_also_fails(self):
        import traceback

        eager, candidate, a, b = self.pair()
        failure = RuntimeError("original forward failure")
        def broken_forward(value):
            raise failure
        eager.forward = broken_forward
        original = TensorOracle.snapshot
        def snapshot(oracle):
            if oracle.context.get("phase") == "forward":
                raise RuntimeError("metadata collector failed")
            return original(oracle)
        with patch.object(TensorOracle, "snapshot", snapshot):
            try:
                paired_trajectory(eager, candidate, a, b, torch.ones(1, 3, 4, 4), [0.],
                                  {"enabled": False}, device="cpu", managed=False)
            except RuntimeError as error:
                self.assertIs(error, failure)
                self.assertIn("broken_forward", [frame.name for frame in traceback.extract_tb(error.__traceback__)])
                self.assertEqual(error.qualification_evidence["oracle_report"]["diagnostic_collection_error"], "RuntimeError")
            else:
                self.fail("Original execution failure must propagate")


class VAEOracleTests(unittest.TestCase):
    def setUp(self):
        # Oracle fixtures construct managed autocast scopes with an explicit
        # policy. They must not inherit defaults installed by a previous test.
        from deepopt import reset_config
        reset_config()
        self.addCleanup(reset_config)

    def test_optimizer_scalar_sequences_preserve_double_precision_identity(self):
        first = dict(named_tree({"betas": [.9, .99]}, "optimizer/param_groups"))
        second = dict(named_tree({"betas": [.9 + 1e-12, .99]}, "optimizer/param_groups"))
        oracle = TensorOracle()
        for name, tensor in first.items():
            self.assertEqual(tensor.dtype, torch.float64)
            oracle.record(name, tensor)
        with self.assertRaises(AssertionError):
            for name, tensor in second.items():
                oracle.compare(name, tensor)

    def test_shared_eager_dtype_inventory_does_not_change_caller_state_or_rng(self):
        from examples.vae.qualification import eager_dtype_diagnostic

        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("calls", torch.zeros((), dtype=torch.long))

            def forward(self, x):
                self.calls.add_(1)
                return x + torch.randn_like(x)

        model = Stateful()
        before = rng_state()
        report = eager_dtype_diagnostic(model, torch.zeros(1, 3, 17, 23),
            {"enabled": False, "group_size": 1024}, device="cpu", managed=False)
        self.assertTrue(report["complete"])
        self.assertFalse(report["timing_eligible"])
        self.assertFalse(report["cuda_execution_verified"])
        self.assertEqual(model.calls.item(), 0)
        self.assertTrue(torch.equal(before["torch"], rng_state()["torch"]))
        self.assertTrue(all(edge["bf16_policy"] == {"name": "torch_amp", "version": 1}
                            for edge in report["edges"]))
        with self.assertRaisesRegex(MemoryError, "bounded CPU state"):
            eager_dtype_diagnostic(model, torch.zeros(1), {"enabled": False, "group_size": 1024},
                                   device="cpu", max_bytes=1)

    def test_real_cpu_inductor_adapter_forward_backward_obeys_shared_context(self):
        import torch._functorch.config as functorch
        from benchmarks.compiler.inductor_benchmark_artifacts import InductorBenchmarkArtifacts

        self.addCleanup(torch.compiler.reset)
        previous = functorch.backward_pass_autocast
        eager = torch.nn.Linear(16, 12, bias=False)
        candidate = deepcopy(eager)
        inputs = torch.randn(8, 16, requires_grad=True)
        control = inputs.detach().clone().requires_grad_()
        upstream = torch.randn(8, 12)
        expected = eager(control)
        expected.backward(upstream)
        task = benchmark.inductor_artifact_task(benchmark.settings(benchmark.parser().parse_args([])), "cast-preserving")
        task["workload"] = {"name": "tiny-linear-cpu-diagnostic", "shape": [8, 16], "gradient_mode": "all",
                            "full_model_reference": False}
        with tempfile.TemporaryDirectory() as directory, InductorBenchmarkArtifacts(directory, task, enabled=True) as artifacts:
            controls = artifacts.configuration["numerical_controls"]
            compiled = benchmark.compile_torch_model(candidate, {"mode": "default"})
            actual = compiled(inputs)
            self.assertEqual(functorch.backward_pass_autocast, "off")
            actual.backward(upstream)
            self.assertIn("CompiledFunctionBackward", type(actual.grad_fn).__name__)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(inputs.grad, control.grad)
            torch.testing.assert_close(candidate.weight.grad, eager.weight.grad)
            self.assertEqual(controls["effective"]["backward_pass_autocast"], "off")
            # This is untimed CPU diagnostic execution, never an eligible
            # full-model GPU runtime reference or fabricated timing receipt.
            report = artifacts.finish()
            self.assertFalse(report["reference_valid"])
            self.assertEqual(report["timing_scope"], task["timing_scope"])
            self.assertFalse(report["cuda_execution_observed"])
            self.assertEqual(report["tensor_files_saved"], 0)
            self.assertEqual(report["native_operator_audit"]["status"], "passed", report["native_operator_audit"])
            self.assertTrue(all(report["compiled_phase_artifacts"][phase] for phase in ("forward", "backward")))
        self.assertEqual(functorch.backward_pass_autocast, previous)

    def test_small_named_snapshots_batch_without_aliasing_or_reordering(self):
        from examples.vae.qualification import ChunkTransport
        from benchmarks.analysis import numerical_ipc

        oracle = TensorOracle(max_bytes=4096)
        parameter = torch.ones(17)
        update = torch.zeros(17)
        with patch.object(numerical_ipc, "send_snapshot", wraps=numerical_ipc.send_snapshot) as send:
            with ChunkTransport(oracle.record, {"probe": "batched"}) as transport:
                transport("parameter/w", parameter)
                transport("delta/w", update)
                parameter.zero_()
            self.assertEqual(send.call_count, 1)
            # Alphabetical wire order places delta first. The receiving adapter
            # preserves emission order for matching post-parameter ULP budgets.
            with ChunkTransport(oracle.compare, {"probe": "batched"}) as transport:
                transport("parameter/w", torch.ones(17))
                transport("delta/w", update)
            self.assertEqual(send.call_count, 2)
        self.assertTrue(oracle.finish()["passed"])

    def test_transport_aggregates_complete_named_tensors_independent_of_wire_chunks(self):
        from examples.vae.qualification import ChunkTransport

        reports = []
        value = torch.arange(73, dtype=torch.float32)
        for chunk in (7, 19):
            oracle = TensorOracle(max_bytes=4096)
            with ChunkTransport(oracle.record, {"probe": "whole-named"}, chunk_elements=chunk) as transport:
                transport("gradient/w", value)
            with ChunkTransport(oracle.compare, {"probe": "whole-named"}, chunk_elements=chunk) as transport:
                transport("gradient/w", value)
            comparisons = oracle.finish()["comparisons"]
            row = next(row for row in comparisons if row["name"] == "gradient/w")
            self.assertEqual(row["elements"], 73)
            self.assertFalse(any("#chunk" in row["name"] for row in comparisons))
            reports.append(row)
        self.assertEqual(reports[0], reports[1])

    def test_independent_capacity_probe_keeps_gate_unavailability_and_oom_phase_separate(self):
        from contextlib import ExitStack
        from unittest.mock import Mock
        from deepopt.scheduling.controller import ScheduleController

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        images = torch.ones(1, 3, 17, 23)
        compression = {"enabled": True, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}
        schedules = ScheduleController.from_config({"beta": {"type": "constant", "value": 0.}})
        with ExitStack() as stack:
            for name in ("empty_cache", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch("torch.cuda." + name))
            stack.enter_context(patch("torch.cuda.mem_get_info", return_value=(20_000_000_000, 80_000_000_000)))
            stack.enter_context(patch("torch.cuda.memory_reserved", return_value=0))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=1_000_000_000))
            stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=2_000_000_000))
            update = stack.enter_context(patch("examples.vae.qualification.emit_update", return_value={}))
            report = benchmark.measure_candidate_capacity(model, optimizer, images, compression, "torch_amp",
                schedules, [1, 128], "cpu")
            self.assertTrue(report["batch1_admitted"])
            self.assertEqual(report["same_batch_eager_gate"], "not_measured_by_capacity_probe")
            self.assertEqual(report["numerical_trajectory_physical_batches"], [])
            self.assertEqual(report["measurements"][-1]["peak_allocated_target_bytes"], 87_990_000_000)
            self.assertTrue(report["measurements"][-1]["batch128_fits"])
            self.assertEqual(update.call_count, 2)
            def oom(*args, **kwargs):
                kwargs["progress"]({"phase": "backward"})
                raise torch.cuda.OutOfMemoryError("simulated CPU metadata probe")
            update.side_effect = oom
            progress = Mock()
            report = benchmark.measure_candidate_capacity(model, optimizer, images, compression, "torch_amp",
                ScheduleController.from_config({"beta": {"type": "constant", "value": 0.}}),
                [1, 2], "cpu", progress=progress)
            self.assertFalse(report["batch1_admitted"])
            self.assertEqual(len(report["measurements"]), 1)
            self.assertEqual(report["measurements"][0]["failure_context"]["phase"], "backward")
            self.assertEqual(report["measurements"][0]["failure_context"]["route"], "native")

    def test_actual_cpu_policy_eager_native_compression_matrix(self):
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model
        from examples.vae.mbconv_vae import MBConvVAE_V2

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.compiler.reset)
        self.addCleanup(restore_rng, rng_state())
        for policy in ("torch_amp", "bf16_storage"):
            for compressed in (False, True):
                with self.subTest(policy=policy, compressed=compressed):
                    torch.compiler.reset()
                    torch.manual_seed(5201)
                    compression = {"enabled": compressed, "bits": 2, "group_size": 1024,
                                   "norm_xhat_recenter": True}
                    config = {"runtime": {"device": "cuda", "precision": "bf16", "bf16_policy": policy,
                                          "fp8": {"enabled": False}, "activation_compression": compression}}
                    eager = prepare_fp8_model(MBConvVAE_V2(base_channels=4, max_channels=16,
                        blocks_per_stage=1, latent_channels=2), False, config=config)[0]
                    native = deepcopy(eager)
                    compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        options={"require_full_aot": True, "use_native_autodiff": True}), config=config)
                    # Explicit tiny CPU diagnostic with SGD, never original
                    # architecture/custom optimizer or CUDA qualification.
                    result = paired_trajectory(eager, native,
                        torch.optim.SGD(eager.parameters(), lr=.001),
                        torch.optim.SGD(native.parameters(), lr=.001),
                        torch.randn(1, 3, 17, 19), [0.], compression, device="cpu", bf16_policy=policy)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["bf16_policy"], {"name": policy, "version": 1})

    def test_policy_cli_snapshot_and_all_eight_coordinator_cells(self):
        args = benchmark.parser().parse_args(["--bf16-policy", "bf16_storage"])
        raw = benchmark.settings(args)
        self.assertEqual(raw["runtime"]["bf16_policy"], "bf16_storage")
        self.assertEqual(benchmark.mode_settings(raw, "deepopt-bf16-native")["runtime"]["bf16_policy"], "bf16_storage")
        self.assertEqual(benchmark.mode_settings(raw, "torch-bf16-compiled")["runtime"]["bf16_policy"], "torch_amp")
        plan = coordinator.make_plan(coordinator.parser().parse_args(["--profile", "gpu-full", "--vae-target"]))
        cells = set()
        for stage in plan["stages"]:
            if stage["validator"] == "vae_target":
                command = stage["command"]
                policy = command[command.index("--bf16-policy") + 1]
                compressed = "--qualification-compressed" in command
                cells.update((policy, route, compressed) for route in ("eager", "native"))
        self.assertEqual(len(cells), 8)

    def test_eager_gate_oom_is_not_a_compiled_capacity_failure(self):
        status = benchmark.gate_failure_status(torch.cuda.OutOfMemoryError("simulated"),
                                              {"route": "eager", "phase": "backward"})
        self.assertEqual(status, "unavailable_eager_oom")

    def test_update_floor_does_not_relax_independent_fp32_gradients(self):
        oracle = TensorOracle(max_bytes=4096)
        parameter = torch.ones(32)
        expected = torch.zeros(32)
        actual = torch.full((32,), torch.finfo(torch.float32).eps)
        oracle.record("parameter/w", parameter)
        oracle.record("delta/w", expected)
        oracle.compare("parameter/w", parameter)
        oracle.compare("delta/w", actual)
        evidence = oracle.finish()["comparisons"][-1]
        self.assertFalse(evidence["direction_defined"])
        self.assertIsNone(evidence["cosine"])
        self.assertIn("raw_cosine", evidence)
        oracle = TensorOracle()
        oracle.record("gradient/w", expected)
        with self.assertRaises(AssertionError):
            oracle.compare("gradient/w", actual)

    def test_bf16_domain_contract_rejects_simulated_fp16_or_disabled_amp(self):
        # Dtype contract simulation only; this does not execute CUDA.
        dtypes = dict(reconstruction="torch.bfloat16", mu="torch.bfloat16", logvar="torch.bfloat16",
                      loss="torch.float32", l1="torch.float32", kl="torch.float32")
        assert_bf16_forward_domains(torch.float32, dtypes, True, torch.bfloat16)
        for enabled, dtype, outputs in ((False, torch.bfloat16, dtypes), (True, torch.float16, dtypes),
                                        (True, torch.bfloat16, dict(dtypes, mu="torch.float32"))):
            with self.assertRaisesRegex(AssertionError, "BF16 recipe differs"):
                assert_bf16_forward_domains(torch.float32, outputs, enabled, dtype)
    def test_archived_fp32_affine_drift_cannot_borrow_a_bf16_budget(self):
        oracle = TensorOracle()
        oracle.record("gradient/norm.weight", torch.tensor([333.], dtype=torch.float32))
        with self.assertRaisesRegex(AssertionError, "relative_l2"):
            oracle.compare("gradient/norm.weight", torch.tensor([334.], dtype=torch.float32))
        self.assertEqual(oracle.errors[-1]["max_abs"], 1.)
        self.assertLess(oracle.errors[-1]["relative_l2_limit"], 1e-5)

    def test_throwaway_gate_restores_process_rng_even_when_it_fails(self):
        import numpy as np
        import random

        before = rng_state()
        with self.assertRaisesRegex(RuntimeError, "gate failed"):
            with preserve_rng():
                torch.rand(15)
                np.random.random(13)
                random.random()
                raise RuntimeError("gate failed")
        oracle = TensorOracle()
        for name, value in named_tree(before, "rng"):
            oracle.record(name, value)
        for name, value in named_tree(rng_state(), "rng"):
            oracle.compare(name, value)
        self.assertTrue(oracle.finish()["passed"])

    def test_resume_perturbs_and_restores_non_tensor_optimizer_counters(self):
        class ScalarOptimizer(torch.optim.SGD):
            def load_state_dict(self, state):
                self.before_restore = {key: dict(value) for key, value in self.state.items()}
                return super().load_state_dict(state)

        model = torch.nn.Linear(2, 1)
        optimizer = ScalarOptimizer(model.parameters(), lr=.1)
        for parameter in model.parameters():
            optimizer.state[parameter] = {"counter": 3, "coefficient": .25, "ready": True}
        report = resume_roundtrip(model, optimizer, 2**20)
        self.assertTrue(report["passed"])
        for parameter in model.parameters():
            self.assertNotEqual(optimizer.before_restore[parameter]["counter"], 3)
            self.assertNotEqual(optimizer.before_restore[parameter]["coefficient"], .25)
            self.assertFalse(optimizer.before_restore[parameter]["ready"])
            self.assertEqual(optimizer.state[parameter], {"counter": 3, "coefficient": .25, "ready": True})

    def test_oracle_is_exhaustive_bounded_and_exact_for_rng_and_geometry(self):
        oracle = TensorOracle(max_bytes=32, chunk_elements=2)
        oracle.record("gradient/weight", torch.arange(7.).reshape(1, 7))
        with self.assertRaises(MemoryError):
            oracle.record("too_large", torch.ones(2))
        with self.assertRaises(AssertionError):
            oracle.finish()
        oracle.compare("gradient/weight", torch.arange(7.).reshape(1, 7))
        report = oracle.finish()
        self.assertTrue(report["exhaustive"])
        self.assertEqual(report["comparisons"][0]["elements"], 7)
        self.assertEqual(report["reference_bytes"], 28)
        oracle = TensorOracle()
        oracle.record("rng/state", torch.tensor([1.]))
        with self.assertRaises(AssertionError):
            oracle.compare("rng/state", torch.tensor([1.00001]))

    def test_none_masks_dtype_names_and_nonfinite_fail_closed(self):
        for expected, actual in ((None, torch.ones(1)), (torch.ones(1), None),
                                 (torch.ones(1), torch.ones(1, dtype=torch.bfloat16)),
                                 (torch.tensor([float("inf")]), torch.tensor([float("inf")]))):
            with self.subTest(expected=expected, actual=actual):
                oracle = TensorOracle()
                oracle.record("gradient/x", expected)
                with self.assertRaises(AssertionError):
                    oracle.compare("gradient/x", actual)

    def test_rng_restore_preserves_all_generators_exactly(self):
        state = rng_state()
        first = dict(named_tree(state, "rng"))
        torch.randn(100)
        restore_rng(state)
        second = dict(named_tree(rng_state(), "rng"))
        oracle = TensorOracle()
        for name, value in first.items():
            oracle.record(name, value)
        for name, value in second.items():
            oracle.compare(name, value)
        self.assertTrue(oracle.finish()["passed"])

    def test_tiny_stochastic_trajectory_compares_every_gradient_and_optimizer_state(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(3, 3, 1)
                self.register_buffer("calls", torch.zeros((), dtype=torch.long))

            def forward(self, x):
                self.calls.add_(1)
                mu = self.projection(x)
                logvar = mu * .01
                return (mu + torch.randn_like(mu) * .01).tanh(), mu, logvar

        eager = Tiny()
        native = deepcopy(eager)
        # This is an oracle transport/trajectory test, explicitly not original
        # optimizer or native backend qualification.
        a = torch.optim.AdamW(eager.parameters(), lr=.001)
        b = torch.optim.AdamW(native.parameters(), lr=.001)
        report = paired_trajectory(eager, native, a, b, torch.randn(1, 3, 19, 19),
            [0., 1e-7], dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), device="cpu")
        self.assertTrue(report["passed"])
        names = {row["name"].split("#chunk")[0] for row in report["updates"][0]["oracle"]["comparisons"]}
        for name in ("output/reconstruction", "output/mu", "output/logvar", "output/loss", "output/l1", "output/kl",
                     "gradient/input", "gradient/projection.weight", "gradient/projection.bias",
                     "delta/projection.weight", "buffer/calls", "rng/torch", "optimizer/state/0/exp_avg"):
            self.assertIn(name, names)

    def test_no_cuda_target_is_unqualified_and_never_writes_tensor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            args = benchmark.parser().parse_args(["--qualify", "--output", str(output)])
            with patch("benchmarks.compiler.installed_runtime.installed_identity", return_value={"version": "test"}), \
                    patch.object(torch.cuda, "is_available", return_value=False):
                self.assertEqual(benchmark.qualify(args), 1)
            evidence = json.loads(output.read_text())
            self.assertFalse(evidence["qualified"])
            self.assertFalse(evidence["cuda_verified"])
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_actual_cpu_native_vae_uses_chunked_full_tensor_gate_and_typed_saves(self):
        from tests.compiler.test_native_vae import NativeVAETests
        from deepopt.compiler import compilation_report

        fixture = NativeVAETests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        native, eager, compression = fixture.pair()
        a = torch.optim.SGD(eager.parameters(), lr=.001, momentum=.9)
        b = torch.optim.SGD(native.parameters(), lr=.001, momentum=.9)
        report = paired_trajectory(eager, native, a, b, torch.randn(1, 3, 17, 19),
            [0., 1e-7], compression, device="cpu")
        self.assertTrue(report["passed"])
        self.assertEqual(report["transport"], "numerical_ipc_named_chunks")
        fixture.assert_native(native)
        self.assertTrue(saved_state_evidence(compilation_report(native).as_dict(), True)["passed"])

    def test_saved_state_gate_rejects_untyped_or_unexplained_dense_saves(self):
        report = {"graphs": [{"native_autodiff_graphs": [{"saved_state": [
            {"name": "activation", "shape": [2048], "dtype": "torch.float32"}],
            "saved_value_bindings": [{"source": "activation", "representation": "dense", "provenance": "activation"}]}]}]}
        with self.assertRaisesRegex(AssertionError, "large dense"):
            saved_state_evidence(report, True)
        report["graphs"][0]["native_autodiff_graphs"][0]["saved_state"][0]["dtype"] = None
        with self.assertRaisesRegex(AssertionError, "Untyped"):
            saved_state_evidence(report, False)


class VAECoordinatorTests(unittest.TestCase):
    def test_fusion_target_opt_in_preserves_original_cases_and_adds_storage_axis(self):
        plan = coordinator.make_plan(coordinator.parser().parse_args(["--profile", "gpu-full", "--only", "fusion"]))
        stages = {stage["name"]: stage for stage in plan["stages"]}
        base = [stage for stage in stages.values()
                if "benchmarks/compiler/isolated_fusion.py" in stage["command"]
                and not stage["name"].endswith("-bf16-storage")]
        self.assertTrue(base)
        for stage in base:
            command = list(stage["command"])
            self.assertIn("--qualify-native-group-fusion", command)
            self.assertEqual(command[command.index("--bf16-policy") + 1], "torch_amp")
            storage = stages[stage["name"] + "-bf16-storage"]
            alternate = list(storage["command"])
            self.assertEqual(alternate[alternate.index("--bf16-policy") + 1], "bf16_storage")
            alternate[alternate.index("--bf16-policy") + 1] = "torch_amp"
            alternate[alternate.index("--output") + 1] = command[command.index("--output") + 1]
            self.assertEqual(command, alternate)
            self.assertEqual(stage["validator"], storage["validator"])
            self.assertEqual(stage["require_observed"], storage["require_observed"])


    def test_cpu_and_cuda_selectors_and_optional_target_stages_are_distinct(self):
        cpu = coordinator.make_plan(coordinator.parser().parse_args(["--profile", "cpu"]))
        commands = [row["command"] for row in cpu["stages"] if row["validator"] == "tests"]
        self.assertTrue(any("tests.compiler.test_native_vae.NativeVAETests" in command for command in commands))
        gpu = coordinator.make_plan(coordinator.parser().parse_args(["--profile", "gpu-full", "--vae-target"]))
        target = [row for row in gpu["stages"] if row["validator"] == "vae_target"]
        self.assertEqual(len(target), 4)
        for row in target:
            self.assertIn("50", row["command"])
        strict = next(row for row in gpu["stages"] if row["name"] == "vae-cuda")
        self.assertNotIn("tests.compiler.test_native_vae.NativeVAECudaTests", strict["command"])
        self.assertTrue(all("metadata" not in value for value in strict["command"]))
        for row in gpu["stages"]:
            command = row["command"]
            if "--group-size" in command:
                self.assertEqual(command[command.index("--group-size") + 1], "1024")

    def test_target_validator_rejects_missing_numeric_lifetime_or_capacity_evidence(self):
        self.assertFalse(coordinator.validate_report({"validator": "vae_target"}, {"qualified": True})[0])


if __name__ == "__main__":
    unittest.main()
