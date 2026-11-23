"""Recovery diagnostics preserve comparison contracts and session ownership."""

import gc
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.benchmarks import test_do_analysis_benchmark as support
from analysis.engine import Session, run_task
from analysis.measurement import sampled_indices, errors, memory_fill_scope, latency_diagnostics, snapshot, quality_diagnostics


class RecoveryDiagnosticsTests(unittest.TestCase):
    def test_compile_reference_omits_empty_options_when_a_mode_is_selected(self):
        task, options = self.task(case='torch_bf16', update={'runtime': {'execution': 'compile',
            'compile': {'backend': 'eager', 'mode': 'default', 'options': {}}}})
        with patch('torch.compile', side_effect=lambda call, **kwargs: call) as compile_call:
            session = Session(task, options)
            try:
                self.assertEqual(compile_call.call_args.kwargs['mode'], 'default')
                self.assertNotIn('options', compile_call.call_args.kwargs)
            finally:
                session.close()

    def test_latency_probe_uses_graph_replay_and_excludes_capture_setup(self):
        from types import SimpleNamespace
        calls = []
        class SessionProbe:
            runtime = {'execution': 'graph'}
            graph = None
            graph_replay_count = 0
            device = SimpleNamespace(type='cpu')
            def reset(self):
                calls.append('reset')
            def step(self):
                raise AssertionError('graph diagnostic executed eager step')
            def graph_step(self):
                self.graph = object()
                self.graph_replay_count += 1
                calls.append('graph')
        session = SessionProbe()
        session.torch = torch
        result = latency_diagnostics(session, 3)
        self.assertEqual(result['execution'], 'graph')
        self.assertEqual(result['diagnostic_graph_replays'], 3)
        self.assertEqual(calls, ['reset', 'graph', 'reset', 'graph', 'graph', 'graph', 'reset'])
        self.assertEqual([r['step'] for r in result['updates']], [0, 1, 2])
        self.assertTrue(all(r['cuda_stream_ms'] is None for r in result['updates']))

    def test_learning_curve_uses_factory_validation_and_restores_the_baseline(self):
        task, options = self.task(case="do_bf16", update={"runtime": {"optimizer": "adamw", "accumulation_steps": 2}})
        session = Session(task, options)
        initial = {key: value.clone() for key, value in session.model.state_dict().items()}
        modes = []
        def validate(model):
            modes.append(model.training)
            return {"weight_l2": next(model.parameters()).norm()}
        session.validate = validate
        try:
            result = quality_diagnostics(session, 3, 2)
            self.assertEqual(len(result["curve"]), 3)
            self.assertEqual([r["step"] for r in result["validation"]], [0, 2, 3])
            self.assertEqual(result["validation_status"], "factory_callback")
            self.assertEqual(modes, [False, False, False])
            self.assertTrue(session.model.training)
            for key, value in session.model.state_dict().items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
            session.validate = None
            self.assertEqual(quality_diagnostics(session, 1, 1)["validation_status"], "not_provided")
        finally:
            session.close()

    def task(self, **kwargs):
        return support.AnalysisExecutionTests().task(**kwargs)

    def test_stratified_samples_cover_bins_endpoints_and_feature_columns(self):
        n, count = 4096 * 3072, 65536
        a = sampled_indices(n, count, torch.device("cpu"))
        b = sampled_indices(n, count, torch.device("cpu"))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual((int(a[0]), int(a[-1]), len(a.unique())), (0, n - 1, count))
        bins = torch.arange(count)
        self.assertTrue(((a >= bins * n // count) & (a < (bins + 1) * n // count)).all())
        self.assertGreater(len((a % 3072).unique()), 1024)

    def test_qkv_diagnostics_explain_near_zero_directions_without_changing_acceptance(self):
        reference = torch.tensor([1., 1., 1e-10, -1e-10, 1., 1.])
        actual = reference.clone()
        actual[2:4].neg_()
        name = "parameter_grad.attention.in_proj_bias"
        result = errors({name: actual}, {name: reference}, .05, 1e-8,
                        gradient_policy={"max_relative_l2": .1, "min_cosine": .99})
        self.assertTrue(result["passed"])
        key = result["qkv_bias_components"][name][name + ".K"]
        self.assertEqual(key["relative_l2"], 2.)
        self.assertLess(key["reference_l2"], 1e-9)
        self.assertFalse(result["component_diagnostics_change_acceptance"])

    def test_shared_native_warmup_restores_identical_model_and_optimizer_baselines(self):
        fingerprints, states = [], []
        for case in ("torch_bf16", "compression", "fp8_compression"):
            task, options = self.task(case=case, update={"runtime": {"optimizer": "adamw", "optimizer_warmup_steps": 3}})
            session = Session(task, options)
            try:
                fingerprints.append((session.initial_fingerprint, session.optimizer_baseline_fingerprint))
                states.append({k: v.clone() for k, v in session.model.state_dict().items()})
                self.assertTrue(all(int(v["step"]) == 3 for v in session.optimizer.state.values()))
                a, b = snapshot(session), snapshot(session)
                self.assertTrue(torch.equal(a["output.0"], b["output.0"]))
                self.assertTrue(all(int(v["step"]) == 3 for v in session.optimizer.state.values()))
                self.assertTrue(all(v["exp_avg"].device.type == "cpu" for v in session.optimizer_initial["state"].values()))
            finally:
                session.close()
        self.assertEqual(len(set(fingerprints)), 1)
        for other in states[1:]:
            for key, value in states[0].items():
                torch.testing.assert_close(other[key], value, rtol=0, atol=0)

    def test_fill_scope_restores_flags_and_separates_correctness_from_performance(self):
        previous = torch.utils.deterministic.fill_uninitialized_memory
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        self.addCleanup(setattr, torch.utils.deterministic, "fill_uninitialized_memory", previous)
        task, options = self.task(case="do_bf16", update={"runtime": {"timing_fill_uninitialized_memory": False}})
        options.update(memory=False, fill_uninitialized_memory=True, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            result = run_task(task, options, Path(directory))
        self.assertTrue(result["numerical_snapshot"]["fill_uninitialized_memory"])
        self.assertFalse(result["measurement_policy"]["performance_fill_uninitialized_memory"])
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
        with self.assertRaises(RuntimeError), memory_fill_scope(False):
            raise RuntimeError("failure")
        self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)

    def test_gemm_metadata_records_actual_roles_outside_throughput(self):
        task, options = self.task(case="fp8")
        with tempfile.TemporaryDirectory() as directory:
            result = run_task(task, options, Path(directory))
        records = result["execution"]["gemm_executions"]
        self.assertEqual({r["role"] for r in records}, {"forward", "dx", "dw"})
        self.assertTrue(all(r["effective_backend"] == "reference" for r in records))

    def test_gemm_recording_follows_forward_owner_into_an_autograd_worker(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from deepopt.opt.backends._shared.gemm_diagnostics import (
            record_gemms, capture_gemm_recording, restore_gemm_recording, observe_gemm)
        from deepopt.opt.quantization.tensor import QuantizedTensor

        ctx = SimpleNamespace()
        q = QuantizedTensor(torch.ones(2, 2).to(torch.float8_e4m3fn), torch.ones(1, 1))
        @restore_gemm_recording
        def backward(owner):
            observe_gemm(q, q, "reference", torch.ones(2, 2), "dw")
        with record_gemms() as records:
            capture_gemm_recording(ctx)
        with ThreadPoolExecutor(1) as pool:
            pool.submit(backward, ctx).result()
        self.assertEqual([r["role"] for r in records], ["dw"])

    def test_latency_probe_restores_model_and_gc_callbacks(self):
        task, options = self.task(case="do_bf16", update={"runtime": {"optimizer": "sgd"}})
        session = Session(task, options)
        try:
            initial = {name: value.clone() for name, value in session.model.state_dict().items()}
            callbacks = list(gc.callbacks)
            result = latency_diagnostics(session, 3)
            self.assertEqual(len(result["updates"]), 3)
            self.assertEqual(list(gc.callbacks), callbacks)
            for name, value in session.model.state_dict().items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        finally:
            session.close()

    def test_selective_family_recipe_and_exclusion_reuse_resolved_hook_scopes(self):
        task, options = self.task(op="linear_pipeline", case="compression", update={"runtime": {
            "scope_lifetime": "reused", "compression_bits_by_family": {"linear": 8, "activation": 4},
            "compression_exclude_modules": ["2"]}})
        session = Session(task, options)
        try:
            policies = session.module_compression_policies
            self.assertEqual(policies["0"]["bits"], 8)
            self.assertEqual(policies["1"]["bits"], 4)
            self.assertFalse(policies["2"]["enabled"])
            with patch("deepopt.autocast", side_effect=AssertionError("context constructed in hook")):
                session.step()
        finally:
            session.close()

    def test_graph_step_retains_replay_output_for_post_timing_finiteness_checks(self):
        task, options = self.task(case="do_bf16")
        session = Session(task, options)
        try:
            session.graph_output = (torch.tensor([float("nan")]),)
            session.graph = type("Replay", (), {"replay": lambda self: None})()
            session.graph_step()
            self.assertIs(session.last_output, session.graph_output)
        finally:
            session.close()
