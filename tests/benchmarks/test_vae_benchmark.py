"""Workload fairness, real optimizer updates, and failure reporting without CUDA."""

from contextlib import ExitStack, redirect_stdout
import copy
from functools import partial
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import deepopt
from deepopt import TrainingRuntime
from examples.vae import benchmark_vae as benchmark
from examples.vae.mbconv_vae import MBConvVAE_V2


class VAEBenchmarkTests(unittest.TestCase):
    @staticmethod
    def native_report():
        return {"graphs": [{"eager_autograd_operations": [], "legacy_backward_operations": []}],
                "native_autodiff_operations": 3, "native_autodiff_regions": 1,
                "legacy_backward_operations": [], "runtime_fallbacks": 0}

    @staticmethod
    def result_row(mode="torch-bf16", *, repeat=0, seconds=2.0):
        """A complete worker result with a shared, explicitly sized workload."""
        native = benchmark.MODES[mode]["native"]
        return {
            "mode": mode, "repeat": repeat, "status": "ok", "wall_seconds": seconds,
            "wall_ms_per_batch": seconds * 100, "cuda_model_ms_per_batch": seconds * 95,
            "images_per_second": 80 / seconds, "peak_allocated_gib": 10.0,
            "peak_reserved_gib": 12.0, "batch_size_per_gpu": 8, "world_size": 1,
            "effective_batch_size": 8, "batches_per_rank": 10, "total_images": 80,
            "image_size": 128, "workload_sha256": "workload", "source_sha256": "source",
            "source_stable": True, "native_ownership_verified": native,
            "same_recipe_oracle": {"passed": True} if native else None,
            "provenance": {"source_sha256": "source"}, "provenance_after": {"source_sha256": "source"},
            "compiler_report": VAEBenchmarkTests.native_report() if native else None,
            "ranks": [{"rank": 0, "initial_state_sha256": "weights", "input_sha256": "images",
                       "source_sha256": "source", "source_stable": True,
                       "same_recipe_oracle": {"passed": True} if native else None,
                       "provenance": {"source_sha256": "source"}, "provenance_after": {"source_sha256": "source"},
                       "compiler_report": VAEBenchmarkTests.native_report() if native else None}],
        }

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def tearDown(self):
        deepopt.reset_config()
        deepopt.shutdown_distributed()

    def test_help_runs_without_torch_deepopt_or_torchvision(self):
        completed = subprocess.run(
            [sys.executable, "-S", str(Path(benchmark.__file__).resolve()), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--optimizer-factory", completed.stdout)
        self.assertIn("--input-mode", completed.stdout)
        self.assertIn("--repeats", completed.stdout)
        self.assertIn("--compile-mode", completed.stdout)
        self.assertNotIn("deepopt-fp8", completed.stdout)

    def test_six_bf16_configs_share_max_speed_defaults(self):
        args = benchmark.parser().parse_args([])
        raw = benchmark.settings(args)
        # A caller-supplied FP8 setting must never enable it in this BF16 suite.
        raw.setdefault("runtime", {}).setdefault("fp8", {})["enabled"] = True
        before = copy.deepcopy(raw)
        self.assertEqual((args.batches, args.warmup, args.repeats), (10, 10, 3))
        self.assertEqual(args.compile_mode, "max-autotune-no-cudagraphs")
        self.assertEqual(args.modes, list(benchmark.MODES))
        self.assertEqual(set(benchmark.MODES), {
            "torch-bf16", "torch-bf16-compiled", "deepopt-bf16", "deepopt-bf16-compressed",
            "deepopt-bf16-native", "deepopt-bf16-native-compressed",
        })
        self.assertEqual(raw["training"]["batch_size"], 8)
        self.assertEqual(raw["training"]["optimizer"], "custom")
        self.assertEqual(raw["application"]["data"]["image_size"], 128)
        for mode, options in benchmark.MODES.items():
            with self.subTest(mode=mode):
                config = deepopt.load_config(benchmark.mode_settings(raw, mode))
                self.assertFalse(config.runtime.fp8.enabled)
                self.assertEqual(config.runtime.precision, "bf16")
                self.assertEqual(config.runtime.activation_compression.enabled, options["compressed"])
                self.assertEqual(config.runtime.activation_compression.group_size, 1024)
                self.assertEqual(config.runtime.activation_compression.bits, 2)
                self.assertTrue(config.runtime.activation_compression.norm_xhat_recenter)
                self.assertEqual(config.runtime.compile.enabled, options["compiled"])
                if options["native"]:
                    self.assertEqual(config.runtime.compile.backend, "deepopt")
                    self.assertTrue(config.runtime.compile.fullgraph)
                    self.assertFalse(config.runtime.compile.dynamic)
                    self.assertIsNone(config.runtime.compile.mode)
                    self.assertTrue(config.runtime.compile.options["use_native_autodiff"])
                    self.assertTrue(config.runtime.compile.options["require_full_aot"])
                elif options["compiled"]:
                    self.assertEqual(config.runtime.compile.backend, "inductor")
                    self.assertEqual(config.runtime.compile.mode, args.compile_mode)
                self.assertEqual(config.runtime.execution_mode, "max_speed")
                self.assertFalse(config.runtime.checks)
                self.assertFalse(config.runtime.deterministic)
                self.assertFalse(config.runtime.fill_uninitialized_memory)
        self.assertEqual(raw, before)

    def test_native_cases_preserve_requested_kernel_options_and_require_strict_ownership(self):
        args = benchmark.parser().parse_args([])
        with tempfile.TemporaryDirectory() as directory:
            source = args.config.read_text(encoding="utf-8")
            source += '\n[runtime.compile.options]\nkernel_backend="triton"\nbf16_decode="materialize"\nrequire_full_aot=false\nuse_native_autodiff=false\n'
            args.config = Path(directory) / "native.toml"
            args.config.write_text(source, encoding="utf-8")
            raw = benchmark.settings(args)
        before = copy.deepcopy(raw)
        for mode in ("deepopt-bf16-native", "deepopt-bf16-native-compressed"):
            options = benchmark.mode_settings(raw, mode)["runtime"]["compile"]["options"]
            self.assertEqual(options["kernel_backend"], "triton")
            self.assertEqual(options["bf16_decode"], "materialize")
            self.assertTrue(options["require_full_aot"])
            self.assertTrue(options["use_native_autodiff"])
        self.assertEqual(benchmark.mode_settings(raw, "torch-bf16-compiled")["runtime"]["compile"]["options"], {})
        self.assertEqual(raw, before)

    def test_native_bf16_config_preserves_target_recipe_with_batch_one_and_strict_compiler(self):
        path = Path(benchmark.__file__).with_name("native_bf16.toml")
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        target = tomllib.loads(path.with_name("activation_memory_local128.toml").read_text(encoding="utf-8"))
        config = deepopt.load_config(raw)
        self.assertEqual(config.training.batch_size, 1)
        self.assertEqual({key: value for key, value in raw["training"].items() if key != "batch_size"},
                         {key: value for key, value in target["training"].items() if key != "batch_size"})
        self.assertEqual(raw["application"], target["application"])
        self.assertEqual(raw["schedules"], target["schedules"])
        self.assertEqual(raw["runtime"]["activation_compression"], target["runtime"]["activation_compression"])
        self.assertEqual(config.runtime.precision, "bf16")
        self.assertFalse(config.runtime.fp8.enabled)
        self.assertEqual(config.runtime.kernels.backend, "triton")
        self.assertEqual(raw["application"]["data"]["train_dir"], "")
        self.assertEqual(raw["application"]["data"]["val_dir"], "")
        self.assertTrue(config.runtime.compile.enabled)
        self.assertEqual(config.runtime.compile.backend, "deepopt")
        self.assertTrue(config.runtime.compile.fullgraph)
        self.assertFalse(config.runtime.compile.dynamic)
        self.assertEqual(config.runtime.compile.options,
                         dict(use_native_autodiff=True, require_full_aot=True, share_compatible_packs=True))
        args = benchmark.parser().parse_args(["--config", str(path)])
        resolved = benchmark.mode_settings(benchmark.settings(args), "deepopt-bf16-native-compressed")
        self.assertEqual(resolved["training"]["batch_size"], 1)
        self.assertTrue(resolved["runtime"]["compile"]["options"]["share_compatible_packs"])

    def test_native_evidence_requires_graphs_and_excludes_all_derivative_escapes(self):
        self.assertIsNone(benchmark.native_report_error(self.native_report()))
        cases = [None, {}, {**self.native_report(), "graphs": []},
                 {**self.native_report(), "runtime_fallbacks": 1},
                 {**self.native_report(), "native_autodiff_regions": 0},
                 {**self.native_report(), "native_autodiff_operations": 0},
                 {**self.native_report(), "legacy_backward_operations": ["linear"]},
                 {**self.native_report(), "graphs": [{"eager_autograd_operations": ["silu"]}]}]
        for report in cases:
            with self.subTest(report=report):
                self.assertIsNotNone(benchmark.native_report_error(report))

    def test_native_evidence_is_available_after_real_compiled_backward(self):
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model = torch.nn.Linear(4, 3).bfloat16()
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options={"require_full_aot": True}))
        with self.assertRaisesRegex(RuntimeError, "no graph evidence"):
            benchmark.native_compilation_evidence(model)
        model(torch.randn(2, 4, dtype=torch.bfloat16)).sum().backward()
        self.assertIsNotNone(model.weight.grad)
        evidence = benchmark.native_compilation_evidence(model)
        self.assertGreater(evidence["native_autodiff_operations"], 0)

    def test_worker_receipts_require_requested_identity_stable_sources_and_each_rank_native_report(self):
        mode = "deepopt-bf16-native-compressed"
        self.assertIsNone(benchmark.worker_result_error(self.result_row(mode), mode, 0, "workload"))
        for defect in ("mode", "repeat", "workload", "source", "rank_source", "top_source", "report", "owner"):
            row = self.result_row(mode)
            if defect == "mode":
                row["mode"] = "deepopt-bf16"
            elif defect == "repeat":
                row["repeat"] = 1
            elif defect == "workload":
                row["workload_sha256"] = "old-workload"
            elif defect == "source":
                row["source_stable"] = False
            elif defect == "rank_source":
                row["ranks"][0]["provenance_after"]["source_sha256"] = "changed-source"
            elif defect == "top_source":
                row["provenance_after"]["source_sha256"] = "changed-source"
            elif defect == "report":
                row["ranks"][0]["compiler_report"] = None
            else:
                row["ranks"][0]["compiler_report"]["graphs"][0]["legacy_backward_operations"] = ["linear"]
            with self.subTest(defect=defect):
                self.assertIsNotNone(benchmark.worker_result_error(row, mode, 0, "workload"))

    def test_max_speed_flags_override_input_config_and_allow_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark.parser().parse_args([])
            source = args.config.read_text(encoding="utf-8")
            args.config = Path(directory) / "config.toml"
            # Supply the opposite runtime values to exercise CLI default precedence.
            source = source.replace("deterministic = false", "deterministic = true")
            source = source.replace("fill_uninitialized_memory = false", "fill_uninitialized_memory = true")
            if "fill_uninitialized_memory" not in source:
                source = source.replace("[runtime]\n", "[runtime]\nfill_uninitialized_memory = true\n")
            if "[runtime.fp8]" not in source:
                source += "\n[runtime.fp8]\nenabled = true\n"
            else:
                source = re.sub(r"(\[runtime\.fp8\]\s*\nenabled\s*=\s*)false", r"\g<1>true", source)
            args.config.write_text(source, encoding="utf-8")
            raw = benchmark.settings(args)
            self.assertFalse(raw["runtime"]["deterministic"])
            self.assertFalse(raw["runtime"]["fill_uninitialized_memory"])
            self.assertFalse(raw["runtime"]["fp8"]["enabled"])
            opted_in = benchmark.parser().parse_args([
                "--config", str(args.config), "--deterministic", "--fill-uninitialized-memory",
                "--compile-mode", "default",
            ])
            raw = benchmark.settings(opted_in)
            self.assertTrue(raw["runtime"]["deterministic"])
            self.assertTrue(raw["runtime"]["fill_uninitialized_memory"])
            compiled = benchmark.mode_settings(raw, "torch-bf16-compiled")
            self.assertEqual(compiled["runtime"]["compile"]["mode"], "default")

    def test_cli_overrides_and_accumulation_rejection(self):
        args = benchmark.parser().parse_args([
            "--batch-size", "12", "--workers", "0", "--optimizer", "sgd", "--train-dir", "/data/train",
        ])
        raw = benchmark.settings(args)
        self.assertEqual(raw["training"]["batch_size"], 12)
        self.assertEqual(raw["training"]["optimizer"], "sgd")
        self.assertEqual(raw["loader"]["num_workers"], 0)
        self.assertEqual(raw["application"]["data"]["train_dir"], "/data/train")
        with tempfile.TemporaryDirectory() as directory:
            args.config = Path(directory) / "config.toml"
            args.config.write_text(
                '[training]\nbatch_size=1\naccumulation_steps=2\n'
                '[application.data]\nimage_size=16\nresize_size=16\n', encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "accumulation_steps=1"):
                benchmark.settings(args)

    def test_cli_rejects_invalid_repeats_duplicate_modes_and_cold_compilation(self):
        for argv in (["--repeats", "0"], ["--modes", "torch-bf16", "torch-bf16"],
                     ["--warmup", "0", "--modes", "torch-bf16-compiled"],
                     ["--warmup", "0", "--modes", "deepopt-bf16-native"],
                     ["--warmup", "0", "--modes", "deepopt-bf16-native-compressed"]):
            with self.subTest(argv=argv), patch.object(sys, "stderr", io.StringIO()), \
                    self.assertRaises(SystemExit) as caught:
                benchmark.main(argv)
            self.assertEqual(caught.exception.code, 2)

    def test_objective_is_fp32_l1_plus_beta_times_kl_sum_per_image(self):
        reconstruction = torch.ones(2, 3, 2, 2, dtype=torch.bfloat16, requires_grad=True)
        mu = torch.ones(2, 2, 2, 2, dtype=torch.bfloat16, requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        model = lambda images: (reconstruction, mu, logvar)
        loss = benchmark.objective(model, torch.zeros_like(reconstruction), 0.25)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(mu.grad, torch.full_like(mu, 0.125))
        torch.testing.assert_close(logvar.grad, torch.zeros_like(logvar))

    def test_native_and_runtime_really_train_the_same_vae(self):
        torch.manual_seed(9)
        original = MBConvVAE_V2(base_channels=2, max_channels=8, blocks_per_stage=1, latent_channels=1)
        images = torch.rand(2, 3, 16, 16) * 2 - 1
        after = []
        losses = []
        for managed in (False, True):
            model = copy.deepcopy(original)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            runtime = (TrainingRuntime(model, optimizer, device="cpu", precision="fp32",
                                       fp8=False, checks=False, execution_mode="max_speed",
                                       grad_clip_norm=0.01,
                                       config={"runtime": {"activation_compression": {"enabled": False}}})
                       if managed else None)
            torch.manual_seed(10)
            losses.append(benchmark.train_step(model, optimizer, runtime, images, 1e-6,
                                               device_type="cpu", grad_clip_norm=0.01))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            self.assertTrue(optimizer.state)
            self.assertFalse(torch.equal(model.encoder[0].weight, original.encoder[0].weight))
            after.append(model.state_dict())
        torch.testing.assert_close(losses[0], losses[1])
        for name in after[0]:
            torch.testing.assert_close(after[0][name], after[1][name])

    def test_custom_optimizer_receives_the_same_training_arguments(self):
        model = torch.nn.Linear(2, 1)
        training = {"optimizer": "custom", "learning_rate": 0.001, "weight_decay": 1e-5, "momentum": 0.9}
        optimizer = benchmark.make_optimizer(model, training, "torch.optim:SGD")
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 0.001)
        self.assertEqual(optimizer.defaults["weight_decay"], 1e-5)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)
        with self.assertRaisesRegex(ImportError, "PYTHONPATH"):
            benchmark.make_optimizer(model, training, "unavailable_vae_optimizer:Optimizer")

    def test_source_provenance_supports_partial_model_factories(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        provenance = benchmark.source_provenance(deepopt, partial(MBConvVAE_V2, base_channels=2), optimizer)
        self.assertTrue(provenance["model_source_sha256"])
        self.assertTrue(provenance["optimizer_source_sha256"])
        self.assertEqual(len(provenance["source_sha256"]), 64)

    def test_failed_candidate_report_keeps_partial_graph_without_success_gate(self):
        from deepopt.compiler.reporting import CompilationReport, GraphReport

        original, candidate = torch.nn.Identity(), torch.nn.Identity()
        candidate._deepopt_compile_report = CompilationReport(graphs=[GraphReport(operations=1)])
        inventory = {"complete": True, "edges": [{"module": "decoder.0", "dtype": "torch.float32"}]}
        self.assertIsNone(benchmark.failure_candidate_diagnostics(original)["compiler_report"])
        report = benchmark.failure_candidate_diagnostics(candidate, inventory)
        self.assertEqual(report["compiler_report_scope"], "failed_candidate")
        self.assertEqual(report["compiler_report_status"], "available")
        self.assertEqual(report["compiler_report"]["captured_graphs"], 1)
        self.assertEqual(report["eager_dtype_edges"], inventory)
        with self.assertRaises(RuntimeError):
            benchmark.native_compilation_evidence(candidate)
        json.dumps(report, allow_nan=False)

    def test_diagnostic_failure_keeps_eager_inventory_and_original_dtype_error(self):
        from deepopt.compiler.codegen.wrapper import intermediate_dtype_error

        inventory = {"complete": True, "edges": []}
        with patch("deepopt.compiler.compilation_report", side_effect=RuntimeError("diagnostic failed")):
            report = benchmark.failure_candidate_diagnostics(torch.nn.Identity(), inventory)
        self.assertEqual(report["compiler_report_status"], "collection_failed")
        self.assertEqual(report["eager_dtype_edges"], inventory)
        contract = dict(value="input_163", operation="nearest2d", module="decoder.0", stage="forward",
                        expected={"dtype": "torch.bfloat16"}, bf16_policy={}, primitive_recipe=3)
        try:
            try:
                raise intermediate_dtype_error(torch.ones(2), contract)
            except TypeError as cause:
                raise RuntimeError("candidate gate failed") from cause
        except RuntimeError as error:
            details = benchmark.failure_error_details(error)
        self.assertIn("candidate gate failed", details["error"])
        self.assertIn("torch.float32", details["traceback"])
        self.assertEqual(details["dtype_contract"]["value"], "input_163")

    def test_worker_failure_receipt_uses_gate_candidate_even_if_cuda_diagnostics_fail(self):
        from deepopt.compiler.reporting import CompilationReport, GraphReport

        args = benchmark.parser().parse_args([])
        args._case, args._repeat = "deepopt-bf16-native-compressed", 0
        args.model_factory = "test:factory"
        inventory = {"complete": True, "edges": [{"module": "decoder.0", "dtype": "torch.float32"}],
                     "storage_policy_validation": {"valid": True}}
        created = []

        def factory(**kwargs):
            model = torch.nn.Linear(2, 2)
            created.append(model)
            return model

        def compile_candidate(candidate, *args, **kwargs):
            candidate._deepopt_compile_report = CompilationReport(graphs=[GraphReport(operations=37)])

        def fail_gate(*args, **kwargs):
            kwargs["progress"]({"route": "native", "phase": "forward", "update": 1})
            raise TypeError("original candidate dtype failure")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args._snapshot, args._result = Path(directory) / "snapshot.json", Path(directory) / "failure.json"
            benchmark.write_json(args._snapshot, benchmark.settings(args))
            context = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
            for target, replacement in (
                ("benchmarks.compiler.installed_runtime.installed_identity", lambda: {}),
                ("deepopt.initialize_distributed", lambda *_: context),
                ("deepopt.shutdown_distributed", lambda: None),
                ("deepopt.runtime.seed_everything", lambda *_: None),
                ("deepopt.opt.policy.prepare_fp8_model", lambda model, *args, **kwargs: (model, None)),
                ("deepopt.engine.execution.compilation.compile_model", compile_candidate),
                ("examples.vae.qualification.eager_dtype_diagnostic", lambda *args, **kwargs: inventory),
                ("examples.vae.qualification.paired_trajectory", fail_gate),
            ):
                stack.enter_context(patch(target, side_effect=replacement))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
            stack.enter_context(patch.object(torch.cuda, "is_bf16_supported", return_value=True))
            stack.enter_context(patch.object(benchmark, "import_factory", return_value=factory))
            stack.enter_context(patch.object(benchmark, "build_loader", return_value=[torch.zeros(1, 3, 16, 16)]))
            stack.enter_context(patch.object(benchmark, "source_provenance", return_value={"source_sha256": "source"}))
            stack.enter_context(patch.object(benchmark, "make_optimizer",
                side_effect=lambda model, *args: torch.optim.SGD(model.parameters(), lr=.01)))
            stack.enter_context(patch.object(benchmark, "cuda_failure_diagnostics", side_effect=RuntimeError("secondary")))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(TypeError, "original candidate dtype failure"):
                benchmark.worker(args)
            receipt = json.loads(args._result.read_text())
        self.assertIsNone(getattr(created[0], "_deepopt_compile_report", None))
        self.assertEqual(receipt["compiler_report"]["graphs"][0]["operations"], 37)
        self.assertEqual(receipt["eager_dtype_edges"], inventory)
        self.assertEqual(receipt["failure_context"]["route"], "native")
        self.assertEqual(receipt["cuda_diagnostics_collection_error"], "RuntimeError")
        self.assertIn("original candidate dtype failure", receipt["traceback"])

    def test_loader_replays_identical_full_batches_and_shards_ranks(self):
        args = benchmark.parser().parse_args(["--batches", "3", "--batch-size", "2", "--workers", "0"])
        raw = benchmark.settings(args)
        raw["loader"]["pin_memory"] = False
        images = torch.arange(30).view(30, 1)
        ranks = []
        with patch.object(benchmark, "Images", return_value=images):
            for rank in (0, 1):
                context = SimpleNamespace(world_size=2, global_rank=rank)
                rng_before = torch.get_rng_state().clone()
                first = list(benchmark.build_loader(raw, args, context))
                second = list(benchmark.build_loader(raw, args, context))
                torch.testing.assert_close(torch.get_rng_state(), rng_before)
                self.assertEqual(len(first), 3)
                self.assertTrue(all(batch.shape == (2, 1) for batch in first))
                self.assertEqual(benchmark.fingerprint(first), benchmark.fingerprint(second))
                ranks.append(set(torch.cat(first).view(-1).tolist()))
            self.assertFalse(ranks[0] & ranks[1])
        with patch.object(benchmark, "Images", return_value=images[:3]), self.assertRaisesRegex(ValueError, "distinct images"):
            benchmark.build_loader(raw, args, context)

    def test_speedups_require_success_and_identical_inputs_and_weights(self):
        baseline = self.result_row()
        candidate = self.result_row("deepopt-bf16", seconds=1)
        rows = benchmark.comparisons([baseline, candidate])
        self.assertEqual(rows[1]["speedup_vs_torch"], 2)
        candidate["ranks"][0]["input_sha256"] = "different"
        benchmark.comparisons(rows)
        self.assertIsNone(candidate["speedup_vs_torch"])
        self.assertFalse(candidate["same_initial_weights_and_inputs"])
        candidate["status"] = "nonfinite"
        benchmark.comparisons(rows)
        self.assertIsNone(candidate["speedup_vs_torch"])

    def test_speedups_reject_different_workload_sizes_even_when_tensor_hashes_match(self):
        mismatches = {
            "batch_size_per_gpu": 32, "world_size": 4, "batches_per_rank": 20,
            "total_images": 320, "image_size": 224,
            "workload_sha256": "different-workload", "source_sha256": "different-source",
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                baseline = self.result_row()
                candidate = self.result_row("deepopt-bf16", seconds=1)
                candidate[field] = value
                benchmark.comparisons([baseline, candidate])
                self.assertIsNone(candidate["speedup_vs_torch"])

    def test_speedups_require_complete_provenance_and_all_rank_results(self):
        for scenario in ("missing_source", "missing_rank", "duplicate_rank", "wrong_rank"):
            with self.subTest(scenario=scenario):
                baseline = self.result_row()
                candidate = self.result_row("deepopt-bf16", seconds=1)
                for row in (baseline, candidate):
                    if scenario == "missing_source":
                        row.pop("source_sha256")
                    elif scenario == "missing_rank":
                        row["world_size"] = 2
                    elif scenario == "duplicate_rank":
                        row["world_size"] = 2
                        row["ranks"] *= 2
                    else:
                        row["ranks"][0]["rank"] = 1
                benchmark.comparisons([baseline, candidate])
                self.assertIsNone(candidate["speedup_vs_torch"])

    def test_compiled_speedup_requires_a_matching_compiled_baseline(self):
        baseline = self.result_row(seconds=2.0)
        compiled = self.result_row("torch-bf16-compiled", seconds=1.0)
        candidate = self.result_row("deepopt-bf16-compressed", seconds=0.5)
        rows = [baseline, compiled, candidate]
        benchmark.comparisons(rows)
        self.assertEqual(candidate["speedup_vs_torch"], 4.0)
        self.assertEqual(candidate["speedup_vs_torch_compiled"], 2.0)
        compiled["batch_size_per_gpu"] = 32
        benchmark.comparisons(rows)
        self.assertEqual(candidate["speedup_vs_torch"], 4.0)
        self.assertIsNone(candidate["speedup_vs_torch_compiled"])
        candidate["status"] = "nonfinite"
        benchmark.comparisons(rows)
        self.assertIsNone(candidate["speedup_vs_torch"])
        self.assertIsNone(candidate["speedup_vs_torch_compiled"])

    def test_compiled_torch_case_uses_requested_inductor_mode_without_cuda_graphs(self):
        model = torch.nn.Linear(2, 1)
        compiled = object()
        args = benchmark.parser().parse_args([])
        options = benchmark.mode_settings(benchmark.settings(args), "torch-bf16-compiled")["runtime"]["compile"]
        with patch.object(torch, "compile", return_value=compiled) as compile_model:
            self.assertIs(benchmark.compile_torch_model(model, options), compiled)
        compile_model.assert_called_once_with(
            model, backend="inductor", mode="max-autotune-no-cudagraphs", fullgraph=False, dynamic=False,
        )

    def test_repeated_trials_report_medians_and_retain_raw_samples(self):
        runs = [self.result_row(repeat=i, seconds=value) for i, value in enumerate((3.0, 1.0, 2.0))]
        before = copy.deepcopy(runs)
        rows = benchmark.aggregate_runs(runs, ["torch-bf16"], 3)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["wall_seconds"], 2.0)
        self.assertEqual(row["wall_seconds_samples"], [3.0, 1.0, 2.0])
        self.assertEqual((row["wall_seconds_min"], row["wall_seconds_max"]), (1.0, 3.0))
        self.assertEqual(row["ranks"][0]["initial_state_sha256"], "weights")
        self.assertNotIn("compiler_report", row["ranks"][0])
        self.assertEqual(row["workload_sha256"], "workload")
        self.assertEqual(runs, before)

    def test_aggregate_requires_all_distinct_successful_consistent_trials(self):
        valid = [self.result_row(repeat=i) for i in range(3)]
        cases = {"missing": valid[:2], "duplicate": [valid[0], valid[0], valid[2]]}
        failed = copy.deepcopy(valid)
        failed[1] = {"mode": "torch-bf16", "repeat": 1, "status": "failed", "returncode": 1}
        cases["failed"] = failed
        changed_inputs = copy.deepcopy(valid)
        changed_inputs[1]["ranks"][0]["input_sha256"] = "different"
        cases["changed_inputs"] = changed_inputs
        changed_batch = copy.deepcopy(valid)
        changed_batch[1]["batch_size_per_gpu"] = 32
        cases["changed_batch"] = changed_batch
        for name, runs in cases.items():
            with self.subTest(case=name):
                rows = benchmark.aggregate_runs(runs, ["torch-bf16"], 3)
                self.assertNotEqual(rows[0]["status"], "ok")

    def test_supervisor_isolates_ddp_modes_and_keeps_results_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            argv = ["--gpus", "4", "--repeats", "1", "--output", str(output),
                    "--modes", "torch-bf16", "deepopt-bf16-compressed"]
            args = benchmark.parser().parse_args(argv)
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                mode = command[command.index("--_case") + 1]
                destination = Path(command[command.index("--_result") + 1])
                if mode == "deepopt-bf16-compressed":
                    # The prior successful result is already durable before this failure.
                    self.assertEqual(json.loads(output.read_text())["results"][0]["status"], "ok")
                    return SimpleNamespace(returncode=1)
                row = self.result_row(mode)
                snapshot = Path(command[command.index("--_snapshot") + 1])
                row["workload_sha256"] = benchmark.json_fingerprint(json.loads(snapshot.read_text()))
                benchmark.write_json(destination, row)
                return SimpleNamespace(returncode=0)

            with patch.dict("os.environ", {"WORLD_SIZE": "1"}), patch.object(subprocess, "run", side_effect=run), \
                    redirect_stdout(io.StringIO()):
                status = benchmark.suite(args, argv)
            self.assertEqual(status, 1)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(command[0] == sys.executable for command in calls))
            self.assertTrue(all("--nproc-per-node=4" in command for command in calls))
            result = json.loads(output.read_text())
            self.assertEqual([row["status"] for row in result["results"]], ["ok", "failed"])
            self.assertEqual([row["status"] for row in result["runs"]], ["ok", "failed"])

    def test_supervisor_repeats_fresh_processes_with_one_frozen_config(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            original = directory / "original.toml"
            original.write_text(benchmark.parser().parse_args([]).config.read_text(), encoding="utf-8")
            output = directory / "results.json"
            argv = ["--config", str(original), "--output", str(output), "--repeats", "3"]
            args = benchmark.parser().parse_args(argv)
            expected = benchmark.settings(args)
            calls = []
            snapshots = []

            def run(command, **kwargs):
                calls.append(command)
                mode = command[command.index("--_case") + 1]
                repeat = int(command[command.index("--_repeat") + 1])
                snapshot = Path(command[command.index("--_snapshot") + 1])
                snapshots.append(snapshot)
                self.assertEqual(json.loads(snapshot.read_text()), expected)
                # The frozen config must survive arbitrary edits to its source file.
                original.write_text("this is no longer valid TOML", encoding="utf-8")
                destination = Path(command[command.index("--_result") + 1])
                row = self.result_row(mode, repeat=repeat, seconds=(3.0, 1.0, 2.0)[repeat])
                row["workload_sha256"] = benchmark.json_fingerprint(expected)
                benchmark.write_json(destination, row)
                return SimpleNamespace(returncode=0)

            with patch.dict("os.environ", {"WORLD_SIZE": "1"}), patch.object(subprocess, "run", side_effect=run), \
                    redirect_stdout(io.StringIO()):
                status = benchmark.suite(args, argv)
            self.assertEqual(status, 0)
            self.assertEqual(len(calls), len(benchmark.MODES) * 3)
            self.assertEqual(len(set(snapshots)), 1)
            cases = {(command[command.index("--_case") + 1], int(command[command.index("--_repeat") + 1]))
                     for command in calls}
            self.assertEqual(cases, {(mode, repeat) for mode in benchmark.MODES for repeat in range(3)})
            destinations = [command[command.index("--_result") + 1] for command in calls]
            self.assertEqual(len(set(destinations)), len(benchmark.MODES) * 3)
            result = json.loads(output.read_text())
            self.assertEqual(result["settings"], expected)
            self.assertEqual(len(result["runs"]), len(benchmark.MODES) * 3)
            self.assertEqual(len(result["results"]), len(benchmark.MODES))
            self.assertTrue(all(row["status"] == "ok" and row["wall_seconds"] == 2.0
                                for row in result["results"]))

    def test_supervisor_keeps_missing_corrupt_and_wrong_native_receipts_as_failures(self):
        for defect in ("missing", "corrupt", "wrong_mode", "missing_native_report", "failed_exit"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "results.json"
                mode = "deepopt-bf16-native-compressed"
                argv = ["--output", str(output), "--repeats", "1", "--modes", mode]
                args = benchmark.parser().parse_args(argv)

                def run(command, **kwargs):
                    destination = Path(command[command.index("--_result") + 1])
                    if defect == "missing":
                        return SimpleNamespace(returncode=0)
                    if defect == "corrupt":
                        destination.write_text("{", encoding="utf-8")
                        return SimpleNamespace(returncode=0)
                    row = self.result_row(mode)
                    snapshot = Path(command[command.index("--_snapshot") + 1])
                    row["workload_sha256"] = benchmark.json_fingerprint(json.loads(snapshot.read_text()))
                    if defect == "wrong_mode":
                        row["mode"] = "deepopt-bf16-compressed"
                    elif defect == "missing_native_report":
                        row["ranks"][0]["compiler_report"] = None
                    benchmark.write_json(destination, row)
                    return SimpleNamespace(returncode=1 if defect == "failed_exit" else 0)

                with patch.dict("os.environ", {"WORLD_SIZE": "1"}), patch.object(subprocess, "run", side_effect=run), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(benchmark.suite(args, argv), 1)
                result = json.loads(output.read_text())
                self.assertEqual(result["runs"][0]["mode"], mode)
                self.assertEqual(result["runs"][0]["status"], "failed")
                self.assertEqual(result["results"][0]["successful_repeats"], 0)
                self.assertEqual(result["status"], "failed")

    def test_worker_uses_frozen_settings_when_original_config_is_unavailable(self):
        args = benchmark.parser().parse_args([])
        raw = benchmark.settings(args)
        with tempfile.TemporaryDirectory() as directory:
            args._snapshot = Path(directory) / "settings.json"
            benchmark.write_json(args._snapshot, raw)
            args.config = Path(directory) / "removed-original.toml"
            args._case = "torch-bf16"
            # Reading the original config here would fail before the CUDA check.
            with patch.object(torch.cuda, "is_available", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "CUDA"):
                benchmark.worker(args)


if __name__ == "__main__":
    unittest.main()
