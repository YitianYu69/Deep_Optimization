"""One matched VAE compiler-parity route; code and metadata only are retained."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "deepopt.vae.compiler_parity.worker.v1"
# Arm manifests pin the exact wheel for qualification. Unpinned runs default to
# the recorded parity release; a host may select another installed candidate
# explicitly, which the installed-identity check still verifies.
EXPECTED_VERSION = os.environ.get(
    "DEEPOPT_VAE_EXPECTED_VERSION", "0.1.0+nativeautodiff.20261120.vaeopt.3")
ROUTES = ("native_torch_eager", "native_torch_inductor", "deepopt_eager", "deepopt_native")
NATIVE_OPTIONS = {"enable_aot_autodiff": True, "use_native_autodiff": True,
    "require_full_aot": True, "capture_deepopt": True, "mutation_backend": "native",
    "share_compatible_packs": False, "fuse_packed_multiply": False}
SOURCES = ("examples/vae/parity_execution.py", "examples/vae/benchmark_parity.py",
    "examples/vae/native_execution.py", "examples/vae/execution_memory.py", "examples/vae/execution_profile.py",
    "examples/vae/benchmark_execution.py", "examples/vae/memory_comparison.py",
    "examples/vae/benchmark_vae.py", "examples/vae/mbconv_vae.py", "examples/vae/qualification.py", "benchmarks/compiler/installed_runtime.py",
    "benchmarks/compiler/inductor_benchmark_artifacts.py", "benchmarks/compiler/inductor_reference_store.py",
    "benchmarks/compiler/save_inductor_references.py")


def parser():
    from examples.vae.native_execution import parser as existing

    result = existing()
    result.description = __doc__
    result.set_defaults(batch_size=16, warmup=5, updates=10, benchmark=True,
                        output=Path("work/vae_parity_worker.json"))
    result.add_argument("--route", choices=ROUTES, default="deepopt_native")
    result.add_argument("--inductor-preset", choices=("default", "cast-preserving"), default="cast-preserving")
    result.add_argument("--cpu-tiny", action="store_true", help="Explicit reduced VAE/16x16 CPU diagnostic")
    return result


def source_identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def execution_plan(args):
    from examples.vae import native_execution as existing

    if not args.benchmark or not args.memory_audit or args.warmup < 1:
        raise ValueError("Parity workers require benchmark timing, staged memory audit and positive warmup")
    if args.cpu_tiny != (args.device == "cpu"):
        raise ValueError("CPU diagnostics require explicit --cpu-tiny; target CUDA retains the original model")
    if args.native_options_json is not None and args.route != "deepopt_native":
        raise ValueError("Native compiler overrides belong only to deepopt_native")
    base = deepcopy(args)
    base.execution_mode = "native" if args.route == "deepopt_native" else "eager"
    plan = existing.execution_plan(base)
    original = plan["target_settings"]
    if plan["model_factory"] is not None:
        raise ValueError("Parity uses the supplied MBConvVAE_V2, without a replacement model factory")
    common = deepcopy(plan["settings"])
    if not args.cpu_tiny and (common["application"]["model"] != existing.ORIGINAL_MODEL
            or common["application"]["data"]["image_size"] != 224):
        raise ValueError("Target parity requires the original VAE architecture and image224")
    if args.cpu_tiny:
        common["application"]["model"] = dict(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=1)
        common["application"]["data"].update(image_size=16, resize_size=20)
    runtime = common["runtime"]
    runtime.update(precision="bf16", bf16_policy="torch_amp", fp8={"enabled": False})
    runtime["activation_compression"] = {**runtime.get("activation_compression", {}), "enabled": False, "group_size": 1024}
    runtime["activation_memory"] = {"multiply_backward": "materialized", "pack_reuse": "off",
        "reduction_scratch_bytes": 8 * 2**20, "recompute_modules": []}
    runtime["compile"] = {"enabled": False, "backend": "deepopt", "mode": None,
        "fullgraph": True, "dynamic": False, "options": dict(NATIVE_OPTIONS)}
    if args.native_options_json is not None:
        runtime["compile"]["options"].update(plan["native_options_override"]["options"])
    if runtime["compile"]["options"].get("share_compatible_packs") is not False or runtime["compile"]["options"].get("fuse_packed_multiply") is not False:
        raise ValueError("Common AMP parity cannot enable compressed pack sharing or packed multiplication")
    expected = plan["expected_installed_version"] if plan["experiment_arm"] is not None else EXPECTED_VERSION
    identity_settings = deepcopy(common)
    identity_settings["runtime"].pop("compile")
    return {"parity_schema": SCHEMA, "schema_version": 1, "route": args.route,
        "inductor_preset": args.inductor_preset, "common_settings": common, "target_settings": original,
        "common_recipe_sha256": hashlib.sha256(json.dumps(identity_settings, sort_keys=True).encode()).hexdigest(),
        "origin": plan["origin"], "device": args.device, "warmup": args.warmup, "updates": args.updates,
        "requested_input_shape": [args.batch_size, 3, common["application"]["data"]["image_size"], common["application"]["data"]["image_size"]],
        "expected_installed_version": expected, "experiment_arm": plan["experiment_arm"],
        "native_options_override": plan["native_options_override"], "project_root": plan["project_root"],
        "optimizer_factory": plan["optimizer_factory"], "input_mode": plan["input_mode"],
        "diagnostic_overrides": list(dict.fromkeys([reason for reason in plan["diagnostic_overrides"] if reason != "non_target_geometry"]
            + (["explicit_tiny_CPU_model"] if args.cpu_tiny else []))),
        "compiled_scope": {"model_forward": True, "model_backward": True, "loss": "eager_FP32_L1_KL", "optimizer": "eager"},
        "external_input_requires_grad": False, "compression_enabled": False, "group_size": 1024,
        "profiling": plan["profiling"], "qualified": False, "correctness_assessed": False,
        "first_update_trace": "deferred_separate_fresh_worker_not_implemented", "typed_storage_owner_ledger": "deferred",
        "tensor_files_written": 0}


def _software(torch, device):
    result = {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        result["triton"] = version("triton")
    except PackageNotFoundError:
        result["triton"] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": props.name, "uuid": str(props.uuid), "capability": [props.major, props.minor], "total_memory": props.total_memory}
    return result


def _rng(torch, device, benchmark):
    values = [torch.get_rng_state()]
    if device.type == "cuda":
        values.append(torch.cuda.get_rng_state(device))
    return benchmark.fingerprint(values)


def run_worker(args):
    from examples.vae import benchmark_vae as benchmark, native_execution as existing
    from examples.vae.execution_memory import ExecutionMemoryAudit

    plan = execution_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh parity output; historical captures are immutable")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "execution_completed": False, "warmup_completed": 0,
        "updates_completed": 0, "steps": [], "memory": {}, "helper_sha256": source_identity(),
        "execution_mode": "native" if args.route == "deepopt_native" else "eager",
        "native_ownership_verified": False, "route_ownership_verified": False, "timing_eligible": False,
        "measurement": {"valid": False, "invalid_reasons": [], "samples": [],
            "timing_scope": "device_resident_model_loss_backward_optimizer_schedule",
            "wall_scope": "model/loss/backward/optimizer/zero_grad plus schedules",
            "cuda_event_scope": "current_stream_train_step_excluding_schedules_and_H2D"},
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    context = {"stage": "setup", "phase": "installed_identity", "update": None}
    report["failure_context"] = context
    audit = ExecutionMemoryAudit()
    report["memory_audit"] = audit.report
    benchmark.write_json(output, report)
    stack = ExitStack()
    torch = deepopt = model = optimizer = runtime = images = cpu_images = loader = loss = losses = None
    device = hook = recorder = artifact_capture = schedules = current = before_step = progress = None
    captures = []
    managed, native = args.route.startswith("deepopt_"), args.route == "deepopt_native"
    compiled = args.route == "native_torch_inductor"
    try:
        report["source_artifacts"] = []
        for name, digest in report["helper_sha256"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Parity helper changed before its source snapshot")
            target = directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            report["source_artifacts"].append(existing._save_bytes(target.parent, target.name, data))
        original_bytes = args.from_result.read_bytes()
        if hashlib.sha256(original_bytes).hexdigest() != plan["origin"]["sha256"]:
            raise RuntimeError("Original parity receipt changed after planning")
        report["original_receipt_artifact"] = existing._save_bytes(directory, "original-receipt.json", original_bytes)
        existing._verify_experiment_inputs(plan)
        for key, name in (("native_options_override", "native-options.json"), ("experiment_arm", "arm-manifest.json")):
            value = plan.get(key)
            if value is not None:
                source = value["source"] if key == "experiment_arm" else value
                report[key + "_artifact"] = existing._save_bytes(directory, name, Path(source["path"]).read_bytes())
        if plan["project_root"] and plan["project_root"] not in sys.path:
            sys.path.append(plan["project_root"])
        cache = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="deepopt-parity-cache-")))
        cache_env = {"TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TRITON_CACHE_DIR": str(cache / "triton"),
            "TRITON_DUMP_DIR": str(cache / "triton-dumps"), "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0", "TORCHINDUCTOR_SAVE_COMPILE_ARGS": "0",
            "TORCH_COMPILE_DEBUG_SAVE_REAL": "0", "TORCHDYNAMO_REPRO_LEVEL": "0"}
        removed = ("TORCH_TRACE", "TRITON_OVERRIDE_DIR", "TORCHDYNAMO_REPRO_AFTER")
        previous_env = {name: os.environ.get(name) for name in (*cache_env, *removed)}
        def restore_environment():
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        stack.callback(restore_environment)
        for name in removed:
            os.environ.pop(name, None)
        os.environ.update(cache_env)
        report["compiler_cache_protocol"] = {"fresh_per_worker": True, "persistent_cache_used": False,
            "temporary_working_cache": True, "native_cache_IR_not_copied": True}
        os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
        if plan["common_settings"]["runtime"].get("deterministic"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from benchmarks.compiler.installed_runtime import installed_identity

        report["installed"] = installed_identity(ROOT, full=True)
        if report["installed"]["version"] != plan["expected_installed_version"]:
            raise RuntimeError("Installed wheel differs from the pinned parity version")
        if plan["experiment_arm"]:
            pin = plan["experiment_arm"]["manifest"]["expected_installed"]
            if any(report["installed"][name] != pin[name] for name in ("version", "source_sha256", "files_hashed")):
                raise RuntimeError("Installed parity package differs from its arm pin")
        import torch
        import deepopt
        from deepopt.runtime import seed_everything
        from deepopt.compiler.runtime import native_boundary_observation
        from benchmarks.compiler.save_inductor_references import disable_dynamo_reproduction, native_graph_provenance, inductor_compile_options
        from benchmarks.compiler.inductor_benchmark_artifacts import InductorBenchmarkArtifacts, inductor_reference_context, validate_inductor_artifacts

        report["compiler_environment"] = disable_dynamo_reproduction()
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU substitution")
            torch.cuda.set_device(device)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Selected CUDA GPU does not support BF16")
            torch.cuda.reset_peak_memory_stats(device)
        raw = deepcopy(plan["common_settings"])
        raw["runtime"]["precision"] = "fp32" if device.type == "cpu" else "bf16"
        raw["runtime"]["compile"].update(enabled=native, options=dict(raw["runtime"]["compile"]["options"]) if native else {}, mode=None)
        config = deepopt.load_config(raw)
        if native:
            from deepopt.compiler import BackendConfig
            BackendConfig(**raw["runtime"]["compile"]["options"])
        seed_everything(raw.get("seed", 42), raw["runtime"].get("deterministic", False))
        torch.utils.deterministic.fill_uninitialized_memory = raw["runtime"].get("fill_uninitialized_memory", False)
        audit.bind(torch, device)
        audit.prepare_baseline()
        report["software"] = _software(torch, device)
        context["phase"] = "input_loading"
        if args.synthetic:
            cpu_images = torch.rand(plan["requested_input_shape"], generator=torch.Generator().manual_seed(raw.get("seed", 42))) * 2 - 1
        else:
            loader = benchmark.build_loader(raw, SimpleNamespace(batches=1), SimpleNamespace(world_size=1, global_rank=0))
            cpu_images = next(iter(loader))
        if list(cpu_images.shape) != plan["requested_input_shape"] or not cpu_images.is_contiguous() or cpu_images.dtype != torch.float32:
            raise ValueError("Parity input must be the declared dense FP32 physical batch")
        report["input"] = {"sha256": benchmark.fingerprint([cpu_images]), "shape": list(cpu_images.shape),
            "dtype": str(cpu_images.dtype), "requires_grad": False, "mode": plan["input_mode"]}
        context["phase"] = "model_initialization"
        from examples.vae.mbconv_vae import MBConvVAE_V2

        model = MBConvVAE_V2(**raw["application"]["model"]).to(device).train()
        report["model"] = {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
            "initial_state_sha256": benchmark.fingerprint(model.state_dict().values())}
        if not args.cpu_tiny and (report["model"]["parameters"] != 32317999 or report["model"]["trainable_parameter_tensors"] != 332):
            raise RuntimeError("Original VAE parameter geometry changed")
        optimizer = benchmark.make_optimizer(model, raw["training"], plan["optimizer_factory"])
        report["optimizer"] = {"name": raw["training"]["optimizer"], "initial_state_sha256": existing._state_fingerprint(optimizer.state_dict(), torch, benchmark)}
        report["provenance"] = benchmark.source_provenance(deepopt, MBConvVAE_V2, optimizer)
        images = cpu_images.to(device, non_blocking=raw["runtime"].get("non_blocking", True)).requires_grad_(False)
        report["input"].update(stride=list(images.stride()), contiguous=images.is_contiguous(), storage_bytes=images.untyped_storage().nbytes())
        if managed:
            runtime = deepopt.TrainingRuntime(model, optimizer, device=device, precision=raw["runtime"]["precision"], config=config)
            model = runtime.model.train()
            if not native:
                existing._verify_eager_ownership(model, runtime)
        report["settings"] = raw
        report["runtime_config"] = runtime.config if runtime is not None else raw
        schedules = deepopt.ScheduleController.from_config(config.schedules)
        if compiled:
            task = benchmark.inductor_artifact_task(raw, args.inductor_preset)
            task["workload"]["gradient_mode"] = "trainable_parameters"
            artifact_capture = stack.enter_context(InductorBenchmarkArtifacts(directory, task, enabled=True))
            report["numerical_controls"] = artifact_capture.configuration["numerical_controls"]
            def backend(graph, example_inputs, **kwargs):
                evidence = native_graph_provenance(graph)
                entry = {"code": graph.code, "provenance": evidence}
                captures.append(entry)
                if not evidence["native_ops_only"]:
                    raise RuntimeError("Opaque/custom target in native Torch model; fallback is forbidden")
                return torch._inductor.compile(graph, example_inputs, options=inductor_compile_options(torch, "default", kwargs))
            model = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        else:
            report["numerical_controls"] = stack.enter_context(inductor_reference_context(args.inductor_preset))
        recorder = existing._ArtifactRecorder(torch, context) if native else None

        def ownership():
            if native:
                evidence = benchmark.native_compilation_evidence(model)
                return {"owner": "deepopt_native_autodiff", "captured_graphs": evidence["captured_graphs"], "runtime_fallbacks": evidence["runtime_fallbacks"]}
            if managed:
                existing._verify_eager_ownership(model, runtime)
            elif not compiled and any(getattr(module, "_compiled_call_impl", None) is not None for module in model.modules()):
                raise RuntimeError("Native Torch eager unexpectedly contains a compiled module")
            return {"owner": "pytorch_aotautograd" if compiled else "pytorch_autograd",
                    "captured_graphs": len(captures), "runtime_fallbacks": 0}

        def before_step(owner, positional, keyword):
            gradients = [p.grad for p in model.parameters() if p.requires_grad]
            current["gradient_parameters"] = len(gradients)
            current["all_trainable_gradients_present"] = all(value is not None for value in gradients)
            current["fp32_gradients"] = all(value is not None and value.dtype == torch.float32 for value in gradients)
            current["finite_gradients"] = existing._finite(torch, gradients)
            if not all(current[name] for name in ("all_trainable_gradients_present", "fp32_gradients", "finite_gradients")):
                raise RuntimeError("Parity requires present finite FP32 trainable gradients")
        hook = optimizer.register_step_pre_hook(before_step)
        audit.capture("after_setup", update=0)
        seed_everything(raw.get("seed", 42), raw["runtime"].get("deterministic", False))
        report["rng_before_warmup_sha256"] = _rng(torch, device, benchmark)
        with native_boundary_observation(recorder) if native else nullcontext():
            for number in range(1, args.warmup + 1):
                current = {"number": number, "stage": "warmup"}
                report["steps"].append(current)
                context.update(stage="warmup", phase="schedule", update=number)
                schedules.prepare_update(number, epoch=1)
                def progress(event):
                    context.update(event)
                start = time.perf_counter()
                loss = benchmark.train_step(model, optimizer, runtime, images, schedules["beta"], device_type=device.type,
                    grad_clip_norm=config.training.grad_clip_norm, progress=progress)
                if not existing._finite(torch, (loss,)) or not existing._finite(torch, model.parameters()):
                    raise RuntimeError("Nonfinite warmup loss or parameters")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                current["wall_seconds_including_checks"] = time.perf_counter() - start
                schedules.complete_update(number)
                report["warmup_completed"] += 1
                if number == 1:
                    audit.capture("after_first_update", update=number)
            audit.capture("after_warmup", update=args.warmup)
        hook.remove()
        hook = None
        report["rng_after_warmup_sha256"] = _rng(torch, device, benchmark)
        before = ownership()
        if compiled and (not captures or any(not item["provenance"]["native_ops_only"] for item in captures)):
            raise RuntimeError("Native Torch capture lacks a complete native operator audit")
        report["measurement"]["native_graph_count_before"] = before["captured_graphs"]
        losses = existing._benchmark_updates(torch=torch, device=device, benchmark=benchmark, model=model,
            optimizer=optimizer, runtime=runtime, images=images, schedules=schedules, grad_clip_norm=config.training.grad_clip_norm,
            warmup=args.warmup, updates=args.updates, report=report, context=context, memory_audit=audit)
        report["measurement"]["postchecks"] = {"finite_losses": existing._finite(torch, losses), "finite_parameters": existing._finite(torch, model.parameters())}
        if not all(report["measurement"]["postchecks"].values()):
            raise RuntimeError("Parity measured postchecks found nonfinite state")
        for value, sample in zip(losses, report["measurement"]["samples"]):
            sample["loss"] = float(value)
        losses = loss = None
        report["input_stable"] = benchmark.fingerprint([images]) == report["input"]["sha256"]
        report["rng_after_measurement_sha256"] = _rng(torch, device, benchmark)
        after = ownership()
        report["measurement"]["native_graph_count_after"] = after["captured_graphs"]
        if before != after or not report["input_stable"]:
            raise RuntimeError("Measured compiler ownership/graph count or resident input changed")
        report["ownership"] = after
        report["native_ownership_verified"] = native
        report["route_ownership_verified"] = True
        if native:
            evidence = benchmark.native_compilation_evidence(model)
            report["compiler_report_artifact"] = existing._save_bytes(directory, "compiler-report.json", json.dumps(existing._metadata(evidence, torch), indent=2, allow_nan=False).encode())
            report["compiler_report_summary"] = {key: value for key, value in evidence.items() if key != "graphs"}
        report["memory"]["postchecks"] = existing._memory(torch, device)
        report["memory"]["total"] = {key: max((row[key] for row in report["memory"].values() if row.get(key) is not None), default=None)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
        report["cuda_allocator_diagnostics"] = benchmark.cuda_failure_diagnostics(device)
        audit.capture("after_postchecks", update=args.warmup + args.updates)
        report["execution_completed"] = True
        report["measurement"]["valid"] = True
        report["status"] = "completed"
        report["compile_costs"] = {"first_update_seconds_including_checks": report["steps"][0]["wall_seconds_including_checks"],
            "torch_compile_times": torch._dynamo.utils.compile_times()}
        report["measurement_frozen_artifact"] = existing._save_bytes(directory, "measurement-frozen.json",
            json.dumps({key: report[key] for key in ("measurement", "memory", "ownership", "cuda_allocator_diagnostics")}, indent=2, allow_nan=False).encode())
        if args.profile_after_measurement:
            from examples.vae.execution_profile import run_execution_profile
            audit.capture("profile_start", update=args.warmup + args.updates)
            profile_losses = []
            def update(number, progress):
                schedules.prepare_update(number, epoch=1)
                profile_losses.append(benchmark.train_step(model, optimizer, runtime, images, schedules["beta"], device_type=device.type,
                    grad_clip_norm=config.training.grad_clip_norm, progress=progress))
                schedules.complete_update(number)
            try:
                report["profile"] = run_execution_profile(torch=torch, device=device, directory=directory / "profile",
                    update=update, first_update=args.warmup + args.updates + 1, updates=args.profile_updates,
                    record_memory=args.profile_memory, record_shapes=args.profile_record_shapes, ownership=ownership,
                    expected_ownership=after, allocator_diagnostics=lambda: benchmark.cuda_failure_diagnostics(device),
                    postchecks=lambda: {"finite_losses": existing._finite(torch, profile_losses), "finite_parameters": existing._finite(torch, model.parameters())})
            except Exception as error:
                report["profile"] = {"complete": False, "status": "incomplete", "measurement_eligible": False,
                    "updates_completed": 0, **benchmark.failure_error_details(error)}
            finally:
                profile_losses.clear()
                audit.capture("profile_end", update=args.warmup + args.updates + report["profile"].get("updates_completed", 0))
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        report["measurement"]["valid"] = False
        if type(error).__name__ == "OutOfMemoryError":
            report["capacity_status"] = "cuda_oom"
    finally:
        if hook is not None:
            try:
                hook.remove()
            except Exception as error:
                audit.record_error("cleanup", "optimizer_hook", error)
        if torch is not None:
            if recorder is not None:
                try:
                    report["artifacts"] = recorder.save(directory)
                except Exception as error:
                    report.setdefault("collection_errors", []).append(benchmark.failure_error_details(error))
                    report["measurement"]["valid"] = False
            for index, capture in enumerate(captures):
                try:
                    capture["artifact"] = existing._save_bytes(directory, f"dynamo-{index}.py", capture.pop("code").encode())
                except Exception as error:
                    report.setdefault("collection_errors", []).append(benchmark.failure_error_details(error))
                    report["measurement"]["valid"] = False
            report["native_torch_captures"] = captures
            if artifact_capture is not None:
                try:
                    report["inductor_artifacts"] = artifact_capture.finish(row={"status": "ok" if report["execution_completed"] else "failed",
                        "timing": report["measurement"], "finite_after_timing": report["measurement"].get("postchecks", {}).get("finite_parameters"),
                        "execution": {"compile_counters": {}}, "environment": {"device": str(device)}})
                    report["inductor_validation"] = validate_inductor_artifacts(report["inductor_artifacts"], directory)
                    if not report["inductor_validation"]["passed"]:
                        report["measurement"]["valid"] = False
                except Exception as error:
                    report.setdefault("collection_errors", []).append(benchmark.failure_error_details(error))
                    report["measurement"]["valid"] = False
            try:
                report["installed_after"] = installed_identity(ROOT, full=True)
                existing._verify_experiment_inputs(plan)
                report["helper_sha256_after"] = source_identity()
                if optimizer is not None:
                    report["provenance_after"] = benchmark.source_provenance(deepopt, MBConvVAE_V2, optimizer)
                    report["source_stable"] = report.get("provenance") == report["provenance_after"]
                report["installed_stable"] = report.get("installed") == report["installed_after"]
                report["helper_source_stable"] = report["helper_sha256"] == report["helper_sha256_after"]
                if not report["installed_stable"] or not report["helper_source_stable"] or report.get("source_stable") is False:
                    raise RuntimeError("Parity installed package or helper source changed")
            except Exception as error:
                report.setdefault("collection_errors", []).append(benchmark.failure_error_details(error))
                report["measurement"]["valid"] = False
        try:
            stack.close()
        except Exception as error:
            audit.record_error("cleanup", "compiler_context", error)
        model = optimizer = runtime = images = cpu_images = loader = loss = losses = recorder = hook = None
        schedules = current = before_step = progress = artifact_capture = None
        actions = []
        if deepopt is not None:
            actions.append(("shutdown_distributed", deepopt.shutdown_distributed))
        if torch is not None:
            actions += [("compiler_reset", torch.compiler.reset), ("clear_autocast_cache", torch.clear_autocast_cache)]
        audit.cleanup(release_actions=actions)
        stages = ["before_model", "after_setup", "after_first_update", "after_warmup", "measurement_start", "measurement_end", "after_postchecks", "cleanup"]
        if args.profile_after_measurement:
            stages += ["profile_start", "profile_end"]
        audit.finalize(expected_stages=stages)
        if not report["measurement"]["valid"]:
            report["status"] = "failed"
        report["timing_eligible"] = report["measurement"]["valid"]
        report["target_CUDA_execution"] = report["execution_completed"] is True and device is not None and device.type == "cuda" and not report["diagnostic_overrides"]
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_worker(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
