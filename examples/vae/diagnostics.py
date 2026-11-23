"""Untimed, installed-only diagnostics for the existing VAE benchmark.

Planning is standard-library only. Each comparison gets a fresh process and
retains scalar evidence, generated code and metadata, never tensor files.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRACE_MODULES = ("encoder.0", "decoder.0", "decoder.4", "decoder.8")
PROFILES = {
    "dense": (False, "materialized"),
    "materialized": (True, "materialized"),
    "streamed": (True, "packed_streaming"),
}
HELPERS = ("examples/vae/benchmark_vae.py", "examples/vae/qualification.py",
           "examples/vae/diagnostics.py", "examples/vae/diagnostic_operators.py",
           "examples/vae/diagnostic_localization.py", "examples/vae/diagnostic_repeatability.py",
           "examples/vae/diagnostic_reporting.py", "benchmarks/compiler/installed_runtime.py")


def _helper_identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in HELPERS}


def diagnostic_cases():
    cases = [{"name": "streamed_eager_native_untraced", "kind": "trajectory", "observe_boundaries": False,
              "reference": {"execution": "eager", "profile": "streamed"},
              "candidate": {"execution": "native", "profile": "streamed"}}]
    for profile in PROFILES:
        for label, reference, candidate in (("eager_repeat", "eager", "eager"),
                                            ("native_repeat", "native", "native"),
                                            ("eager_native", "eager", "native")):
            cases.append({"name": f"{profile}_{label}", "kind": "trajectory",
                          "reference": {"execution": reference, "profile": profile},
                          "candidate": {"execution": candidate, "profile": profile}})
    for execution in ("eager", "native"):
        cases.append({"name": execution + "_materialized_streamed", "kind": "trajectory",
                      "reference": {"execution": execution, "profile": "materialized"},
                      "candidate": {"execution": execution, "profile": "streamed"}})
    cases.append({"name": "operator_replays", "kind": "operators"})
    return cases


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def diagnostic_plan(args):
    from examples.vae import benchmark_vae as benchmark

    receipt_path = getattr(args, "diagnostic_from_result", None)
    if receipt_path is not None:
        data = Path(receipt_path).read_bytes()
        receipt = json.loads(data)
        raw = deepcopy(receipt["settings"])
        origin = {"path": str(Path(receipt_path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                  "workload_sha256": receipt.get("workload_sha256")}
        optimizer_factory = receipt.get("optimizer_factory", benchmark.CUSTOM_OPTIMIZER)
        original_factory = receipt.get("model_factory")
        if args.optimizer_factory != benchmark.CUSTOM_OPTIMIZER:
            optimizer_factory = args.optimizer_factory
    else:
        raw = benchmark.settings(args)
        optimizer_factory = args.optimizer_factory
        original_factory = None
        origin = {"config": str(args.config.resolve())}
    target = deepcopy(raw)
    batch = args.diagnostic_batch_size
    if type(batch) is not int or batch < 1:
        raise ValueError("diagnostic batch size must be positive")
    if batch > raw["training"]["batch_size"]:
        raise ValueError("diagnostic batch size cannot exceed the recorded target batch")
    if args.diagnostic_max_bytes < 1 or args.diagnostic_trace_max_bytes < 1:
        raise ValueError("diagnostic byte budgets must be positive")
    if raw["training"].get("accumulation_steps", 1) != 1:
        raise ValueError("VAE diagnostics require accumulation_steps=1")
    runtime = raw.setdefault("runtime", {})
    deterministic = getattr(args, "diagnostic_deterministic", None)
    if deterministic is None and args.deterministic:
        deterministic = True
    if deterministic is not None:
        runtime["deterministic"] = deterministic
    compression = runtime.setdefault("activation_compression", {})
    if compression.get("group_size", 1024) != 1024 or compression.get("bits", 2) != 2:
        raise ValueError("VAE diagnostics require two-bit/group1024 compression")
    if runtime.get("activation_memory", {}).get("recompute_modules"):
        raise ValueError("Diagnostic replay currently requires no block recomputation")
    memory = runtime.get("activation_memory", {})
    scratch = memory.get("reduction_scratch_bytes", 8388608)
    if type(scratch) is not int or scratch < 4:
        raise ValueError("reduction_scratch_bytes must be an integer >= 4")
    if memory.get("pack_reuse", "inherit") not in ("inherit", "off"):
        raise ValueError("Diagnostic replay requires supported inherit/off pack reuse")
    if raw["application"]["data"]["image_size"] < 16:
        raise ValueError("VAE diagnostic image size must be at least 16")
    raw["training"]["batch_size"] = batch
    if args.train_dir is not None:
        raw["application"]["data"]["train_dir"] = str(args.train_dir)
    if args.optimizer is not None:
        raw["training"]["optimizer"] = args.optimizer
    if args.bf16_policy is not None:
        runtime["bf16_policy"] = args.bf16_policy
    runtime.setdefault("bf16_policy", "bf16_storage")
    if runtime["bf16_policy"] != "bf16_storage":
        raise ValueError("This diagnostic stage requires --bf16-policy bf16_storage")
    runtime["fp8"] = {"enabled": False}
    compression.update(group_size=1024, bits=2, norm_xhat_recenter=True)
    runtime.setdefault("activation_memory", {}).update(
        pack_reuse=memory.get("pack_reuse", "inherit"), recompute_modules=[], reduction_scratch_bytes=scratch)
    raw.setdefault("loader", {}).update(num_workers=0, persistent_workers=False, drop_last=True)
    choices = diagnostic_cases()
    requested = getattr(args, "diagnostic_cases", None)
    if requested:
        available = [{"name": "backward_repeatability", "kind": "repeatability"}, *choices]
        names = {case["name"] for case in available}
        if len(set(requested)) != len(requested) or set(requested) - names:
            raise ValueError(f"Choose unique diagnostic cases from {sorted(names)}")
        choices = [case for case in available if case["name"] in requested]
    project = getattr(args, "diagnostic_project_root", None)
    if project is None:
        project = next((parent for parent in ROOT.parents
                        if (parent / "source/training/geo_optimizer.py").is_file()), None)
    if project is not None and not Path(project).is_dir():
        raise ValueError("Diagnostic project root must be an existing directory")
    return {"schema_version": 1, "scope": "untimed_numerical_diagnostics",
            "qualification_eligible": False, "timing_eligible": False, "cuda_qualified": False,
            "origin": origin, "target_settings": target, "settings": raw,
            "diagnostic_overrides": {"batch_size": batch, "loader_workers": 0,
                "input": "seeded_synthetic" if args.diagnostic_synthetic else "first_dataset_batch",
                "original_image_size_preserved": True, "original_model_options_preserved": True,
                "deterministic_algorithms": {"effective": runtime.get("deterministic", False),
                    "source": "recorded_settings" if deterministic is None else "explicit_diagnostic_control",
                    "original": target.get("runtime", {}).get("deterministic", False)}},
            "device": args.diagnostic_device, "optimizer_factory": optimizer_factory,
            "model_factory": args.model_factory or (original_factory if original_factory not in
                (None, "mbconv_vae:MBConvVAE_V2", "examples.vae.mbconv_vae:MBConvVAE_V2") else None),
            "project_root": None if project is None else str(Path(project).resolve()),
            "max_bytes": args.diagnostic_max_bytes, "trace_max_bytes": args.diagnostic_trace_max_bytes,
            "trace_modules": list(args.diagnostic_trace_modules or TRACE_MODULES),
            "large_operators": bool(args.diagnostic_large_operators), "cases": choices,
            "settings_sha256": _fingerprint(raw), "tensor_files_written": 0}


def _write(path, value):
    from examples.vae.benchmark_vae import write_json

    write_json(Path(path), value)


def run_diagnostics(args):
    plan = diagnostic_plan(args)
    if args.diagnostic_dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    from benchmarks.compiler.installed_runtime import child_environment, installed_identity
    from examples.vae.diagnostic_reporting import summary_for_parent

    identity = installed_identity(ROOT)
    output = args.output.resolve()
    artifacts = output.parent / (output.stem + "-artifacts")
    if output.exists() or artifacts.exists():
        raise ValueError("Choose a fresh diagnostic output; historical results are preserved")
    artifacts.mkdir(parents=True)
    report = {**plan, "started_at_utc": datetime.now(timezone.utc).isoformat(), "installed": identity,
              "helper_sha256": _helper_identity(),
              "status": "running", "runs": [], "all_cases_passed": False,
              "candidate_capacity_status": "not_measured"}
    _write(output, report)
    for case in plan["cases"]:
        directory = artifacts / case["name"]
        directory.mkdir()
        task = {**plan, "case": case, "installed": identity, "directory": str(directory),
                "helper_sha256": report["helper_sha256"]}
        task_path, result_path = directory / "task.json", directory / "result.json"
        _write(task_path, task)
        environment = child_environment(ROOT, identity=identity)
        if plan["project_root"]:
            environment["PYTHONPATH"] += os.pathsep + plan["project_root"]
        for key in ("TORCHDYNAMO_REPRO_AFTER", "TORCH_TRACE", "TRITON_OVERRIDE_DIR"):
            environment.pop(key, None)
        environment.update(TORCHDYNAMO_REPRO_LEVEL="0", TORCH_COMPILE_DEBUG_SAVE_REAL="0",
                           PYTHONDONTWRITEBYTECODE="1")
        if plan["settings"]["runtime"].get("deterministic", False):
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command = [sys.executable, "-B", str(ROOT / "examples/vae/benchmark_vae.py"),
                   "--_diagnostic-task", str(task_path), "--output", str(result_path)]
        print(f"VAE diagnostic: {case['name']} ({plan['device']}, untimed)", flush=True)
        with (directory / "worker.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            result = json.loads(result_path.read_text())
            if result.get("case") != case or result.get("settings_sha256") != plan["settings_sha256"]:
                raise ValueError("Worker diagnostic identity differs from the frozen task")
            if result.get("qualification_eligible") is not False or result.get("timing_eligible") is not False:
                raise ValueError("Diagnostic worker must explicitly exclude qualification and timing")
            if result.get("status") not in ("passed", "failed", "blocked"):
                raise ValueError("Diagnostic worker status is invalid")
            if result.get("status") == "passed" and result.get("installed") != identity:
                raise ValueError("Passing diagnostic worker has a different installed wheel identity")
            if completed.returncode != 0 and result.get("status") == "passed":
                raise ValueError("Worker exited unsuccessfully after reporting a pass")
        except (OSError, ValueError) as error:
            result = {"case": case, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        if result_path.is_file():
            try:
                result = summary_for_parent(result, result_path, output.parent)
            except Exception as reporting_error:
                result = {"case": case, "status": "failed",
                          "error": result.get("error") or "Diagnostic result summary failed",
                          "diagnostic_reporting_error": f"{type(reporting_error).__name__}: {reporting_error}"}
        result.update(returncode=completed.returncode, result_path=str(result_path),
                      worker_log=str(directory / "worker.log"))
        report["runs"].append(result)
        comparison = result.get("comparison") or {}
        detail = (f"; first mismatch={comparison.get('name')} relative_l2={comparison.get('relative_l2')}"
                  if comparison else "")
        if result.get("error") and (not comparison or not str(result["error"]).startswith("NumericalComparisonError:")):
            detail += "; " + str(result["error"]).splitlines()[0][:600]
        elif not comparison and result.get("repeatability"):
            probe = result["repeatability"]
            controls = probe.get("controls", ())
            detail = "; " + "; ".join(
                f"{control.get('name')}: repeat_stable={control.get('repeat_stable')} "
                f"deterministic_blocked={control.get('deterministic_blocked')}" for control in controls)
            if not controls:
                detail = f"; repeat_stable={probe.get('repeat_stable')} deterministic_blocked={probe.get('deterministic_blocked')}"
        label = ("completed (evidence collected)" if result["status"] == "passed"
                 and result.get("repeatability") else result["status"])
        print(f"  {label}{detail}", flush=True)
        _write(output, report)
    report["all_cases_passed"] = bool(report["runs"]) and all(row["status"] == "passed" for row in report["runs"])
    if _helper_identity() != report["helper_sha256"]:
        report["helper_source_stable"] = False
        report["all_cases_passed"] = False
    else:
        report["helper_source_stable"] = True
    report["status"] = "passed" if report["all_cases_passed"] else "failed"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "qualification_eligible": False}), flush=True)
    return 0 if report["all_cases_passed"] else 1


def _route_settings(raw, descriptor):
    selected = deepcopy(raw)
    enabled, algorithm = PROFILES[descriptor["profile"]]
    runtime = selected["runtime"]
    runtime["activation_compression"]["enabled"] = enabled
    runtime["activation_memory"].update(multiply_backward=algorithm, recompute_modules=[])
    compiler = runtime.setdefault("compile", {})
    compiler.update(enabled=False, backend="deepopt", fullgraph=True, dynamic=False, mode=None)
    compiler.setdefault("options", {}).update(use_native_autodiff=True, require_full_aot=True,
        fuse_packed_multiply=algorithm == "packed_streaming", collect_saved_state_metadata=True)
    return selected


def _software(torch, device):
    from importlib.metadata import PackageNotFoundError, version

    result = {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version,
              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
              "numerics_environment": {"CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
              "allocator_environment": {name: os.environ.get(name) for name in
                    ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "TRITON_DEFAULT_FP_FUSION")}}
    result["packages"] = {}
    for name in ("triton", "torchvision", "numpy"):
        try:
            result["packages"][name] = version(name)
        except PackageNotFoundError:
            result["packages"][name] = None
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": prop.name, "uuid": str(prop.uuid),
                         "total_memory": prop.total_memory, "capability": [prop.major, prop.minor]}
    return result


def _strict_control_applied(software):
    return (software.get("deterministic_algorithms") is True
            and software.get("deterministic_warn_only") is False
            and software.get("cudnn_deterministic") is True
            and software.get("cudnn_benchmark") is False)


def _images(task, benchmark, torch):
    raw = task["settings"]
    size, batch = raw["application"]["data"]["image_size"], raw["training"]["batch_size"]
    if task["diagnostic_overrides"]["input"] == "seeded_synthetic":
        return torch.rand((batch, 3, size, size), generator=torch.Generator().manual_seed(raw.get("seed", 42))) * 2 - 1
    from types import SimpleNamespace

    context = SimpleNamespace(world_size=1, global_rank=0)
    return next(iter(benchmark.build_loader(raw, SimpleNamespace(batches=1), context)))


def _run_trajectory(task, result, device, torch, benchmark):
    import deepopt
    from deepopt.config import CompileConfig
    from deepopt.engine.execution.compilation import compile_model
    from deepopt.opt.policy import prepare_fp8_model
    from examples.vae.qualification import paired_trajectory
    from examples.vae.diagnostic_localization import BoundaryTrace
    from examples.vae.mbconv_vae import MBConvVAE_V2

    raw, case = task["settings"], task["case"]
    result["failure_context"] = {"phase": "schedule_setup", "route": None, "update": None}
    # Recorded benchmark settings retain both the Inductor mode and native
    # options until a route is selected. Only a normalized route is a valid
    # DeepOpt config; preserve the recorded envelope for provenance.
    reference_config = deepopt.config_from_dict(_route_settings(raw, case["reference"]))
    schedules = deepopt.ScheduleController.from_config(reference_config.schedules)
    schedules.prepare_update(1, epoch=1)
    result["beta"] = schedules["beta"]
    result["failure_context"] = {"phase": "model_setup", "route": None, "update": None}
    factory = benchmark.import_factory(task["model_factory"]) if task["model_factory"] else MBConvVAE_V2
    original = factory(**raw["application"]["model"])
    parameters = sum(value.numel() for value in original.parameters())
    if (not task["model_factory"] and raw["application"]["model"] ==
            dict(base_channels=32, max_channels=512, latent_channels=4, blocks_per_stage=2) and parameters != 32317999):
        raise AssertionError("Original VAE parameter count changed")
    result["model"] = {"parameters": parameters, "options": raw["application"]["model"],
                       "initial_state_sha256": benchmark.fingerprint(original.state_dict().values())}
    images = _images(task, benchmark, torch)
    result["input"] = {"shape": list(images.shape), "sha256": benchmark.fingerprint([images]),
                       "scope": task["diagnostic_overrides"]["input"]}
    models, optimizers, reports = [], [], []
    result["routes"] = reports
    observe = case.get("observe_boundaries", True)
    result["boundary_observation_enabled"] = observe
    traces = []
    if observe:
        traces.append(BoundaryTrace(modules=task["trace_modules"], max_bytes=task["trace_max_bytes"],
                      artifact_dir=Path(task["directory"]) / "reference-code"))
        traces.append(BoundaryTrace(reference=traces[0], modules=task["trace_modules"],
            max_bytes=task["trace_max_bytes"], artifact_dir=Path(task["directory"]) / "candidate-code"))
    try:
        for slot in ("reference", "candidate"):
            descriptor = case[slot]
            selected = _route_settings(raw, descriptor)
            model = prepare_fp8_model(deepcopy(original), False,
                activation_compression=selected["runtime"]["activation_compression"], config=selected)[0].train()
            optimizer = benchmark.make_optimizer(model, selected["training"], task["optimizer_factory"])
            models.append(model)
            optimizers.append(optimizer)
            reports.append({"slot": slot, **descriptor,
                "activation_memory": selected["runtime"]["activation_memory"],
                "compression": selected["runtime"]["activation_compression"]})
            if descriptor["execution"] == "native":
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True, dynamic=False,
                    options=selected["runtime"]["compile"]["options"]), config=selected)
        optimizers[1].load_state_dict(deepcopy(optimizers[0].state_dict()))
        result["provenance"] = benchmark.source_provenance(deepopt, factory, optimizers[0])
        result["software_before_trajectory"] = _software(torch, device)
        if raw["runtime"].get("deterministic", False) and not _strict_control_applied(result["software_before_trajectory"]):
            raise RuntimeError("Requested deterministic control was overridden during model/optimizer setup")
        def progress(event):
            result["failure_context"] = dict(event)
            print(f"{case['name']}: {event}", flush=True)

        def route_context(label, model):
            index = 0 if label == "eager" else 1
            descriptor = case["reference" if index == 0 else "candidate"]
            return traces[index].native() if descriptor["execution"] == "native" else traces[index].eager(model)

        def evidence(value):
            result["qualification_evidence"] = value

        compression = _route_settings(raw, case["reference"])["runtime"]["activation_compression"]
        result["trajectory"] = paired_trajectory(models[0], models[1], optimizers[0], optimizers[1], images,
            [result["beta"]], compression, device=device, bf16_policy=raw["runtime"]["bf16_policy"],
            max_bytes=task["max_bytes"], grad_clip_norm=raw["training"].get("grad_clip_norm"),
            diagnostic=True, evidence_callback=evidence,
            route_context=route_context if observe else None, progress=progress)
        for model, descriptor in zip(models, (case["reference"], case["candidate"])):
            if descriptor["execution"] == "native":
                benchmark.native_compilation_evidence(model)
    finally:
        try:
            result["software_after_trajectory"] = _software(torch, device)
            if raw["runtime"].get("deterministic", False):
                result["deterministic_control_preserved"] = _strict_control_applied(result["software_after_trajectory"])
                if not result["deterministic_control_preserved"]:
                    result.setdefault("diagnostic_collection_errors", []).append({"field": "deterministic_control_preserved"})
        except Exception as collection_error:
            result.setdefault("diagnostic_collection_errors", []).append(
                {"field": "software_after_trajectory", "error_type": type(collection_error).__name__})
        result["compiler_reports"] = [{**benchmark.failure_candidate_diagnostics(model), "slot": slot,
                                       "compiler_report_scope": "diagnostic_route"}
                                      for slot, model in zip(("reference", "candidate"), models)]
        result["boundary_traces"] = []
        result["localization"] = {"scope": "matched_selected_boundaries_only",
                                  "enabled": observe, "global_first_divergence_verified": False}
        for trace in traces:
            try:
                result["boundary_traces"].append(trace.report())
            except Exception as collection_error:
                result.setdefault("diagnostic_collection_errors", []).append(
                    {"field": "boundary_trace", "error_type": type(collection_error).__name__})
        if len(result["boundary_traces"]) == 2:
            compared = result["boundary_traces"][1]
            result["localization"] = {
                "scope": "matched_selected_boundaries_only", "enabled": True,
                "global_first_divergence_verified": False,
                "complete": compared["complete"],
                "coverage_complete": compared.get("coverage_complete", compared["complete"]),
                "context_succeeded": compared.get("context_succeeded", not compared.get("errors")),
                "first_tolerance_failure": compared["first_differing_boundary"],
                "first_bitwise_difference": next((row for row in compared["boundaries"]
                    if row.get("bitwise_equal") is False), None)}
        if models and optimizers and result.get("provenance"):
            result["provenance_after"] = benchmark.source_provenance(deepopt, factory, optimizers[0])
            result["source_stable"] = result["provenance_after"] == result["provenance"]
            if not result["source_stable"]:
                result.setdefault("diagnostic_collection_errors", []).append({"field": "source_stable", "value": False})


def diagnostic_worker(args):
    from examples.vae import benchmark_vae as benchmark

    task = json.loads(args._diagnostic_task.read_text())
    result = {"schema_version": 1, "case": task["case"], "settings_sha256": task["settings_sha256"],
              "status": "failed", "scope": "untimed_numerical_diagnostics", "qualification_eligible": False,
              "cuda_qualified": False, "timing_eligible": False, "tensor_files_written": 0,
              "candidate_capacity_status": "not_measured"}
    device = None
    try:
        from benchmarks.compiler.installed_runtime import installed_identity

        identity = installed_identity(ROOT)
        result["installed"] = identity
        if identity != task["installed"]:
            raise RuntimeError("Installed wheel identity changed between parent and worker")
        if _helper_identity() != task["helper_sha256"]:
            raise RuntimeError("Diagnostic helper sources changed between parent and worker")
        import torch
        from benchmarks.compiler.save_inductor_references import disable_dynamo_reproduction
        from deepopt.runtime import seed_everything

        result["compiler_environment"] = disable_dynamo_reproduction()
        device = torch.device(task["device"])
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Diagnostic device must be CPU or CUDA")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                result["status"] = "blocked"
                raise RuntimeError("CUDA unavailable; target diagnostics remain unqualified")
            torch.cuda.set_device(device)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA BF16 is required")
        else:
            torch.set_num_threads(1)
        seed_everything(task["settings"].get("seed", 42), task["settings"]["runtime"].get("deterministic", False))
        torch.utils.deterministic.fill_uninitialized_memory = task["settings"]["runtime"].get("fill_uninitialized_memory", False)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = not task["settings"]["runtime"].get("deterministic", False)
        result["software"] = _software(torch, device)
        if task["case"]["kind"] == "repeatability":
            from examples.vae.diagnostic_repeatability import run_repeatability_diagnostics

            result["repeatability"] = run_repeatability_diagnostics(device=str(device),
                seed=task["settings"].get("seed", 42),
                original_shapes=task["diagnostic_overrides"]["input"] != "seeded_synthetic")
            if not result["repeatability"].get("completed"):
                raise RuntimeError("Backward repeatability probes did not complete; see per-probe evidence")
        elif task["case"]["kind"] == "operators":
            from examples.vae.diagnostic_operators import run_packed_diagnostics
            from examples.vae.diagnostic_localization import nearest_replay_diagnostics, chain_replay_diagnostics

            result["operators"] = {}
            probes = (("packed", run_packed_diagnostics,
                       dict(device=str(device), group_size=1024, seed=task["settings"].get("seed", 42),
                            scratch_bytes=task["settings"]["runtime"]["activation_memory"]["reduction_scratch_bytes"],
                            full_shape=task["large_operators"])),
                      ("nearest", nearest_replay_diagnostics,
                       dict(device=str(device), large=task["large_operators"],
                            max_bytes=task["trace_max_bytes"],
                            artifact_dir=Path(task["directory"]) / "nearest-code")),
                      ("chain", chain_replay_diagnostics, dict(device=str(device),
                            max_bytes=task["trace_max_bytes"],
                            artifact_dir=Path(task["directory"]) / "chain-code")))
            for name, probe, options in probes:
                try:
                    result["operators"][name] = probe(**options)
                except Exception as probe_error:
                    result["operators"][name] = {"passed": False, **benchmark.failure_error_details(probe_error)}
            if not all(item.get("passed") is True for item in result["operators"].values()):
                raise AssertionError("One or more isolated operator replays failed; see per-case evidence")
        else:
            _run_trajectory(task, result, device, torch, benchmark)
            trajectory = result.get("trajectory", {})
            if trajectory.get("numerical_comparisons_passed") is not True:
                raise AssertionError("Trajectory did not establish complete numerical comparisons")
            if trajectory.get("qualification_eligible") is not False or trajectory.get("diagnostic_only") is not True:
                raise RuntimeError("Diagnostic trajectory must explicitly exclude qualification")
        if result.get("diagnostic_collection_errors"):
            raise RuntimeError("Diagnostic evidence collection is incomplete; see collection errors")
        if installed_identity(ROOT, full=True) != identity:
            raise RuntimeError("Installed wheel changed during diagnostic worker")
        result["status"] = "passed"
    except Exception as error:
        result.update(benchmark.failure_error_details(error))
        if (task["settings"]["runtime"].get("deterministic", False)
                and "does not have a deterministic implementation" in str(error)):
            result["status"] = "blocked"
            result["determinism_blocker"] = {"error_type": type(error).__name__, "message": str(error)}
    finally:
        try:
            result["cuda_failure_diagnostics"] = benchmark.cuda_failure_diagnostics(device)
        except Exception as diagnostic_error:
            result["cuda_diagnostics_collection_error"] = type(diagnostic_error).__name__
        try:
            from examples.vae.diagnostic_reporting import externalize_compiler_reports

            externalize_compiler_reports(result, Path(task["directory"]))
        except Exception as reporting_error:
            result["diagnostic_reporting_error"] = f"{type(reporting_error).__name__}: {reporting_error}"
            if result["status"] == "passed":
                result["status"] = "failed"
        _write(args.output, result)
    return 0 if result["status"] == "passed" else 1
