"""Execute one installed eager or native compiled VAE, without a numerical oracle.

The default is the original physical [128, 3, 224, 224] CUDA workload. Saved
settings supply its precision, compression, memory policy and custom optimizer.
Completion establishes execution and finite-state evidence, not correctness.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
# Arm manifests pin the exact wheel for qualification. Unpinned runs default to
# the recorded VAE release; a host may select another installed candidate
# explicitly, which the installed-identity check still verifies.
EXPECTED_VERSION = os.environ.get(
    "DEEPOPT_VAE_EXPECTED_VERSION", "0.1.0+nativeautodiff.20261119.vaememory.4")
ORIGINAL_MODEL = dict(base_channels=32, max_channels=512, latent_channels=4, blocks_per_stage=2)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--from-result", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=128, help="Physical batch; no accumulation or microbatching")
    result.add_argument("--warmup", type=int, default=1, help="Complete warmup optimizer updates")
    result.add_argument("--updates", type=int, default=2, help="Complete updates after warmup")
    result.add_argument("--execution-mode", choices=("native", "eager"), default="native")
    result.add_argument("--benchmark", action="store_true", help="Measure a separate steady window without validation scans")
    result.add_argument("--profile-after-measurement", action="store_true",
                        help="Profile extra updates after freezing a successful benchmark")
    result.add_argument("--profile-updates", type=int, default=1, help="Extra untimed profile updates (1-3)")
    result.add_argument("--profile-memory", action="store_true", help="Include bounded allocator metadata and memory events")
    result.add_argument("--profile-record-shapes", action="store_true", help="Record shapes only in the extra profile window")
    result.add_argument("--diagnostic-phase", choices=("none", "cold", "steady"), default="none",
                        help="Separate cold first-update or steady diagnostic worker; never a measured sample")
    result.add_argument("--storage-ledger", action="store_true", help="Record bounded weak physical storage in diagnostic updates")
    result.add_argument("--inventory-output", type=Path, help="Export compact compiled model-scope inventory; requires available metadata")
    result.add_argument("--host-headroom-mib", type=int, default=8192,
                        help="Reserve host/job headroom outside a version2 physical pinned capacity")
    result.add_argument("--memory-audit", action=argparse.BooleanOptionalAction, default=True,
                        help="Collect staged allocator metadata outside measured updates (default: enabled)")
    result.add_argument("--native-options-json", type=Path, help="Recorded, validated native compiler overrides")
    result.add_argument("--packed-offload-json", type=Path, help="Pinned packed-save transport policy; numerical recipe is unchanged")
    result.add_argument("--arm-manifest", type=Path, help="Pin an experiment arm's installed version and package hash")
    result.add_argument("--expected-version", help="Expected version; requires a matching pinned arm manifest")
    result.add_argument("--project-root", type=Path, help="Project containing source/training/geo_optimizer.py")
    result.add_argument("--train-dir", type=Path)
    result.add_argument("--optimizer", choices=("custom", "sgd"), help="Explicit diagnostic optimizer override")
    result.add_argument("--synthetic", action="store_true", help="Explicit seeded synthetic input diagnostic")
    result.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="Explicit override; otherwise retain the recorded algorithm policy")
    result.add_argument("--output", type=Path, default=Path("work/vae_native_execution.json"))
    result.add_argument("--dry-run", action="store_true", help="Print the plan without importing Torch")
    result.add_argument("--configuration-preflight", action="store_true",
                        help="Validate the pinned native worker configuration without allocating a model or input")
    return result


def _json_input(path):
    """Read bounded data, rejecting duplicate keys and non-JSON numbers."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    path = path.resolve()
    with path.open("rb") as stream:
        payload = stream.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("Experiment JSON input exceeds 1 MiB")
    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    if type(value) is not dict:
        raise ValueError("Experiment JSON input must be an object")
    return value, {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _experiment_inputs(args, execution):
    options, override = {}, None
    path = getattr(args, "native_options_json", None)
    if path is not None:
        if execution != "native":
            raise ValueError("--native-options-json is valid only for native execution")
        options, override = _json_input(path)
        if any(name in options for name in ("bf16_policy", "activation_memory", "packed_activation_offload")):
            raise ValueError("Native compiler overrides cannot replace the recorded storage or activation-memory policy")
        for name in ("enable_aot_autodiff", "use_native_autodiff", "require_full_aot", "capture_deepopt"):
            if name in options and options[name] is not True:
                raise ValueError(f"Native compiler override {name} must remain True")
        if options.get("mutation_backend", "native") not in ("native", "error"):
            raise ValueError("Native compiler overrides cannot select an eager/Inductor mutation owner")
        override["options"] = options
    version = getattr(args, "expected_version", None)
    manifest_path = getattr(args, "arm_manifest", None)
    if manifest_path is None:
        if version is not None:
            raise ValueError("--expected-version requires a pinned --arm-manifest")
        return options, override, None, EXPECTED_VERSION
    manifest, source = _json_input(manifest_path)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Arm manifest schema_version must be 1")
    if type(manifest.get("arm")) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", manifest["arm"]) is None:
        raise ValueError("Arm manifest requires a scalar arm label")
    installed = manifest.get("expected_installed")
    if not isinstance(installed, dict) or not all(name in installed for name in ("version", "source_sha256", "files_hashed")):
        raise ValueError("Arm manifest requires expected_installed version, source_sha256 and files_hashed")
    if type(installed["version"]) is not str or not installed["version"].strip():
        raise ValueError("Arm manifest installed version must be nonempty")
    if type(installed["files_hashed"]) is not int or installed["files_hashed"] < 1:
        raise ValueError("Arm manifest files_hashed must be a positive integer")
    for name, digest in (("source_sha256", installed["source_sha256"]),
                         ("wheel_sha256", manifest.get("wheel_sha256")),
                         ("native_options_sha256", manifest.get("native_options_sha256")),
                         ("resolved_native_options_sha256", manifest.get("resolved_native_options_sha256")),
                         ("packed_offload_sha256", manifest.get("packed_offload_sha256"))):
        if (name == "source_sha256" or digest is not None) and (
                type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None):
            raise ValueError(f"Arm manifest {name} must be a lowercase SHA256")
    if version is not None and version != installed["version"]:
        raise ValueError("--expected-version differs from the arm manifest")
    expected_options = manifest.get("native_options_sha256")
    if expected_options != (override["sha256"] if override is not None else None):
        raise ValueError("Native options bytes do not match the arm manifest pin")
    return options, override, {"source": source, "manifest": manifest}, installed["version"]


def _verify_experiment_inputs(plan):
    sources = [plan.get("native_options_override"), plan.get("packed_offload_override"), (plan.get("experiment_arm") or {}).get("source")]
    for source in sources:
        if source is not None:
            data = Path(source["path"]).read_bytes()
            if len(data) != source["bytes"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise RuntimeError("Experiment input changed after planning: " + source["path"])


def resolved_native_options_sha256(settings):
    options = deepcopy(settings["runtime"]["compile"].get("options", {}))
    options.pop("packed_activation_offload", None)
    return hashlib.sha256(json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_plan(args):
    """Resolve the original settings envelope using standard-library code only."""
    from examples.vae import benchmark_vae as benchmark

    diagnostic_phase = getattr(args, "diagnostic_phase", "none")
    if getattr(args, "storage_ledger", False) and diagnostic_phase == "none" and not getattr(args, "profile_after_measurement", False):
        raise ValueError("--storage-ledger requires an explicit diagnostic or profile window")

    if args.batch_size < 1 or args.warmup < 0 or args.updates < 1:
        raise ValueError("batch-size/updates must be positive and warmup nonnegative")
    if type(getattr(args, "host_headroom_mib", 8192)) is not int or getattr(args, "host_headroom_mib", 8192) < 0:
        raise ValueError("host-headroom-mib must be nonnegative")
    benchmarking = getattr(args, "benchmark", False)
    execution = getattr(args, "execution_mode", "native")
    if diagnostic_phase != "none" and execution != "native":
        raise ValueError("Native storage diagnostics require native execution")
    if getattr(args, "configuration_preflight", False) and execution != "native":
        raise ValueError("--configuration-preflight requires native execution")
    profiling = getattr(args, "profile_after_measurement", False) or diagnostic_phase == "steady"
    profile_updates = getattr(args, "profile_updates", 1)
    profile_memory = getattr(args, "profile_memory", False)
    profile_shapes = getattr(args, "profile_record_shapes", False)
    if not 1 <= profile_updates <= 3:
        raise ValueError("--profile-updates must be between 1 and 3")
    if profiling and not benchmarking:
        raise ValueError("--profile-after-measurement requires --benchmark")
    if not profiling and diagnostic_phase != "cold" and (profile_memory or profile_shapes or profile_updates != 1):
        raise ValueError("Profile options require --profile-after-measurement")
    native_options, override, arm, expected_version = _experiment_inputs(args, execution)
    packed_source, packed_policy = None, None
    packed_path = getattr(args, "packed_offload_json", None)
    if packed_path is not None:
        if execution != "native":
            raise ValueError("--packed-offload-json requires native compiled execution")
        from examples.vae.benchmark_packed_offload import normalize_policy

        value, packed_source = _json_input(packed_path)
        packed_policy = normalize_policy(value)
        packed_source["policy"] = packed_policy
        if arm is not None and arm["manifest"].get("packed_offload_sha256") != packed_source["sha256"]:
            raise ValueError("Packed offload bytes do not match the arm manifest pin")
    elif arm is not None and arm["manifest"].get("packed_offload_sha256") is not None:
        raise ValueError("The arm manifest requires its pinned packed offload policy")
    if benchmarking and args.warmup < 1:
        raise ValueError("--benchmark requires at least one warmup update")
    if args.device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", args.device) is None:
        raise ValueError("Choose an explicit CPU diagnostic or CUDA device; no substitution is performed")
    payload = args.from_result.read_bytes()
    previous = json.loads(payload)
    target = deepcopy(previous["settings"])
    raw = deepcopy(target)
    if raw["training"].get("accumulation_steps", 1) != 1:
        raise ValueError("Native execution requires accumulation_steps=1")
    compression = raw["runtime"].setdefault("activation_compression", {})
    if compression.setdefault("group_size", 1024) != 1024:
        raise ValueError("The recorded compression group_size must be 1024")
    compressed = compression.get("enabled", False)
    if type(compressed) is not bool:
        raise ValueError("Recorded activation compression enabled must be Boolean")
    mode = ("deepopt-bf16-native" if execution == "native" else "deepopt-bf16") + ("-compressed" if compressed else "")
    raw = benchmark.mode_settings(raw, mode)
    raw["runtime"]["compile"]["options"].update(native_options)
    if packed_policy is not None:
        raw["runtime"]["packed_activation_offload"] = dict(packed_policy)
        raw["runtime"]["compile"]["options"]["packed_activation_offload"] = dict(packed_policy)
    if execution == "eager":
        raw["runtime"]["compile"].update(enabled=False, mode=None, options={})
    raw["training"]["batch_size"] = args.batch_size
    raw["runtime"]["device"] = args.device
    if raw["runtime"].get("precision", "bf16") != "bf16":
        raise ValueError("This runner executes the recorded BF16 VAE recipe")
    raw["runtime"]["precision"] = "bf16"
    if args.optimizer is not None:
        raw["training"]["optimizer"] = args.optimizer
    if args.train_dir is not None:
        raw["application"]["data"]["train_dir"] = str(args.train_dir)
    if args.deterministic is not None:
        raw["runtime"]["deterministic"] = args.deterministic
    if benchmarking and (raw["runtime"].get("checks", False) is not False
            or raw["runtime"].get("diagnostics") is not None or raw["runtime"].get("mode_schedule") is not None
            or raw["runtime"].get("execution_mode") not in (None, "max_speed")):
        raise ValueError("Benchmark timing requires runtime checks/diagnostics/mode schedules disabled in the recorded recipe")
    raw.setdefault("loader", {}).update(num_workers=0, persistent_workers=False, drop_last=True)
    size = raw["application"]["data"]["image_size"]
    if size < 16:
        raise ValueError("VAE image_size must be at least 16")
    project = args.project_root
    if project is None:
        project = next((parent for parent in ROOT.parents
                        if (parent / "source/training/geo_optimizer.py").is_file()), None)
    if project is not None and not project.is_dir():
        raise ValueError("project-root must be an existing directory")
    original_geometry = args.batch_size == 128 and size == 224 and raw["application"]["model"] == ORIGINAL_MODEL
    reasons = []
    if args.device == "cpu":
        reasons.append("explicit_CPU_execution")
        # TrainingRuntime's supported CPU path is FP32. This is an explicit
        # local diagnostic, never a substitute for the recorded CUDA recipe.
        raw["runtime"].update(precision="fp32", bf16_policy="torch_amp")
        reasons.append("CPU_runtime_fp32_torch_amp")
    if args.synthetic:
        reasons.append("explicit_synthetic_input")
    if args.optimizer is not None:
        reasons.append("explicit_optimizer_override")
    if args.deterministic is not None:
        reasons.append("explicit_deterministic_algorithm_override")
    if not original_geometry:
        reasons.append("non_target_geometry")
    factory = previous.get("model_factory")
    if factory in ("mbconv_vae:MBConvVAE_V2", "examples.vae.mbconv_vae:MBConvVAE_V2"):
        factory = None
    resolved_options_hash = resolved_native_options_sha256(raw)
    pinned_resolved = (arm or {}).get("manifest", {}).get("resolved_native_options_sha256")
    if pinned_resolved is not None and pinned_resolved != resolved_options_hash:
        raise ValueError("Resolved native compiler options differ from the arm manifest pin")
    return {"schema_version": 1, "scope": "native_compiled_single_model_execution" if execution == "native" else "eager_single_model_execution",
        "qualified": False, "qualification_eligible": False, "correctness_assessed": False,
        "correctness_not_assessed": True, "timing_eligible": False,
        "origin": {"path": str(args.from_result.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                   "workload_sha256": previous.get("workload_sha256")},
        "target_settings": target, "settings": raw, "mode": mode, "device": args.device,
        "execution_mode": execution, "benchmark": benchmarking,
        "autodiff_owner": "deepopt_native_autodiff" if execution == "native" else "pytorch_autograd_with_DeepOpt_custom_gradients",
        "requested_input_shape": [args.batch_size, 3, size, size], "original_geometry": original_geometry,
        "input_mode": "seeded_synthetic" if args.synthetic else "first_dataset_batch_repeated",
        "diagnostic_overrides": reasons, "warmup": args.warmup, "updates": args.updates,
        "optimizer_factory": previous.get("optimizer_factory", benchmark.CUSTOM_OPTIMIZER),
        "model_factory": factory, "project_root": None if project is None else str(project.resolve()),
        "gradient_mode": "trainable_parameters", "expected_installed_version": expected_version,
        "native_options_override": override, "experiment_arm": arm,
        "resolved_native_options_sha256": resolved_options_hash,
        "host_capacity_policy": {"reserved_headroom_bytes": args.host_headroom_mib * 2**20,
            "headroom_mib": args.host_headroom_mib, "read_only_preallocation_check": True,
            "requested_capacity_clamped": False},
        "packed_offload_override": packed_source,
        "profiling": {"enabled": profiling or diagnostic_phase == "cold", "updates": profile_updates,
            "memory": profile_memory or diagnostic_phase == "cold", "record_shapes": profile_shapes,
            "phase": "cold_first_update" if diagnostic_phase == "cold" else "after_measurement_and_postchecks"},
        "memory_audit_options": {"enabled": getattr(args, "memory_audit", True),
            "scope": "staged_allocator_metadata_outside_measurement",
            "first_update_profile_implemented": True, "typed_storage_owner_ledger_implemented": True},
        "numerical_comparisons_performed": False, "tensor_files_written": 0}


def _metadata(value, torch):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (torch.dtype, torch.device, torch.memory_format)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_metadata(item, torch) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _metadata(item, torch) for key, item in value.items()}
    raise TypeError(f"Compiler metadata contains unsupported {type(value).__name__}")


def _save_bytes(directory, name, payload):
    path = directory / name
    with path.open("xb") as stream:
        stream.write(payload)
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class _ArtifactRecorder:
    """Collect code/IR only, then call the ordinary native artifact function."""

    metadata_only = True
    supports_packed_transport = True

    def __init__(self, torch, context):
        self.torch, self.context = torch, context
        self.seen, self.payloads, self.calls = set(), {}, {"forward": 0, "backward": 0}

    def record_artifact(self, artifact, *, phase):
        self.calls[phase] += 1
        identity = (id(artifact), phase)
        if identity not in self.seen:
            self.seen.add(identity)
            source = artifact.source
            digest = hashlib.sha256(source.encode()).hexdigest()
            prefix = f"native_{phase}_{digest[:16]}"
            metadata = _metadata({"phase": phase, "final_schedule": artifact.final_schedule,
                "logical_lifetimes": artifact.logical_lifetimes,
                "boundary_metadata": getattr(artifact, "boundary_metadata", ()),
                **{name: getattr(artifact, name, None) for name in ("native_region_index", "native_scope_identity",
                    "native_forward_signature", "native_backward_signature", "native_recipe_identity_sha256")},
                "tensor_files_saved": 0}, self.torch)
            serialized = json.dumps(metadata, indent=2, allow_nan=False).encode()
            prefix += "_" + hashlib.sha256(serialized).hexdigest()[:12]
            self.payloads[prefix + ".py"] = source.encode()
            self.payloads[prefix + ".json"] = serialized
            for family in ("triton_artifacts", "tilelang_artifacts"):
                for index, kernel in enumerate(getattr(artifact, family, ())):
                    if isinstance(getattr(kernel, "source", None), str):
                        self.payloads[f"{prefix}_{family}_{index}.py"] = kernel.source.encode()
        self.context["native_artifact_phase"] = phase
        self.context["native_artifact_source_sha256"] = hashlib.sha256(artifact.source.encode()).hexdigest()

    def run_artifact(self, artifact, *args, phase, **runtime_kwargs):
        self.record_artifact(artifact, phase=phase)
        # No observed wrapper, boundary callbacks, tensor copies or retention.
        return artifact.fn(*args, **runtime_kwargs)

    def save(self, directory):
        files = [_save_bytes(directory, name, data) for name, data in self.payloads.items()]
        return {"files": files, "execution_calls": dict(self.calls), "tensor_files_saved": 0,
                "ordinary_artifact_fn_used": True, "numerical_boundary_observation": False}


def _finite(torch, values):
    values = tuple(value for value in values if value is not None)
    return bool(values) and bool(torch.stack([torch.isfinite(value).all() for value in values]).all())


def _memory(torch, device):
    if device.type != "cuda":
        return {"scope": "CPU_no_CUDA_allocator_evidence", "peak_allocated_bytes": None,
                "peak_reserved_bytes": None}
    return {"scope": "CUDA_allocator", "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def _helper_identity():
    names = ("examples/vae/native_execution.py", "examples/vae/execution_profile.py", "examples/vae/execution_storage.py", "examples/vae/execution_inventory.py", "examples/vae/execution_memory.py", "examples/vae/benchmark_vae.py",
             "examples/vae/mbconv_vae.py", "benchmarks/compiler/installed_runtime.py")
    if (ROOT / "examples/vae/benchmark_packed_offload.py").is_file():
        names += ("examples/vae/benchmark_packed_offload.py",)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def _packed_offload_snapshot():
    try:
        from deepopt.compiler.runtime.manager import packed_offload_runtime_snapshot
    except ImportError:
        return {"schema_version": 1, "available": False, "reason": "installed_runtime_API_unavailable"}
    value = json.loads(json.dumps(packed_offload_runtime_snapshot(), allow_nan=False))
    return {"schema_version": 1, "available": True, "snapshot": value}


def _drain_packed_offload():
    from deepopt.compiler.runtime.manager import drain_packed_offload_runtime

    return drain_packed_offload_runtime()


def _close_packed_offload():
    from deepopt.compiler.runtime.manager import close_packed_offload_runtime

    return close_packed_offload_runtime()


def _freeze_packed_offload_profile():
    from deepopt.compiler.runtime.manager import freeze_packed_offload_profiles

    value = json.loads(json.dumps(freeze_packed_offload_profiles(), allow_nan=False))
    if value.get("frozen") is not True:
        raise RuntimeError("Packed offload profile was not frozen before measurement")
    return {"schema_version": 1, "available": True, "snapshot": value,
        "phase": "after_warmup_before_measurement"}


def _rng_fingerprint(torch, device, benchmark):
    states = [torch.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return benchmark.fingerprint(states)


def _state_fingerprint(value, torch, benchmark):
    """Hash initial optimizer state without retaining or writing tensor payloads."""
    def encode(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": benchmark.fingerprint([item])}
        if isinstance(item, dict):
            return {str(key): encode(part) for key, part in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(part) for part in item]
        return _metadata(item, torch)
    return benchmark.json_fingerprint(encode(value))


def _verify_eager_ownership(model, runtime):
    from deepopt.compiler import compilation_report
    from torch._dynamo.eval_frame import OptimizedModule

    if runtime.config.get("runtime", {}).get("compile", {}).get("enabled", False):
        raise RuntimeError("Eager execution inherited enabled compilation")
    if any(getattr(module, "_compiled_call_impl", None) is not None or isinstance(module, OptimizedModule)
           for module in model.modules()):
        raise RuntimeError("Eager execution contains a compiled model handle")
    report = compilation_report(model)
    if report is not None and report.graphs:
        raise RuntimeError("Eager execution contains captured compiler graphs")


def _profile_after_measurement(*, torch, device, benchmark, model, optimizer, runtime, images,
                               schedules, config, report, directory):
    """Freeze canonical evidence before optional updates can mutate any state."""
    frozen = {key: deepcopy(report.get(key)) for key in ("measurement", "memory", "memory_segments",
        "cuda_allocator_diagnostics", "packed_offload_runtime", "packed_offload_profile",
        "native_ownership_verified", "eager_ownership_verified",
        "compiler_report_artifact", "installed", "source_stable",
        "installed_stable", "helper_source_stable", "input_stable", "warmup_completed", "updates_completed")}
    frozen["compiler_report_summary"] = {key: report.get("compiler_report_summary", {}).get(key) for key in
        ("captured_graphs", "native_autodiff_operations", "native_autodiff_regions", "runtime_fallbacks",
         "legacy_backward_operations", "triton_regions", "tilelang_regions", "fused_deepopt_regions")}
    frozen["execution_completed"] = report["execution_completed"]
    report["measurement_frozen_artifact"] = _save_bytes(directory, "measurement-frozen.json",
        json.dumps(frozen, indent=2, allow_nan=False).encode())
    report["profile"] = {"requested": True, "complete": False, "status": "incomplete",
        "measurement_eligible": False, "tensor_files_written": 0}
    # Preserve a usable clean receipt even if the optional profiler or export
    # terminates the worker. The supervisor still records abnormal exit codes.
    benchmark.write_json(Path(report["output_path"]), report)
    from examples.vae.execution_profile import run_execution_profile

    context = {"stage": "profile_after_measurement", "phase": "setup", "update": None}
    native = report["execution_mode"] == "native"
    losses = []
    diagnostic_recorder = _ArtifactRecorder(torch, context) if native else None
    storage_trace = None
    if native and report.get("diagnostic", {}).get("storage_ledger"):
        from examples.vae.execution_storage import NativeStorageTrace

        storage_trace = NativeStorageTrace(torch, device, recorder=diagnostic_recorder,
            runtime_snapshot=_packed_offload_snapshot if report.get("packed_offload_override") is not None else None)
        storage_trace.observe_persistent(model, optimizer, images)

    def ownership():
        if native:
            evidence = benchmark.native_compilation_evidence(model)
            return {"native_ownership_verified": True, "captured_graphs": evidence["captured_graphs"],
                "runtime_fallbacks": evidence["runtime_fallbacks"],
                "native_autodiff_operations": evidence["native_autodiff_operations"],
                "native_autodiff_regions": evidence["native_autodiff_regions"]}
        _verify_eager_ownership(model, runtime)
        return {"eager_ownership_verified": True, "captured_graphs": 0}

    def update(number, progress):
        schedules.prepare_update(number, epoch=1)
        def diagnostic_progress(event):
            if storage_trace is not None and event["phase"] == "optimizer":
                storage_trace.observe_persistent()
            progress(event)
        loss = benchmark.train_step(model, optimizer, runtime, images, schedules["beta"],
            device_type=device.type, grad_clip_norm=config.training.grad_clip_norm, progress=diagnostic_progress)
        schedules.complete_update(number)
        losses.append(loss.detach())

    def postchecks():
        return {"finite_losses": _finite(torch, losses), "finite_parameters": _finite(torch, model.parameters())}

    options = report["profiling"]
    report["profile"] = run_execution_profile(torch=torch, device=device, directory=directory / "profile",
        update=update, first_update=report["warmup"] + report["updates"] + 1, updates=options["updates"],
        record_memory=options["memory"], record_shapes=options["record_shapes"], ownership=ownership,
        expected_ownership=({"native_ownership_verified": True,
            "captured_graphs": report["measurement"]["native_graph_count_after"],
            "runtime_fallbacks": report["compiler_report_summary"]["runtime_fallbacks"],
            "native_autodiff_operations": report["compiler_report_summary"]["native_autodiff_operations"],
            "native_autodiff_regions": report["compiler_report_summary"]["native_autodiff_regions"]}
            if native else {"eager_ownership_verified": True, "captured_graphs": 0}),
        allocator_diagnostics=lambda: benchmark.cuda_failure_diagnostics(device), postchecks=postchecks,
        context=context, storage_trace=storage_trace, diagnostic_phase="steady")
    losses.clear()
    if diagnostic_recorder is not None and storage_trace is not None:
        report["profile"]["native_artifacts"] = diagnostic_recorder.save(directory / "profile")
        report["profile"]["native_artifacts"]["ordinary_artifact_fn_used"] = False
        report["profile"]["native_artifacts"]["metadata_observed_wrapper_used"] = True
    if report.get("packed_offload_override") is not None:
        report["profile"]["packed_offload_runtime_after"] = _packed_offload_snapshot()
    if native:
        try:
            # Optional recompilations belong only to the diagnostic artifact.
            # Never replace the canonical report captured before profiling.
            evidence = benchmark.failure_candidate_diagnostics(model)
            data = evidence.get("compiler_report")
            if data is not None:
                report["profile"]["compiler_report_artifact"] = _save_bytes(directory / "profile",
                    "compiler-report.json", json.dumps(_metadata(data, torch), indent=2, allow_nan=False).encode())
        except Exception as error:
            report["profile"].setdefault("collection_errors", []).append(benchmark.failure_error_details(error))
            report["profile"].update(complete=False, status="incomplete")


def _benchmark_updates(*, torch, device, benchmark, model, optimizer, runtime, images, schedules,
                       grad_clip_norm, warmup, updates, report, context, memory_audit=None):
    """Submit only training/scheduling work while the synchronized timer is open."""
    measurement = report["measurement"]
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(updates)] if device.type == "cuda" else None
    if events is not None:
        for start, end in events:
            start.record()
            end.record()
        torch.cuda.synchronize(device)
    report["memory"]["startup"] = _memory(torch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    losses = []
    if memory_audit is not None:
        memory_audit.capture("measurement_start", update=warmup)
    context.update(stage="measurement", phase="measurement", update=None)
    context.pop("native_artifact_phase", None)
    context.pop("native_artifact_source_sha256", None)
    wall_start = time.perf_counter()
    for index in range(updates):
        number = warmup + index + 1
        context["update"] = number
        schedules.prepare_update(number, epoch=1)
        beta = schedules["beta"]
        row = {"number": number, "stage": "update", "beta": beta, "forward_completed": False,
               "backward_completed": False, "optimizer_completed": False,
               "finite_gradients": None, "gradient_validation": "not_scanned_during_measurement"}
        report["steps"].append(row)
        if events is not None:
            events[index][0].record()
        submitted = time.perf_counter()
        loss = benchmark.train_step(model, optimizer, runtime, images, beta,
            device_type=device.type, grad_clip_norm=grad_clip_norm)
        row["host_submission_seconds"] = time.perf_counter() - submitted
        if events is not None:
            events[index][1].record()
        schedules.complete_update(number)
        losses.append(loss.detach())
        row.update(forward_completed=True, backward_completed=True, optimizer_completed=True)
        report["updates_completed"] += 1
        measurement["samples"].append({"update": number, "measured_index": index + 1,
            "host_submission_seconds": row["host_submission_seconds"]})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - wall_start
    # This snapshot must precede loss/parameter scans and any hashing/copying.
    measurement["memory"] = _memory(torch, device)
    report["memory"]["steady"] = dict(measurement["memory"])
    context.update(phase="measurement_complete", update=None)
    if memory_audit is not None:
        memory_audit.capture("measurement_end", update=warmup + updates)
    measurement.update(wall_seconds=wall, wall_ms_per_update=wall * 1000 / updates,
        images_per_second=updates * images.shape[0] / wall,
        measured_updates=updates, warmup_updates=warmup)
    event_ms = [start.elapsed_time(end) for start, end in events] if events is not None else None
    for index, row in enumerate(measurement["samples"]):
        row["cuda_event_ms"] = event_ms[index] if event_ms is not None else None
    measurement["cuda_event_ms_per_update"] = sum(event_ms) / updates if event_ms is not None else None
    return losses


def _configuration_topology():
    """Read unsupported multi-process requests without initializing a group."""
    raw = os.environ.get("WORLD_SIZE")
    try:
        requested = 1 if raw is None else int(raw)
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be a positive integer for configuration preflight") from error
    if requested < 1:
        raise ValueError("WORLD_SIZE must be a positive integer for configuration preflight")
    if requested > 1:
        raise ValueError("Packed offload configuration preflight requires one process; WORLD_SIZE=" + str(requested))
    import torch

    distributed = torch.distributed
    initialized = distributed.is_available() and distributed.is_initialized()
    size = distributed.get_world_size() if initialized else 1
    if size > 1:
        raise ValueError("Packed offload configuration preflight does not support an initialized multi-rank process group")
    return {"scope": "read_only_process_metadata_no_initialization", "requested_world_size": requested,
        "process_group_initialized": initialized, "process_group_world_size": size}


def _configuration_preflight(args, plan):
    """Exercise the installed schema/option handoff, without a training runtime."""
    from dataclasses import asdict
    from examples.vae import benchmark_vae as benchmark

    output = args.output.resolve()
    if output.exists() or output.parent.joinpath(output.stem + "-artifacts").exists():
        raise FileExistsError("Choose a fresh configuration-preflight output")
    report = {**plan, "scope": "configuration_preflight_only", "status": "running",
        "execution_completed": False, "timing_eligible": False, "CUDA_execution_evidence": False,
        "configuration_preflight": {"requested": True, "valid": False,
            "checks": dict.fromkeys(("installed_identity", "arm_policy_pins", "typed_config", "backend_config",
                                      "single_process_execution"), False),
            "no_model_allocated": True, "no_input_allocated": True, "cuda_execution": False},
        "failure_context": {"stage": "configuration_preflight", "phase": "installed_identity"},
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    checks, context = report["configuration_preflight"]["checks"], report["failure_context"]
    benchmark.write_json(output, report)
    identity_function = None
    try:
        from benchmarks.compiler.installed_runtime import installed_identity

        identity_function = installed_identity
        report["installed"] = installed_identity(ROOT, full=True)
        if report["installed"]["version"] != plan["expected_installed_version"]:
            raise RuntimeError("Install the requested " + plan["expected_installed_version"] + " wheel before validation")
        if plan["experiment_arm"] is not None:
            pin = plan["experiment_arm"]["manifest"]["expected_installed"]
            if any(report["installed"].get(key) != pin[key] for key in ("version", "source_sha256", "files_hashed")):
                raise RuntimeError("Installed package differs from the experiment arm pin")
        checks["installed_identity"] = True
        _verify_experiment_inputs(plan)
        checks["arm_policy_pins"] = True
        report["helper_sha256"] = _helper_identity()
        os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
        context["phase"] = "config_validation"
        import deepopt

        report["validated_config"] = asdict(deepopt.load_config(plan["settings"]))
        checks["typed_config"] = True
        context["phase"] = "distributed_execution_validation"
        report["execution_topology"] = _configuration_topology()
        checks["single_process_execution"] = True
        context["phase"] = "native_options_validation"
        from deepopt.compiler import BackendConfig
        from benchmarks.compiler.save_inductor_references import disable_dynamo_reproduction

        report["compiler_environment"] = disable_dynamo_reproduction()
        BackendConfig(**plan["settings"]["runtime"]["compile"]["options"])
        checks["backend_config"] = True
        report["status"] = "completed"
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
    finally:
        try:
            _verify_experiment_inputs(plan)
            report["helper_sha256_after"] = _helper_identity()
            report["helper_source_stable"] = report.get("helper_sha256") == report["helper_sha256_after"]
            if identity_function is not None:
                report["installed_after"] = identity_function(ROOT, full=True)
            report["installed_stable"] = (report.get("installed") is not None
                                           and report.get("installed") == report.get("installed_after"))
            if not report["helper_source_stable"] or not report["installed_stable"]:
                raise RuntimeError("Configuration-preflight source or installation changed")
        except Exception as error:
            details = benchmark.failure_error_details(error)
            if "error" not in report:
                report.update(details)
            else:
                report.setdefault("collection_errors", []).append(details)
            report["status"] = "failed"
        report["configuration_preflight"]["valid"] = report["status"] == "completed" and all(checks.values())
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    print(json.dumps({"status": report["status"], "configuration_preflight": report["configuration_preflight"],
        "error": report.get("error"), "output": str(output)}, allow_nan=False))
    return 0 if report["configuration_preflight"]["valid"] else 1


def run_native_execution(args):
    """Run one model through complete native updates and save execution evidence."""
    from examples.vae import benchmark_vae as benchmark
    from examples.vae.execution_memory import ExecutionMemoryAudit

    args = deepcopy(args)
    requested_window = {"warmup": args.warmup, "updates": args.updates}
    diagnostic_phase = getattr(args, "diagnostic_phase", "none")
    if diagnostic_phase == "cold":
        args.warmup, args.updates, args.benchmark = 0, 1, False
        args.profile_after_measurement = False
    elif diagnostic_phase == "steady":
        args.profile_after_measurement = True
    plan = execution_plan(args)
    plan["diagnostic"] = {"phase": diagnostic_phase, "requested_clean_window": requested_window,
        "measurement_eligible": False, "storage_ledger": getattr(args, "storage_ledger", False)}
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if getattr(args, "configuration_preflight", False):
        return _configuration_preflight(args, plan)
    output = args.output.resolve()
    directory = output.parent / (output.stem + "-artifacts")
    if output.exists() or directory.exists():
        raise FileExistsError("Choose a fresh output; previous execution evidence is immutable")
    directory.mkdir(parents=True)
    report = {**plan, "status": "running", "execution_completed": False,
        "native_ownership_verified": False, "eager_ownership_verified": False,
        "warmup_completed": 0, "updates_completed": 0,
        "steps": [], "memory": {}, "capacity_status": "not_established",
        "helper_sha256": _helper_identity(),
        "started_at_utc": datetime.now(timezone.utc).isoformat()}
    report["output_path"] = str(output)
    memory_audit = ExecutionMemoryAudit(enabled=plan["memory_audit_options"]["enabled"],
        runtime_snapshot=_packed_offload_snapshot if plan.get("packed_offload_override") is not None else None)
    report["memory_audit"] = memory_audit.report
    native, benchmarking = plan["execution_mode"] == "native", plan["benchmark"]
    if benchmarking:
        report["measurement"] = {"valid": False, "invalid_reasons": [], "samples": [],
            "numerical_correctness_assessed": False,
            "scope": "descriptive_execution_timing_without_numerical_qualification",
            "timing_scope": "device_resident_train_step_including_loss_backward_optimizer_schedule",
            "wall_scope": "GPU-resident train_step and schedule; forward/loss/backward/optimizer/zero_grad",
            "cuda_event_scope": "current-stream train_step interval; excludes schedule calls and H2D",
            "exclusions": ["input_loading", "H2D", "cold_compile_and_warmup", "finite_checks",
                           "artifact_observation", "source_and_state_hashing", "code_IR_writing"]}
    context = {"stage": "setup", "phase": "installed_identity", "update": None, "route": plan["execution_mode"]}
    report["failure_context"] = context
    benchmark.write_json(output, report)
    model = optimizer = runtime = recorder = hook = torch = deepopt = device = factory = None
    storage_trace = None
    images = cpu_images = loader = loss = losses = schedules = compiler_report = None
    current = before_optimizer = progress = None
    try:
        if plan["project_root"] and plan["project_root"] not in sys.path:
            sys.path.append(plan["project_root"])
        if plan["settings"]["runtime"].get("deterministic", False):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
        from benchmarks.compiler.installed_runtime import installed_identity

        report["installed"] = installed_identity(ROOT, full=True)
        _verify_experiment_inputs(plan)
        expected = plan["expected_installed_version"]
        if report["installed"]["version"] != expected:
            raise RuntimeError(f"Install the requested {expected} wheel before execution")
        if plan["experiment_arm"] is not None:
            pin = plan["experiment_arm"]["manifest"]["expected_installed"]
            if any(report["installed"].get(key) != pin[key] for key in ("version", "source_sha256", "files_hashed")):
                raise RuntimeError("Installed package differs from the experiment arm pin")
        import torch
        import deepopt
        from deepopt.runtime import seed_everything
        from deepopt.compiler.runtime import native_boundary_observation
        from benchmarks.compiler.save_inductor_references import disable_dynamo_reproduction

        report["compiler_environment"] = disable_dynamo_reproduction()
        if native:
            from deepopt.compiler import BackendConfig

            # Unknown options, wrong types and unavailable candidate flags fail
            # against this arm's installed compiler before model allocation.
            context["phase"] = "native_options_validation"
            BackendConfig(**plan["settings"]["runtime"]["compile"]["options"])
        device = torch.device(plan["device"])
        if plan.get("packed_offload_override") is not None:
            from deepopt._packed_offload import validate_packed_offload_execution

            context["phase"] = "packed_offload_device_validation"
            validate_packed_offload_execution(plan["packed_offload_override"]["policy"], device_type=device.type,
                compiler_options=plan["settings"]["runtime"]["compile"]["options"])
            policy = plan["packed_offload_override"]["policy"]
            if device.type == "cuda" and policy.get("version", 1) == 2:
                from examples.vae.execution_memory import host_capacity_admission

                context["phase"] = "physical_host_capacity_preflight"
                report["host_capacity_preflight"] = host_capacity_admission(
                    policy["host_pinned_bytes"] + policy["host_pageable_bytes"] if policy["enabled"] else 0,
                    headroom_bytes=args.host_headroom_mib * 2**20)
                if not report["host_capacity_preflight"]["admitted"]:
                    raise RuntimeError("Physical pinned capacity plus host/job headroom cannot be admitted")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU substitution was performed")
            torch.cuda.set_device(device)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The requested CUDA GPU must support BF16")
            torch.cuda.reset_peak_memory_stats(device)
        raw = plan["settings"]
        seed_everything(raw.get("seed", 42), raw["runtime"].get("deterministic", False))
        torch.utils.deterministic.fill_uninitialized_memory = raw["runtime"].get("fill_uninitialized_memory", False)
        context["phase"] = "config_validation"
        config = deepopt.load_config(raw)
        memory_audit.bind(torch, device)
        memory_audit.prepare_baseline()
        context["phase"] = "input_loading"
        if args.synthetic:
            cpu_images = torch.rand(plan["requested_input_shape"],
                generator=torch.Generator().manual_seed(raw.get("seed", 42)), dtype=torch.float32) * 2 - 1
        else:
            loader = benchmark.build_loader(raw, SimpleNamespace(batches=1),
                                           SimpleNamespace(world_size=1, global_rank=0))
            cpu_images = next(iter(loader))
        if list(cpu_images.shape) != plan["requested_input_shape"]:
            raise ValueError("Loader did not supply the requested physical batch geometry")
        if not cpu_images.is_contiguous() or cpu_images.stride(0) <= 0:
            raise ValueError("Input must be a dense physical batch, not an expanded/broadcast view")
        if cpu_images.dtype != torch.float32:
            raise ValueError("The original VAE input contract requires FP32 images")
        report["input"] = {"shape": list(cpu_images.shape), "dtype": str(cpu_images.dtype),
            "requires_grad": False, "sha256": benchmark.fingerprint([cpu_images]), "mode": plan["input_mode"],
            "logical_bytes": cpu_images.numel() * cpu_images.element_size(),
            "cpu_storage_bytes": cpu_images.untyped_storage().nbytes()}
        context["phase"] = "model_initialization"
        if plan["model_factory"]:
            factory = benchmark.import_factory(plan["model_factory"])
        else:
            from examples.vae.mbconv_vae import MBConvVAE_V2

            factory = MBConvVAE_V2
        model = factory(**raw["application"]["model"]).to(device).train()
        count = sum(value.numel() for value in model.parameters())
        if not plan["model_factory"] and raw["application"]["model"] == ORIGINAL_MODEL and count != 32317999:
            raise AssertionError("Original VAE parameter count changed")
        report["model"] = {"parameters": count, "options": raw["application"]["model"]}
        context["phase"] = "optimizer_initialization"
        optimizer = benchmark.make_optimizer(model, raw["training"], plan["optimizer_factory"])
        if benchmarking or diagnostic_phase == "cold":
            report["model"]["initial_state_sha256"] = benchmark.fingerprint(model.state_dict().values())
            report["optimizer"] = {"name": raw["training"]["optimizer"],
                "initial_state_sha256": _state_fingerprint(optimizer.state_dict(), torch, benchmark)}
        report["provenance"] = benchmark.source_provenance(deepopt, factory, optimizer)
        context["phase"] = "native_runtime_setup"
        runtime = deepopt.TrainingRuntime(model, optimizer, device=device,
                                         precision=raw["runtime"]["precision"], config=config)
        model = runtime.model.train()
        report["runtime_config"] = runtime.config
        if benchmarking and (runtime.execution_mode not in (None, "max_speed")
                or runtime.config["runtime"].get("checks") is not False):
            raise RuntimeError("Effective runtime diagnostics must be disabled for benchmark timing")
        if not native:
            _verify_eager_ownership(model, runtime)
            report["eager_ownership_verified"] = True
        report["software"] = {"python": sys.version, "torch": str(torch.__version__), "torch_git": torch.version.git_version,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}
        try:
            report["software"]["triton"] = version("triton")
        except PackageNotFoundError:
            report["software"]["triton"] = None
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            report["software"]["gpu"] = {"name": props.name, "uuid": str(props.uuid),
                "capability": [props.major, props.minor], "total_memory": props.total_memory}
        images = cpu_images.to(device, non_blocking=raw["runtime"].get("non_blocking", True))
        images.requires_grad_(False)
        report["input"].update(device=str(images.device), stride=list(images.stride()),
            contiguous=images.is_contiguous(), storage_bytes=images.untyped_storage().nbytes())
        optimizer.zero_grad(set_to_none=True)
        schedules = deepopt.ScheduleController.from_config(config.schedules)
        recorder = _ArtifactRecorder(torch, context) if native else None
        current = None

        def before_optimizer(owner, positional, keyword):
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            current["gradient_parameters"] = sum(value is not None for value in gradients)
            current["trainable_parameters"] = len(gradients)
            current["all_trainable_gradients_present"] = all(value is not None for value in gradients)
            current["fp32_gradients"] = all(value is not None and value.dtype == torch.float32 for value in gradients)
            current["finite_gradients"] = _finite(torch, gradients)
            if not (current["finite_gradients"] and current["all_trainable_gradients_present"]
                    and current["fp32_gradients"]):
                raise RuntimeError("Native execution requires present, finite FP32 trainable-parameter gradients")

        hook = optimizer.register_step_pre_hook(before_optimizer)
        memory_audit.capture("after_setup", update=0)
        if args.warmup == 0:
            memory_audit.capture("after_warmup", update=0)
        if benchmarking or diagnostic_phase == "cold":
            # Match the training benchmark's first latent draw independently
            # of preparation, optimizer construction or compiler setup.
            seed_everything(raw.get("seed", 42), raw["runtime"].get("deterministic", False))
            if benchmarking:
                report["measurement"]["rng_seed_before_warmup"] = raw.get("seed", 42)
            else:
                report["diagnostic"]["rng_seed_before_first_update"] = raw.get("seed", 42)
            if plan.get("packed_offload_override") is not None:
                report["rng_before_warmup_sha256"] = _rng_fingerprint(torch, device, benchmark)
        started = time.perf_counter()
        with native_boundary_observation(recorder) if native else nullcontext():
            checked_updates = args.warmup if benchmarking else args.warmup + args.updates
            for number in range(1, checked_updates + 1):
                stage = "warmup" if number <= args.warmup else "update"
                if number == args.warmup + 1:
                    report["memory"]["startup"] = _memory(torch, device)
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    memory_audit.capture("measurement_start", update=args.warmup)
                current = {"number": number, "stage": stage, "forward_completed": False,
                    "backward_completed": False, "optimizer_completed": False}
                report["steps"].append(current)
                context.update(stage=stage, phase="schedule", update=number)
                context.pop("native_artifact_phase", None)
                context.pop("native_artifact_source_sha256", None)
                schedules.prepare_update(number, epoch=1)
                current["beta"] = schedules["beta"]
                def progress(event):
                    context.update(event)
                    if event["phase"] == "backward":
                        current["forward_completed"] = True
                    elif event["phase"] == "optimizer":
                        current["backward_completed"] = True
                step_start = time.perf_counter()
                if diagnostic_phase == "cold":
                    from examples.vae.execution_profile import run_execution_profile
                    from examples.vae.execution_storage import NativeStorageTrace

                    storage_trace = (NativeStorageTrace(torch, device, recorder=recorder,
                        runtime_snapshot=_packed_offload_snapshot if plan.get("packed_offload_override") is not None else None)
                        if plan["diagnostic"]["storage_ledger"] else None)
                    if storage_trace is not None:
                        storage_trace.observe_persistent(model, optimizer, images)
                    result_box = []
                    def cold_update(number, phases):
                        def cold_progress(event):
                            progress(event)
                            if storage_trace is not None and event["phase"] == "optimizer":
                                storage_trace.observe_persistent()
                            phases(event)
                        result_box.append(benchmark.train_step(model, optimizer, runtime, images, current["beta"],
                            device_type=device.type, grad_clip_norm=config.training.grad_clip_norm, progress=cold_progress))
                    def cold_ownership():
                        from deepopt.compiler import compilation_report
                        existing = compilation_report(model)
                        return ({"captured_graphs": 0} if existing is None or not existing.graphs else
                            {key: value for key, value in benchmark.native_compilation_evidence(model).items()
                             if key in ("captured_graphs", "runtime_fallbacks", "native_autodiff_operations", "native_autodiff_regions")})
                    memory_audit.capture("profile_start", update=0)
                    report["profile"] = run_execution_profile(torch=torch, device=device, directory=directory / "cold-profile",
                        update=cold_update, first_update=1, updates=1, record_memory=True,
                        record_shapes=args.profile_record_shapes, ownership=cold_ownership,
                        allocator_diagnostics=lambda: benchmark.cuda_failure_diagnostics(device), context=context,
                        storage_trace=storage_trace, diagnostic_phase="cold", allow_initial_compilation=True)
                    memory_audit.capture("profile_end", update=1)
                    if len(result_box) != 1:
                        report["diagnostic"]["profile_error"] = report["profile"].get("error")
                        raise RuntimeError("Cold first-update diagnostic did not complete: " + str(report["profile"].get("error")))
                    loss = result_box.pop()
                    memory_audit.report["first_update_profile"] = {"status": report["profile"]["status"],
                        "implemented": True, "first_update": 1,
                        "allocator_snapshot_available": device.type == "cuda" and any(
                            Path(item.get("path", "")).name == "allocator.json" for item in report["profile"].get("files", ())),
                        "profile": str(directory / "cold-profile/profile.json")}
                    if "storage_ledger" in report["profile"]:
                        memory_audit.report["typed_storage_owner_ledger"] = report["profile"]["storage_ledger"]
                else:
                    loss = benchmark.train_step(model, optimizer, runtime, images, current["beta"],
                        device_type=device.type, grad_clip_norm=config.training.grad_clip_norm, progress=progress)
                current["optimizer_completed"] = True
                context["phase"] = "finite_state"
                current["finite_loss"] = bool(torch.isfinite(loss).all())
                current["loss"] = float(loss) if current["finite_loss"] else None
                current["finite_parameters"] = _finite(torch, model.parameters())
                if not current["finite_loss"] or not current["finite_parameters"]:
                    raise RuntimeError("Native execution produced nonfinite loss or parameters")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                current["wall_seconds_including_checks"] = time.perf_counter() - step_start
                schedules.complete_update(number)
                report[stage + "_completed" if stage == "warmup" else "updates_completed"] += 1
                if native and not report["native_ownership_verified"]:
                    context["phase"] = "native_ownership_check"
                    benchmark.native_compilation_evidence(model)
                    report["native_ownership_verified"] = True
                if number == 1:
                    memory_audit.capture("after_first_update", update=number)
                if number == args.warmup:
                    memory_audit.capture("after_warmup", update=number)
                print(f"{plan['execution_mode']} VAE {stage} {number}: physical_shape={list(images.shape)} loss={current['loss']}", flush=True)
        if benchmarking:
            # The optimizer hook and native artifact context are absent for
            # every timed update. Ordinary wrapper execution remains intact.
            hook.remove()
            hook = None
            before = benchmark.native_compilation_evidence(model) if native else None
            report["measurement"]["native_graph_count_before"] = before["captured_graphs"] if before else 0
            del before
            if plan.get("packed_offload_override") is not None:
                context.update(stage="after_warmup", phase="packed_offload_profile_freeze")
                report["packed_offload_profile"] = _freeze_packed_offload_profile()
                from examples.vae.benchmark_packed_offload import runtime_budget_errors

                context["phase"] = "packed_offload_budget_preflight"
                errors = runtime_budget_errors(report["packed_offload_profile"], plan["packed_offload_override"]["policy"])
                report["packed_offload_budget_preflight"] = {"valid": not errors, "errors": errors,
                    "scope": "frozen_runtime_metadata_after_warmup_before_measurement"}
                if errors:
                    raise RuntimeError("Packed offload budget preflight failed before measurement: " + ", ".join(errors))
                report["rng_after_warmup_sha256"] = _rng_fingerprint(torch, device, benchmark)
            losses = _benchmark_updates(torch=torch, device=device, benchmark=benchmark, model=model,
                optimizer=optimizer, runtime=runtime, images=images, schedules=schedules,
                grad_clip_norm=config.training.grad_clip_norm, warmup=args.warmup, updates=args.updates,
                report=report, context=context, memory_audit=memory_audit)
            context.update(stage="post_measurement", phase="post_measurement")
            if plan.get("packed_offload_override") is not None:
                report["rng_after_measurement_sha256"] = _rng_fingerprint(torch, device, benchmark)
            postchecks = {"finite_losses": _finite(torch, losses),
                          "finite_parameters": _finite(torch, model.parameters())}
            report["measurement"]["postchecks"] = postchecks
            if not all(postchecks.values()):
                raise RuntimeError("Benchmark postchecks found nonfinite loss or parameters")
            measured_rows = [row for row in report["steps"] if row["stage"] == "update"]
            for loss, row, sample in zip(losses, measured_rows, report["measurement"]["samples"]):
                row["loss"] = sample["loss"] = float(loss)
            del losses
            report["input"]["sha256_after"] = benchmark.fingerprint([images])
            report["input_stable"] = report["input"]["sha256_after"] == report["input"]["sha256"]
            if not report["input_stable"]:
                raise RuntimeError("The repeated resident input changed during execution")
        else:
            memory_audit.capture("measurement_end", update=args.warmup + args.updates)
            if diagnostic_phase == "cold":
                report["input"]["sha256_after"] = benchmark.fingerprint([images])
                report["input_stable"] = report["input"]["sha256_after"] == report["input"]["sha256"]
                report["rng_after_first_update_sha256"] = _rng_fingerprint(torch, device, benchmark)
        report["execution_wall_seconds_including_checks"] = time.perf_counter() - started
        report["execution_completed"] = True
        report["capacity_status"] = "fits_requested_geometry" if device.type == "cuda" else "CPU_diagnostic_only"
        report["physical_target_executed"] = device.type == "cuda" and plan["original_geometry"]
        context.update(stage="finalization", phase="ownership_check", update=None)
        if native:
            compiler_report = benchmark.native_compilation_evidence(model)
            if benchmarking:
                report["measurement"]["native_graph_count_after"] = compiler_report["captured_graphs"]
                if compiler_report["captured_graphs"] != report["measurement"]["native_graph_count_before"]:
                    report["measurement"]["invalid_reasons"].append("native_graph_count_changed_during_measurement")
                    raise RuntimeError("Native graph compilation changed during the measured window")
            context["phase"] = "compiler_report_save"
            compiler_data = json.dumps(_metadata(compiler_report, torch), indent=2, allow_nan=False).encode()
            report["compiler_report_artifact"] = _save_bytes(directory, "compiler-report.json", compiler_data)
            report["compiler_report_summary"] = {key: value for key, value in compiler_report.items() if key != "graphs"}
            from examples.vae.execution_inventory import extract_inventory

            context["phase"] = "compiled_inventory_export"
            inventory = extract_inventory(compiler_report, plan, report["installed"])
            report["packed_offload_inventory"] = inventory
            report["inventory_artifact"] = _save_bytes(directory, "inventory.json",
                json.dumps(inventory, indent=2, allow_nan=False).encode())
            if getattr(args, "inventory_output", None) is not None:
                if inventory["status"] != "available":
                    raise RuntimeError("Compiled model-scope offload inventory is unavailable: "
                        f"{inventory['reason']}; distinct inventories={inventory['model_scope_inventories_found']}; "
                        f"invalid records={len(inventory['inventory_extraction']['errors'])}")
                target = args.inventory_output.resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(json.dumps(inventory, indent=2, allow_nan=False).encode())
                report["inventory_export"] = {"path": str(target), "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        else:
            _verify_eager_ownership(model, runtime)
            report["compiler_report_status"] = "not_applicable_eager_execution"
            if benchmarking:
                report["measurement"]["native_graph_count_after"] = 0
        report["status"] = "completed"
    except Exception as error:
        report.update(benchmark.failure_error_details(error), status="failed")
        if type(error).__name__ == "OutOfMemoryError":
            report["capacity_status"] = "cuda_oom"
    finally:
        def collect(field, function):
            try:
                return function()
            except Exception as error:
                report.setdefault("collection_errors", []).append({"field": field,
                    **benchmark.failure_error_details(error)})
                report["status"] = "failed"
                return None

        if hook is not None:
            collect("optimizer_hook_cleanup", hook.remove)
        if torch is not None and device is not None:
            segment = ("postchecks" if benchmarking and "steady" in report["memory"] else
                       "steady" if "startup" in report["memory"] else "startup")
            report["memory"][segment] = collect("memory", lambda: _memory(torch, device))
            snapshots = [row for row in report["memory"].values() if isinstance(row, dict)]
            report["memory"]["total"] = {name: max((row[name] for row in snapshots if row.get(name) is not None), default=None)
                for name in ("peak_allocated_bytes", "peak_reserved_bytes")}
            report["memory"]["current_segment"] = segment
            report["memory"]["steady_includes_compile"] = args.warmup == 0 if "steady" in report["memory"] else None
            report["memory"]["total_accounting"] = "max of startup and current/steady segment peaks; first-update peaks retained"
            report["memory"]["scope"] = "Torch CUDA allocator; native workspaces and other processes are not fully measured"
            report["cuda_allocator_diagnostics"] = collect("allocator_diagnostics",
                lambda: benchmark.cuda_failure_diagnostics(device))
        if native and model is not None and "compiler_report_artifact" not in report:
            details = collect("partial_compiler_report", lambda: benchmark.failure_candidate_diagnostics(model)) or {}
            compiler_report = details.get("compiler_report")
            if compiler_report is not None:
                report["compiler_report_artifact"] = collect("partial_compiler_report_save", lambda: _save_bytes(
                    directory, "compiler-report.json", json.dumps(_metadata(compiler_report, torch),
                    indent=2, allow_nan=False).encode()))
            report["compiler_report_status"] = details.get("compiler_report_status", "collection_failed")
        if recorder is not None:
            report["artifacts"] = collect("native_code_IR", lambda: recorder.save(directory))
            if diagnostic_phase == "cold" and storage_trace is not None and report["artifacts"] is not None:
                report["artifacts"].update(ordinary_artifact_fn_used=False, metadata_observed_wrapper_used=True)
        elif not native:
            report["artifacts"] = {"files": [], "tensor_files_saved": 0, "not_applicable_to_eager": True}
        try:
            if deepopt is not None and factory is not None and optimizer is not None:
                report["provenance_after"] = benchmark.source_provenance(deepopt, factory, optimizer)
                report["source_stable"] = report["provenance"] == report["provenance_after"]
            if "installed" in report:
                from benchmarks.compiler.installed_runtime import installed_identity

                report["installed_after"] = installed_identity(ROOT, full=True)
                report["installed_stable"] = report["installed_after"] == report["installed"]
            report["helper_sha256_after"] = _helper_identity()
            report["helper_source_stable"] = report["helper_sha256_after"] == report["helper_sha256"]
            if (report.get("source_stable") is False or report.get("installed_stable") is False
                    or report["helper_source_stable"] is False):
                raise RuntimeError("Installed library or workload sources changed during execution")
            _verify_experiment_inputs(plan)
        except Exception as error:
            report.setdefault("collection_errors", []).append({"field": "source_identity",
                **benchmark.failure_error_details(error)})
            report["status"] = "failed"
        if benchmarking:
            measurement = report["measurement"]
            if plan.get("packed_offload_override") is not None:
                report["packed_offload_runtime"] = collect("packed_offload_runtime", _packed_offload_snapshot)
            measurement["valid"] = report["status"] == "completed" and not measurement["invalid_reasons"]
            if not measurement["valid"] and not measurement["invalid_reasons"]:
                measurement["invalid_reasons"].append("worker_execution_or_evidence_failed")
            report["timing_eligible"] = measurement["valid"] and diagnostic_phase == "none"
            report["memory_segments"] = {"startup": report["memory"].get("startup"),
                "measurement": measurement.get("memory"), "postchecks": report["memory"].get("postchecks"),
                "total": report["memory"].get("total")}
        if report.get("execution_completed"):
            memory_audit.capture("after_postchecks", update=args.warmup + args.updates)
        elif device is not None:
            memory_audit.capture("execution_failure", update=context.get("update"))
        if benchmarking:
            if plan["profiling"]["enabled"]:
                if measurement["valid"]:
                    memory_audit.capture("profile_start", update=args.warmup + args.updates)
                    try:
                        _profile_after_measurement(torch=torch, device=device, benchmark=benchmark,
                            model=model, optimizer=optimizer, runtime=runtime, images=images,
                            schedules=schedules, config=config, report=report, directory=directory)
                    except Exception as error:
                        report["profile"] = {"requested": True, "complete": False, "status": "incomplete",
                            "measurement_eligible": False, "tensor_files_written": 0,
                            **benchmark.failure_error_details(error)}
                    finally:
                        memory_audit.capture("profile_end", update=args.warmup + args.updates
                                             + report.get("profile", {}).get("updates_completed", 0))
                    if "storage_ledger" in report.get("profile", {}):
                        memory_audit.report["typed_storage_owner_ledger"] = report["profile"]["storage_ledger"]
                else:
                    report["profile"] = {"requested": True, "complete": False, "status": "not_run",
                        "reason": "clean_measurement_incomplete", "measurement_eligible": False}
        # Canonical evidence is now frozen. Drop worker-owned tensor/closure
        # references before resetting compiler/autocast caches and the allocator.
        # Cleanup errors are diagnostic and cannot replace the execution error
        # or invalidate a completed measured window.
        model = optimizer = runtime = recorder = hook = images = cpu_images = loader = loss = losses = None
        schedules = compiler_report = current = before_optimizer = progress = None
        release_actions = []
        if plan.get("packed_offload_override") is not None and torch is not None:
            release_actions.append(("packed_offload_drain", _drain_packed_offload))
            release_actions.append(("packed_offload_close", _close_packed_offload))
        if deepopt is not None:
            release_actions.append(("shutdown_distributed", deepopt.shutdown_distributed))
        if torch is not None:
            release_actions.extend((("compiler_reset", torch.compiler.reset),
                                    ("clear_autocast_cache", torch.clear_autocast_cache)))
        memory_audit.cleanup(release_actions=release_actions)
        if plan.get("packed_offload_override") is not None:
            report["packed_offload_cleanup"] = deepcopy(memory_audit.report["snapshots"].get("cleanup", {}).get(
                "packed_offload_runtime"))
        expected_stages = ["before_model", "after_setup", "after_first_update", "after_warmup",
                           "measurement_start", "measurement_end", "after_postchecks", "cleanup"]
        if plan["profiling"]["enabled"]:
            expected_stages += ["profile_start", "profile_end"]
        memory_audit.finalize(expected_stages=expected_stages)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        benchmark.write_json(output, report)
    return 0 if report["status"] == "completed" else 1


def main(argv=None):
    return run_native_execution(parser().parse_args(argv))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
