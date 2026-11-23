"""Test the installed DeepOpt library with frozen contracts and benchmarks.

--dry-run and --export use only the standard library. GPU stages fail closed
when required kernels/tests are skipped or selected fusions lack launch proof.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("cpu", "gpu-smoke", "gpu-full", "metadata")
AREAS = ("autodiff", "fusion", "codegen", "memory", "integration", "models")
SOURCE_ROOTS = ("benchmarks", "tests", "examples")
SOURCE_SUFFIXES = {".py", ".toml", ".json", ".yaml", ".yml", ".csv"}
CONTRACTS = {
    "autodiff": (
        "tests.opt.test_bf16_policy.BF16PolicyTests",
        "tests.compiler.test_bf16_storage_native.BF16StorageActivationTests",
        "tests.compiler.test_native_semantic_contracts.NativeSemanticContractTests",
        "tests.compiler.test_native_semantic_contracts.NativeAttentionBroadcastTests",
        "tests.compiler.test_native_prepared_group_norm.PreparedGroupNormTests",
        "tests.compiler.test_native_empty_normalization.EmptyNormalizationTests",
        "tests.compiler.test_attention_saved_state", "tests.compiler.test_saved_activation_contracts",
        "tests.compiler.test_native_multiplication", "tests.compiler.test_native_padding",
        "tests.compiler.test_native_autodiff", "tests.compiler.test_attention_capture",
        "tests.compiler.test_native_fp8_recipes_cuda",
        "tests.compiler.test_pointwise_autograd_inputs", "tests.compiler.test_custom_lowering",
        "tests.opt.test_binary_activation_saving", "tests.opt.test_padding", "tests.opt.test_convolution_native",
        "tests.opt.test_attention_compression",
    ),
    "fusion": (
        "tests.opt.test_bf16_storage_normalization.BF16StorageNormalizationTests",
        "tests.opt.test_native_group_staged.NativeGroupStagedTests",
        "tests.compiler.test_frozen_producer_fusion.FrozenProducerFusionTests",
        "tests.compiler.test_fusion_model_pack_sharing.VAEFusionPackSharingTests",
        "tests.compiler.test_fusion_contracts", "tests.compiler.test_normalization_fusion",
        "tests.compiler.test_packed_multiply_fusion", "tests.compiler.test_pack_fusion",
        "tests.compiler.test_fusion_reporting", "tests.compiler.test_bf16",
        "tests.compiler.test_pack_fusion_cuda",
        "tests.benchmarks.test_fusion_candidate_evidence", "tests.benchmarks.test_native_fusion_fixtures",
    ),
    "codegen": (
        "tests.opt.test_bf16_dtype_diagnostics.DTypeDiagnosticsTests",
        "tests.opt.test_storage_exp_kernels.StorageExpKernelTests",
        "tests.opt.test_storage_exp_policy_guards.StorageExpPolicyGuardTests",
        "tests.benchmarks.test_policy_qualification_selectors.PolicyQualificationSelectorTests",
        "tests.compiler.test_bf16_panel_compiler.BF16PanelCompilerTests",
        "tests.opt.test_bf16_codegen_safety.BF16SafetyTests",
        "tests.opt.test_bf16_gemm_configuration", "tests.compiler.test_kernel_backends",
        "tests.benchmarks.test_bf16_configurations_benchmark", "tests.benchmarks.test_backend_qualification_suite",
        "tests.benchmarks.test_backend_contract_runner",
        "tests.benchmarks.test_installed_backend_runtime",
        "tests.benchmarks.test_numerical_ipc",
        "tests.benchmarks.test_bf16_qualification",
        "tests.benchmarks.test_scheduler_failures.SchedulerFailureTests.test_nonzero_exit_invalidates_fresh_success_json",
    ),
    "memory": (
        "tests.compiler.test_policy_routes_physical_memory.PolicyRouteTests",
        "tests.compiler.test_policy_routes_physical_memory.PhysicalObservationTests",
        "tests.compiler.test_schedule_memory", "tests.memory.test_storage_ledger",
        "tests.memory.test_memory_estimation", "tests.opt.test_spatial_buffer_allocations",
        "tests.benchmarks.test_bf16_measurement", "tests.benchmarks.test_vae_memory_audit", "tests.checkpointing.test_bf16_policy_identity",
    ),
    "integration": (
        "tests.cuda.test_raw_bf16_policy_guard.RawBF16PolicyGuardTests",
        "tests.benchmarks.test_bf16_policy_matrix.BF16PolicyMatrixTests",
        "tests.compiler.test_native_vae.NativeVAETests",
        "tests.compiler.test_native_vae_training.NativeVAETrainingTests",
        "tests.compiler.test_native_vae_primitives.NativeVAEPrimitiveTests",
        "tests.compiler.test_operation_registration.OperationRegistrationTests",
        "tests.compiler.test_library_group_norm.LibraryGroupNormTests",
        "tests.compiler.test_native_global_average.NativeGlobalAverageTests",
        "tests.compiler.test_convolution_pack_sharing.ConvolutionPackSharingTests",
        "tests.benchmarks.test_vae_qualification",
        "tests.compiler.test_training", "tests.checkpointing.test_saved_multiplication_recipe",
        "tests.execution.test_compile_black_box", "tests.execution.test_compile_white_box",
        "tests.scheduling.test_compile_black_box", "tests.distributed.test_compiler",
    ),
}

# Requested feature coverage, not a claim that all future plan work exists.
COVERAGE = (
    ("attention_saved_state", "autodiff", "CPU ABI contracts; forced CUDA math/flash/efficient/cuDNN"),
    ("consumer_specific_residuals", "autodiff", "Exact/encoded uses, provenance, rank/layout, policy guards"),
    ("multiplication_padding_frozen_convolution", "autodiff", "Public eager routes and native VJPs, masks and codec oracles"),
    ("native_fp8", "autodiff", "Current/delayed/rowwise/blockwise CUDA cases, two optimizer updates, cache/scaling/transport diagnostics"),
    ("gemm_activation_residual", "fusion", "Generated on/off and library controls; candidate-specific CUDA evidence"),
    ("layernorm_groupnorm_activation", "fusion", "Forward/backward/both; GroupNorm FP32 AMP and channels-last controls"),
    ("packed_multiply_pack_cse", "fusion", "Decoded coefficients and explicit-IR CSE CUDA proof; public shared-projection model is a rejection control"),
    ("role_shape_gemm_configurations", "codegen", "Forward/dX/dW independent configs; materialized/tiled dW controls"),
    ("final_schedule_storage_lifetimes", "memory", "Actual order, effects/aliases, release and weak physical-storage ledger"),
    ("invocation_integration", "integration", "Retained/interleaved graphs, recipe resume, accumulation and two-rank Gloo"),
    ("pointwise_cuda_graph", "integration", "Existing pointwise replay contract; broader graph capture remains unqualified"),
    ("vae_local128", "memory", "Exact allocation-free metadata audit, not CUDA capacity or full optimizer qualification"),
    ("vae_native_recipe", "integration", "CPU semantic and separate strict CUDA selectors; optional 224-square custom-optimizer trajectories, full named IPC parity and measured allocator capacity"),
    ("bf16_storage_policy", "autodiff", "Immutable policy/scope and same-policy native semantics; separate actual CUDA chain and gradient-mask contracts"),
    ("storage_normalization_exp", "codegen", "CPU semantic/interpreter controls and strict actual CUDA stores, staged native GroupNorm and independent exp coefficient oracles"),
    ("physical_storage_observation_reuse", "memory", "Alias/version/storage observations and experimental invocation-local inference-only generated-buffer reuse; target CUDA proof remains separate"),
    ("raw_policy_replay", "integration", "Retained policy guards and strict actual raw CUDA Graph replay; optional multi-GPU checks are independent"),
)
NOT_IMPLEMENTED_OR_UNQUALIFIED = (
    "Physical storage observation and invocation-local generated-buffer reuse are experimental and inference-only; training scratch pools, automatic reuse promotion and target runtime/memory benefits remain unqualified.",
    "Bounded panel decoding and scratch/resource contracts are implemented experimentally; actual target CUDA correctness, launch/resource evidence and complete-pipeline runtime/memory remain separately qualified. Split-K, persistent GEMMs and automatic winning-route selection are not implemented/qualified.",
    "The opt-in BF16 storage policy has CPU semantic coverage; actual CUDA stores, native differentiation, same-policy model parity and memory/performance require the separate strict target stages.",
    "VAE target paths require explicit --vae-target and actual CUDA execution; CPU contracts do not qualify 224-square custom-optimizer training. NCCL DDP and native-MLP/norm CUDA Graph replay remain separate qualification.",
    "Public shared-projection pack CSE lacks compatible exposed pack IR; explicit-IR CSE is tested separately.",
    "Norm+activation positive controls use declared exact ATen activation state with norm xhat compression; compressed activation-coefficient fusion is a rejection case. Experimental native-statistics GroupNorm fusion requires explicit qualification and positive CUDA replacement evidence; selection alone never qualifies it.",
    "Local128 VAE metadata, saved-state bytes and actual CUDA allocated/reserved peaks are different measurements.",
    "Quantization quality versus uncompressed training is outside this suite; matching-recipe checks remain required.",
    "Wide-coordinate GEMM and operand-device safety contracts are implemented with CPU/interpreter controls; CPU metadata and batch-one tests do not qualify the full large-index CUDA domain. Two-GPU checks require --multi-gpu and are never mandatory single-GPU tests.",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stage_activity(stage, log):
    """Read worker progress without importing CUDA or polling it during timing."""
    activity = {"stage": stage["name"], "kind": (
        "cpu_contracts" if stage["validator"] == "tests" else
        "planning_only" if stage["validator"] == "plan" else
        "metadata_only" if stage["validator"] == "metadata" else "gpu_or_kernel_work")}
    try:
        with Path(log).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4096))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        if lines:
            activity["last_log_line"] = lines[-1][:500]
    except OSError:
        pass
    progress = Path(stage["report"]).parent / "fusion-progress.json"
    try:
        active = json.loads(progress.read_text()).get("active")
        if active:
            activity["worker"] = active
            phase_file = active.get("worker_phase_path") if isinstance(active, dict) else None
            if phase_file:
                activity["worker_phase"] = json.loads(Path(phase_file).read_text())
    except (OSError, ValueError, TypeError):
        pass
    return activity


@contextmanager
def report_stage_progress(summary, stage, log, output, index, total, interval):
    """Publish bounded heartbeats while a fresh worker owns the timing device."""
    stopped = threading.Event()
    began = time.perf_counter()

    def publish():
        activity = stage_activity(stage, log)
        activity.update(index=index, total=total, seconds=time.perf_counter() - began)
        summary["active_stage"] = activity
        write_json(output / "qualification-summary.json", summary)
        phase = activity.get("worker_phase", {}).get("phase", activity.get("last_log_line", "starting"))
        print(f"[{index}/{total}] {stage['name']} ({activity['kind']}, {activity['seconds']:.0f}s): {phase}", flush=True)

    publish()

    def monitor():
        while not stopped.wait(interval):
            publish()

    watcher = threading.Thread(target=monitor, name="qualification-progress", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join()
        summary.pop("active_stage", None)


def source_snapshot(root=ROOT):
    paths = []
    for name in SOURCE_ROOTS:
        paths.extend(p for p in (root / name).rglob("*") if p.is_file()
                     and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.exists()}


def snapshot_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Stage:
    name: str
    area: str
    command: tuple
    report: str
    validator: str
    require_observed: bool = False


def make_plan(args):
    output = args.output.resolve()
    gpu = args.profile.startswith("gpu")
    if getattr(args, "multi_gpu", False) and not gpu:
        raise ValueError("--multi-gpu requires a GPU profile; two-device checks cannot run as CPU contracts")
    if getattr(args, "vae_target", False):
        if not gpu:
            raise ValueError("--vae-target requires a GPU profile; CPU contracts cannot qualify the target VAE")
        batches = args.vae_capacity_batches
        if not batches or batches[0] != 1 or batches != sorted(set(batches)) or any(batch < 1 for batch in batches):
            raise ValueError("VAE capacity batches must increase strictly from physical batch 1")
    full = args.profile == "gpu-full"
    representative = gpu and (args.workload == "representative" or (args.workload is None and full))
    dimensions = {"batch": 32, "tokens": 197, "width": 768, "hidden": 3072,
                  "spatial": (16, 16)} if representative else {
                  "batch": 2, "tokens": 17, "width": 64, "hidden": 96, "spatial": (8, 8)}
    fixture_size = [value for key in ("batch", "tokens", "width", "hidden")
                    for value in ("--" + key, str(dimensions[key]))]
    fixture_size += ["--spatial", *map(str, dimensions["spatial"])]
    fusion_execution = ["--worker-isolation", args.fusion_worker_isolation,
                        "--phase-iterations", str(args.phase_iterations),
                        "--phase-repeats", str(args.phase_repeats), "--bf16-policy", "torch_amp",
                        *(["--qualify-native-group-fusion"] if gpu else [])]
    minimum_ms = (args.min_measure_ms if args.min_measure_ms is not None else (200 if full else 50)) if gpu else 0
    timing = {"warmup": 10 if full else 1, "iterations": 50 if full else 2,
              "repeats": 5 if full else 1, "trials": 3 if full else 1}
    timing.update({name: getattr(args, name) for name in timing if getattr(args, name) is not None})
    selected = set(args.only or AREAS)
    stages = []

    def add(name, area, command, report, validator, observed=False):
        if area in selected:
            stages.append(Stage(name, area, tuple(command), str(report), validator, observed))

    if args.profile != "metadata":
        for area, modules in CONTRACTS.items():
            report = output / "contracts" / f"{area}.json"
            add(f"{area}-contracts", area, ["benchmarks/compiler/qualification_tests.py", "--modules", *modules,
                "--device", "cpu", "--output", str(report)], report, "tests")
    if gpu:
        modules = (
            ("semantic-followup-cuda", "autodiff", (
                "tests.compiler.test_native_semantic_contracts.NativeSemanticContractCUDATests",
                "tests.compiler.test_native_semantic_contracts.NativeAttentionBroadcastCUDATests",
                "tests.compiler.test_native_prepared_group_norm.PreparedGroupNormCUDATests",
                "tests.compiler.test_native_empty_normalization.EmptyNormalizationCUDATests",
            ), False),
            # Explicit methods exclude the inherited FakeTensor metadata audit:
            # running that method in a CUDA class is still CPU evidence.
            ("vae-cuda", "integration", ("tests.benchmarks.test_vae_memory_audit.VAEMemoryAuditPolicyCUDATests",
                "tests.compiler.test_native_vae.NativeVAECudaTests.test_whole_vae_matches_eager_for_even_and_odd_images",
                "tests.compiler.test_native_vae.NativeVAECudaTests.test_complete_saved_state_is_typed_and_compression_reduces_physical_storage",
                "tests.compiler.test_native_vae.NativeVAECudaTests.test_selective_frozen_vae_paths_preserve_named_gradient_masks",
                "tests.compiler.test_native_vae_training.NativeVAETrainingCudaTests",
                "tests.compiler.test_native_vae_primitives.NativeVAEPrimitiveCudaTests",
                "tests.compiler.test_library_group_norm.LibraryGroupNormCudaTests",
            ), False),
            ("attention-cuda", "autodiff", ("tests.compiler.test_attention_saved_state_cuda",
                "tests.compiler.test_attention_capture_cuda"), False),
            ("native-bf16-cuda", "autodiff", (
                "tests.compiler.test_native_autodiff_cuda.NativeAutoDiffCudaTests.test_generated_bf16_saved_state_and_backward_fusion",
                "tests.compiler.test_native_autodiff_cuda.NativeAutoDiffCudaTests.test_native_flash_attention_handles_unaligned_head_dimensions",
                "tests.compiler.test_native_multiplication.NativeMultiplicationCudaTests",
                "tests.opt.test_padding.MetadataPaddingTests.test_cuda_bf16_metadata_padding_matches_native",
                "tests.opt.test_padding.MetadataPaddingTests.test_cuda_padding_geometry_and_rounding_oracle",
                "tests.opt.test_convolution_native.PublicFrozenConvolutionTests.test_cuda_frozen_fp32_parameter_masks_preserve_bf16_amp",
            ), False),
            ("fusion-cuda", "fusion", ("tests.compiler.test_normalization_fusion.NormalizationFusionCudaTests",
                "tests.compiler.test_frozen_producer_fusion.FrozenProducerFusionCudaTests",
                "tests.compiler.test_fusion_model_pack_sharing.VAEFusionPackSharingCudaTests",
                "tests.compiler.test_packed_multiply_fusion.PackedMultiplyFusionTests.test_cuda_broadcast_reduction_and_noncontiguous_gradient",
                "tests.compiler.test_pack_fusion_cuda",
                "tests.benchmarks.test_native_fusion_fixtures.NativeFusionFixtureCudaTests"), False),
            ("configuration-cuda", "codegen", ("tests.opt.test_bf16_gemm_configuration.BF16ConfigurationCUDA",
                "tests.opt.test_bf16_codegen_safety.BF16CodegenCUDATests"), False),
            ("pointwise-replay-cuda", "integration", (
                "tests.compiler.test_cuda.CompilerCUDATests.test_fused_pointwise_values_gradients_and_tail_masks",
                "tests.compiler.test_cuda.CompilerCUDATests.test_compiled_callable_cuda_graph_replay_does_not_compile_again"), False),
            ("bf16-storage-native-cuda", "autodiff", (
                "tests.compiler.test_bf16_storage_native.BF16StorageNativeCUDATests",
            ), False),
            ("bf16-storage-normalization-cuda", "fusion", (
                "tests.opt.test_bf16_storage_normalization.BF16StorageNormalizationCudaTests",
                "tests.opt.test_native_group_staged.NativeGroupStagedCudaTests",
            ), False),
            ("storage-exp-cuda", "codegen", (
                "tests.opt.test_storage_exp_kernels.StorageExpCudaTests",
                "tests.opt.test_storage_exp_policy_guards.StorageExpIndependentOracleCUDA",
            ), False),
            ("physical-memory-cuda", "memory", (
                "tests.compiler.test_policy_routes_physical_memory.PhysicalMemoryCUDATests",
            ), False),
            ("raw-policy-replay-cuda", "integration", (
                "tests.cuda.test_raw_bf16_policy_guard.RawBF16PolicyGuardCUDATests",
            ), False),
        )
        if getattr(args, "multi_gpu", False):
            modules += (
                ("physical-memory-multi-gpu-cuda", "memory", (
                    "tests.compiler.test_policy_routes_physical_memory.PhysicalMemoryMultiGPUCUDATests",
                ), False),
                ("codec-multi-gpu-cuda", "codegen", (
                    "tests.opt.test_bf16_codegen_safety.BF16CodecMultiGPUCUDA",
                ), False),
            )
        if not args.exclude_fp8:
            modules += (("fp8-cuda", "autodiff", (
                "tests.compiler.test_native_autodiff_cuda.NativeAutoDiffCudaTests.test_fp8_native_derived_backward_uses_existing_gpu_kernels",
                "tests.compiler.test_native_fp8_recipes_cuda"), True),)
        for name, area, tests, fp8 in modules:
            report = output / "contracts" / f"{name}.json"
            command = ["benchmarks/compiler/qualification_tests.py", "--modules", *tests, "--device", args.device,
                       "--require-cuda", "--require-triton", "--fail-on-skip", "--output", str(report)]
            if fp8:
                command.append("--require-fp8")
            add(name, area, command, report, "tests_strict")

        if getattr(args, "vae_target", False):
            for policy in ("torch_amp", "bf16_storage"):
                for compressed in (False, True):
                    name = "vae-target-" + policy + "-" + ("compressed" if compressed else "dense")
                    report = output / f"{name}.json"
                    add(name, "integration", ["-m", "examples.vae.benchmark_vae", "--qualify",
                        "--bf16-policy", policy, "--config", "examples/vae/native_bf16.toml", "--qualification-device", args.device,
                        "--qualification-updates", "50", "--qualification-capacity-batches",
                        *map(str, getattr(args, "vae_capacity_batches", [1])),
                        "--qualification-compressed" if compressed else "--no-qualification-compressed",
                        "--optimizer-factory", args.vae_optimizer_factory, "--output", str(report)],
                        report, "vae_target", True)
                    capacity_report = output / (name + "-capacity.json")
                    add(name + "-capacity", "integration", ["-m", "examples.vae.benchmark_vae", "--qualify",
                        "--qualification-capacity-only", "--bf16-policy", policy,
                        "--config", "examples/vae/native_bf16.toml", "--qualification-device", args.device,
                        "--qualification-capacity-batches", *map(str, getattr(args, "vae_capacity_batches", [1])),
                        "--qualification-compressed" if compressed else "--no-qualification-compressed",
                        "--optimizer-factory", args.vae_optimizer_factory, "--output", str(capacity_report)],
                        capacity_report, "vae_capacity", True)
    timer = [value for name, number in timing.items() for value in ("--" + name, str(number))]
    # Role-config controls exercise the new explicit kernel configuration API.
    kernels = output / "kernel-configurations.json"
    if args.profile != "metadata":
        add("role-kernel-benchmarks", "codegen", ["benchmarks/compiler/benchmark_bf16_configurations.py",
            "--device", args.device if gpu else "cpu", "--shapes",
            *(["6304x768x3072", "6304x3072x768"] if representative else ["65x33x37"]),
            "--roles", "forward", "dx", "dw", "--configs", "representative" if gpu else "legacy",
            "--max-configurations", "4" if full else "1", "--bits", "2", "--group-size", "1024",
            *timer, "--min-measure-ms", str(minimum_ms), "--max-iterations", str(args.max_iterations),
            *(["--profile"] if gpu else []), "--output", str(kernels)], kernels, "kernels", gpu)

    # All fixtures use the same generated implementation inside each toggle pair.
    if gpu or args.profile == "cpu":
        for compressed in (False, True):
            directory = output / ("fusion-compressed" if compressed else "fusion-dense")
            command = ["benchmarks/compiler/isolated_fusion.py", "--device", args.device if gpu else "cpu",
                "--fixtures", *(["layernorm-activation", "groupnorm-activation"] if compressed
                                  else ["gemm-activation", "layernorm-activation", "groupnorm-activation"]),
                "--phases", "forward", "backward", "both", "--backends", "triton",
                "--activation-state", "exact",
                "--compression", "on" if compressed else "off", "--bits", *(["1", "2", "4", "8"] if full and compressed else ["2"]),
                "--group-size", "1024", *fixture_size, *fusion_execution,
                "--profile-steps", "1", *timer, "--output", str(directory)]
            if not gpu:
                command.append("--dry-run")
            add("fusion-compressed" if compressed else "fusion-dense", "fusion", command,
                directory / ("fusion-summary.json" if gpu else "fusion-plan.json"), "fusion" if gpu else "plan", gpu)
            if compressed:
                # A compressed Linear forward cannot use the current activation
                # GEMM epilogue. Producer packing/dX fusion need a downstream GEMM.
                for activation in (("silu", "gelu", "gelu-tanh") if full else ("silu",)):
                    extra = output / f"fusion-gemm-compressed-{activation}"
                    gemm_command = list(command)
                    start, end = gemm_command.index("--fixtures") + 1, gemm_command.index("--phases")
                    gemm_command[start:end] = ["gemm-activation"]
                    start, end = gemm_command.index("--phases") + 1, gemm_command.index("--backends")
                    gemm_command[start:end] = ["backward", "both"]
                    gemm_command[gemm_command.index("--output") + 1] = str(extra)
                    gemm_command += ["--activation", activation]
                    add(extra.name, "fusion", gemm_command, extra / ("fusion-summary.json" if gpu else "fusion-plan.json"),
                        "fusion" if gpu else "plan", gpu)
            if full:
                for activation in ("gelu", "gelu-tanh"):
                    extra = output / f"{directory.name}-{activation}"
                    extra_command = [*command[:-2], "--output", str(extra), "--activation", activation]
                    add(extra.name, "fusion", extra_command, extra / "fusion-summary.json", "fusion", True)
        for fixture, phases in (("packed-multiply", ("backward", "both")),
                                ("shared-packs", ("forward", "both"))):
            directory = output / f"fusion-{fixture}"
            add(f"fusion-{fixture}", "fusion", ["benchmarks/compiler/isolated_fusion.py", "--device", args.device if gpu else "cpu",
                "--fixtures", fixture, "--phases", *phases, "--compression", "on", "--bits", "2",
                "--group-size", "1024", "--activation-state", "exact", *fixture_size, *fusion_execution,
                *timer, "--profile-steps", "1", "--output", str(directory),
                *([] if gpu else ["--dry-run"])], directory / ("fusion-summary.json" if gpu else "fusion-plan.json"),
                ("fusion_rejection" if fixture == "shared-packs" else "fusion") if gpu else "plan",
                gpu and fixture != "shared-packs")
        residual = output / "fusion-residual"
        add("fusion-residual", "fusion", ["benchmarks/compiler/isolated_fusion.py", "--device", args.device if gpu else "cpu",
            "--fixtures", "gemm-residual", "--phases", "forward", "both", "--backends", "triton",
            *fixture_size, *fusion_execution, *timer, "--profile-steps", "1", "--output", str(residual), *([] if gpu else ["--dry-run"])],
            residual / ("fusion-summary.json" if gpu else "fusion-plan.json"), "fusion" if gpu else "plan", gpu)
        if full:
            spatial = output / "fusion-groupnorm-amp"
            add("fusion-groupnorm-amp", "fusion", ["benchmarks/compiler/isolated_fusion.py", "--device", args.device,
                "--fixtures", "groupnorm-activation", "--phases", "forward", "backward", "both", "--autocast",
                "--dtype", "fp32", "--layout", "channels-last", "--compression", "on", "--bits", "2",
                "--group-size", "1024", "--activation-state", "exact", *fixture_size, *fusion_execution,
                *timer, "--output", str(spatial)], spatial / "fusion-summary.json", "fusion", True)
    if gpu:
        # Preserve every existing AMP fixture/bit/mask/trial/sample. Storage is
        # an additive recipe axis with its own positive launch evidence.
        for stage in tuple(stages):
            if stage.area != "fusion" or "benchmarks/compiler/isolated_fusion.py" not in stage.command:
                continue
            command = list(stage.command)
            command[command.index("--bf16-policy") + 1] = "bf16_storage"
            directory = Path(stage.report).parent.with_name(Path(stage.report).parent.name + "-bf16-storage")
            command[command.index("--output") + 1] = str(directory)
            add(stage.name + "-bf16-storage", stage.area, command, directory / Path(stage.report).name,
                stage.validator, stage.require_observed)
    if gpu:
        models = output / "model-matrix"
        model_command = ["benchmarks/compiler/benchmark_native_autodiff.py", "--device", args.device,
            "--bf16-policy", "torch_amp",
            "--models", "linear", "mlp", "vit-block", "--batches", *(["8", "32"] if full else ["32"] if representative else ["1"]),
            "--width", "768" if representative else "24", "--hidden", "3072" if representative else "48",
            # Keep the smoke model cases/widths; enlarge the sequence so even
            # batch-one inputs cross the qualification codec's 1024 threshold.
            "--tokens", "197" if representative else "49", "--profile-steps", "1", *timer, "--output", str(models)]
        add("linear-mlp-vit", "models", model_command, models / "comparison.json", "models", True)
        storage_models = models.with_name(models.name + "-bf16-storage")
        storage_command = list(model_command)
        storage_command[storage_command.index("--bf16-policy") + 1] = "bf16_storage"
        storage_command[storage_command.index("--output") + 1] = str(storage_models)
        add("linear-mlp-vit-bf16-storage", "models", storage_command,
            storage_models / "comparison.json", "models", True)
    if args.profile == "metadata" or args.vae_audit:
        for compressed in (False, True):
            report = output / f"vae-local128-{'compressed' if compressed else 'dense'}.json"
            add(f"vae-local128-{'compressed' if compressed else 'dense'}", "memory", ["-m", "examples.vae.audit_activation_memory",
                "--config", "examples/vae/activation_memory_local128.toml", "--output", str(report),
                "--compressed" if compressed else "--no-compressed"], report, "metadata")
    return {"profile": args.profile, "device": args.device, "timing": timing,
            "execution_policy": {"cpu_contracts_requested": any(stage.validator == "tests" for stage in stages),
                "workload": "representative" if representative else "smoke", "fixture_dimensions": dimensions,
                "benchmark_cases_preserved": True,
                "bf16_policies": ("torch_amp", "bf16_storage"),
                "raw_torch_inductor_policy": "torch_amp",
                "multi_gpu_checks_requested": bool(getattr(args, "multi_gpu", False)),
                "multi_gpu_stages": tuple(stage.name for stage in stages if "-multi-gpu-cuda" in stage.name),
                "fusion_worker_isolation": args.fusion_worker_isolation, "gpu_timing_concurrency": 1,
                "phase_iterations": args.phase_iterations, "phase_repeats": args.phase_repeats,
                "kernel_min_measure_ms": minimum_ms, "kernel_max_iterations": args.max_iterations},
            "vae_target_requested": bool(getattr(args, "vae_target", False)),
            "stages": [asdict(stage) for stage in stages], "coverage": [dict(zip(("feature", "area", "evidence"), item)) for item in COVERAGE],
            "limitations": list(NOT_IMPLEMENTED_OR_UNQUALIFIED), "performance_claims_automatic": False}


def validate_report(stage, data):
    """An exit status or a static fusion count alone cannot qualify a stage."""
    kind = stage["validator"]
    if kind in ("vae_target", "vae_capacity") and "--bf16-policy" in stage.get("command", ()):
        command = stage["command"]
        expected_policy = command[command.index("--bf16-policy") + 1]
        expected_compression = "--qualification-compressed" in command
        compression = data.get("compression", {})
        if (data.get("bf16_policy") != {"name": expected_policy, "version": 1}
                or compression.get("enabled") is not expected_compression
                or compression.get("group_size") != 1024 or compression.get("bits") != 2
                or compression.get("norm_xhat_recenter") is not True):
            return False, "vae_policy_or_compression_identity_mismatch"
    if kind == "vae_capacity":
        return (data.get("qualified") is False and data.get("status") == "capacity_measured"
                and data.get("source_stable") is True and data.get("cuda_verified") is True
                and data.get("gate_status") == "not_requested_capacity_only"
                and data.get("capacity", {}).get("batch1_admitted") is True), "candidate_capacity_only"
    if kind == "vae_target":
        return (data.get("qualified") is True and data.get("source_stable") is True
                and data.get("cuda_verified") is True and data.get("original_recipe") is True
                and data.get("geometry") == [1, 3, 224, 224]
                and data.get("same_route_control", {}).get("passed") is True
                and data.get("native_repeat_control", {}).get("passed") is True
                and data.get("numerical_policy") == "strict_storage_dtype_and_relative_l2_with_eager_and_native_repeat_controls"
                and data.get("saved_state", {}).get("passed") is True
                and data.get("odd_geometry", {}).get("passed") is True
                and data.get("trajectory", {}).get("passed") is True
                and (data.get("trajectory", {}).get("resume") or {}).get("passed") is True
                and data.get("lifetime", {}).get("passed") is True
                and len(data.get("trajectory", {}).get("updates", [])) >= 50
                and data.get("capacity", {}).get("batch1_admitted") is True), "vae_target_full_update"
    if kind.startswith("tests"):
        passed = data.get("passed") is True
        if kind == "tests_strict":
            passed &= data.get("qualified") is True and data.get("counts", {}).get("skipped", 1) == 0
        return passed, data.get("status", "invalid_test_report")
    if kind == "plan":
        return bool(data.get("jobs")), "planned_only"
    if kind == "fusion":
        rows = data.get("comparisons", [])
        passed = data.get("correctness_passed") is True and bool(rows)
        if stage["require_observed"]:
            passed &= all(row.get("cuda_fusion_observed") is True for row in rows)
        return passed, "observed" if passed else "failed_or_unobserved_fusion"
    if kind == "fusion_rejection":
        rows = data.get("comparisons", [])
        cse = [row for row in rows if row.get("transformation") == "pack_cse"]
        passed = data.get("correctness_passed") is True and bool(cse) and all(
            row.get("correctness_passed") is True and row.get("status") == "unsupported"
            and row.get("cuda_fusion_observed") is False for row in cse)
        # This model can independently acquire a GEMM/residual epilogue. That
        # is valid kernel fusion, but it cannot certify missing pack sharing.
        for row in rows:
            if row not in cse:
                passed &= row.get("correctness_passed") is True and (
                    row.get("status") == "unsupported" or row.get("cuda_fusion_observed") is True)
        return passed, "expected_unsupported_public_pack_sharing" if passed else "unexpected_pack_sharing_evidence"
    if kind == "models":
        rows = data.get("comparisons", [])
        passed = data.get("passed") is True and bool(rows) and all(row.get("passed") is True for row in rows)
        command = stage.get("command", ())
        if "--bf16-policy" in command:
            requested = command[command.index("--bf16-policy") + 1]
            for row in rows:
                key = row.get("key", ())
                native = (row.get("strategy") == "torch-inductor"
                          or isinstance(key, (tuple, list)) and len(key) > 2 and key[2] == "torch_bf16")
                expected = {"name": "torch_amp" if native else requested, "version": 1}
                passed &= row.get("same_bf16_policy") is True
                for field in ("bf16_policy", "reference_bf16_policy"):
                    receipt = row.get(field, {})
                    policy = receipt.get("policy", {}) if isinstance(receipt, dict) else {}
                    passed &= (isinstance(receipt, dict) and receipt.get("passed") is True
                               and isinstance(policy, dict) and type(policy.get("version")) is int
                               and policy == expected)
        return passed, "model_comparisons" if passed else "failed_or_mismatched_model_comparisons"
    if kind == "metadata":
        return data.get("status") == "ok" and data.get("source_stable") is True and data.get("kind") == "metadata_only", "metadata_only"
    if kind == "kernels":
        workers = data.get("results", [])
        rows = [row for worker in workers for row in worker.get("results", [])]
        passed = data.get("passed") is True and bool(workers) and bool(rows)
        passed &= all(worker.get("status") == "ok" and worker.get("source_unchanged") is True
                      and bool(worker.get("results")) for worker in workers)
        passed &= all(row.get("status") == "ok" for row in rows)
        if stage["require_observed"]:
            generated = [row for row in rows if row.get("backend") != "native"]
            passed &= bool(generated) and all(row.get("execution") == "cuda"
                and row.get("generated_launch_observed") is True
                and (row.get("profile") or {}).get("cuda_kernel_count", 0) > 0 for row in generated)
        return passed, "kernel_controls" if passed else "failed_or_unobserved_kernel"
    return False, "unknown_report_schema"


def export_bundle(destination, root=ROOT):
    """Preserve relative benchmark workers; never bundle DeepOpt source."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = source_snapshot(root)
    files = dict(sources)
    extra = {"README.md": (
        "# Installed DeepOpt qualification\n\n"
        "This bundle contains tests and benchmarks only. Install the current non-editable DeepOpt wheel in your environment.\n\n"
        "Run from this extracted directory:\n\n```sh\n"
        "python benchmarks/compiler/qualify_backend.py --profile cpu --output results/cpu\n"
        "python benchmarks/compiler/qualify_backend.py --profile gpu-smoke --device cuda:0 --output results/smoke\n"
        "python benchmarks/compiler/qualify_backend.py --profile gpu-full --device cuda:0 --vae-audit --output results/full\n"
        "```\n\nSee [the coverage and run guide](docs/execution/backend_qualification.md).\n"
        "The runner records the actual installed version, package location and fingerprint; no checkout library is used.\n"
    ).encode()}
    files.update({name: hashlib.sha256(data).hexdigest() for name, data in extra.items()})
    for name in ("LICENSE", "docs/execution/backend_qualification.md"):
        path = root / name
        if path.exists():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"format": "deepopt-backend-qualification-bundle-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": snapshot_digest(sources), "sources": sources, "files": files,
                "entrypoint": "benchmarks/compiler/qualify_backend.py", "requires": "installed non-editable DeepOpt wheel; PyTorch; Triton for GPU profile",
                "deepopt_source_included": False, "runtime_import_policy": "installed_distribution_only",
                "coverage": [dict(zip(("feature", "area", "evidence"), item)) for item in COVERAGE],
                "qualification_boundaries": list(NOT_IMPLEMENTED_OR_UNQUALIFIED)}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            path = root / name
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in qualification sources: {name}")
            data = extra[name] if name in extra else path.read_bytes()
            if hashlib.sha256(data).hexdigest() != files[name]:
                raise RuntimeError(f"Source changed during export: {name}")
            archive.writestr(name, data)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    if sources != source_snapshot(root):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Source changed during export")
    temporary.replace(destination)
    return {"archive": str(destination), "files": len(files) + 1, "source_sha256": manifest["source_sha256"],
            "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=PROFILES, default="cpu")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--output", type=Path, default=Path("work/backend-qualification"))
    result.add_argument("--only", nargs="+", choices=AREAS)
    result.add_argument("--exclude-fp8", action="store_true", help="Explicitly leave FP8 GPU qualification unrequested")
    result.add_argument("--multi-gpu", action="store_true",
                        help="Also require the separate two-CUDA-device codec and physical-memory tests; never enabled by a single-GPU profile")
    result.add_argument("--vae-audit", action="store_true", help="Also run both exact local128 metadata audits; no real payload allocation")
    result.add_argument("--vae-target", action="store_true", help="Require original 224-square custom-optimizer VAE trajectories and measured capacity")
    result.add_argument("--vae-capacity-batches", nargs="+", type=int, default=[1], help="Increasing physical batches to measure; must start at 1")
    result.add_argument("--vae-optimizer-factory", default="source.training.geo_optimizer:SGD_NS_Overshoot")
    result.add_argument("--workload", choices=("smoke", "representative"),
                        help="Full GPU defaults to representative shapes; smoke defaults to small correctness cases")
    result.add_argument("--fusion-worker-isolation", choices=("trial", "strategy"), default="trial",
                        help="One fresh process per randomized fixture/trial pair, or stricter per-strategy isolation")
    result.add_argument("--phase-iterations", type=int, default=3,
                        help="Synchronized fusion phase diagnostic samples; zero disables this diagnostic")
    result.add_argument("--phase-repeats", type=int, default=1)
    result.add_argument("--min-measure-ms", type=float,
                        help="Total CUDA event window for role kernels; defaults to 50ms smoke/200ms full; zero disables adaptation")
    result.add_argument("--max-iterations", type=int, default=10000,
                        help="Upper bound for adaptive role-kernel measurement iterations per repeat")
    result.add_argument("--progress-interval", type=float, default=15,
                        help="Seconds between stage/worker phase heartbeats")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--export", type=Path, help="Write an installed-library-only test/benchmark ZIP without running stages")
    result.add_argument("--evidence-export", type=Path, help="After running, export complete scalar evidence to a fresh ZIP outside the report directory")
    result.add_argument("--stage-timeout", type=float, default=21600)
    for name in ("trials", "warmup", "iterations", "repeats"):
        result.add_argument("--" + name, type=int)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 0 < args.stage_timeout < float("inf") or any(getattr(args, name) is not None and getattr(args, name) < (0 if name == "warmup" else 1)
                                     for name in ("trials", "warmup", "iterations", "repeats")):
        arguments.error("Timeout/counts must be positive; warmup may be zero")
    requested_iterations = args.iterations if args.iterations is not None else (50 if args.profile == "gpu-full" else 2)
    if (args.phase_iterations < 0 or args.phase_repeats < 1 or args.max_iterations < requested_iterations
            or not 0 < args.progress_interval < float("inf")
            or args.min_measure_ms is not None and not 0 <= args.min_measure_ms < float("inf")):
        arguments.error("Phase counts/minimum duration must be nonnegative, repeats/progress positive, and max iterations at least the requested timing count")
    if args.export:
        print(json.dumps(export_bundle(args.export), indent=2))
        return 0
    run_id = uuid.uuid4().hex
    plan_arguments = argparse.Namespace(**vars(args))
    if not args.dry_run:
        plan_arguments.output = args.output / "runs" / run_id
    try:
        plan = make_plan(plan_arguments)
    except ValueError as error:
        arguments.error(str(error))
    if not plan["stages"]:
        arguments.error("Selected profile/areas contain no stages")
    output = args.output.resolve()
    if args.dry_run:
        write_json(output / "qualification-plan.json", plan)
        print(json.dumps({"status": "planned_only", "stages": len(plan["stages"]), "output": str(output)}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if (output / "qualification-summary.json").exists():
        arguments.error("Output already contains a run; use a fresh output directory")
    summary = {"run_id": run_id, "plan": plan, "status": "running", "stages": [],
               "started_utc": datetime.now(timezone.utc).isoformat(), "gpu_performance_qualified": False}
    try:
        if args.profile.startswith("gpu"):
            if not args.device.startswith("cuda"):
                raise ValueError("GPU profiles require an explicit CUDA device")
            sys.path.insert(0, str(ROOT / "benchmarks"))
            from _gpu_lease import ensure
            leased = ensure(__file__, argv, args.device)
            if leased is not None:
                return leased
        before = source_snapshot()
        bundled = ROOT / "bundle-manifest.json"
        if bundled.exists() and json.loads(bundled.read_text())["sources"] != before:
            raise RuntimeError("Bundle sources differ from the exported manifest; export a fresh bundle")
        sys.path.insert(0, str(ROOT / "benchmarks"))
        from compiler.installed_runtime import installed_identity, child_environment

        installed = installed_identity(ROOT)
        import torch
        summary["environment"] = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                                  "deepopt_path": installed["import_path"], "installed_deepopt": installed,
                                  "source_sha256": snapshot_digest(before)}
        write_json(output / "qualification-plan.json", plan)
        environment = child_environment(ROOT, identity=installed)
        from analysis.scheduler import run_command

        # Reuse the existing whole-process-group cleanup and timeout protocol.
        # Workers use repository-relative configuration and helper paths.
        os.chdir(ROOT)
        for index, stage in enumerate(plan["stages"], 1):
            log = output / "logs" / f"{stage['name']}.log"
            log.parent.mkdir(exist_ok=True)
            began = time.perf_counter()
            try:
                stage_environment = dict(environment)
                if stage["validator"] == "tests":
                    # Mixed CPU/CUDA test modules must remain CPU-only here;
                    # their requested device tests run in separate strict stages.
                    stage_environment["CUDA_VISIBLE_DEVICES"] = ""
                stage_environment["PYTHONUNBUFFERED"] = "1"
                with log.open("w") as stream, report_stage_progress(summary, stage, log, output,
                        index, len(plan["stages"]), args.progress_interval):
                    exit_code = run_command([sys.executable, *stage["command"]], env=stage_environment, log=stream,
                        timeout=args.stage_timeout, cancel_file=output / "cancel")
                report = Path(stage["report"])
                data = json.loads(report.read_text()) if report.is_file() else {}
                passed, evidence = validate_report(stage, data)
                passed &= exit_code == 0
                record = {"name": stage["name"], "area": stage["area"], "passed": passed,
                          "exit_code": exit_code, "evidence": evidence, "counts": data.get("counts"),
                          "report": str(report), "log": str(log)}
            except Exception as error:
                record = {"name": stage["name"], "area": stage["area"], "passed": False,
                          "reason": f"{type(error).__name__}: {error}", "log": str(log)}
                if getattr(error, "_deepopt_cleanup_failed", False):
                    record["cleanup_uncertain"] = True
            record["seconds"] = time.perf_counter() - began
            summary["stages"].append(record)
            write_json(output / "qualification-summary.json", summary)
            print(f"[{index}/{len(plan['stages'])}] {stage['name']}: {'passed' if record['passed'] else 'failed'} ({record['seconds']:.1f}s)", flush=True)
            if record.get("cleanup_uncertain"):
                raise RuntimeError("Owned worker cleanup is uncertain; aborting remaining stages")
        summary["source_unchanged"] = before == source_snapshot()
        summary["installed_unchanged"] = installed == installed_identity(ROOT)
        summary["passed"] = summary["source_unchanged"] and summary["installed_unchanged"] and all(row["passed"] for row in summary["stages"])
        summary["status"] = "passed" if summary["passed"] else "failed"
        summary["qualification_scope"] = ("gpu_execution_checks" if args.profile.startswith("gpu")
                                          else "metadata_only" if args.profile == "metadata" else "cpu_contracts")
        summary["cpu_contracts_requested"] = plan["execution_policy"]["cpu_contracts_requested"]
        summary["cpu_contracts_passed"] = (all(row["passed"] for row in summary["stages"] if row["name"].endswith("-contracts"))
                                           if summary["cpu_contracts_requested"] else None)
        summary["gpu_execution_checks_passed"] = args.profile.startswith("gpu") and summary["passed"]
        requested_multi_gpu = plan["execution_policy"]["multi_gpu_stages"]
        multi_gpu_rows = [row for row in summary["stages"] if row["name"] in requested_multi_gpu]
        summary["multi_gpu_checks_requested"] = plan["execution_policy"]["multi_gpu_checks_requested"]
        summary["multi_gpu_checks_passed"] = (len(multi_gpu_rows) == len(requested_multi_gpu)
            and all(row["passed"] for row in multi_gpu_rows)) if requested_multi_gpu else None
        target_rows = [row for row in summary["stages"] if row["name"].startswith("vae-target-")
                       and not row["name"].endswith("-capacity")]
        summary["vae_capacity_probes_passed"] = all(row["passed"] for row in summary["stages"]
            if row["name"].startswith("vae-target-") and row["name"].endswith("-capacity")) if target_rows else None
        summary["vae_target_qualified"] = (len(target_rows) == 4 and all(row["passed"] for row in target_rows))
        summary["vae_target_qualification_scope"] = "original224_batch1_numerics_and_lifetime;batch128_capacity_and_allocated_target_are_separate"
    except KeyboardInterrupt:
        summary.update(status="interrupted", passed=False, reason="Qualification interrupted; worker group cleanup requested")
    except Exception as error:
        summary.update(status="preflight_failed", passed=False, reason=f"{type(error).__name__}: {error}")
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "qualification-summary.json", summary)
    exported = None
    if args.evidence_export:
        try:
            from compiler.export_evidence import export_evidence
            exported = export_evidence(output, args.evidence_export)
        except Exception as error:
            print(json.dumps({"status": "evidence_export_failed", "error": f"{type(error).__name__}: {error}",
                              "summary": str(output / "qualification-summary.json")}, indent=2))
            return 1
    print(json.dumps({"status": summary["status"], "summary": str(output / "qualification-summary.json"),
                      "evidence_export": exported}, indent=2))
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
