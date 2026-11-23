"""Real distributed contracts for native AD and scheduler-owned packed saves.

The CPU transport case is an explicit testing backend below the public runtime
policy. It never qualifies pinned memory, CUDA copies, NCCL, or CPU offload.
"""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("TORCHDYNAMO_REPRO_AFTER", None)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F

import deepopt
from deepopt import Trainer, TrainingRuntime, initialize_distributed, shutdown_distributed
from deepopt.compiler import compilation_report
from deepopt.compiler.runtime.manager import (close_packed_offload_runtime,
    drain_packed_offload_runtime, freeze_packed_offload_profiles, packed_offload_runtime_snapshot)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--case", choices=("suite", "native", "transport", "preflight", "forward_error"),
                        default="suite")
    result.add_argument("--require-cuda", action="store_true")
    result.add_argument("--allow-source", action="store_true", help="CPU source integration only")
    return result


def _identity(allow_source):
    if not allow_source:
        from benchmarks.compiler.installed_runtime import installed_identity

        return installed_identity(ROOT, full=True)
    directory = Path(deepopt.__file__).resolve().parent
    digest, count = hashlib.sha256(), 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
        count += 1
    return {"import_path": str(Path(deepopt.__file__).resolve()), "package_path": str(directory),
            "source_sha256": digest.hexdigest(), "files_hashed": count,
            "source_testing_only": True, "editable": None}


def _policy(mode="demand", *, enabled=True):
    return dict(enabled=enabled, policy=mode, version=2, host_pinned_bytes=16 << 20,
                copy_chunk_bytes=1 << 20, admission="error", max_live_forwards=1,
                max_backward_tasks=1)


def _configuration(device, mode=None, bf16_policy=None):
    cuda = device.type == "cuda"
    runtime = dict(device=str(device), precision="bf16" if cuda else "fp32",
        bf16_policy=bf16_policy or ("bf16_storage" if cuda else "torch_amp"), checks=False,
        execution_mode="max_speed", deterministic=True,
        activation_compression=dict(enabled=cuda, bits=2, group_size=1024),
        activation_memory=dict(multiply_backward="packed_streaming"),
        ddp=dict(gradient_as_bucket_view=True, broadcast_buffers=False, bucket_cap_mb=.0001),
        compile=dict(enabled=True, backend="deepopt", fullgraph=True, dynamic=False,
            options=dict(use_native_autodiff=True, require_full_aot=True,
                         share_compatible_packs=True, packed_offload_selection_mode="all_eligible")))
    if mode is not None:
        runtime["packed_activation_offload"] = _policy(mode)
    return dict(training=dict(epochs=2, batch_size=2, learning_rate=.001,
                              accumulation_steps=2, monitor=None),
                runtime=runtime, logging=dict(level="ERROR"))


def _model(device):
    torch.manual_seed(9721)
    if device.type == "cpu":
        # Transparent FP32 operations expose real DDPOptimizer partitions.
        # Packed BF16 CPU saved ownership is tested separately below.
        return nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 16))
    return nn.Sequential(nn.Conv2d(4, 8, 1), nn.SiLU(), nn.Conv2d(8, 4, 1)).to(device)


def _mse(output, target):
    return F.mse_loss(output.float(), target)


def _batch(device, rank, index=0):
    source = (torch.arange(64, dtype=torch.float32).reshape(2, 32) if device.type == "cpu"
              else torch.arange(120, dtype=torch.float32).reshape(2, 4, 3, 5))
    source = (source - 31.17) / 47.3 + rank * .071 + index * .113
    target = (torch.arange(32, dtype=torch.float32).reshape(2, 16) if device.type == "cpu"
              else torch.arange(120, dtype=torch.float32).reshape(2, 4, 3, 5))
    target = torch.sin(target * .193 + index * .21 + rank * .31) * .173
    return source.to(device), target.to(device)


def _compare(actual, expected, role, comparisons):
    difference = actual.detach().double() - expected.detach().double()
    norm = torch.linalg.vector_norm(expected.detach().double())
    comparisons.append(dict(role=role, dtype=str(actual.dtype), shape=list(actual.shape),
        bitwise_equal=torch.equal(actual, expected), max_abs=difference.abs().max().item(),
        relative_l2=(torch.linalg.vector_norm(difference) / norm.clamp_min(1e-300)).item(),
        rtol=0, atol=0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=role)


def _equal_state(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _equal_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal_state(left, right)
    else:
        assert actual == expected, (actual, expected)


def _bucket_manifest(wrapper, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    buckets = wrapper.reducer._get_zeros_like_grad_buckets()
    manifest = []
    for bucket in buckets:
        manifest.append(dict(index=bucket.index(), bytes=bucket.buffer().numel() * bucket.buffer().element_size(),
            dtype=str(bucket.buffer().dtype), parameters=[dict(name=names[id(parameter)],
                shape=list(gradient.shape), stride=list(gradient.stride()), offset=gradient.storage_offset())
                for parameter, gradient in zip(bucket.parameters(), bucket.gradients())]))
    return buckets, manifest


def _reduce_reference_buckets(wrapper, model, reference, context):
    """Independent source values, matching the reducer's collective geometry.

    The zero buffers and their views describe only message ABI. No candidate
    gradient value is read. Predivision follows Torch's default averaging hook.
    """
    expected = {id(actual): target for actual, target in zip(model.parameters(), reference.parameters())}
    buckets, manifest = _bucket_manifest(wrapper, model)
    manifests = [None] * context.world_size
    dist.all_gather_object(manifests, manifest)
    assert all(value == manifest for value in manifests)
    covered = []
    for bucket in buckets:
        for parameter, view in zip(bucket.parameters(), bucket.gradients()):
            gradient = expected[id(parameter)].grad
            assert gradient is not None and view.dtype == gradient.dtype == torch.float32
            view.copy_(gradient)
            covered.append(id(parameter))
        bucket.buffer().div_(context.world_size)
        dist.all_reduce(bucket.buffer())
        for parameter, view in zip(bucket.parameters(), bucket.gradients()):
            expected[id(parameter)].grad.copy_(view)
    assert len(covered) == len(expected) and set(covered) == set(expected)
    return manifest


def _assert_parameter_gradients(model, reference, context, comparisons, role, *, wrapper=None):
    views = 0
    if context.world_size > 2:
        assert wrapper is not None
        manifest = _reduce_reference_buckets(wrapper, model, reference, context)
        comparisons.append(dict(role=f"{role}.independent_bucket_collective_ABI", world_size=context.world_size,
            same_sized_zero_bucket_views_filled_only_from_reference_gradients=True,
            default_average_order="divide_then_all_reduce", buckets=manifest))
    for (name, parameter), expected in zip(model.named_parameters(), reference.parameters()):
        assert parameter.grad is not None and expected.grad is not None, name
        assert parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr(), name
        if context.world_size == 2:
            dist.all_reduce(expected.grad)
            expected.grad.div_(context.world_size)
        _compare(parameter.grad, expected.grad, f"{role}.{name}", comparisons)
        views += parameter.grad._base is not None
    return views


def _assert_rank_parameters(model, context):
    vector = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    other = [torch.empty_like(vector) for _ in range(context.world_size)]
    dist.all_gather(other, vector)
    for value in other:
        torch.testing.assert_close(value, vector, rtol=0, atol=0)


def _native_evidence(model):
    report = compilation_report(model).as_dict()
    assert report["captured_graphs"] > 0
    assert report["native_autodiff_operations"] > 0
    assert report["native_autodiff_regions"] > 0
    assert report["runtime_fallbacks"] == 0
    assert report["legacy_backward_operations"] == []
    settings = model._deepopt_compile_config
    assert settings["fullgraph"] and settings["options"]["require_full_aot"]
    assert settings["options"]["use_native_autodiff"]
    return {key: report[key] for key in ("captured_graphs", "native_autodiff_operations",
        "native_autodiff_regions", "runtime_fallbacks", "legacy_backward_operations")}


def _reference(model, config):
    from deepopt.engine.execution.configuration import prepare_execution_model

    resident = deepcopy(config)
    resident["runtime"].pop("ddp", None)
    resident["runtime"].pop("packed_activation_offload", None)
    return prepare_execution_model(model, resident)[0]


def _reference_context(device, bf16_policy=None):
    return deepopt.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16,
        fp8=False, check_finite=False,
        bf16_policy=bf16_policy or ("bf16_storage" if device.type == "cuda" else "torch_amp"),
        activation_compression=dict(enabled=True, bits=2, group_size=1024),
        activation_memory=dict(multiply_backward="packed_streaming"))


def native_training(context, mode=None, bf16_policy=None):
    config = _configuration(context.device, mode, bf16_policy)
    model = _model(context.device)
    reference = _reference(deepcopy(model), config)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.001, momentum=.9)
    trainer = Trainer(model, optimizer, loss_fn=_mse, config=deepcopy(config))
    runtime, comparisons = trainer.runtime, []
    parameters = tuple(model.parameters())
    bucket_views = 0
    try:
        # The first synchronized update establishes DDP bucket views.
        batch = _batch(context.device, context.global_rank)
        loss = runtime.compute(batch, lambda value: _mse(model(value[0]), value[1]))
        with _reference_context(context.device, bf16_policy):
            expected = _mse(reference(batch[0]), batch[1])
        _compare(loss, expected, "sync.loss", comparisons)
        runtime.backward(loss)
        expected.backward()
        bucket_views += _assert_parameter_gradients(model, reference, context, comparisons, "sync.grad",
                                                    wrapper=runtime.core.wrapper)
        runtime.step()
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        del loss, expected
        for index, (actual, expected) in enumerate(zip(model.parameters(), reference.parameters())):
            _compare(actual, expected, f"sync.parameter.{index}", comparisons)

        # Retained backward is confined to a valid no_sync window; a fresh
        # synchronized forward then reduces the accumulated contributions.
        with runtime.core.accumulation_context(synchronize=False):
            batch = _batch(context.device, context.global_rank, 1)
            loss = runtime.compute(batch, lambda value: _mse(model(value[0]), value[1]) / 3)
            with _reference_context(context.device, bf16_policy):
                expected = _mse(reference(batch[0]), batch[1]) / 3
            runtime.backward(loss, retain_graph=True)
            runtime.backward(loss)
            expected.backward(retain_graph=True)
            expected.backward()
            del loss, expected
        batch = _batch(context.device, context.global_rank, 2)
        loss = runtime.compute(batch, lambda value: _mse(model(value[0]), value[1]) / 3)
        with _reference_context(context.device, bf16_policy):
            expected = _mse(reference(batch[0]), batch[1]) / 3
        runtime.backward(loss)
        expected.backward()
        bucket_views += _assert_parameter_gradients(model, reference, context, comparisons, "accumulated.grad",
                                                    wrapper=runtime.core.wrapper)
        runtime.step()
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        del loss, expected
        for index, (actual, expected) in enumerate(zip(model.parameters(), reference.parameters())):
            _compare(actual, expected, f"accumulated.parameter.{index}", comparisons)
        assert all(a is b for a, b in zip(parameters, model.parameters()))
        assert bucket_views > 0
        _assert_rank_parameters(model, context)
        native = _native_evidence(model)
        scope_checks, compiler_scope = {}, None
        if mode is None:
            assert native["captured_graphs"] >= 2, "Dynamo DDPOptimizer must actually split this model"
            scope_checks["ddp_optimizer_graph_split"] = True
        else:
            assert native["captured_graphs"] == 1, "Offload must preserve whole-model selection and pre-admission"
            compiler_scope = runtime.core._packed_offload_preparation["compiler_scope"]
            assert compiler_scope["ddp_optimizer_partitioning"] == "disabled"
            assert compiler_scope["communication_overlap_qualified"] is False
            scope_checks["whole_model_offload_plan_preserved"] = True
        buckets, bucket_metadata = _bucket_manifest(runtime.core.wrapper, model)
        assert len(buckets) >= 2, "The tiny-bucket fixture must exercise split DDP graph compilation"
        del buckets
        drain_packed_offload_runtime(runtime)
        snapshot = freeze_packed_offload_profiles()
        transport_owner = None
        if mode is not None:
            transport_owner = _assert_transport(snapshot, runtime)
            scope_checks["one_training_runtime_transport_owner_per_rank"] = True
        return dict(name=f"native_{mode or 'resident'}" + ("_torch_amp" if bf16_policy == "torch_amp" else ""),
            checks=dict(native_fullgraph=True,
            native_autodiff=True, fp32_parameter_gradients=True, exact_manual_all_reduce_oracle=True,
            no_sync_accumulation=True, retained_backward_in_no_sync=True, gradient_bucket_views=True,
            parameter_identity=True, rank_parameters_equal=True, multiple_ddp_buckets=True,
            **scope_checks), compiler=native, compiler_scope=compiler_scope,
            bucket_abi=bucket_metadata, bf16_policy=config["runtime"]["bf16_policy"],
            transport_owner=transport_owner, preparation=runtime.core._packed_offload_preparation,
            updates=2, comparisons=comparisons, runtime=snapshot,
            test_transport=False, cuda_execution_verified=context.device.type == "cuda")
    finally:
        close_packed_offload_runtime(runtime)


_CURRENT_LIFECYCLE = ("live_forwards", "regional_frames", "backward_tasks", "graph_leases", "task_leases", "dma_leases")
_CURRENT_DEVICE = ("staging_bytes", "pending_retire_bytes", "d2h_inflight_bytes")
_IDLE_HOST = ("pinned_active_bytes", "pinned_cached_bytes", "pinned_peak_bytes",
              "pageable_active_bytes", "pageable_cached_bytes", "pageable_peak_bytes")
_PHYSICAL_ZERO = ("pinned_physical_active_bytes", "pinned_physical_pending_retirement_bytes",
                  "pinned_physical_unowned_retired_bytes", "pinned_cleanup_failures",
                  "pinned_omitted_arenas", "pinned_omitted_spans")
_SHARED_PHYSICAL = ("pinned_allocator_domain_id", "pinned_physical_capacity_bytes", "pinned_physical_peak_bytes",
    "pinned_physical_active_bytes", "pinned_physical_pending_retirement_bytes", "pinned_physical_cached_bytes",
    "pinned_physical_unowned_retired_bytes", "pinned_accounting_verified", "pinned_accounting",
    "pinned_cleanup_failures", "pinned_omitted_arenas", "pinned_omitted_spans", "pinned_cached_bytes")


def _zero_fields(value, names):
    return isinstance(value, dict) and all(type(value.get(name)) is int and value[name] == 0 for name in names)


def _zero_current_manager(manager):
    lifecycle, host, device = (manager.get(name, {}) for name in ("lifecycle", "host", "device"))
    return (_zero_fields(lifecycle, _CURRENT_LIFECYCLE) and lifecycle.get("quiescent") is True
        and _zero_fields(device, _CURRENT_DEVICE)
        and _zero_fields(host, ("pinned_active_bytes", "pageable_active_bytes", "pageable_cached_bytes"))
        and not manager.get("error") and not manager.get("cleanup_errors"))


def _audit_transport_managers(snapshot, owned):
    """Count live owners; retain complete proof for idle and closed histories.

    V2 repositories retain charged cached arenas across closed managers. Only
    exact physical-domain matches may deduplicate that shared capacity.
    """
    own_host = owned["host"]
    assert type(own_host.get("pinned_allocator_domain_id")) is int
    active, idle, closed_zero, closed_shared = [], 0, 0, 0
    for manager in snapshot["managers"]:
        life, host = manager.get("lifecycle", {}), manager.get("host", {})
        assert manager.get("profile_frozen") is True and manager.get("profile_learning") is False
        if life.get("closed") is True:
            assert _zero_current_manager(manager), "Closed history retains current resources or leases"
            if set(host) == set(_IDLE_HOST) and _zero_fields(host, _IDLE_HOST):
                closed_zero += 1
                continue
            assert host.get("pinned_accounting_verified") is True, "Closed initialized owner lacks physical proof"
            assert _zero_fields(host, _PHYSICAL_ZERO), "Closed initialized owner retains physical bindings"
            if _zero_fields(host, ("pinned_cached_bytes", "pinned_physical_capacity_bytes", "pinned_physical_cached_bytes")):
                closed_zero += 1
                continue
            assert all(name in host and name in own_host and host[name] == own_host[name]
                       for name in _SHARED_PHYSICAL), "Closed cached capacity is not the actual owner's exact shared domain"
            closed_shared += 1
        elif manager.get("plans"):
            active.append(manager)
        else:
            assert (_zero_current_manager(manager) and set(host) == set(_IDLE_HOST)
                and _zero_fields(host, _IDLE_HOST)
                and _zero_fields(manager.get("device"), (*_CURRENT_DEVICE, "staging_peak_bytes", "peak_d2h_inflight_bytes"))
                and not manager.get("counters") and not manager.get("events") and not manager.get("profiles")
                and not manager.get("admission_counts") and not manager.get("admission_history")
                and manager.get("admission_history_truncated") is False), "Unused facade is not a complete frozen zero-resource schema"
            idle += 1
    assert len(active) == 1, "Expected one open initialized transport owner per rank"
    assert active[0]["host"]["pinned_allocator_domain_id"] == own_host["pinned_allocator_domain_id"]
    assert active[0]["counters"] == owned["counters"]
    return active[0], dict(unused_zero_resource_facades=idle, closed_zero_resource_histories=closed_zero,
                          closed_shared_allocator_histories=closed_shared)


def _assert_transport(snapshot, owner):
    from deepopt.compiler.runtime.manager import current_runtime_manager

    with owner._packed_offload_scope():
        owned = current_runtime_manager(owner._packed_offload_policy).snapshot()
    assert owned["plans"]
    manager, audit = _audit_transport_managers(snapshot, owned)
    assert manager["testing_backend"] is False
    assert manager["host"]["pinned_accounting_verified"] is True
    assert manager["host"]["pinned_physical_peak_bytes"] <= manager["policy"]["host_pinned_bytes"]
    assert manager["counters"].get("d2h_bytes", 0) > 0
    assert manager["counters"].get("h2d_bytes", 0) > 0
    assert manager["lifecycle"]["quiescent"] is True
    limits = manager["transport_limits"]
    assert limits["device_staging_bytes"] == max(row["required_staging_bytes"] for row in manager["plans"])
    assert limits["d2h_inflight_bytes"] == max(row["required_d2h_inflight_bytes"] for row in manager["plans"])
    assert manager["device"]["staging_peak_bytes"] <= limits["device_staging_bytes"]
    assert manager["device"]["peak_d2h_inflight_bytes"] <= limits["d2h_inflight_bytes"]
    return dict(active_transport_managers=1, actual_training_runtime_owner=True,
        pinned_allocator_domain_id=owned["host"]["pinned_allocator_domain_id"],
        selected_native_regions=len(owned["plans"]), cuda_backend=True, budgets_verified=True, **audit)


def checkpoint_roundtrip(context, mode=None, bf16_policy=None):
    from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint

    config = _configuration(context.device, mode, bf16_policy)
    # Different rank inputs are covered by the independent gradient oracle.
    # Keep this checkpoint fixture identical across ranks so fresh-vs-rebuilt
    # DDP bucket message geometry cannot change the FP32 summation order.
    batches = [_batch(context.device, 0, index) for index in (3, 4)]
    temporary = tempfile.TemporaryDirectory(prefix="deepopt-ddp-packed-checkpoint-") if context.is_main_process else None
    shared = [temporary.name if temporary is not None else None]
    dist.broadcast_object_list(shared, src=0)
    path = Path(shared[0]) / "state.pt"
    trainer = restored = None
    try:
        def components():
            model = _model(context.device)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
            return Trainer(model, optimizer, loss_fn=_mse, config=deepcopy(config))

        trainer = components()
        torch.manual_seed(20261121 + context.global_rank)
        trainer.fit(batches, epochs=1, progress_bar=False)
        assert trainer.state.optimizer_steps == 1
        save_checkpoint(path, trainer, generators={}, history=[{}])
        saved = read_checkpoint(path)
        assert saved["distributed"]["world_size"] == context.world_size
        assert len(saved["distributed"]["ranks"]) == context.world_size
        trainer.fit(batches, epochs=2, progress_bar=False)
        expected_model = deepcopy(trainer.model.state_dict())
        expected_optimizer = deepcopy(trainer.optimizer.state_dict())
        evidence = _native_evidence(trainer.model)
        close_packed_offload_runtime(trainer.runtime)
        trainer = None
        gc.collect()
        restored = components()
        restore_checkpoint(saved, restored, generators={}, batch_size_mode="manual")
        assert restored.state.optimizer_steps == 1
        saved_rank_rng = saved["distributed"]["ranks"][context.global_rank]["rng"]["torch"]
        assert torch.equal(torch.get_rng_state(), saved_rank_rng)
        cuda_rng_hash = None
        if context.device.type == "cuda":
            saved_cuda_rng = saved["distributed"]["ranks"][context.global_rank]["rng"]["cuda"]
            assert len(saved_cuda_rng) == 1
            assert torch.equal(torch.cuda.get_rng_state(context.device), saved_cuda_rng[0])
            cuda_rng_hash = hashlib.sha256(saved_cuda_rng[0].numpy().tobytes()).hexdigest()
        for name, value in restored.model.state_dict().items():
            _equal_state(value, saved["model"][name].to(value.device) if isinstance(value, torch.Tensor)
                         else saved["model"][name])
        restored.fit(batches, epochs=2, progress_bar=False)
        assert restored.state.optimizer_steps == 2
        _equal_state(restored.model.state_dict(), expected_model)
        _equal_state(restored.optimizer.state_dict(), expected_optimizer)
        _assert_rank_parameters(restored.model, context)
        _native_evidence(restored.model)
        drain_packed_offload_runtime(restored.runtime)
        snapshot = freeze_packed_offload_profiles()
        transport_owner = None
        if mode is not None:
            transport_owner = _assert_transport(snapshot, restored.runtime)
        return dict(name=f"checkpoint_{mode or 'resident'}" + ("_torch_amp" if bf16_policy == "torch_amp" else ""),
            checks=dict(checkpoint_restore=True,
            checkpoint_next_update=True, exact_optimizer_state=True, rank_local_checkpoint_state=True,
            rank_local_rng_restored=True,
            **({"rank_local_cuda_rng_restored": True} if cuda_rng_hash is not None else {}),
            native_fullgraph=True, native_autodiff=True, rank_parameters_equal=True),
            optimizer_steps=2, compiler=evidence, runtime=snapshot,
            transport_owner=transport_owner, preparation=restored.runtime.core._packed_offload_preparation,
            checkpoint_input_scope="identical_rank_inputs_isolate_restore_from_bucket_rebuild_rounding",
            rank_local_rng_sha256=hashlib.sha256(saved_rank_rng.numpy().tobytes()).hexdigest(),
            rank_local_cuda_rng_sha256=cuda_rng_hash,
            test_transport=False, cuda_execution_verified=context.device.type == "cuda")
    finally:
        if trainer is not None:
            close_packed_offload_runtime(trainer.runtime)
        if restored is not None:
            close_packed_offload_runtime(restored.runtime)
        if temporary is not None:
            temporary.cleanup()


def generated_transport(context):
    from deepopt.compiler.autodiff.joint import JointProgram, _JointFunction
    from deepopt.compiler.registry import KernelCodegenContext
    from tests.compiler.test_compiled_packed_offload import _cpu_fixture

    assert context.device.type == "cpu", "Testing transport must never satisfy a CUDA gate"
    torch.manual_seed(3191)
    source, region, resident, plan, manager = _cpu_fixture("demand")

    class Region(SimpleNamespace):
        __call__ = JointProgram.__call__

    target = Region(**vars(region))
    target.context = KernelCodegenContext(packed_activation_offload=plan.policy)
    target.offload_runtime, target.offload_plan, target.specifications = manager, plan, []

    class Model(nn.Module):
        def __init__(self, selected):
            super().__init__()
            self.weight = nn.Parameter(source.detach().float().clone())
            self.selected = selected

        def forward(self, value):
            activation = (self.weight + value).to(torch.bfloat16)
            return target(activation) if self.selected else _JointFunction.apply(resident, activation)[0]

    model, reference = Model(True), Model(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    runtime = TrainingRuntime(model, optimizer, device="cpu", checks=False, execution_mode="max_speed")
    comparisons, views = [], 0
    try:
        value = torch.full(source.shape, .137 + context.global_rank * .031)
        with runtime.core.accumulation_context(synchronize=False):
            loss = runtime.compute(value, lambda part: model(part).float().square().mean() / 3)
            expected = reference(value).float().square().mean() / 3
            runtime.backward(loss, retain_graph=True)
            runtime.backward(loss)
            expected.backward(retain_graph=True)
            expected.backward()
            del loss, expected
        loss = runtime.compute(value * .7, lambda part: model(part).float().square().mean() / 3)
        expected = reference(value * .7).float().square().mean() / 3
        runtime.backward(loss)
        expected.backward()
        views += _assert_parameter_gradients(model, reference, context, comparisons, "testing_transport.grad",
                                             wrapper=runtime.core.wrapper)
        assert views > 0
        runtime.step()
        del loss, expected
        gc.collect()
        manager.drain()
        state = manager.snapshot()
        assert state["testing_backend"] is True
        assert state["counters"]["d2h_bytes"] > 0 and state["counters"]["h2d_bytes"] > 0
        assert state["host"]["pinned_active_bytes"] == 0 and state["lifecycle"]["quiescent"]
        return dict(name="generated_testing_transport", checks=dict(generated_joint_saved_abi=True,
            exact_manual_all_reduce_oracle=True, no_sync_accumulation=True,
            retained_backward_in_no_sync=True, gradient_bucket_views=True, graph_leases_released=True),
            comparisons=comparisons, runtime=state, test_transport=True,
            cuda_execution_verified=False, generated_source_sha256={
                "forward": hashlib.sha256(region.forward_artifact.source.encode()).hexdigest(),
                "backward": hashlib.sha256(region.backward_artifact.source.encode()).hexdigest()})
    finally:
        manager.close()


def preflight(context):
    from deepopt.distributed.offload import prepare_packed_offload

    recipe = dict(native_autodiff=True, fullgraph=True, group_size=1024, bits=2,
                  precision="bf16" if context.device.type == "cuda" else "fp32")
    disabled = prepare_packed_offload(_policy(enabled=False), context, recipe=recipe)
    mismatch = _policy("demand" if context.global_rank == 0 else "memory_first_overlap")
    mismatch_error = None
    try:
        prepare_packed_offload(mismatch, context, recipe=recipe)
    except (ValueError, RuntimeError) as error:
        mismatch_error = dict(class_name=type(error).__name__, message=str(error))
    errors = [None] * context.world_size
    dist.all_gather_object(errors, mismatch_error)
    assert all(errors), errors
    checks = dict(policy_mismatch_cooperative=True, collective_usable_after_rejection=True)
    assert all("matching" in value["message"] for value in errors), errors
    result = dict(name="preflight", checks=checks, disabled=disabled,
                  mismatch_errors=list(errors), test_transport=False, cuda_execution_verified=False)
    if context.device.type == "cpu":
        model = _model(context.device)
        error = None
        try:
            TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001),
                config=_configuration(context.device, "demand"))
        except (ValueError, RuntimeError) as problem:
            error = dict(class_name=type(problem).__name__, message=str(problem))
        errors = [None] * context.world_size
        dist.all_gather_object(errors, error)
        assert all(errors), errors
        checks["cpu_offload_rejected"] = True
        result["cpu_offload_errors"] = errors
    else:
        receipt = prepare_packed_offload(_policy(), context, recipe=recipe)
        checks["actual_cuda_startup_preflight"] = True
        result["enabled"] = receipt
    return result


def forward_error(context):
    config = _configuration(context.device, "demand" if context.device.type == "cuda" else None)
    model = _model(context.device)
    runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), config=config)
    if context.device.type == "cpu":
        from deepopt._packed_offload import resolve_packed_offload

        # Exercise the enabled-policy coordination control flow without
        # claiming supported CPU transport or creating a CUDA runtime owner.
        runtime._packed_offload_policy = resolve_packed_offload(_policy())
        runtime._packed_offload_scope = nullcontext
    batch = _batch(context.device, context.global_rank)
    error, backward_calls = None, []
    hooks = [parameter.register_hook(lambda value: backward_calls.append(True)) for parameter in model.parameters()]
    try:
        def step(value):
            loss = _mse(model(value[0]), value[1])
            if context.global_rank == 0:
                raise ValueError("rank-local-after-forward-before-backward")
            return loss

        try:
            runtime.compute(batch, step)
        except (ValueError, RuntimeError) as problem:
            error = dict(class_name=type(problem).__name__, message=str(problem))
        errors = [None] * context.world_size
        dist.all_gather_object(errors, error)
        assert all(errors), errors
        assert runtime._failed and not runtime._forward_pending and not runtime._pending_gradients
        assert not runtime.core.forward_pending
        assert all(parameter.grad is None for parameter in model.parameters())
        assert not backward_calls
        return dict(name="forward_error", checks=dict(cooperative_forward_error=True,
            no_backward_after_forward_error=not backward_calls, collective_usable_after_rejection=True),
            errors=errors, boundary="runtime_enabled_offload" if context.device.type == "cuda"
                else "CPU_host_boundary_marker_nullscope_fault_injection", test_transport=False,
            cuda_execution_verified=context.device.type == "cuda")
    finally:
        for hook in hooks:
            hook.remove()
        close_packed_offload_runtime(runtime)


def main(argv=None):
    args = parser().parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    record = dict(schema_version=1, status="failed", rank=rank, local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")), requested_device=args.device,
        scope="CPU_Gloo_native_and_explicit_test_transport" if args.device == "cpu" else "CUDA_NCCL_real_packed_offload",
        cuda_execution_verified=False, test_transport=args.device == "cpu", checks={}, cases=[], tensor_files_saved=0,
        performance_qualified=False, full_model_qualified=False)
    context, code, started = None, 0, time.monotonic()
    first_bucket = getattr(dist, "_DEFAULT_FIRST_BUCKET_BYTES", None)
    try:
        if args.require_cuda and args.device != "cuda":
            raise ValueError("--require-cuda requires --device cuda; CPU substitution is forbidden")
        if args.allow_source and args.device != "cpu":
            raise ValueError("--allow-source is CPU testing only; CUDA qualification requires the installed wheel")
        if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < 2
                                     or not dist.is_nccl_available()):
            raise RuntimeError("Two visible CUDA devices and NCCL are required; CPU substitution is forbidden")
        torch.set_num_threads(1)
        torch._dynamo.config.repro_after, torch._dynamo.config.repro_level = None, 0
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        record["algorithm_controls_before_numerics"] = dict(
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_enabled=torch.backends.cudnn.enabled, cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
            cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        # Torch's Dynamo DDPOptimizer reads this default at compilation,
        # independently of the public small bucket cap. The process-local
        # fixture must keep both caps consistent to exercise genuine splits.
        dist._DEFAULT_FIRST_BUCKET_BYTES = 64
        record["torch_ddp_test_first_bucket_bytes"] = 64
        record["installed"] = _identity(args.allow_source)
        context = initialize_distributed(args.device, timeout=90)
        assert context.world_size >= 2, "This worker requires real torchrun ranks"
        record.update(context.as_dict())
        selected = ("native", "transport", "preflight", "forward_error") if args.case == "suite" else (args.case,)
        if "preflight" in selected:
            record["cases"].append(preflight(context))
        if "native" in selected:
            for mode in (None,) if args.device == "cpu" else ("demand", "memory_first_overlap"):
                record["cases"].append(native_training(context, mode))
                record["cases"].append(checkpoint_roundtrip(context, mode))
            if args.device == "cuda":
                record["cases"].append(native_training(context, "demand", "torch_amp"))
                record["cases"].append(checkpoint_roundtrip(context, "demand", "torch_amp"))
        if "transport" in selected and args.device == "cpu":
            record["cases"].append(generated_transport(context))
        if "forward_error" in selected:
            record["cases"].append(forward_error(context))
        record["checks"] = {name: all(case["checks"].get(name, True) for case in record["cases"])
            for name in {name for case in record["cases"] for name in case["checks"]}}
        assert record["cases"] and all(record["checks"].values())
        record["installed_after"] = _identity(args.allow_source)
        record["checks"]["source_identity_stable"] = record["installed"] == record["installed_after"]
        assert record["checks"]["source_identity_stable"]
        record["cuda_execution_verified"] = any(case["cuda_execution_verified"] for case in record["cases"])
        record["status"] = "completed"
    except BaseException as error:
        code = 1
        record["exception"] = dict(class_name=type(error).__name__, message=str(error),
                                   traceback="".join(traceback.format_exception(error)))
        print(f"rank {rank}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        try:
            drain_packed_offload_runtime()
            close_packed_offload_runtime()
            gc.collect()
            record["runtime_after_cleanup"] = packed_offload_runtime_snapshot()
        except BaseException as error:
            code = 1
            record["status"] = "failed"
            record["cleanup_exception"] = dict(class_name=type(error).__name__, message=str(error))
        try:
            shutdown_distributed()
        except BaseException as error:
            code = 1
            record["status"] = "failed"
            record["shutdown_exception"] = dict(class_name=type(error).__name__, message=str(error))
        if first_bucket is not None:
            dist._DEFAULT_FIRST_BUCKET_BYTES = first_bucket
        record["group_initialized_after_shutdown"] = dist.is_initialized()
        record["elapsed_seconds"] = time.monotonic() - started
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / f"rank-{rank}.json").open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"rank": rank, "status": record["status"], "cases":len(record["cases"]),
                          "device":record.get("device"), "cuda_execution_verified":record["cuda_execution_verified"]}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
