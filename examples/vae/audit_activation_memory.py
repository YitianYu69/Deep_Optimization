"""Allocation-free MBConv VAE saved-state and visible-tensor metadata audit.

This executes the maintained model and L1/KL objective with strict FakeTensors.
It never authorizes a CUDA run and does not estimate opaque CUDA workspaces.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tomllib


DTYPE_POLICIES = ("bf16-reference", "cpu-autocast", "cuda-groupnorm-emulated")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path(__file__).with_name("activation_memory_local128.toml"))
    result.add_argument("--dtype-policy", choices=DTYPE_POLICIES, default="cuda-groupnorm-emulated")
    result.add_argument("--bf16-policy", choices=("torch_amp", "bf16_storage"))
    result.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--optimizer-update", action="store_true",
                        help="Trace the configured optimizer too; a missing custom factory is an error")
    result.add_argument("--optimizer-factory", default="source.training.geo_optimizer:SGD_NS_Overshoot")
    result.add_argument("--update", type=int, default=1, help="One-based beta schedule update")
    result.add_argument("--max-records", type=int, default=50000)
    result.add_argument("--output", type=Path, default=Path("work/vae_local128_metadata.json"))
    return result


def settings(path):
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"base_channels": 32, "blocks_per_stage": 2, "latent_channels": 4, "max_channels": 512}
    compression = raw["runtime"]["activation_compression"]
    if (raw["application"]["model"] != expected or raw["training"]["batch_size"] != 128
            or raw["application"]["data"]["image_size"] != 224
            or raw["training"].get("accumulation_steps", 1) != 1
            or compression["bits"] != 2 or compression["group_size"] != 1024
            or compression["norm_xhat_recenter"] is not True):
        raise ValueError("The target audit requires the exact MBConv V2 local128/image224/two-bit/group1024/recenter recipe")
    return raw


@contextmanager
def _fake_initialization(factory):
    """Skip the expensive orthogonal QR only while constructing fake weights."""
    from torch._subclasses.fake_tensor import FakeTensor

    original = factory.init_weights

    def initialize(self, module):
        for parameter in module.parameters(recurse=False):
            if not isinstance(parameter, FakeTensor):
                raise RuntimeError("Initialization may only be skipped for strict fake parameters")

    factory.init_weights = initialize
    try:
        yield
    finally:
        factory.init_weights = original


def emulated_groupnorm_dtype(bf16_policy):
    """Requested CUDA storage table; actual-device tests must validate it."""
    import torch
    from deepopt._bf16_policy import resolve_bf16_policy

    return torch.bfloat16 if resolve_bf16_policy(bf16_policy).storage_enabled else torch.float32


def audit(raw, *, dtype_policy="cuda-groupnorm-emulated", compressed=True, optimizer_update=False,
          optimizer_factory="source.training.geo_optimizer:SGD_NS_Overshoot", update=1, max_records=50000, bf16_policy=None):
    """Trace the exact target; return metadata and explicit coverage limitations."""
    import torch
    import deepopt
    from torch._subclasses.fake_tensor import FakeTensorMode
    from deepopt.memory.estimation import _snapshot_rng, _restore_rng
    from deepopt.memory.ledger import StorageLedger, StorageObserver
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.scheduling.controller import fake_tensor_reads
    from benchmarks.analysis.measurement import SavedStorageLifetimes
    from examples.vae import benchmark_vae as benchmark
    from examples.vae.mbconv_vae import MBConvVAE_V2

    if dtype_policy not in DTYPE_POLICIES or type(update) is not int or update < 1:
        raise ValueError("Choose a supported dtype policy and a positive schedule update")
    from deepopt._bf16_policy import resolve_bf16_policy

    resolved = deepcopy(raw)
    policy = resolve_bf16_policy(bf16_policy if bf16_policy is not None else resolved["runtime"].get("bf16_policy"))
    resolved["runtime"]["bf16_policy"] = policy.as_dict()
    resolved["runtime"]["activation_compression"]["enabled"] = bool(compressed)
    # Kernel policies apply to CPU reference metadata paths; no generated/CUDA
    # launch is inferred from the model's adapter transformation report.
    resolved["runtime"]["compile"]["enabled"] = False
    resolved["runtime"]["fp8"]["enabled"] = False
    ledger = StorageLedger(max_records=max_records)
    tracker = SavedStorageLifetimes(ledger=ledger)
    before = benchmark.source_provenance(deepopt, MBConvVAE_V2, None)
    audit_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rng = _snapshot_rng()
    handles, stack, saved = [], [], []
    model_outputs = []
    norm_dtypes = set()
    optimizer = None
    mode = FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=False)
    if mode.propagate_real_tensors:
        raise RuntimeError("Metadata auditing requires fake real-tensor propagation disabled")
    try:
        # Construction is outside the observed training interval. All tensor
        # factories, including default module initialization, remain fake.
        with mode, _fake_initialization(MBConvVAE_V2):
            model = MBConvVAE_V2(**resolved["application"]["model"]).train()
        with mode:
            if dtype_policy == "bf16-reference":
                model = model.to(torch.bfloat16)
            model, transformation = prepare_fp8_model(model, config=resolved)
            images = torch.empty(128, 3, 224, 224,
                                 dtype=torch.bfloat16 if dtype_policy == "bf16-reference" else torch.float32)
            if optimizer_update:
                optimizer = benchmark.make_optimizer(model, resolved["training"], optimizer_factory)
            ledger.observe(tuple(model.parameters()), owner="model_parameter")
            ledger.observe(tuple(model.buffers()), owner="model_buffer")
            ledger.observe(images, owner="caller_input")

            def before_module(module, args, name):
                stack.append(ledger.phase)
                ledger.phase = f"forward:{name}"
                if dtype_policy == "cuda-groupnorm-emulated" and isinstance(module, torch.nn.GroupNorm):
                    return (args[0].to(emulated_groupnorm_dtype(policy)), *args[1:])

            def after_module(module, args, output):
                if isinstance(module, torch.nn.GroupNorm) and output is not None:
                    if dtype_policy == "cuda-groupnorm-emulated" and policy.storage_enabled:
                        output = output.to(torch.bfloat16)
                    norm_dtypes.add(str(output.dtype))
                if module is model and output is not None:
                    ledger.observe(output, owner="model_return")
                    model_outputs.extend({"shape": list(value.shape), "dtype": str(value.dtype)}
                                         for value in output)
                ledger.phase = stack.pop()
                return output

            for name, module in model.named_modules():
                handles.append(module.register_forward_pre_hook(
                    lambda current, args, name=name: before_module(current, args, name or "<root>")))
                handles.append(module.register_forward_hook(after_module, always_call=True))

            def pack(value):
                token = tracker.pack(value)
                if len(saved) < max_records:
                    saved.append({"source": ledger.phase, "storage_ids": list(token.key),
                                  "shape": list(value.shape), "dtype": str(value.dtype),
                                  "logical_bytes": value.numel() * value.element_size()})
                return token

            schedules = deepopt.ScheduleController.from_config(resolved["schedules"])
            # Select a diagnostic beta without pretending earlier model updates
            # occurred. The controller still advances in its required order.
            for previous_update in range(1, update):
                schedules.prepare_update(previous_update, epoch=1)
                schedules.complete_update(previous_update)
            schedules.prepare_update(update, epoch=1)
            beta = schedules["beta"]
            with (fake_tensor_reads(mode), StorageObserver(ledger),
                  torch.autograd.graph.saved_tensors_hooks(pack, tracker.unpack)):
                with ledger.scope("forward_and_loss"), deepopt.autocast(
                        "cpu", dtype=torch.bfloat16, enabled=dtype_policy != "bf16-reference",
                        fp8=False, check_finite=False, config=resolved):
                    loss = benchmark.objective(model, images, beta)
                    ledger.observe(loss, owner="escaped_output")
                after_forward = tracker.snapshot()
                with ledger.scope("backward"):
                    loss.backward()
                    ledger.observe(tuple(parameter.grad for parameter in model.parameters()), owner="leaf_gradient")
                if optimizer is not None:
                    with ledger.scope("optimizer"):
                        optimizer.step()
                        ledger.observe(optimizer.state, owner="optimizer_state")
            snapshot = ledger.snapshot()
            loss_dtype = str(loss.dtype)
            schedules.complete_update(update)
    finally:
        for handle in handles:
            handle.remove()
        _restore_rng(rng)
    after = benchmark.source_provenance(deepopt, MBConvVAE_V2, optimizer)
    stable = all(before[key] == after[key] for key in (
        "deepopt_source_sha256", "benchmark_source_sha256", "model_source_sha256"))
    stable &= audit_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {"status": "ok" if stable else "invalid_changed_source", "kind": "metadata_only",
            "resolved_config": resolved, "requested_physical_local_batch": 128,
            "input_shape": [128, 3, 224, 224], "source": after, "source_stable": stable,
            "audit_source_sha256": audit_hash, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "execution": "strict FakeTensor CPU reference", "compiler_mode": "eager_metadata",
            "cuda_execution_observed": False, "cuda_capacity_admitted": False,
            "dtype_policy": dtype_policy, "bf16_policy": policy.as_dict(),
            "cuda_dtype_table_validated": False,
            "expected_cuda_groupnorm_output_dtype": "torch.bfloat16" if policy.storage_enabled else "torch.float32", "groupnorm_output_dtypes": sorted(norm_dtypes),
            "model_outputs": model_outputs,
            "dtype_emulation": ("Policy-aware CPU GroupNorm dtype emulation; cast tensors are included. "
                                "This is metadata emulation, not CUDA execution."
                                if dtype_policy == "cuda-groupnorm-emulated" else None),
            "initialization": "default module initialization is fake; model-specific orthogonal QR skipped only on fake parameters",
            "objective": "stochastic reparameterization; FP32 L1 + beta * FP32 KL", "loss_dtype": loss_dtype,
            "beta": beta, "beta_update": update, "model_updates_before_audit": 0,
            "optimizer": {"requested": resolved["training"]["optimizer"], "executed": optimizer is not None,
                          "factory": optimizer_factory if resolved["training"]["optimizer"] == "custom" else None,
                          "state_bytes": None,
                          "state_coverage": "included_in_ledger" if optimizer is not None else "unknown_not_executed",
                          "note": "No substitute optimizer is selected; default audit covers forward/backward only."},
            "saved_tensors": saved, "after_forward_saved_storage": after_forward,
            "invocation_storage_ledger": snapshot,
            "transformation_report": transformation.as_dict() if hasattr(transformation, "as_dict") else str(transformation),
            "limitations": ["No payloads, numerical validation, device launches, timings or CUDA allocator peaks were measured.",
                            "Opaque native workspaces and internal operator allocations remain unknown.",
                            "Historical 208.6632/103.1841/91.8257/87.9901 GiB are counterfactual references, not expectations forced on this run."]}


def main(argv=None):
    args = parser().parse_args(argv)
    raw = settings(args.config)
    result = audit(raw, dtype_policy=args.dtype_policy, compressed=args.compressed,
                   optimizer_update=args.optimizer_update, optimizer_factory=args.optimizer_factory,
                   update=args.update, max_records=args.max_records, bf16_policy=args.bf16_policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "kind": result["kind"], "physical_local_batch": 128,
                      "saved_storage": result["after_forward_saved_storage"]}, sort_keys=True))
    return 0 if result["source_stable"] else 1


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
