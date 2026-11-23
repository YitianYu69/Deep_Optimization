"""Opt-in, metadata-only backend observations outside throughput timing."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from itertools import count
from threading import local


_records = ContextVar("deepopt_gemm_records", default=None)
_module = ContextVar("deepopt_gemm_module", default=None)
_launch = ContextVar("deepopt_gemm_launch", default=None)
_scratch_invocations = count(1)


def record_gemm_launch(function):
    """Bind launch observations to one dispatch, including nested fallbacks."""
    @wraps(function)
    def call(*args, **kwargs):
        if _records.get() is None:
            return function(*args, **kwargs)
        token = _launch.set({})
        try:
            return function(*args, **kwargs)
        except Exception as error:
            evidence = getattr(error, "evidence", None)
            if isinstance(evidence, dict):
                evidence.update({key: value for key, value in (_launch.get() or {}).items()
                                 if key in ("bf16_policy", "route_contract")})
                _records.get().append({"status": "rejected", "execution": "not_launched",
                                      "module": _module.get(), "resource_admission": dict(evidence)})
            raise
        finally:
            _launch.reset(token)
    return call


def note_gemm_launch(**metadata):
    """Opt-in metadata only; never retain launch arguments or tensor storage."""
    current = _launch.get()
    if current is not None:
        current.update(metadata)
        if "scratch_bytes" in metadata and "resources" in current:
            current["resources"] = dict(current["resources"], explicit_workspace_bytes=metadata["scratch_bytes"])


def gemm_launch_is_recording():
    return _launch.get() is not None


def observe_bf16_scratch(buffers):
    """Physical call-local scratch receipt; never a total allocator peak.

    Invoked only by diagnostic dispatches. Storage identities/generations are
    scoped to this receipt; no tensor or storage ownership escapes the call.
    """
    if not gemm_launch_is_recording():
        return
    import torch
    from deepopt.memory.ledger import StorageLedger

    device = next(iter(buffers.values())).device
    if any(value.device != device for value in buffers.values()):
        raise ValueError("Scratch observation requires one device")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Scratch observation is outside CUDA Graph capture")
            stream = torch.cuda.current_stream(device).cuda_stream
    else:
        stream = "host_ordered"
    ledger = StorageLedger()
    ledger.advance("bf16_scratch_allocated")
    ledger.observe(buffers, owner="invocation_scratch", producer="bf16_panel")
    note_gemm_launch(scratch_storage_receipt=dict(
        scope="one_product_invocation", invocation=next(_scratch_invocations),
        device=str(device), stream=stream, ledger=ledger.snapshot(),
        bindings=tuple(dict(name=name, version=value._version, shape=tuple(value.shape),
                            strides=tuple(value.stride()), storage_offset=value.storage_offset(),
                            storage_bytes=value.untyped_storage().nbytes())
                       for name, value in buffers.items()),
        accounting="backing_storage_observation_not_total_peak",
        unknown_native_workspaces=True, capture_observation=False))


@contextmanager
def gemm_module_scope(name):
    """Compiler numerical regions supply names without traced module hooks."""
    if _records.get() is None:
        yield
        return
    token = _module.set(name or "<root>")
    try:
        yield
    finally:
        _module.reset(token)


@contextmanager
def record_gemms(model=None):
    records = []
    token = _records.set(records)
    handles, owner = [], local()
    module_token = _module.set(None) if model is not None else None
    compilation = getattr(model, "_deepopt_compile_config", {}) if model is not None else {}
    compiler_names = (compilation.get("backend") == "deepopt"
                      and compilation.get("options", {}).get("use_native_autodiff", True))
    if model is not None and not compiler_names:
        for name, module in model.named_modules():
            if not any(word in type(module).__name__.lower() for word in ("linear", "conv")):
                continue
            def before(current, args, name=name):
                if not hasattr(owner,"stacks"):
                    owner.stacks={}
                token= _module.set(name or "<root>") if _records.get() is records else None
                owner.stacks.setdefault(id(current), []).append(token)
            def after(current, args, output):
                values = getattr(owner,"stacks",{}).get(id(current), [])
                if values:
                    token=values.pop()
                    if token is not None:
                        _module.reset(token)
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after, always_call=True))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()
        if module_token is not None:
            _module.reset(module_token)
        _records.reset(token)


def capture_gemm_recording(ctx):
    # CUDA autograd worker threads do not inherit the forward ContextVar.
    # Retain the diagnostic list only; it contains no tensor/storage objects.
    ctx.gemm_diagnostics = _records.get()
    ctx.gemm_module_name = _module.get()


def restore_gemm_recording(function):
    @wraps(function)
    def call(ctx, *args):
        records = getattr(ctx, "gemm_diagnostics", None)
        name = getattr(ctx, "gemm_module_name", None)
        if records is None or records is _records.get() and name == _module.get():
            return function(ctx, *args)
        token = _records.set(records)
        module_token = _module.set(name)
        try:
            return function(ctx, *args)
        finally:
            _module.reset(module_token)
            _records.reset(token)
    return call


def observe_gemm(a, b, requested, output, role, preparation=None, *, bias=None, routing=None):
    records = _records.get()
    if records is None:
        return
    reason = None
    effective = requested
    if requested in ("native_fast", "native_precise"):
        from deepopt.opt.backends.native.fp8 import unsupported_reason

        reason = unsupported_reason(a, b,bias=bias,out_dtype=output.dtype)
        if reason:
            from deepopt.kernel_policy import kernel_backend
            effective = kernel_backend("gemm", role=f"linear.{role}")

    def operand(q):
        return {"shape": tuple(q.data.shape), "strides": tuple(q.data.stride()),
                "dtype": str(q.data.dtype), "scale_shape": tuple(q.scale_inv.shape),
                "block_shape": q.block_shape}

    strategy=None
    if bias is not None and requested in ("native_fast","native_precise"):
        from deepopt.opt.backends.native.fp8 import bias_strategy
        strategy=bias_strategy(bias,output.dtype)
    records.append({"role": role, "requested_backend": requested, "effective_backend": effective,
                    "bias_strategy":strategy,
                    "module": _module.get(), "preparation": preparation,
                    "fallback_reason": reason, "a": operand(a), "b": operand(b),
                    "output_shape": tuple(output.shape), "output_dtype": str(output.dtype),
                    "output_storage_dtype": str(getattr(output, "_data", output).dtype)})
    if _launch.get():
        records[-1]["kernel_launch"] = dict(_launch.get())
    from deepopt.kernel_policy import current_kernel_policy
    policy = current_kernel_policy()
    if policy.backend != "triton":
        records[-1].update(kernel_mode=policy.backend, kernel_policy=policy.fingerprint)
    if routing is not None:
        records[-1].update(routing)


def observe_bf16_gemm(a, b, requested, output, role, *, packed_b=None, execution):
    """Use the same opt-in recording stream for BF16 and FP8 products."""
    records = _records.get()
    if records is None:
        return

    def operand(value):
        return None if value is None else {
            "shape": tuple(value.shape), "strides": tuple(value.stride()), "dtype": str(value.dtype)}

    record = {"role": role, "precision": "bf16", "requested_backend": requested,
              "effective_backend": requested if execution != "semantic_reference" else "reference",
              "execution": execution, "module": _module.get(), "fallback_reason": None,
              "a": operand(a), "b": operand(b), "output_shape": tuple(output.shape),
              "output_dtype": str(output.dtype), "output_storage_dtype": str(output.dtype),
              "packed_read": packed_b is not None}
    if packed_b is not None:
        record["packed_operand"] = {**operand(packed_b.data), "logical_shape": tuple(packed_b.shape),
                                    "bits": packed_b.bits, "group_size": packed_b.group_size,
                                    "orientation": "source_KN_right_KN"}
    if _launch.get():
        metadata = dict(_launch.get())
        for key in ("bf16_policy", "route_contract"):
            if key in metadata:
                record[key] = metadata[key]
        if "kernel" in metadata or "resources" in metadata:
            record["kernel_launch"] = metadata
    records.append(record)
