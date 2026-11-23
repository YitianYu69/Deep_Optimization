"""Lazy dispatch for custom preparation kernels, independent of GPU compilers."""

from importlib import import_module
from importlib.util import find_spec
from functools import wraps

from deepopt.kernel_policy import (
    current_kernel_policy, current_kernel_role, kernel_backend, select_kernel_backend,
)


class _MixedModule:
    """Validate and observe a metadata decision before any kernel executes."""

    def __init__(self, family, path, module, selected, policy, role):
        self.family, self.path, self.module = family, path, module
        self.selected, self.policy, self.role = selected, policy, role
        self.effective_backend, self.fallback_reason = selected, None

    def __getattr__(self, name):
        value = getattr(self.module, name, None)
        if value is None and (name.startswith("__") or self.policy.fallback != "other"
                              or self.policy.pinned(self.family, self.role)):
            raise AttributeError(f"{getattr(self.module, '__name__', self.path)} has no attribute {name!r}")
        if value is not None and (not callable(value) or isinstance(value, type) or name.startswith("__")):
            return value

        def call(*args, **kwargs):
            module, selected = self.module, self.selected
            if self.policy.fallback == "other" and not self.policy.pinned(self.family, self.role):
                other = "tilelang" if selected == "triton" else "triton"
                if find_spec(other) is None:
                    raise RuntimeError(f"Mixed fallback=other requires the optional {other} package")
            checker = getattr(module, "unsupported_reason", None)
            reason = ("operation unavailable" if value is None else
                      checker(name, *args, **kwargs) if checker is not None else None)
            if reason is not None:
                other = "tilelang" if selected == "triton" else "triton"
                eligibility = {selected: reason}
                if self.policy.fallback == "other" and not self.policy.pinned(self.family, self.role):
                    candidate = import_module(f"deepopt.opt.backends.{other}.{self.path}")
                    check_other = getattr(candidate, "unsupported_reason", None)
                    eligibility[other] = ("operation unavailable" if not hasattr(candidate, name) else
                                          check_other(name, *args, **kwargs) if check_other else None)
                    selected = select_kernel_backend(self.family, eligibility, policy=self.policy, role=self.role)
                    module = candidate
                else:
                    select_kernel_backend(self.family, eligibility, policy=self.policy, role=self.role)
            self.effective_backend, self.fallback_reason = selected, reason
            if selected == "triton":
                self._validate_fusion(name, args, kwargs)
            # No exception handler: compile/runtime/OOM errors remain visible.
            return getattr(module, name)(*args, **kwargs)

        return wraps(value)(call) if value is not None else call

    def _validate_fusion(self, operation, args, kwargs):
        pins = dict(self.policy.families)
        codec = kwargs.get("packed") or kwargs.get("output_pack") is not None
        compression = kwargs.get("compression")
        if compression is None and self.family == "normalization" and operation == "forward" and len(args) > 4:
            compression = args[4]
        codec = codec or isinstance(compression, dict) and compression.get("enabled", False)
        if codec and pins.get("activation_codec", "triton") != "triton":
            raise ValueError("Conflicting Mixed routes: fused Triton saved-state packing requires activation_codec=triton")
        dtype = kwargs.get("dtype", kwargs.get("out_dtype"))
        if str(dtype).startswith("torch.float8") and pins.get("quantization", "triton") != "triton":
            raise ValueError("Conflicting Mixed routes: fused Triton FP8 output requires quantization=triton")
        if self.path == "quantization.activation.fp8" and pins.get("quantization", "triton") != "triton":
            raise ValueError("Conflicting Mixed routes: fused Triton FP8 packing requires quantization=triton")


def backend_module(family, module, *, backend=None, role=None):
    selected = kernel_backend(family, backend=backend, role=role)
    implementation = import_module(f"deepopt.opt.backends.{selected}.{module}")
    policy = current_kernel_policy()
    if policy.backend == "mixed":
        return _MixedModule(family, module, implementation, selected, policy, role or current_kernel_role())
    return implementation


def call_kernel(family, operation, *args, backend=None, role=None, module=None, **kwargs):
    """Operation is a module-qualified canonical name, e.g. quantization.fp8.quantize."""
    if module is None:
        module, _, operation = operation.rpartition(".")
        if not module:
            raise ValueError("Kernel operation must include its implementation module")
    implementation = backend_module(family, module, backend=backend, role=role)
    function = getattr(implementation, operation, None)
    if function is None:
        raise NotImplementedError(f"{implementation.__name__} does not implement {operation}")
    return function(*args, **kwargs)
