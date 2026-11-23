"""Optional FX compiler reusing DeepOpt's numerical operations and autograd."""

from .backend import BackendConfig, compile_fx, do_backend
from .registry import BackendRegistry, KernelCodegenContext, OperatorExtension
from .reporting import CompilationReport, compilation_report
from threading import Lock

_registration_lock = Lock()


def register_torch_backend(name: str = "deepopt") -> None:
    """Register this compiler for ``torch.compile(backend=name)``."""
    from torch._dynamo.backends.registry import _COMPILER_FNS, register_backend

    with _registration_lock:
        if name in _COMPILER_FNS:
            if _COMPILER_FNS[name] is not do_backend:
                raise ValueError(
                    f"Torch compiler backend {name!r} is already registered by another implementation"
                )
            return
        register_backend(do_backend, name=name)


__all__ = [
    "BackendConfig",
    "BackendRegistry",
    "KernelCodegenContext",
    "OperatorExtension",
    "compile_fx",
    "do_backend",
    "register_torch_backend",
    "CompilationReport",
    "compilation_report",
]
