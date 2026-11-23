"""Static contiguous pointwise TileLang emission and lazy CUDA compilation."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import threading

from ...ir import TensorValue
from ...registry import BackendRegistry
from ...scheduler import SinglePointwiseOp, external_values
from ..source import load_source, write_source
from .expressions import expression


@dataclass
class TileLangKernelArtifact:
    name: str
    kernel: object
    inputs: list[TensorValue]
    output: TensorValue
    source: str
    path: Path
    backend: str = "tilelang"


def emit_scalar(value):
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    raise TypeError(f"Unsupported scalar in TileLang source: {value!r}")


def emit_arg(value, names):
    if isinstance(value, TensorValue):
        return names[value]
    if isinstance(value, tuple):
        body = ", ".join(emit_arg(item, names) for item in value)
        return f"({body}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return "[" + ", ".join(emit_arg(item, names) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{emit_arg(k, names)}: {emit_arg(v, names)}" for k, v in value.items()) + "}"
    return emit_scalar(value)


def generate_kernel_source(partition, name="pointwise", *, registry=None, block_size=1024):
    registry = registry if registry is not None else BackendRegistry()
    nodes = [partition.node] if isinstance(partition, SinglePointwiseOp) else partition.nodes
    inputs = external_values(nodes)
    if not inputs:
        raise ValueError("A TileLang pointwise region needs tensor inputs")
    meta = partition.output_value.meta
    if meta is None:
        raise ValueError("A TileLang pointwise region needs output metadata")
    count = math.prod(meta.shape)
    dtype = str(meta.dtype).removeprefix("torch.")
    signature = [f'a{i}: T.Tensor(({count},), "{dtype}")' for i in range(len(inputs))]
    signature.append(f'output: T.Tensor(({count},), "{dtype}")')
    lines = [
        f"def build_{name}(device):",
        "    from deepopt.opt.backends.tilelang._runtime import runtime, compile_kernel",
        "    _, T = runtime()",
        "    @T.prim_func",
        f"    def kernel({', '.join(signature)}):",
        f"        with T.Kernel({(count + block_size - 1) // block_size}, threads=128) as block:",
        f"            for lane in T.Parallel({block_size}):",
        f'                offset = T.cast(block, "int64") * {block_size} + lane',
        f"                if offset < {count}:",
    ]
    names = {value: f"x{i}" for i, value in enumerate(inputs)}
    for index in range(len(inputs)):
        lines.append(f"                    x{index} = a{index}[offset]")
    for index, node in enumerate(nodes):
        args = [emit_arg(arg, names) for arg in node.args]
        kwargs = {key: emit_arg(arg, names) for key, arg in node.kwargs.items()}
        for key in ("approximate", "negative_slope", "min_val", "max_val", "beta", "threshold"):
            target = getattr(node, "target", None)
            if key not in kwargs and hasattr(target, key):
                kwargs[key] = emit_scalar(getattr(target, key))
        extension = registry.for_node(node)
        if extension is not None:
            if extension.tilelang_codegen is None:
                raise NotImplementedError(f"No TileLang expression for extension {extension.name!r}")
            rendered = extension.tilelang_codegen(node, args, kwargs)
        else:
            rendered = expression(node.op, args, kwargs)
        if not isinstance(rendered, str) or not rendered.strip():
            raise TypeError("TileLang expression callbacks must return nonempty source")
        output = node.output()
        result_dtype = str(output.meta.dtype).removeprefix("torch.")
        lines.append(f'                    v{index} = T.cast({rendered}, "{result_dtype}")')
        names[output] = f"v{index}"
    lines.extend((
        f"                    output[offset] = {names[partition.output_value]}",
        "    return compile_kernel(kernel, device)",
        "",
    ))
    return "\n".join(lines), inputs


class _LazyKernel:
    def __init__(self, builder):
        self.builder = builder
        self.executables = {}
        self.lock = threading.RLock()

    def __call__(self, *tensors):
        import torch
        from deepopt.opt.backends.tilelang._runtime import require_cuda, storage_view

        device = require_cuda(*tensors)
        key = str(device)
        with self.lock:
            executable = self.executables.get(key)
            if executable is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("TileLang compiler specialization was not warmed before CUDA capture")
                executable = self.builder(device)
                self.executables[key] = executable
        executable(*(storage_view(tensor) for tensor in tensors))


class TileLangCodegen:
    def __init__(self, *, registry=None, cache_dir=None, block_size=1024, kernel_policy=None):
        self.registry = registry if registry is not None else BackendRegistry()
        self.cache_dir, self.block_size = cache_dir, block_size
        self.kernel_policy = kernel_policy

    def compile_partitions(self, partitions):
        if not partitions:
            return []
        import torch

        try:
            compiler_version = version("tilelang")
        except PackageNotFoundError:
            compiler_version = "unavailable"
        fingerprint = getattr(self.kernel_policy, "fingerprint", "tilelang")
        header = (f"# DeepOpt TileLang ABI 1; tilelang={compiler_version}; "
                  f"torch={torch.__version__}; cuda={torch.version.cuda}; policy={fingerprint}\n")
        artifacts = []
        for index, partition in enumerate(partitions):
            name = f"pointwise_{index}"
            source, inputs = generate_kernel_source(
                partition, name, registry=self.registry, block_size=self.block_size,
            )
            source = header + source
            path = write_source(source, self.cache_dir)
            module = load_source(path)
            artifacts.append(TileLangKernelArtifact(
                name, _LazyKernel(getattr(module, f"build_{name}")), inputs,
                partition.output_value, source, path,
            ))
        return artifacts
