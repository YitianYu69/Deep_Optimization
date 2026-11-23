from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import triton
except ImportError:  # Native-only graphs can still use the backend without Triton.
    triton = None

from ...ir import IRNode, PointwiseNode, TensorValue
from ...registry import BackendRegistry
from ...scheduler import FusionGroup, SinglePointwiseOp, external_values
from .register_pointwise_ops import get_pointwise_lowering


_MODULE_LOAD_LOCK = threading.Lock()


# Inductor's pointwise heuristic base: max(256, min(numel // 128, 1024)),
# rounded down to a power of two. Chosen per partition and baked into the
# emitted source, which lets the mask be dropped when the element count is a
# provable multiple of the block.
MIN_POINTWISE_BLOCK = 256
MAX_POINTWISE_BLOCK = 1024
MAX_INT32_INDEX = (1 << 31) - 1
DEFAULT_POINTWISE_WARPS = 4


@dataclass(frozen=True)
class PointwiseConfig:
    """Per-partition launch geometry derived from static metadata only."""

    block_size: int
    num_warps: int
    index_dtype: str
    masked: bool
    numel: int | None = None

    def as_dict(self) -> dict:
        return {"block_size": self.block_size, "num_warps": self.num_warps,
                "index_dtype": self.index_dtype, "masked": self.masked,
                "numel": self.numel}


def select_pointwise_config(numel: int | None, max_block: int = MAX_POINTWISE_BLOCK) -> PointwiseConfig:
    """Choose block, warps, index width and masking from a static element count.

    Mirrors TorchInductor's 1-D pointwise heuristic (size-derived block, 32-bit
    indexing below 2**31 elements) but bakes the result, so a provably exact
    grid also removes the bounds mask instead of folding it to a constant.
    """
    if numel is None or numel <= 0:
        return PointwiseConfig(MIN_POINTWISE_BLOCK, DEFAULT_POINTWISE_WARPS,
                               "tl.int32", True, numel)
    upper = max(1, min(int(max_block), MAX_POINTWISE_BLOCK))
    lower = min(MIN_POINTWISE_BLOCK, upper)
    candidate = max(lower, min(numel // 128, upper))
    block = 1 << (candidate.bit_length() - 1)
    block = max(lower, min(block, upper))
    # Blocks are capped at 1024, so four warps is the floor for every
    # admissible geometry; num_warps stays a field for a future autotuner.
    warps = DEFAULT_POINTWISE_WARPS
    index_dtype = "tl.int64" if numel > MAX_INT32_INDEX else "tl.int32"
    return PointwiseConfig(block, warps, index_dtype, numel % block != 0, numel)


@dataclass
class TritonKernelArtifact:
    """Loaded Triton kernel and the IR values required to invoke it."""

    name: str
    kernel: Any
    inputs: list[TensorValue]
    output: TensorValue
    source: str
    path: Path
    # Geometry baked into the source. The launcher must use exactly these
    # values; a different block would make an unmasked store write past the end.
    config: PointwiseConfig | None = None
    block_size: int = 1024
    num_warps: int = DEFAULT_POINTWISE_WARPS
    baked_block_size: bool = True


def emit_scalar(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError(
                f"Non-finite scalar is unsupported in Triton source: {value!r}"
            )
        return repr(value)
    raise TypeError(f"Unsupported scalar in Triton pointwise kernel: {value!r}")


def emit_arg(arg: Any, value_names: dict[TensorValue, str]) -> str:
    if isinstance(arg, TensorValue):
        try:
            return value_names[arg]
        except KeyError as exc:
            raise RuntimeError(
                f"TensorValue {arg.name!r} has no generated Triton expression yet"
            ) from exc
    if isinstance(arg, tuple):
        body = ", ".join(emit_arg(item, value_names) for item in arg)
        return f"({body}{',' if len(arg) == 1 else ''})"
    if isinstance(arg, list):
        return "[" + ", ".join(emit_arg(item, value_names) for item in arg) + "]"
    if isinstance(arg, dict):
        return (
            "{"
            + ", ".join(
                f"{emit_arg(key, value_names)}: {emit_arg(value, value_names)}"
                for key, value in arg.items()
            )
            + "}"
        )
    return emit_scalar(arg)


def emit_expr(
    node: IRNode, value_names: dict[TensorValue, str], registry: BackendRegistry
) -> str:
    args = getattr(node, "args", None)
    kwargs = getattr(node, "kwargs", None)
    op = getattr(node, "op", None)
    if args is None or kwargs is None or not isinstance(op, str):
        raise TypeError(
            f"Fused node {type(node).__name__} must expose op, args, and kwargs"
        )

    from deepopt._bf16_policy import resolve_bf16_policy
    import torch

    policy = resolve_bf16_policy(getattr(node, "bf16_policy", None))
    def render_operand(arg):
        rendered = emit_arg(arg, value_names)
        if (policy.storage_enabled and isinstance(arg, TensorValue) and arg.meta is not None
                and arg.meta.dtype in (torch.bfloat16, torch.float16)):
            rendered = f"({rendered}).to(tl.float32)"
        return rendered

    rendered_args = [render_operand(arg) for arg in args]
    rendered_kwargs = {
        key: render_operand(value) for key, value in kwargs.items()
    }
    extension = registry.for_node(node)
    if extension is not None:
        if extension.triton_codegen is None:
            raise NotImplementedError(
                f"No Triton expression registered for {type(node).__name__}"
            )
        expression = extension.triton_codegen(node, rendered_args, rendered_kwargs)
    elif isinstance(node, PointwiseNode):
        # call_module nodes keep their options on the module object rather than
        # in FX args/kwargs. Add those scalar options for the Triton expression
        # without changing the callable arguments used by native fallback.
        for name in (
            "approximate",
            "negative_slope",
            "min_val",
            "max_val",
            "beta",
            "threshold",
        ):
            if name not in rendered_kwargs and hasattr(node.target, name):
                rendered_kwargs[name] = emit_arg(
                    getattr(node.target, name), value_names
                )
        expression = get_pointwise_lowering(op)(rendered_args, rendered_kwargs)
    else:
        raise NotImplementedError(
            f"No Triton expression registered for {type(node).__name__}"
        )
    if not isinstance(expression, str) or not expression.strip():
        raise TypeError(
            f"Triton codegen for {op!r} must return a non-empty expression string"
        )
    return expression


def _partition_nodes(partition: SinglePointwiseOp | FusionGroup) -> list[IRNode]:
    return (
        [partition.node]
        if isinstance(partition, SinglePointwiseOp)
        else partition.nodes
    )


def generate_single_kernel_source(
    partition: SinglePointwiseOp | FusionGroup,
    kernel_name: str,
    *,
    registry: BackendRegistry | None = None,
    config: PointwiseConfig | None = None,
) -> tuple[str, list[TensorValue]]:
    registry = registry if registry is not None else BackendRegistry()
    nodes = _partition_nodes(partition)
    inputs = external_values(nodes)
    if not inputs:
        raise RuntimeError(
            f"Triton partition has no external tensor inputs: {partition!r}"
        )
    output = partition.output_value
    if output.meta is None:
        raise ValueError(f"Triton output {output.name!r} has no tensor metadata")

    if config is None:
        from ...heuristic import numel_of

        config = select_pointwise_config(numel_of(output.meta.shape))
    value_names = {value: f"x{index}" for index, value in enumerate(inputs)}
    pointer_args = [f"x{index}_ptr" for index in range(len(inputs))]
    # N stays in the signature so the launcher ABI is unchanged; a provably
    # exact grid does not read it.
    signature = pointer_args + ["out_ptr", "N"]
    lines = [
        "@triton.jit",
        f"def {kernel_name}({', '.join(signature)}):",
        f"    # pointwise_config={config.as_dict()!r}",
    ]
    if config.index_dtype == "tl.int64":
        lines.append("    pid = tl.program_id(0).to(tl.int64)")
    else:
        # 32-bit program ids need no widening; Inductor does the same.
        lines.append("    pid = tl.program_id(0)")
    lines.append(f"    offs = pid * {config.block_size} + tl.arange(0, {config.block_size})")
    if config.masked:
        lines.append("    mask = offs < N")
    for index in range(len(inputs)):
        if config.masked:
            lines.append(f"    x{index} = tl.load(x{index}_ptr + offs, mask=mask, other=0.0)")
        else:
            lines.append(f"    x{index} = tl.load(x{index}_ptr + offs)")

    for index, node in enumerate(nodes):
        try:
            expression = emit_expr(node, value_names, registry)
        except Exception as exc:
            op = getattr(node, "op", type(node).__name__)
            raise RuntimeError(
                f"Failed to generate Triton code for {type(node).__name__} ({op})"
            ) from exc
        from deepopt._bf16_policy import resolve_bf16_policy
        policy = resolve_bf16_policy(getattr(node, "bf16_policy", None))
        lines.append(f"    # bf16_policy={policy.name}; version={policy.version}; preserve_node_rounding")
        generated_name = f"v{index}"
        value = node.output()
        if value is None:
            raise TypeError(f"Fused node {type(node).__name__} has no output")
        # Materialized eager intermediates round at every operation. Preserve
        # that rounding when several BF16/FP16 operations share one kernel.
        dtype = str(value.meta.dtype).removeprefix("torch.")
        lines.append(f"    {generated_name} = ({expression}).to(tl.{dtype})")
        value_names[value] = generated_name

    if config.masked:
        lines.append(f"    tl.store(out_ptr + offs, {value_names[output]}, mask=mask)")
    else:
        lines.append(f"    tl.store(out_ptr + offs, {value_names[output]})")
    return "\n".join(lines) + "\n", inputs


def generate_entire_kernel_source(
    partitions: list[SinglePointwiseOp | FusionGroup],
    *,
    registry: BackendRegistry | None = None,
    max_block: int = MAX_POINTWISE_BLOCK,
) -> tuple[str, list[tuple[str, SinglePointwiseOp | FusionGroup, list[TensorValue], PointwiseConfig]]]:
    registry = registry if registry is not None else BackendRegistry()
    lines = [
        "import triton",
        "import triton.language as tl",
        "from triton.language.extra import libdevice",
        "",
    ]
    kernel_infos = []
    for index, partition in enumerate(partitions):
        kernel_name = f"pointwise_kernel_{index}"
        output = partition.output_value
        from ...heuristic import numel_of

        config = select_pointwise_config(numel_of(output.meta.shape) if output.meta else None,
                                         max_block=max_block)
        generated_source, inputs = generate_single_kernel_source(
            partition, kernel_name, registry=registry, config=config
        )
        lines.extend((generated_source, ""))
        kernel_infos.append((kernel_name, partition, inputs, config))
    return "\n".join(lines), kernel_infos


def write_module_source(source: str, cache_dir: str | Path | None = None) -> Path:
    """Write generated source atomically to a configurable, external cache."""
    root = Path(
        cache_dir
        or os.environ.get("DEEPOPT_COMPILER_CACHE_DIR")
        or (Path(tempfile.gettempdir()) / "deepopt_compiler")
    ).expanduser()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]
    path = root / f"module_{digest}.py"
    if path.exists():
        if path.read_text(encoding="utf-8") != source:
            raise RuntimeError(
                f"Generated kernel cache entry does not match its digest: {path}"
            )
        return path

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=root
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary_name, path)
        except FileExistsError:
            # Another process generated the same deterministic artifact.
            os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def load_module_from_file(path: Path):
    module_name = f"do_compiler_triton_module_{path.stem}"
    with _MODULE_LOAD_LOCK:
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load generated Triton module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module


class TritonCodegen:
    def __init__(
        self,
        *,
        registry: BackendRegistry | None = None,
        cache_dir: str | Path | None = None,
        block_size: int = 1024,
        kernel_policy=None,
    ):
        if block_size < 1 or block_size & (block_size - 1):
            raise ValueError("Triton block_size must be a positive power of two")
        self.registry = registry if registry is not None else BackendRegistry()
        self.cache_dir = cache_dir
        self.block_size = block_size
        self.kernel_policy = kernel_policy

    def compile_partitions(
        self,
        partitions: list[SinglePointwiseOp | FusionGroup],
    ) -> list[TritonKernelArtifact]:
        if not partitions:
            return []
        if triton is None:
            raise RuntimeError(
                "Triton is required for this graph because it contains pointwise kernels"
            )
        module_source, kernel_infos = generate_entire_kernel_source(
            partitions, registry=self.registry, max_block=self.block_size
        )
        import torch

        module_source = (
            # ABI 2 bakes the launch block and index width into each kernel and
            # removes the BLOCK_SIZE constexpr argument.
            f"# DeepOpt compiler ABI 2; torch={torch.__version__}; "
            f"triton={triton.__version__}; cuda={torch.version.cuda}; "
            f"block_size={self.block_size}\n" + module_source
        )
        if self.kernel_policy is not None:
            module_source = f"# kernel_policy={self.kernel_policy.fingerprint}\n" + module_source
        path = write_module_source(module_source, self.cache_dir)
        module = load_module_from_file(path)
        artifacts = []
        for kernel_name, partition, inputs, config in kernel_infos:
            artifacts.append(
                TritonKernelArtifact(
                    name=kernel_name,
                    kernel=getattr(module, kernel_name),
                    inputs=inputs,
                    output=partition.output_value,
                    source=module_source,
                    path=path,
                    config=config,
                    block_size=config.block_size,
                    num_warps=config.num_warps,
                )
            )
        return artifacts
