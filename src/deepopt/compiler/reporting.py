"""Serializable evidence of compiler work, without retaining executable graphs."""

from dataclasses import asdict, dataclass, field

from .saved_state import SAVED_STATE_SCHEMA_VERSION as SAVED_STATE_SCHEMA_VERSION
from .saved_state import classify_saved_state as classify_saved_state


@dataclass(frozen=True)
class GraphReport:
    operations: int
    triton_regions: int = 0
    custom_kernels: int = 0
    compiler_autograd_regions: int = 0
    deepopt_operations: tuple[str, ...] = ()
    eager_autograd_operations: tuple[str, ...] = ()
    native_operations: tuple[str, ...] = ()
    reason: str = ""
    lowered_deepopt_operations: tuple[str, ...] = ()
    compiled_deepopt_backward_regions: int = 0
    unsupported_deepopt_operations: tuple[str, ...] = ()
    fused_deepopt_regions: int = 0
    fusion_patterns: tuple[str, ...] = ()
    program_schedules: tuple[dict, ...] = ()
    compile_time_ms: dict[str, float] = field(default_factory=dict)
    tilelang_regions: int = 0
    kernel_backend: str = "triton"
    kernel_policy: dict = field(default_factory=dict)
    kernel_routes: tuple[dict, ...] = ()
    effective_compiler_options: dict = field(default_factory=dict)
    fusion_candidates: tuple[dict, ...] = ()
    pack_sharing_decisions: tuple[dict, ...] = ()
    static_regions: dict[str, int] = field(default_factory=dict)
    # Only a profiler can supply runtime launch evidence. Static region counts
    # cannot be substituted for observed CUDA launches.
    observed_cuda_launches: int | None = None
    native_autodiff_operations: int = 0
    native_autodiff_regions: int = 0
    differentiation_rules: tuple[dict, ...] = ()
    legacy_backward_operations: tuple[str, ...] = ()
    native_autodiff_graphs: tuple[dict, ...] = ()
    final_schedule: tuple[dict, ...] = ()
    logical_lifetimes: tuple[dict, ...] = ()
    precision_contracts: tuple[dict, ...] = ()
    memory_optimization_regions: int = 0
    # Static admission predictions per pointwise partition. These are estimates
    # derived from declared geometry, never measured time or observed launches.
    pointwise_cost_estimates: tuple[dict, ...] = ()


@dataclass
class CompilationReport:
    """One entry per backend invocation, including guarded recompilations."""

    graphs: list[GraphReport] = field(default_factory=list)
    runtime_fallbacks: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        # A copied nn.Module discards its compiled callable, so its old
        # model's compilation evidence must not appear as work on this copy.
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    def fallback(self, reason):
        self.runtime_fallbacks += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    @property
    def captured_graphs(self):
        return len(self.graphs)

    @property
    def triton_regions(self):
        return sum(graph.triton_regions for graph in self.graphs)

    @property
    def tilelang_regions(self):
        return sum(graph.tilelang_regions for graph in self.graphs)

    @property
    def deepopt_operations(self):
        return sum(len(graph.deepopt_operations) for graph in self.graphs)

    @property
    def lowered_deepopt_operations(self):
        return sum(len(graph.lowered_deepopt_operations) for graph in self.graphs)

    @property
    def compiled_deepopt_backward_regions(self):
        return sum(graph.compiled_deepopt_backward_regions for graph in self.graphs)

    @property
    def fused_deepopt_regions(self):
        return sum(graph.fused_deepopt_regions for graph in self.graphs)

    def as_dict(self):
        static_regions = {}
        source_module_sites = {"train": set(), "eval": set()}
        candidates = []
        for index, graph in enumerate(self.graphs):
            for name, count in graph.static_regions.items():
                static_regions[name] = static_regions.get(name, 0) + count
            for candidate in graph.fusion_candidates:
                candidates.append({"graph_index": index, **candidate})
                phase = candidate.get("phase")
                path = candidate.get("module_path")
                if phase in source_module_sites and isinstance(path, str):
                    source_module_sites[phase].add(path)
        return {
            "captured_graphs": self.captured_graphs,
            "triton_regions": self.triton_regions,
            "tilelang_regions": self.tilelang_regions,
            "deepopt_operations": self.deepopt_operations,
            "lowered_deepopt_operations": self.lowered_deepopt_operations,
            "compiled_deepopt_backward_regions": self.compiled_deepopt_backward_regions,
            "native_autodiff_operations": sum(graph.native_autodiff_operations for graph in self.graphs),
            "native_autodiff_regions": sum(graph.native_autodiff_regions for graph in self.graphs),
            "legacy_backward_operations": [operation for graph in self.graphs for operation in graph.legacy_backward_operations],
            "fused_deepopt_regions": self.fused_deepopt_regions,
            "memory_optimization_regions": sum(graph.memory_optimization_regions for graph in self.graphs),
            "fusion_candidates": candidates,
            "pack_sharing_decisions": [{"graph_index": index, **decision}
                for index, graph in enumerate(self.graphs) for decision in graph.pack_sharing_decisions],
            "effective_compiler_options": [graph.effective_compiler_options for graph in self.graphs],
            "static_regions": static_regions,
            "pointwise_cost_estimates": [
                {"graph_index": index, **estimate}
                for index, graph in enumerate(self.graphs)
                for estimate in graph.pointwise_cost_estimates
            ],
            "source_module_sites": {phase: sorted(paths) for phase, paths in source_module_sites.items()},
            "observed_cuda_launches": None,
            **asdict(self),
        }


def compilation_report(model):
    """Return the report attached by DeepOpt's compilation helper, if selected."""
    return getattr(model, "_deepopt_compile_report", None)
