"""Bounded eager dtype receipts, deliberately outside compiled/timed execution."""

from collections.abc import Mapping
from copy import deepcopy

from deepopt._bf16_policy import resolve_bf16_policy


class DTypeEdgeRecorder:
    """Observe module edges without retaining tensors, gradients or storages.

    This is an eager module inventory, not generated-kernel launch evidence or
    an allocator observer.  Full compiled graphs must use compiler/runtime
    receipts; adding hooks to a compilation would change the graph being tested.
    """

    def __init__(self, model, *, max_edges=4096, max_values_per_call=128, max_depth=12):
        for name, value in (("max_edges", max_edges), ("max_values_per_call", max_values_per_call),
                            ("max_depth", max_depth)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model
        self.max_edges, self.max_values_per_call, self.max_depth = max_edges, max_values_per_call, max_depth
        self._handles, self._edges, self._pending = [], [], {}
        self._omitted, self._errors, self._calls = 0, [], 0
        self._contracts = []
        self._active = False

    def _tensors(self, value):
        import torch

        remaining = self.max_values_per_call

        def visit(item, path, depth):
            nonlocal remaining
            if remaining <= 0 or depth > self.max_depth:
                self._omitted += 1
                return
            remaining -= 1
            if isinstance(item, torch.Tensor):
                yield path, item
            elif isinstance(item, Mapping):
                for index, (key, child) in enumerate(item.items()):
                    if remaining <= 0:
                        self._omitted += 1
                        break
                    label = key[:128] if isinstance(key, str) else str(index)
                    yield from visit(child, f"{path}.{label}", depth + 1)
            elif isinstance(item, (tuple, list)):
                for index, child in enumerate(item):
                    if remaining <= 0:
                        self._omitted += 1
                        break
                    yield from visit(child, f"{path}[{index}]", depth + 1)

        yield from visit(value, "value", 0)

    def _record(self, name, role, values, policy, call):
        for path, tensor in self._tensors(values):
            if len(self._edges) >= self.max_edges:
                self._omitted += 1
                break
            try:
                self._edges.append({"module": name[:1024], "role": role, "path": path,
                    "call": call, "bf16_policy": policy.as_dict(), "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype), "device": str(tensor.device), "layout": str(tensor.layout),
                    "requires_grad": bool(tensor.requires_grad), "is_leaf": bool(tensor.is_leaf),
                    "logical_bytes": int(tensor.numel() * tensor.element_size()),
                    "physical_storage_observed": False})
            except (TypeError, ValueError, RuntimeError) as exc:
                if len(self._errors) < 16:
                    self._errors.append(f"{name[:128]}: {type(exc).__name__}")

    def __enter__(self):
        if self._active:
            raise RuntimeError("DTypeEdgeRecorder is not reentrant")
        modules = tuple(self.model.named_modules())
        if any(getattr(module, "_compiled_call_impl", None) is not None for _, module in modules):
            raise ValueError("Dtype hooks require a separate eager diagnostic; use compiled runtime receipts")
        self._edges, self._pending, self._errors, self._contracts = [], {}, [], []
        self._omitted, self._calls, self._active = 0, 0, True
        try:
            for name, module in modules:
                path = name or "<root>"

                def before(owner, args, kwargs, path=path):
                    from .bf16 import invocation_bf16_policy

                    self._calls += 1
                    tensors = self._tensors((args, kwargs))
                    first = next(tensors, None)
                    policy = resolve_bf16_policy()
                    if first is not None:
                        plan = getattr(owner, "plan", None)
                        if plan is None:
                            plan = {"bf16_policy": getattr(owner, "_deepopt_bf16_policy", None)}
                        try:
                            policy = invocation_bf16_policy(plan, first[1])
                        except ValueError as exc:
                            if len(self._errors) < 16:
                                self._errors.append(f"{path[:128]}: {exc}")
                    import torch

                    nearest = (isinstance(owner, torch.nn.Upsample) and owner.mode == "nearest"
                               and policy.storage_enabled and first is not None
                               and first[1].dtype == torch.bfloat16)
                    self._pending.setdefault(path, []).append((self._calls, policy, nearest))
                    self._record(path, "input", (args, kwargs), policy, self._calls)

                def after(owner, args, kwargs, output, path=path):
                    call, policy, nearest = self._pending[path].pop()
                    self._record(path, "output", output, policy, call)
                    if nearest and len(self._contracts) >= self.max_edges:
                        self._omitted += 1
                    elif nearest:
                        import torch

                        self._contracts.append({"module": path, "call": call,
                            "operation": "nearest2d", "expected_dtype": "torch.bfloat16",
                            "actual_dtype": str(output.dtype) if isinstance(output, torch.Tensor) else None,
                            "valid": isinstance(output, torch.Tensor) and output.dtype == torch.bfloat16})

                self._handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                self._handles.append(module.register_forward_hook(after, with_kwargs=True, always_call=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()
        self._active = False
        if exc_type is not None and len(self._errors) < 16:
            self._errors.append(f"forward: {exc_type.__name__}")

    def report(self):
        complete = bool(self._edges) and not self._omitted and not self._errors
        return deepcopy({"scope": "eager_module_edges_only", "timing_eligible": False,
            "cuda_execution_verified": False, "complete": complete,
            "edge_count": len(self._edges), "omitted": self._omitted, "errors": self._errors,
            "edges": self._edges,
            "storage_policy_validation": {"scope": "declared_nearest_module_edges_only",
                "full_activation_graph_verified": False, "checked_edges": len(self._contracts),
                "valid": complete and all(item["valid"] for item in self._contracts),
                "contracts": self._contracts}})
