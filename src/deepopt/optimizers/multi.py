"""Present named, disjoint optimizers as one training-engine optimizer."""

from collections.abc import Mapping
from types import MappingProxyType

import torch


class MultiOptimizer(torch.optim.Optimizer):
    """Coordinate child optimizers whose layer assignments are already chosen.

    The caller's optimizer_factory(model, settings) owns parameter routing.
    Children must have disjoint parameters and support step() without a closure.
    The engine checks all gradients before stepping the wrapper. An exception
    inside a child's step is not transactional: restore the last checkpoint
    before continuing.
    """

    def __init__(self, optimizers: Mapping[str, torch.optim.Optimizer]):
        if not isinstance(optimizers, Mapping) or not optimizers:
            raise ValueError("MultiOptimizer requires a nonempty mapping of names to optimizers")
        self._optimizers = dict(optimizers)
        seen = set()
        for name, optimizer in self._optimizers.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Child optimizer names must be nonempty strings")
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise ValueError(f"Child {name!r} must be a torch.optim.Optimizer")
            if isinstance(optimizer, torch.optim.LBFGS):
                raise ValueError("MultiOptimizer requires closure-free optimizers; LBFGS is unsupported")
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in seen:
                        raise ValueError("Child optimizers must not share or repeat parameters")
                    seen.add(id(parameter))
        groups = [group for optimizer in self._optimizers.values() for group in optimizer.param_groups]
        self._initializing = True
        super().__init__(groups, defaults={})
        self._initializing = False
        self._bind_groups()

    @property
    def optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        return MappingProxyType(self._optimizers)

    def _bind_groups(self):
        # Keep the SAME dictionaries: a scheduler on the wrapper must change
        # the learning rates that each child actually uses during step().
        self.param_groups = [group for child in self._optimizers.values() for group in child.param_groups]
        self._group_structure = self._structure()

    def _structure(self):
        return tuple((id(group), tuple(id(p) for p in group["params"]))
                     for child in self._optimizers.values() for group in child.param_groups)

    def _validate_structure(self):
        if self._structure() != self._group_structure or tuple(id(g) for g in self.param_groups) != tuple(g[0] for g in self._group_structure):
            raise ValueError("Optimizer parameter groups changed; rebuild MultiOptimizer after changing layer assignments")

    def add_param_group(self, param_group):
        if not getattr(self, "_initializing", False):
            raise ValueError("Configure child parameter groups before constructing MultiOptimizer")
        super().add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True):
        self._validate_structure()
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MultiOptimizer does not support closures")
        self._validate_structure()
        for optimizer in self._optimizers.values():
            optimizer.step()

    @staticmethod
    def _type_name(optimizer):
        return f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"

    @staticmethod
    def _parameter_layout(optimizer):
        return [[{"shape": list(p.shape), "dtype": str(p.dtype)} for p in group["params"]]
                for group in optimizer.param_groups]

    def state_dict(self):
        self._validate_structure()
        return {"format_version": 1, "optimizers": {
            name: {"type": self._type_name(optimizer), "layout": self._parameter_layout(optimizer),
                   "state_dict": optimizer.state_dict()}
            for name, optimizer in self._optimizers.items()
        }}

    def load_state_dict(self, state_dict):
        self._validate_structure()
        if not isinstance(state_dict, dict) or state_dict.get("format_version") != 1:
            raise ValueError("Unsupported MultiOptimizer checkpoint format")
        saved = state_dict.get("optimizers")
        if not isinstance(saved, dict) or list(saved) != list(self._optimizers):
            raise ValueError("Child optimizer names and order must match the checkpoint")
        # Check every child before loading any state. Types and layouts prevent
        # accidentally loading Adam moments into an SGD branch or another shape.
        for name, optimizer in self._optimizers.items():
            record = saved[name]
            if not isinstance(record, dict) or record.get("type") != self._type_name(optimizer):
                raise ValueError(f"Optimizer type for {name!r} differs from checkpoint")
            if record.get("layout") != self._parameter_layout(optimizer):
                raise ValueError(f"Parameter layout for {name!r} differs from checkpoint")
            if not isinstance(record.get("state_dict"), dict):
                raise ValueError(f"Missing optimizer state for {name!r}")
        for name, optimizer in self._optimizers.items():
            optimizer.load_state_dict(saved[name]["state_dict"])
        # Child load_state_dict replaces group dictionaries; rebind so scheduler
        # updates continue to reach the restored children.
        self._bind_groups()
