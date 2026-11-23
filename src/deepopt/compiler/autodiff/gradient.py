"""Per-invocation backward edge selection; no live tensors in compile caches."""

import torch


class GradientPolicy:
    def __init__(self, module, root, internal_input, occurrence=0):
        self.module, self.root = module, root
        self.training, self.internal_input, self.occurrence = module.training, internal_input, occurrence
        self.paths = tuple(getattr(module, "_gradient_slots", {}).get(self.training, ()))
        self.key = getattr(module, "_compiler_gradient_key", "gradient:" + getattr(module, "_gradient_key", ""))
        self.slots = tuple(root.get_submodule(path) for path in self.paths) if self.paths else ()
        self.enabled = any(slot.edge.precision == "fp8" for slot in self.slots)

    def select(self, x):
        from deepopt.opt.precision.planner import _frame
        from deepopt.opt._autocast import _get_fp8_plan

        frame = _frame.get()
        index = self.occurrence
        if frame is not None and self.paths:
            index = frame.positions.get(self.key, 0)
            frame.positions[self.key] = index + 1
        if not self.paths:
            return None
        leaf_fp8 = (0 <= index < len(self.slots) and self.slots[index].edge.precision == "fp8"
                    and not self.internal_input and x.is_leaf and x.requires_grad)
        if (_get_fp8_plan(x.device.type) is False or leaf_fp8
                or index >= len(self.paths)):
            index = -1
        # Host metadata stays on CPU: reading the selector in backward never
        # synchronizes a CUDA tensor or retains an activation.
        return torch.tensor(index, device="cpu", dtype=torch.int64)

    def slot(self, selector):
        if selector is None:
            return None
        index = int(selector.item())
        if index < 0 or index >= len(self.slots):
            return None
        slot = self.slots[index]
        return slot if slot.edge.precision == "fp8" else None

    def record(self, selector, *, reused, transpose=False):
        index = int(selector.item()) if selector is not None else -1
        slot = self.slots[index] if 0 <= index < len(self.slots) else None
        if slot is not None:
            slot.last_input_execution = {"reused_fp8": reused, "independent_transpose_stages": int(transpose)}
