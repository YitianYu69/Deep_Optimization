"""Scale-aware gradient transport and per-call-site output scaling ownership."""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

from .activation_tensor import FP8Activation, activation_matrix, restore_activation
from .quantizer import quantize_fp8


def gradient_operand(value, layout):
    """Reuse physical storage only when its matrix orientation is compatible."""
    if isinstance(value, FP8Activation) and (value._activation_layout == layout or value.ndim == 2):
        return value.quantized(layout)
    return None


def gradient_matrix(value, layout):
    if isinstance(value, FP8Activation):
        value = value.dequantize(torch.float32)
    return activation_matrix(value, layout)


def gradient_bias(operand):
    if operand.data.device.type == "cuda" and not isinstance(operand.data, FakeTensor):
        from ..backends.triton.gradient import bias

        return bias(operand)
    return operand.dequantize().sum(0, dtype=torch.float32)


def gradient_transpose(value, dtype, blocks, check_finite=None, *, layout="last"):
    from .._execution import finite_checks_enabled

    checked = finite_checks_enabled() if check_finite is None else check_finite
    q = gradient_operand(value, layout)
    if (q is not None and q.data.device.type == "cuda" and not isinstance(q.data, FakeTensor)
            and not checked and blocks is not None and blocks[0] * blocks[1] <= 16384):
        from ..backends.triton.gradient import transpose_quantize

        return transpose_quantize(q, dtype, blocks)
    return quantize_fp8(gradient_matrix(value, layout).T, fp8_dtype=dtype, block_shape=blocks, check_finite=checked)


class GradientSlot(nn.Module):
    """One immutable edge decision and its independent delayed state/report."""

    def __init__(self, plan, options, *, device):
        super().__init__()
        self.edge = plan
        self.scaling = None
        self.last_execution = None
        self.last_input_execution = None
        if plan.precision == "fp8" and plan.scaling_recipe == "delayed":
            from ..scaling.delayed import DelayedScalingState

            dtype = getattr(torch, "float8_" + plan.dtype)
            self.scaling = DelayedScalingState(options, {role: dtype for role in ("activation", "weight", "grad")}, device=device)

    @contextmanager
    def output(self, shape, logical_dtype, *, layout=None, check_finite=None, training=None):
        from ..scaling.delayed import fp8_scaling_scope, current_scaling_batch

        state = self.scaling
        training = self.training if training is None else training
        with fp8_scaling_scope() if state is not None else nullcontext():
            with state._lock if state is not None else nullcontext():
                if state is not None:
                    state.ordered()
                    if state._pending_owner is not None and state._pending_owner is not current_scaling_batch():
                        raise RuntimeError("Overlapping backward boundaries cannot share gradient scales")
                output = GradientOutput(self, tuple(shape), logical_dtype, layout or self.edge.layout, check_finite, training)
                yield output
                if state is not None and output.observations is not None and training:
                    current_scaling_batch().record(state, 2, output.observations[:output.observation_count])
                self.last_execution = dict(output.info)


class GradientSlots(nn.ModuleDict):
    """Numerical state remains an identity if a custom container visits children."""

    def forward(self, input):
        return input


class GradientOutput:
    """Ephemeral storage bound to a cached edge; never cached with live activations."""

    def __init__(self, slot, shape, logical_dtype, layout, check_finite=None, training=True):
        from .._execution import finite_checks_enabled

        self.checked = finite_checks_enabled() if check_finite is None else check_finite
        self.training = training
        self.slot, self.shape, self.logical_dtype, self.layout = slot, shape, logical_dtype, layout
        self.dtype = getattr(torch, "float8_" + slot.edge.dtype)
        self.block_shape = slot.edge.block_shape
        self.scale = slot.scaling.scales[2:3] if slot.scaling is not None else None
        self.inverse = self.observations = None
        self.observation_count = 0
        self.info = {"precision": "fp8", "layout": layout, "dtype": str(self.dtype),
                     "scale_mode": slot.edge.scaling_recipe, "fused": False, "output_cast_launches": 0,
                     "scale_reduction_launches": 0, "reason": slot.edge.reason}

    def allocate(self, rows, cols, device, programs=0):
        if self.scale is not None and self.scale.device != torch.device(device):
            raise ValueError("Gradient output scales and operands must share a device")
        grid = ((1, 1) if self.block_shape is None else
                ((rows + self.block_shape[0] - 1) // self.block_shape[0],
                 (cols + self.block_shape[1] - 1) // self.block_shape[1]))
        self.inverse = torch.empty(grid, device=device, dtype=torch.float32)
        if self.scale is not None and self.training and programs:
            self.observations = torch.empty(programs, device=device, dtype=torch.float32)
            self.observation_count = programs

    def can_fuse(self, value, *, bm=16, bn=16):
        storage = value._data if isinstance(value, FP8Activation) else value
        if storage.device.type != "cuda" or isinstance(storage, FakeTensor) or self.checked:
            return False
        if self.scale is not None:
            return True
        return (self.block_shape is not None and bm % self.block_shape[0] == 0
                and bn % self.block_shape[1] == 0)

    def wrap(self, matrix):
        if self.checked and matrix.numel() and not isinstance(matrix, FakeTensor) and matrix.device.type != "meta":
            if not torch.isfinite(matrix.float()).all() or not torch.isfinite(self.inverse).all():
                raise ValueError("FP8 gradient transport requires finite values and scales")
        self.info["fused"] = matrix.device.type == "cuda" and not isinstance(matrix, FakeTensor)
        return FP8Activation(matrix, self.inverse, self.shape, self.block_shape, self.layout, self.logical_dtype)

    def convert(self, value, *, matrix=False):
        values = value if matrix else gradient_matrix(value, self.layout)
        state = self.slot.scaling
        if not values.numel() or isinstance(values, FakeTensor) or values.device.type == "meta":
            self.allocate(*values.shape, values.device)
            return FP8Activation(torch.empty_like(values, dtype=self.dtype), self.inverse, self.shape,
                                 self.block_shape, self.layout, self.logical_dtype)
        if state is not None:
            q = state.quantize(values, "grad", training=self.training, check_finite=self.checked)[0]
        else:
            q = quantize_fp8(values.float(), fp8_dtype=self.dtype, block_shape=self.block_shape, check_finite=self.checked)
        self.info.update(fused=False, output_cast_launches=int(values.device.type == "cuda" and not isinstance(values, FakeTensor)),
                         output_cast_stages=1,
                         workspace_dtype=str(values.dtype), scale_mode="delayed" if state is not None else "current")
        if values.device.type == "cuda" and not isinstance(values, FakeTensor) and state is None:
            rows, cols = values.shape
            br, bc = self.block_shape or (rows, cols)
            elements = min(br, rows) * min(bc, cols)
            limit = 4096 if self.block_shape is None else 16384
            if elements > limit:
                count = (elements + 4095) // 4096
                reductions = 2
                while count > 1024:
                    count = (count + 1023) // 1024
                    reductions += 1
                self.info["scale_reduction_launches"] = reductions
        return FP8Activation(q.data, q.scale_inv, self.shape, q.block_shape, self.layout, self.logical_dtype)


def enabled(slot):
    return getattr(getattr(slot, "edge", None), "precision", None) == "fp8"


def native_gradient(value):
    return value.dequantize() if isinstance(value, FP8Activation) else value


def matrix_restore(value, shape, layout):
    return restore_activation(value, shape, layout)
