"""Interpret installed pooling launch policies/JIT bodies; no CUDA emulation claim."""

import ast
from importlib.resources import files
from math import prod, isinf
from types import SimpleNamespace
import warnings

import torch

from tests.opt.test_activation_kernels import Language
from tests.opt.test_relu_kernels import CPUCast, TypedPointer
from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.nn.pooling.state import PoolState
from deepopt.opt.quantization.indices import PackedIndices, empty_indices
from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.backends.reference.pooling import coefficient_geometry


class PoolLanguage(Language):
    uint32, uint64 = torch.uint32, torch.int64
    ceil = staticmethod(torch.ceil)
    trans = staticmethod(lambda value: value.transpose(0, 1))

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis, dtype=value.dtype)

    @staticmethod
    def gather(value, indices, axis):
        return torch.gather(value, axis, indices.long())

    @staticmethod
    def store(pointer, value, mask=True):
        return Language.store(pointer, torch.as_tensor(value), mask)

    @staticmethod
    def device_assert(condition, message):
        if not torch.as_tensor(condition).all():
            raise RuntimeError(message)


class Launch:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.owner.events.append((self.name, grid))
            for key in ("num_warps", "enable_fp_fusion", "debug"):
                kwargs.pop(key, None)
            self.owner.launch(self.name, grid, *args, **kwargs)
        return run


class KernelHarness:
    def __init__(self):
        self.language = PoolLanguage()
        self.events = []
        api = SimpleNamespace(cdiv=self.language.cdiv, next_power_of_2=lambda v: 1 << (int(v) - 1).bit_length())
        self.kernels = {"torch": torch, "tl": self.language, "triton": api, "libdevice": SimpleNamespace(pow=torch.pow)}
        root = files("deepopt.opt.backends.triton")
        for path in ("quantization/fp8.py", "quantization/activation/affine.py", "activation/common.py",
                     "activation/elementwise_forward.py", "quantization/activation/indices.py",
                     "pooling/common/addressing.py", "pooling/common/windows.py", "pooling/common/output.py",
                     "pooling/common/global_reduce.py",
                     "pooling/window/forward.py", "pooling/power/factors.py", "pooling/unpool/forward.py",
                     "pooling/window/backward.py"):
            nodes = [node for node in ast.parse(root.joinpath(path).read_text()).body if isinstance(node, ast.FunctionDef)
                     and any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list)]
            for node in nodes:
                node.decorator_list = []
            tree = CPUCast().visit(ast.Module(body=nodes, type_ignores=[]))
            exec(compile(ast.fix_missing_locations(tree), path, "exec"), self.kernels)
        self.host = dict(torch=torch, triton=api, prod=prod, isinf=isinf, warnings=warnings,
                         ReLUGeometry=ReLUGeometry, tensor_arguments=tensor_arguments, PoolState=PoolState,
                         PackedActivation=PackedActivation, PackedIndices=PackedIndices, empty_indices=empty_indices,
                         coefficient_geometry=coefficient_geometry, quantize_fp8=quantize_fp8)
        for name in self.kernels:
            if name.startswith("_") and callable(self.kernels[name]):
                self.host[name] = Launch(self, name)
        self.host["_pool_forward"] = Launch(self, "_forward")
        self.host["_input_gradient"] = Launch(self, "_backward")
        for path, names in (("pooling/common/launch.py", ("kernel_geometry", "candidate_count", "can_inline", "output_storage", "finish_output")),
                            ("pooling/launch.py", ("coefficient_state", "power_program_block", "fused_coefficient_block", "global_reduction", "launch_global", "_forward")),
                            ("pooling/backward_launch.py", ("_backward",))):
            nodes = [node for node in ast.parse(root.joinpath(path).read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
            for node in nodes:
                node.decorator_list = []
            exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), self.host)

    def launch(self, name, grid, *args, **kwargs):
        if isinstance(grid, int):
            grid = (grid,)
        converted = tuple(TypedPointer(v) if isinstance(v, torch.Tensor) else v for v in args)
        for p in range(grid[0]):
            for q in range(grid[1] if len(grid) > 1 else 1):
                self.language.programs = (p, q, 0)
                self.kernels[name](*converted, **{k: TypedPointer(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
