"""CPU execution of the installed kernels' indexing and scale arithmetic.

This interprets the Python bodies of @triton.jit functions with torch tensors.
It does not compile Triton or validate CUDA instructions. It catches address,
mask, scale-boundary, output-ownership and split-reduction errors in the actual
kernel source, rather than testing a separate implementation of convolution.
"""

import ast
from importlib.resources import files
from types import SimpleNamespace

import torch

from deepopt.opt.backends.triton.convolution.scales import scale_chunk_length


class Pointer:
    def __init__(self, tensor=None, *, storage=None, offset=0):
        if storage is None:
            elements = tensor.untyped_storage().nbytes() // tensor.element_size()
            storage = tensor.as_strided((elements,), (1,), storage_offset=0)
            offset = tensor.storage_offset()
        self.storage, self.offset = storage, offset

    def __add__(self, offset):
        return Pointer(storage=self.storage, offset=self.offset + offset)


class KernelLanguage:
    constexpr = object
    float32, int32, int64 = torch.float32, torch.int32, torch.int64

    def __init__(self):
        self.programs = (0, 0, 0)
        self.dots = 0

    def program_id(self, axis):
        return torch.tensor(self.programs[axis], dtype=torch.int32)

    @staticmethod
    def arange(start, end):
        return torch.arange(start, end, dtype=torch.int32)

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def cdiv(value, divisor):
        return (value + divisor - 1) // divisor

    @staticmethod
    def minimum(a, b):
        return torch.minimum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def load(pointer, mask=True, other=0.0):
        indices, mask = torch.broadcast_tensors(torch.as_tensor(pointer.offset).long(), torch.as_tensor(mask))
        if ((indices[mask] < 0) | (indices[mask] >= pointer.storage.numel())).any():
            raise AssertionError("unmasked out-of-bounds kernel load")
        safe = torch.where(mask, indices, 0)
        if pointer.storage.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # CPU advanced indexing is implemented for bytes on every supported
            # torch version, including versions lacking FP8 index kernels.
            values = pointer.storage.view(torch.uint8)[safe].view(pointer.storage.dtype)
            return torch.where(mask, values.float(), other).to(values.dtype)
        return torch.where(mask, pointer.storage[safe], other)

    @staticmethod
    def store(pointer, values, mask=True):
        indices, values, mask = torch.broadcast_tensors(
            torch.as_tensor(pointer.offset).long(), values, torch.as_tensor(mask))
        if ((indices[mask] < 0) | (indices[mask] >= pointer.storage.numel())).any():
            raise AssertionError("unmasked out-of-bounds kernel store")
        pointer.storage[indices[mask]] = values[mask].to(pointer.storage.dtype)

    def dot(self, a, b, acc=None, *, out_dtype, **kwargs):
        if a.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or b.dtype not in (
            torch.float8_e4m3fn, torch.float8_e5m2,
        ):
            raise AssertionError("convolution multiplication operands must remain FP8")
        if kwargs:
            raise AssertionError("FP8 dots must not select a floating-point emulation precision")
        self.dots += 1
        result = a.float() @ b.float()
        return (result if acc is None else result + acc).to(out_dtype)

    @staticmethod
    def sum(value, axis):
        return value.sum(dim=axis)


def kernel_namespace():
    language = KernelLanguage()
    namespace = {"tl": language, "_scale_chunk_length": scale_chunk_length,
                 "triton": SimpleNamespace(cdiv=language.cdiv),
                 "range": lambda *args: (torch.tensor(value, dtype=torch.int32)
                                         for value in range(*map(int, args)))}
    package = files("deepopt.opt.backends.triton.convolution")
    for filename in ("common.py", "forward.py", "backward.py", "weight_gradient_blockwise.py"):
        tree = ast.parse(package.joinpath(filename).read_text())
        functions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(decorator, ast.Attribute) and decorator.attr == "jit"
                for decorator in node.decorator_list
            ):
                node.decorator_list = []
                functions.append(node)
        exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    return language, namespace
