"""Interpret real forward launch policies and JIT bodies without a CUDA device."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.nn.activation.geometry import tensor_arguments
from deepopt.opt.nn.normalization.geometry import NormGeometry, output_strides
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from deepopt.opt.nn.normalization.layer_state import LayerNormState
from deepopt.opt.quantization.activation import PackedActivation
from tests.opt import test_layer_norm_kernels as layer_cases
from tests.opt import test_normalization_kernels as norm_cases


class Launch:
    def __init__(self, case, name, events):
        self.case, self.name, self.events = case, name, events

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.events.append(self.name)
            kwargs.pop("num_warps", None)
            kwargs.pop("enable_fp_fusion", None)
            self.case.launch(self.name, grid if len(grid) > 1 else (grid[0], 1), *args, **kwargs)
        return call


def host_namespace(case, layer):
    events = []
    namespace = dict(torch=torch, prod=prod, PackedActivation=PackedActivation, LayerNormState=LayerNormState,
                     tensor_arguments=tensor_arguments, output_strides=output_strides,
                     triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                            next_power_of_2=lambda v: 1 << (v - 1).bit_length()))
    for name, value in case.k.items():
        if callable(value):
            namespace[name] = Launch(case, name, events)
    # Addressing metadata is Python, not a GPU kernel.
    if not layer:
        namespace["input_arguments"] = case.k["input_arguments"]
    root = files("deepopt.opt.backends.triton").joinpath("normalization")
    paths = ("recenter.py", "layer/launch.py" if layer else "__init__.py")
    for path in paths:
        nodes = [node for node in ast.parse(root.joinpath(path).read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("reduce_center", "_reduce", "_forward", "_reduce_scales")]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace, events


class NormRecenterLaunchTests(unittest.TestCase):
    def test_batch_and_group_complete_and_partial_domains(self):
        for kind in ("batch", "group"):
            for shape, group in (((2, 4, 4, 4), 512), ((2, 4, 3, 5), 17), ((2, 4, 1, 2501), 17003)):
                case = norm_cases.NormalizationKernelTests()
                case.setUp()
                host, events = host_namespace(case, False)
                geometry = NormGeometry(kind, shape, 2)
                source = torch.randn(shape, dtype=torch.bfloat16)
                results, records = [], []
                for enabled in (False, True):
                    events.clear()
                    info = {}
                    shift = torch.empty(geometry.domains) if enabled else None
                    results.append(host["_forward"](source, None, None, geometry,
                                   dict(enabled=True, bits=2, group_size=min(group, geometry.elements)),
                                   center_shift=shift, execution_info=info)[0])
                    records.append(list(events))
                    complete = min(group, geometry.elements) == geometry.elements and geometry.elements <= 4096
                    self.assertEqual(info["recenter_launches"], int(enabled and not complete))
                    self.assertEqual(info["recenter_scratch_bytes"] > 0, enabled and not complete)
                torch.testing.assert_close(*results, rtol=0, atol=0)
                self.assertEqual([name for name in records[1] if name != "_merge_center"], records[0])

    def test_layer_row_prepass_and_chunked_epilogues(self):
        cases = ((33, 17, torch.bfloat16, None, False),
                 (33, 17, torch.float8_e4m3fn, None, False),  # tensorwise row prepass
                 (33, 17, torch.float8_e5m2, (2, 8), False),  # general scale tiles
                 (768, 512, torch.float8_e4m3fn, None, True),
                 (17003, 17003, torch.bfloat16, None, False))
        for width, group, dtype, block, provided in cases:
            case = layer_cases.LayerNormKernelTests()
            case.setUp()
            host, events = host_namespace(case, True)
            geometry = LayerNormGeometry((3, width), (width,))
            source = torch.randn(3, width, dtype=torch.bfloat16)
            results, records = [], []
            for enabled in (False, True):
                events.clear()
                shift = torch.empty(3) if enabled else None
                result = host["_forward"](source, geometry, weight=None, bias=None, dtype=dtype, block_shape=block,
                                           compression=dict(enabled=True, bits=2, group_size=group), check_finite=False,
                                           scale=torch.ones(1, 1) if provided else None, center_shift=shift)
                results.append(result[:2])
                records.append(list(events))
                expected_extra = enabled and (width > 8192 or block is not None)
                self.assertEqual(result[-1]["recenter_launches"], int(expected_extra))
                self.assertEqual(result[-1]["recenter_scratch_bytes"] > 0, expected_extra)
            torch.testing.assert_close(results[0][0].float(), results[1][0].float(), rtol=0, atol=0)
            if results[0][1] is not None:
                torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
            self.assertEqual([name for name in records[1] if name != "_merge_center"], records[0])


if __name__ == "__main__":
    unittest.main()
