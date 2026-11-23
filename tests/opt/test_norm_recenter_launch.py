"""Interpret real forward launch policies and JIT bodies without a CUDA device."""

import ast
from importlib.resources import files
from math import prod
from types import SimpleNamespace
import unittest

import torch

from deepopt.opt.nn.activation.geometry import tensor_arguments, ReLUGeometry
from deepopt.opt.nn.normalization.geometry import NormGeometry, output_strides
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from deepopt.opt.nn.normalization.layer_state import LayerNormState
from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt.backends._shared.normalization import normalization_variant, normalization_kernel_scope, normalization_tuning, layer_backward_programs
from deepopt.opt.backends._shared.normalization import measured_layer_variant, measured_norm_device, measured_group_forward, channels_last_batch
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
                     normalization_variant=normalization_variant,
                     normalization_tuning=normalization_tuning,
                     layer_backward_programs=layer_backward_programs,
                     measured_layer_variant=measured_layer_variant, measured_norm_device=measured_norm_device,
                     measured_group_forward=measured_group_forward,
                     channels_last_batch=channels_last_batch,
                     ReLUGeometry=ReLUGeometry,
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
    paths = ("recenter.py", "layer/moments.py", "layer/launch.py", "layer/backward_launch.py") if layer else ("recenter.py", "__init__.py")
    for path in paths:
        nodes = [node for node in ast.parse(root.joinpath(path).read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("reduce_center", "reduce_layer_moments", "_reduce", "_forward", "_backward", "_reduce_scales")]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace, events


class NormRecenterLaunchTests(unittest.TestCase):
    def test_mandatory_layer_moments_fuse_rows_and_merge_wide_tiles(self):
        from deepopt.opt.backends.reference.normalization import layer as reference

        for bits in (1, 2, 4, 8):
            for width, group in ((33, 17), (768, 768), (5003, 17), (17003, 17003)):
                with self.subTest(bits=bits, width=width, group=group):
                    case = layer_cases.LayerNormKernelTests()
                    case.setUp()
                    host, events = host_namespace(case, True)
                    geometry = LayerNormGeometry((2, width), (width,), eps=.003)
                    x = torch.randn(2, width) * .02
                    dy, weight = torch.randn_like(x), torch.randn(width)
                    shifts, gains = [torch.empty(2) for _ in range(2)], [torch.empty(2) for _ in range(2)]
                    options = dict(weight=weight, bias=None, dtype=torch.float32, block_shape=None,
                        compression=dict(enabled=True, bits=bits, group_size=group), check_finite=False)
                    expected = reference.forward(x, geometry, center_shift=shifts[0], moment_gain=gains[0], **options)
                    actual = host["_forward"](x, geometry, center_shift=shifts[1], moment_gain=gains[1], **options)
                    torch.testing.assert_close(actual[0], expected[0], rtol=3e-5, atol=3e-6)
                    torch.testing.assert_close(shifts[1], shifts[0], rtol=3e-5, atol=3e-6)
                    torch.testing.assert_close(gains[1], gains[0], rtol=3e-5, atol=3e-6)
                    if width <= 768:
                        self.assertEqual(events, ["_row_forward"])
                        self.assertEqual(actual[-1]["moment_launches"], 0)
                    else:
                        self.assertIn("_merge_layer_moments", events)
                        self.assertGreater(actual[-1]["moment_scratch_bytes"], 0)
                    for variant in ("baseline", "layer_fused", "layer_tiled"):
                        args = dict(packed=actual[2].packed, x=x, mean=actual[2].mean, inverse=actual[2].inverse,
                            need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32,
                            center_shift=shifts[1], moment_gain=gains[1])
                        wanted = reference.backward(dy, weight, geometry, **args)
                        with normalization_kernel_scope(variant, row_chunk=1, max_programs=2):
                            got = host["_backward"](dy, weight, geometry, **args)
                        for a, b in zip(got, wanted):
                            torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)

    def test_new_layer_variants_match_reference_across_row_tiles_and_partial_gradients(self):
        from deepopt.opt.backends.reference.normalization import layer as reference

        for variant in ("layer_fused", "layer_tiled"):
            for chunk in (1, 4, 8):
                for compressed in (False, True):
                    case = layer_cases.LayerNormKernelTests()
                    case.setUp()
                    host, events = host_namespace(case, True)
                    geometry = LayerNormGeometry((37, 33), (33,))
                    x, dy, weight = torch.randn(37, 33), torch.randn(37, 33), torch.randn(33)
                    shift = torch.empty(37) if compressed else None
                    _, _, state, _, _, _ = reference.forward(x, geometry, weight=weight, bias=None,
                        dtype=torch.float32, block_shape=None, compression=dict(enabled=compressed, bits=4, group_size=17),
                        check_finite=False, center_shift=shift)
                    for dw, db in ((True, True), (True, False), (False, True)):
                        args = dict(packed=state.packed, x=x, mean=state.mean, inverse=state.inverse,
                            need_dx=True, need_dw=dw, need_db=db, input_dtype=torch.float32, center_shift=shift)
                        expected = reference.backward(dy, weight, geometry, **args)
                        events.clear()
                        with normalization_kernel_scope(variant, row_chunk=chunk, max_programs=7):
                            actual = host["_backward"](dy, weight, geometry, **args)
                        self.assertIn("_tiled_backward" if variant == "layer_tiled" else "_fused_backward", events)
                        for got, wanted in zip(actual, expected):
                            if wanted is None:
                                self.assertIsNone(got)
                            else:
                                torch.testing.assert_close(got, wanted, rtol=3e-5, atol=3e-5)

    def test_batch_shared_variant_keeps_the_original_forward_statistics(self):
        case = norm_cases.NormalizationKernelTests()
        case.setUp()
        host, events = host_namespace(case, False)
        geometry = NormGeometry("batch", (2, 4, 3, 5), 2)
        x, dy, w = torch.randn(geometry.shape), torch.randn(geometry.shape), torch.tensor([0., -.5, 1., 2.])
        with normalization_kernel_scope("batch_shared"):
            _, mean, inverse, packed = host["_forward"](x, w, None, geometry,
                dict(enabled=True, bits=2, group_size=17))
            self.assertNotIn("_small_forward", events)
            events.clear()
            host["_backward"](dy, w, geometry, packed=packed, x=x, mean=mean, inverse=inverse,
                input_dtype=torch.float32)
            self.assertNotIn("_partial_backward", events)
            self.assertEqual(events.count("_partial_affine"), 1)

    def test_candidate_layer_fixed_fp8_output_keeps_channel_layout_scale_and_observations(self):
        from deepopt.opt.backends.reference.normalization import layer as reference

        case = layer_cases.LayerNormKernelTests()
        case.setUp()
        host, _ = host_namespace(case, True)
        shape = (2, 3, 7)
        geometry = LayerNormGeometry(shape, (3, 7))
        x, dy, weight = torch.randn(shape), torch.randn(shape), torch.randn(3, 7)
        _, _, state, _, _, _ = reference.forward(x, geometry, weight=weight, bias=None,
            dtype=torch.bfloat16, block_shape=None, compression=dict(enabled=True, bits=4, group_size=17), check_finite=False)
        args = dict(packed=state.packed, x=x, mean=state.mean, inverse=state.inverse,
                    need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
        expected = reference.backward(dy, weight, geometry, **args)
        out = SimpleNamespace(checked=False, scale=torch.tensor([[2.]]), block_shape=None,
            dtype=torch.float8_e5m2, layout="channels", inverse=torch.empty(1, 1), observations=torch.empty(1), info={},
            allocate=lambda *args: None, wrap=lambda value: value)
        with normalization_kernel_scope("candidate"):
            actual = host["_backward"](dy, weight, geometry, gradient_output=out, **args)
        target = ReLUGeometry(shape, "channels")
        wanted = (target.matrix(expected[0]) * 2).clamp(-57344, 57344).to(torch.float8_e5m2)
        torch.testing.assert_close(actual[0].float(), wanted.float(), rtol=.25, atol=.001)
        torch.testing.assert_close(out.inverse, torch.tensor([[.5]]), rtol=0, atol=0)
        torch.testing.assert_close(out.observations, expected[0].abs().max().reshape(1), rtol=3e-5, atol=3e-5)
        for value, wanted in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(value, wanted, rtol=3e-5, atol=3e-5)

    def test_large_statistics_domains_do_not_leak_part_counts_into_small_codec_groups(self):
        from deepopt.opt.backends.reference import normalization as reference
        from deepopt.opt.quantization.activation import dequantize_unpack_activation

        for kind in ("batch", "group"):
            case = norm_cases.NormalizationKernelTests()
            case.setUp()
            host, events = host_namespace(case, False)
            geometry = NormGeometry(kind, (2, 4, 17, 65), 2)
            source = torch.randn(geometry.shape)
            compression = dict(enabled=True, bits=2, group_size=512)
            actual = host["_forward"](source, None, None, geometry, compression)
            wanted = reference.forward(source, None, None, geometry, compression)
            torch.testing.assert_close(actual[0], wanted[0], rtol=3e-5, atol=3e-5)
            decoded = dequantize_unpack_activation(actual[3], torch.float32)
            expected = dequantize_unpack_activation(wanted[3], torch.float32)
            torch.testing.assert_close(decoded, expected, rtol=3e-5, atol=3e-5)
            self.assertTrue(torch.isfinite(decoded).all())

    def test_candidate_small_domains_fuse_state_updates_and_pack_without_changing_forward_policy(self):
        from deepopt.opt.backends.reference import normalization as reference

        for kind in ("batch", "group"):
            for shape in ((2, 4, 3, 5), (2, 4, 17, 19)):
                case = norm_cases.NormalizationKernelTests()
                case.setUp()
                host, events = host_namespace(case, False)
                geometry = NormGeometry(kind, shape, 2)
                x = torch.randn(shape)
                weight, bias = torch.randn(4), torch.randn(4)
                outputs = []
                for bits, group in ((1, 17), (2, 512), (4, 1024), (8, 17)):
                    center = torch.empty(geometry.domains)
                    for compressed in (False, True):
                        running = dict(running_mean=torch.zeros(4), running_var=torch.ones(4), momentum=.1) if kind == "batch" else {}
                        expected_running = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in running.items()}
                        compression = dict(enabled=compressed, bits=bits, group_size=group)
                        wanted = reference.forward(x, weight, bias, geometry, compression, **expected_running)
                        events.clear()
                        info = {}
                        with normalization_kernel_scope("candidate"):
                            actual = host["_forward"](x, weight, bias, geometry, compression,
                                center_shift=center if compressed else None, execution_info=info, **running)
                        torch.testing.assert_close(actual[0], wanted[0], rtol=3e-5, atol=3e-5)
                        outputs.append(actual[0])
                        self.assertEqual(events, ["_small_forward"])
                        self.assertEqual(info["forward_launches"], 1)
                        for key, value in running.items():
                            if isinstance(value, torch.Tensor):
                                torch.testing.assert_close(value, expected_running[key], rtol=3e-5, atol=3e-5)
                        if compressed:
                            from deepopt.opt.quantization.activation import dequantize_unpack_activation
                            decoded = dequantize_unpack_activation(actual[3], torch.float32)
                            h = (geometry.matrix(x) - actual[1][:, None]) * actual[2][:, None]
                            torch.testing.assert_close(center, (decoded - h).mean(1), rtol=3e-5, atol=3e-6)
                for output in outputs[1:]:
                    torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

    def test_candidate_layer_backward_shares_decoded_rows_and_bounds_affine_scratch(self):
        from deepopt.opt.backends.reference.normalization import layer as reference

        for bits in (1, 2, 4, 8):
            for recenter in (False, True):
                case = layer_cases.LayerNormKernelTests()
                case.setUp()
                host, events = host_namespace(case, True)
                shape = (257 if bits == 8 else 5, 33)
                geometry = LayerNormGeometry(shape, (33,))
                x, dy = torch.randn(shape), torch.randn(shape)
                weight = torch.randn(33)
                shift = torch.empty(shape[0]) if recenter else None
                _, _, state, _, _, _ = reference.forward(x, geometry, weight=weight, bias=None,
                    dtype=torch.bfloat16, block_shape=None, compression=dict(enabled=True, bits=bits, group_size=17),
                    check_finite=False, center_shift=shift)
                for needs in ((True, True, True), (True, True, False), (True, False, True)):
                    args = dict(packed=state.packed, x=x, mean=state.mean, inverse=state.inverse,
                                need_dx=needs[0], need_dw=needs[1], need_db=needs[2], input_dtype=torch.float32,
                                center_shift=shift)
                    expected = reference.backward(dy, weight, geometry, **args)
                    info = {}
                    events.clear()
                    with normalization_kernel_scope("candidate"):
                        actual = host["_backward"](dy, weight, geometry, execution_info=info, **args)
                    for value, wanted in zip(actual, expected):
                        if wanted is None:
                            self.assertIsNone(value)
                        else:
                            torch.testing.assert_close(value, wanted, rtol=3e-5, atol=3e-5)
                    parts = (shape[0] + 127) // 128
                    self.assertEqual(events, ["_fused_backward"] + (["_affine_merge"] if parts > 1 else []))
                    self.assertEqual(info["backward_variant"], "fused_rows")
                    self.assertEqual(info["backward_scratch_bytes"], 2 * 33 * 4 * (parts + int(parts > 1)))

    def test_batch_backward_shares_one_unweighted_reduction_including_zero_gamma(self):
        from deepopt.opt.backends.reference import normalization as reference

        for kind in ("batch", "group"):
            for training in (False, True):
                for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
                    for compressed in (False, True):
                        with self.subTest(kind=kind, training=training, needs=needs, compressed=compressed):
                            case = norm_cases.NormalizationKernelTests()
                            case.setUp()
                            host, events = host_namespace(case, False)
                            geometry = NormGeometry(kind, (2, 4, 3, 5), 2, batch_stats=training)
                            x, dy = torch.randn(geometry.shape), torch.randn(geometry.shape)
                            weight = torch.tensor([0., -0.5, 1., 2.])
                            compression = dict(enabled=compressed, group_size=17, bits=2)
                            options = dict(running_mean=torch.zeros(4), running_var=torch.ones(4)) if not training else {}
                            _, mean, inverse, packed = reference.forward(x, weight, None, geometry, compression, **options)
                            arguments = dict(packed=packed, x=x, mean=mean, inverse=inverse,
                                             need_dx=needs[0], need_dw=needs[1], need_db=needs[2], input_dtype=torch.float32)
                            expected = reference.backward(dy, weight, geometry, **arguments)
                            with normalization_kernel_scope("candidate"):
                                actual = host["_backward"](dy, weight, geometry, **arguments)
                            for value, wanted in zip(actual, expected):
                                if wanted is None:
                                    self.assertIsNone(value)
                                else:
                                    torch.testing.assert_close(value, wanted, rtol=4e-5, atol=4e-5)
                            if kind == "batch" and training and needs[0]:
                                self.assertEqual(events.count("_partial_affine"), 1)
                                self.assertNotIn("_partial_backward", events)

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
