"""Private packed-source workspace reuse; CPU checks are not CUDA evidence."""

from contextlib import contextmanager, nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import os
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import convolution_reuse
from deepopt.opt.backends.native.convolution import ops
from deepopt.opt.nn.convolution import source_matrix
from deepopt.opt.nn.convolution.geometry import ConvGeometry
from deepopt.opt.quantization.activation import (quantize_pack_activation,
    _dequantize_unpack_spatial_activation)
from tests.compiler.test_native_convolution_order import _metadata_fixture, _TiedProjections


def _original_cuda_dx_batch_size():
    value = os.environ.get("DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE", "1")
    if (not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 127
            or str(int(value)) != value):
        raise ValueError("DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE must be a canonical integer from 1 through 127")
    return int(value)


def _rows(report):
    return [row for row in report.as_dict()["fusion_candidates"] if row["pattern"] == convolution_reuse.PATTERN]


@contextmanager
def _convolution_test_controls(*, allow_tf32):
    """Declare the intended library as well as its deterministic controls."""
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=allow_tf32):
        yield


def _convolution_control_snapshot():
    return dict(cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_available=torch.backends.cudnn.is_available(), cudnn_version=torch.backends.cudnn.version(),
        torch_version=str(torch.__version__), torch_git=str(torch.version.git_version),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_v8_disabled_environment=os.environ.get("TORCH_CUDNN_V8_API_DISABLED"))


def _cuda_convolution_backend_preflight(model, value, *, depthwise, mode, batch_size):
    """Inspect actual CUDA descriptors before a large numerical operation."""
    convolution = model[0]
    if value.device.type != "cuda" or any(parameter.device != value.device for parameter in model.parameters()):
        raise ValueError("Convolution qualification requires real CUDA operands on the same device")
    with torch.no_grad():
        weight = convolution.weight.to(value.dtype)
        backend = torch._C._select_conv_backend(value, weight, None,
            list(convolution.stride), list(convolution.padding), list(convolution.dilation),
            False, [0, 0], convolution.groups)
    row = dict(schema_version=1, scope="pre_forward_backend_selection_not_completed_VJP",
        input_shape=list(value.shape), input_stride=list(value.stride()), input_dtype=str(value.dtype),
        weight_shape=list(weight.shape), weight_stride=list(weight.stride()), weight_dtype=str(weight.dtype),
        master_weight_dtype=str(convolution.weight.dtype),
        family="depthwise3x3" if depthwise else "pointwise1x1", mode=mode, dx_batch_size=batch_size,
        selected_backend=str(backend), input_elements=value.numel(),
        exceeds_int32_element_count=value.numel() > (1 << 31) - 1,
        algorithm_controls=_convolution_control_snapshot(), tensor_files_written=0)
    print(json.dumps({"convolution_backend_preflight": row}), flush=True)
    destination = os.environ.get("DEEPOPT_TEST_CUDA_CONV_ARTIFACT_DIR")
    if destination:
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        stem = row["family"] + "-" + "x".join(map(str, value.shape)) + f"-dx{batch_size}-{mode}-backend.json"
        with (folder / stem).open("x") as stream:
            json.dump(row, stream, indent=2)
            stream.write("\n")
    if not row["algorithm_controls"]["cudnn_enabled"]:
        raise RuntimeError("Positive convolution qualification requires cuDNN enabled")
    if row["exceeds_int32_element_count"] and backend != torch._C._ConvBackend.Cudnn:
        raise RuntimeError("Original large convolution did not select the required cuDNN route: " + json.dumps(row))
    return row


def _checkpoint_model_optimizer(device):
    model = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False),
                          nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
    return model, optimizer


def _compile(model, x, mode, *, batch_size=2, shared=False):
    graph = torch.fx.symbolic_trace(deepcopy(model))
    report, captures = CompilationReport(), []
    transform = convolution_reuse.backward
    def capture(backward, context, **options):
        before = tuple(value for node in backward.nodes if getattr(node, "op", None) in
            ("convolution.native_dx_db", "convolution.native_dw") for value in node.outputs())
        result = transform(backward, context, **options)
        captures.append((backward, before))
        return result
    with patch.object(convolution_reuse, "backward", capture), autocast(x.device.type, dtype=torch.bfloat16,
            fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024),
            bf16_policy="bf16_storage" if x.device.type == "cuda" else "torch_amp"):
        ShapeProp(graph).propagate(x)
        executable = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True,
            native_conv_workspace_reuse=mode, native_conv_dx_batch_size=batch_size,
            share_compatible_packs=shared), report=report)
        output = executable(x)
    return graph, executable, output, report, captures


def _fixture(*, depthwise=False, shape=(5, 4, 7, 9), outputs=6, batch_size=2, mode="tiled_dx"):
    graph, context, spec, dx, dw = _metadata_fixture(shape=shape, outputs=shape[1] if depthwise else outputs,
        kernel=3 if depthwise else 1, padding=1 if depthwise else 0, groups=shape[1] if depthwise else 1)
    context = replace(context, native_conv_backward_order="source", native_conv_workspace_reuse=mode,
                      native_conv_dx_batch_size=batch_size)
    spec.context = context
    return graph, context, spec, dx, dw


def _model(depthwise=False, *, bias=True):
    return nn.Sequential(nn.Conv2d(4, 4 if depthwise else 6, 3 if depthwise else 1,
        padding=1 if depthwise else 0, groups=4 if depthwise else 1, bias=bias))


def _cotangent(shape, device, layout="contiguous"):
    value = torch.linspace(-.93, .81, torch.tensor(shape).prod().item(), device=device,
                           dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    if layout == "expanded":
        return value[:1].expand(shape)
    if layout == "transposed":
        return value.transpose(2, 3).contiguous().transpose(2, 3)
    return value


class NativeConvolutionReuseTests(unittest.TestCase):
    def test_positive_convolution_controls_enable_cudnn_and_restore_parent_settings(self):
        def flags():
            return (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32)

        original = flags()
        with torch.backends.cudnn.flags(enabled=False, benchmark=True, deterministic=False, allow_tf32=True):
            parent = flags()
            with _convolution_test_controls(allow_tf32=False):
                self.assertEqual(flags(), (True, False, True, False))
            self.assertEqual(flags(), parent)
        self.assertEqual(flags(), original)

    def setUp(self):
        torch.manual_seed(260621)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_default_typed_controls_and_unsupported_whole_batch_are_explicit(self):
        self.assertEqual(BackendConfig().native_conv_workspace_reuse, "off")
        self.assertEqual(BackendConfig().native_conv_dx_batch_size, 1)
        for invalid in (True, None, 1, "out", "TILED_DX"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "native_conv_workspace_reuse"):
                BackendConfig(native_conv_workspace_reuse=invalid)
        for invalid in (True, 0, -1, 1.5, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "native_conv_dx_batch_size"):
                BackendConfig(native_conv_dx_batch_size=invalid)
        with self.assertRaisesRegex(ValueError, "native AutoDiff"):
            BackendConfig(native_conv_workspace_reuse="tiled_dx", use_native_autodiff=False)
        for other in (dict(cuda_conv_dx_geometry=True), dict(native_conv_backward_order="dw_first"),
                      dict(donate_fresh_backward_add=True)):
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "qualified composition"):
                BackendConfig(native_conv_workspace_reuse="tiled_dx", **other)
        graph, context, _, dx, dw = _fixture(mode="whole_batch")
        original = tuple(graph.nodes)
        diagnostics = []
        self.assertEqual(convolution_reuse.backward(graph, context, diagnostics=diagnostics), ())
        self.assertEqual(tuple(graph.nodes), original)
        self.assertEqual(diagnostics[0]["reason"], "whole_batch_destination_unqualified")
        self.assertFalse(diagnostics[0]["whole_batch_destination_supported"])
        self.assertIs(graph.nodes[-3], dx)
        self.assertIs(graph.nodes[-2], dw)

    def test_original_cuda_contract_bound_is_explicit_and_strict(self):
        name = "DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE"
        with patch.dict(os.environ):
            os.environ.pop(name, None)
            self.assertEqual(_original_cuda_dx_batch_size(), 1)
            for value in ("1", "16", "127"):
                os.environ[name] = value
                self.assertEqual(_original_cuda_dx_batch_size(), int(value))
            for value in ("", "0", "128", "-1", "+1", "01", "1.0", "True", " 16", "１６"):
                os.environ[name] = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, name):
                    _original_cuda_dx_batch_size()

    def test_original_metadata_preserves_output_ids_and_has_bounded_partial_dx(self):
        for depthwise in (False, True):
            with self.subTest(depthwise=depthwise):
                graph, context, _, dx, dw = _fixture(depthwise=depthwise,
                    shape=(128, 768, 224, 224), outputs=112, batch_size=1)
                original = (*dx.values, *dw.outputs())
                diagnostics = []
                self.assertEqual(convolution_reuse.backward(graph, context, diagnostics=diagnostics),
                                 (convolution_reuse.PATTERN,))
                node = graph.nodes[-2]
                self.assertEqual(node.values, original)
                self.assertTrue(all(value.producer is node for value in original))
                self.assertTrue(all(value.fresh_allocation is None for value in original))
                self.assertTrue(convolution_reuse.is_private_workspace_consumer(node))
                self.assertEqual(node.scheduling_constraints["effects"], "private_decoded_workspace_overwrite")
                row = diagnostics[0]
                self.assertFalse(row["kernel_fusion"])
                contract = row["replacement_contract"]
                self.assertEqual(contract["decoded_workspace_bytes"], 9_865_003_008)
                self.assertEqual(contract["max_partial_dx_bytes"], 77_070_336)
                self.assertEqual(contract["metadata_placeholder_bytes"], 0)
                self.assertIsNone(contract["vendor_workspace_bytes"])
                self.assertIsNone(row["global_peak_saving_bytes"])

    def test_private_consumer_cannot_be_spoofed_or_gain_input_alias_permission(self):
        for mutation in ("target", "alias", "stream", "effect", "batch"):
            with self.subTest(mutation=mutation):
                graph, context, spec, _, _ = _fixture()
                convolution_reuse.backward(graph, context)
                node = graph.nodes[-2]
                self.assertTrue(convolution_reuse.is_private_workspace_consumer(node))
                if mutation == "target":
                    node.target = lambda *inputs: inputs
                elif mutation == "alias":
                    node.scheduling_constraints["owned_output_alias"] = {"donor": node.args[0].name}
                elif mutation == "stream":
                    node.stream = object()
                elif mutation == "effect":
                    node.scheduling_constraints["dW_before_overwrite"] = False
                else:
                    spec.context = replace(context, native_conv_dx_batch_size=5)
                self.assertFalse(convolution_reuse.is_private_workspace_consumer(node))

    def test_slice_bound_changes_final_schedule_and_implementation_identity(self):
        from deepopt.compiler.offload_plan import _digest, _schedule_descriptor
        from deepopt.compiler.scheduler import Scheduler

        signatures, fingerprints = [], []
        for batch_size in (1, 2):
            graph, context, _, _, _ = _fixture(batch_size=batch_size)
            diagnostics = []
            convolution_reuse.backward(graph, context, diagnostics=diagnostics)
            signatures.append(_digest(_schedule_descriptor(Scheduler(graph).run())))
            fingerprints.append(diagnostics[0]["implementation_fingerprint"])
            self.assertEqual(graph.nodes[-2].scheduling_constraints["dx_batch_size"], batch_size)
        self.assertNotEqual(*signatures)
        self.assertNotEqual(*fingerprints)

    def test_rejects_unsupported_geometry_selectivity_and_existing_effects(self):
        for case in ("batch", "compression", "bits", "dtype", "layout", "effect", "dx_only", "output_alias", "executor"):
            with self.subTest(case=case):
                graph, context, spec, dx, dw = _fixture()
                original = tuple(graph.nodes)
                if case == "batch":
                    context = replace(context, native_conv_dx_batch_size=5)
                elif case == "compression":
                    spec.compressed = False
                elif case == "bits":
                    spec.compression["bits"] = 4
                elif case == "dtype":
                    spec.compute_dtype = torch.float32
                elif case == "layout":
                    spec.input_format = torch.channels_last
                elif case == "effect":
                    dx.scheduling_constraints["mutation"] = True
                elif case == "dx_only":
                    spec.needs = (True, False, False)
                elif case == "output_alias":
                    dx.values[0].storage_source = "caller"
                else:
                    dw.target = lambda *args: args[0]
                diagnostics = []
                self.assertEqual(convolution_reuse.backward(graph, context, diagnostics=diagnostics), ())
                self.assertEqual(tuple(graph.nodes), original)
                self.assertFalse(diagnostics[0]["selected"])
                self.assertTrue(diagnostics[0]["reason"])

    def test_same_saved_pack_native_reductions_and_tail_storage_reuse(self):
        for depthwise in (False, True):
            with self.subTest(depthwise=depthwise):
                source = torch.randn(5, 4, 7, 9, dtype=torch.bfloat16)
                weight = torch.randn((4, 1, 3, 3) if depthwise else (6, 4, 1, 1), dtype=source.dtype)
                geometry = ConvGeometry(source.shape, weight.shape, (1, 1), (1, 1) if depthwise else (0, 0),
                    (1, 1), 4 if depthwise else 1)
                packed = quantize_pack_activation(source_matrix(source), group_size=1024, bits=2)
                original = tuple(value.clone() for value in (packed.data, packed.step, packed.minimum))
                restored = _dequantize_unpack_spatial_activation(packed, source.shape, dtype=source.dtype)
                gradient = _cotangent(geometry.output_shape, "cpu")
                calls, info = [], {}
                backward = ops.native_backward
                def observe(g, x, w, geo, mask, **kwargs):
                    values = backward(g, x, w, geo, mask, **kwargs)
                    calls.append((geo.batch_size, mask, x.untyped_storage().data_ptr(),
                        values[0].numel() * values[0].element_size() if values[0] is not None else None))
                    return values
                with torch.no_grad():
                    expected_dw = ops.native_weight_gradient(restored, gradient, geometry)
                    expected_dx, _, expected_db = ops.native_input_bias_gradients(gradient, weight, geometry, (True, False, True))
                    pointer = restored.untyped_storage().data_ptr()
                    with patch.object(ops, "native_backward", side_effect=observe):
                        dx, dw, db = ops._native_full_dw_tiled_dx(restored, gradient, weight, geometry,
                            batch_size=2, need_bias=True, execution_info=info)
                self.assertEqual(dx.untyped_storage().data_ptr(), pointer)
                self.assertEqual([row[0] for row in calls], [5, 5, 2, 2, 1])
                self.assertEqual([row[1] for row in calls], [(False, True, False), (False, False, True),
                    (True, False, False), (True, False, False), (True, False, False)])
                self.assertTrue(all(row[2] == pointer for row in calls))
                self.assertEqual(info["convolution_partial_dx_max_bytes"], 2 * 4 * 7 * 9 * 2)
                for actual, expected in ((dx, expected_dx), (dw, expected_dw), (db, expected_db)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for actual, expected in zip((packed.data, packed.step, packed.minimum), original):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_native_ad_preserves_full_reductions_strided_gradients_and_retained_graphs(self):
        for depthwise in (False, True):
            for layout in ("contiguous", "expanded", "transposed"):
                with self.subTest(depthwise=depthwise, layout=layout):
                    model, sample = _model(depthwise), torch.randn(5, 4, 7, 9, dtype=torch.bfloat16)
                    results = []
                    for mode in ("off", "tiled_dx"):
                        x = sample.clone().requires_grad_()
                        gm, _, output, report, captures = _compile(model, x, mode)
                        seed = _cotangent(output.shape, x.device, layout)
                        targets = (x, *gm.parameters())
                        first = torch.autograd.grad(output, targets, seed, retain_graph=True)
                        again = torch.autograd.grad(output, targets, seed * -.73)
                        self.assertTrue(all(value.dtype == torch.float32 for value in first[1:]))
                        self.assertEqual(report.runtime_fallbacks, 0)
                        self.assertEqual(any(row["selected"] for row in _rows(report)), mode == "tiled_dx")
                        if mode == "tiled_dx":
                            nodes = [node for node in captures[0][0].nodes if getattr(node, "op", None) == convolution_reuse.OPERATION]
                            self.assertEqual(len(nodes), 1)
                            self.assertTrue(convolution_reuse.is_private_workspace_consumer(nodes[0]))
                        results.append((output.detach(), first, again))
                    for actual, expected in zip((results[1][0], *results[1][1], *results[1][2]),
                                                (results[0][0], *results[0][1], *results[0][2])):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_packs_tied_weights_and_interleaved_forward_owners(self):
        model = _TiedProjections()
        sample = torch.randn(5, 4, 7, 9, dtype=torch.bfloat16)
        results = []
        for mode in ("off", "tiled_dx"):
            x = sample.clone().requires_grad_()
            gm, executable, first, report, _ = _compile(model, x, mode, shared=True)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                    activation_compression=dict(enabled=True, bits=2, group_size=1024)):
                second = executable(x * .61)
            targets = (x, *gm.parameters())
            second_grads = torch.autograd.grad(second, targets, _cotangent(second.shape, "cpu"))
            first_grads = torch.autograd.grad(first, targets, _cotangent(first.shape, "cpu"), retain_graph=True)
            repeated = torch.autograd.grad(first, targets, _cotangent(first.shape, "cpu"))
            for a, b in zip(first_grads, repeated):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            if mode == "tiled_dx":
                self.assertEqual(sum(row["selected"] for row in _rows(report)), 2)
            results.append((*first_grads, *second_grads))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_external_input_does_not_gain_an_unrequested_gradient(self):
        model = nn.Sequential(nn.Conv2d(4, 4, 1, bias=False), nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False))
        x = torch.randn(5, 4, 7, 9, dtype=torch.bfloat16)
        gm, _, output, report, _ = _compile(model, x, "tiled_dx")
        gradients = torch.autograd.grad(output, tuple(gm.parameters()), _cotangent(output.shape, "cpu"))
        self.assertTrue(all(value.dtype == torch.float32 and bool(value.isfinite().all()) for value in gradients))
        self.assertIsNone(x.grad)
        self.assertEqual(sum(row["selected"] for row in _rows(report)), 1)

    def test_selective_masks_stay_on_the_original_native_path(self):
        for need_x, need_w, need_b in ((False, True, True), (True, False, True), (True, False, False)):
            with self.subTest(needs=(need_x, need_w, need_b)):
                model = _model()
                model[0].weight.requires_grad_(need_w)
                model[0].bias.requires_grad_(need_b)
                x = torch.randn(5, 4, 7, 9, dtype=torch.bfloat16, requires_grad=need_x)
                gm, _, output, report, _ = _compile(model, x, "tiled_dx")
                targets = tuple(value for value in (x, *gm.parameters()) if value.requires_grad)
                gradients = torch.autograd.grad(output, targets, _cotangent(output.shape, "cpu"))
                self.assertFalse(any(row["selected"] for row in _rows(report)))
                self.assertEqual(len(gradients), len(targets))

    def test_CPU_singleton_decode_view_rejects_before_runtime_without_full_clone(self):
        for shape, outputs in (((5, 4, 1, 1), 6), ((5, 1, 7, 9), 3)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                gm, _, output, report, _ = _compile(nn.Sequential(nn.Conv2d(shape[1], outputs, 1, bias=False)), x, "tiled_dx")
                gradients = torch.autograd.grad(output, (x, *gm.parameters()), _cotangent(output.shape, "cpu"))
                self.assertEqual(_rows(report)[0]["reason"], "cpu_singleton_decode_layout_not_qualified")
                self.assertFalse(_rows(report)[0]["selected"])
                self.assertTrue(all(bool(value.isfinite().all()) for value in gradients))
                self.assertEqual(report.runtime_fallbacks, 0)

    def test_partial_failure_does_not_mutate_original_owners_or_retained_saved_state(self):
        x = torch.randn(5, 4, 7, 9, dtype=torch.bfloat16, requires_grad=True)
        gm, _, output, _, _ = _compile(_model(True), x, "tiled_dx")
        owners = (x, *gm.parameters())
        snapshots = tuple(value.detach().clone() for value in owners)
        versions = tuple(value._version for value in owners)
        seed = _cotangent(output.shape, "cpu")
        backward = ops.native_backward
        calls = []
        def fail(g, source, weight, geometry, mask, **kwargs):
            if mask == (True, False, False) and geometry.batch_size < 5:
                calls.append(geometry.batch_size)
                if len(calls) == 2:
                    raise RuntimeError("partial-dx-primary")
            return backward(g, source, weight, geometry, mask, **kwargs)
        with patch.object(ops, "native_backward", side_effect=fail), self.assertRaisesRegex(RuntimeError, "partial-dx-primary"):
            torch.autograd.grad(output, owners, seed, retain_graph=True)
        self.assertEqual(tuple(value._version for value in owners), versions)
        for actual, expected in zip(owners, snapshots):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertIsNone(actual.grad)
        recovered = torch.autograd.grad(output, owners, seed, retain_graph=True)
        repeated = torch.autograd.grad(output, owners, seed)
        for actual, expected in zip(recovered, repeated):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_step_checkpoint_restore_and_next_native_update_keeps_workspace_control(self):
        self._checkpoint_roundtrip("cpu")

    def test_checkpoint_factory_places_parameters_before_optimizer_binding(self):
        # Meta exercises a real non-CPU placement that replaces Parameter
        # owners, without presenting this structural check as CUDA execution.
        model, optimizer = _checkpoint_model_optimizer(torch.device("meta"))
        parameters = tuple(model.parameters())
        self.assertTrue(all(value.device.type == "meta" for value in parameters))
        self.assertTrue(all(value.dtype == torch.float32 for value in parameters))
        self.assertEqual(tuple(id(value) for group in optimizer.param_groups for value in group["params"]),
                         tuple(id(value) for value in parameters))

    def _checkpoint_roundtrip(self, device):
        import deepopt
        from deepopt.compiler import compilation_report
        from deepopt.data import build_loaders
        from deepopt.engine.checkpointing.checkpoint import read_checkpoint
        from tests.checkpointing.test_compiled_operation_recipe import _SavedUpdate, _StopAfterSavedUpdate

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        self.addCleanup(deepopt.reset_config)
        self.addCleanup(deepopt.shutdown_distributed)
        raw = dict(training=dict(epochs=1, batch_size=5, learning_rate=.001, checkpoint_every_steps=1, monitor=None),
            runtime=dict(device=device, precision="bf16" if device == "cuda" else "fp32",
                bf16_policy="bf16_storage" if device == "cuda" else "torch_amp", checks=False,
                activation_compression=dict(enabled=True, bits=2, group_size=1024),
                compile=dict(enabled=True, backend="deepopt", dynamic=False, fullgraph=True,
                    options=dict(require_full_aot=True, use_native_autodiff=True,
                        native_conv_workspace_reuse="tiled_dx", native_conv_dx_batch_size=2))),
            logging=dict(level="ERROR"))
        images = torch.randn(10, 4, 7, 9)
        def step(model, batch):
            # On CPU this is an explicit BF16 operator scope within its FP32
            # training runtime; CUDA uses the configured BF16 storage recipe.
            with autocast(batch.device.type, dtype=torch.bfloat16, fp8=False,
                    bf16_policy="bf16_storage" if device == "cuda" else "torch_amp",
                    activation_compression=dict(enabled=True, bits=2, group_size=1024)):
                value = model(batch)
            loss = value.float().square().mean()
            return {"loss": deepopt.LossResult.from_mean(loss, weight=batch.shape[0]), "sample_count": batch.shape[0]}
        def components():
            model, optimizer = _checkpoint_model_optimizer(device)
            trainer = deepopt.Trainer(model, optimizer, step_fn=step, config=deepcopy(raw))
            loader, _, _ = build_loaders(images, images[:5], batch_size=5, seed=91,
                resumable=True, num_workers=0, persistent_workers=False)
            return trainer, loader
        def assert_selected(trainer):
            report = compilation_report(trainer.model)
            self.assertTrue(any(row["selected"] for row in _rows(report)))
            self.assertEqual(report.runtime_fallbacks, 0)
            self.assertEqual(trainer.model._deepopt_compile_config["options"]["native_conv_dx_batch_size"], 2)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer, loader = components()
            with self.assertRaises(_SavedUpdate):
                trainer.fit(loader, epochs=1, callbacks=[_StopAfterSavedUpdate(Path(directory))], progress_bar=False)
            payload = read_checkpoint(Path(directory) / "last_step.pt")
            self.assertEqual(payload["state"]["optimizer_steps"], 1)
            assert_selected(trainer)
            restored, restored_loader = components()
            checkpoint = deepopt.Checkpoint(directory)
            checkpoint.restore(restored, restored_loader)
            self.assertEqual(restored.state.optimizer_steps, 1)
            restored.fit(restored_loader, epochs=1, callbacks=[checkpoint], progress_bar=False)
            final = read_checkpoint(Path(directory) / "last.pt")
            self.assertEqual(final["state"]["optimizer_steps"], 2)
            assert_selected(restored)

    def test_runtime_rejects_views_aliases_and_nonfirst_order_before_overwrite(self):
        geometry = ConvGeometry((5, 4, 7, 9), (4, 4, 1, 1), (1, 1), (0, 0), (1, 1), 1)
        source = torch.randn(geometry.input_shape, dtype=torch.bfloat16)
        weight = torch.randn(geometry.weight_shape, dtype=torch.bfloat16)
        for case in ("view", "seed_alias", "dtype", "bound", "grad"):
            with self.subTest(case=case), torch.no_grad():
                x, g, batch = source.clone(), torch.randn_like(source), 2
                if case == "view":
                    x = x.view_as(x)
                elif case == "seed_alias":
                    g = x
                elif case == "dtype":
                    g = g.float()
                elif case == "bound":
                    batch = 5
                else:
                    x.requires_grad_()
                before = x.clone()
                with patch.object(ops, "native_backward", side_effect=AssertionError("must reject before numerical work")), self.assertRaises(ValueError):
                    ops._native_full_dw_tiled_dx(x, g, weight, geometry, batch_size=batch)
                torch.testing.assert_close(x, before, rtol=0, atol=0)


def _cuda_compare(test, depthwise, *, shape=(5, 4, 7, 9), outputs=6, batch_size=2, profile=False):
    torch.manual_seed(171128)
    channels = shape[1]
    model = nn.Sequential(nn.Conv2d(channels, channels if depthwise else outputs,
        3 if depthwise else 1, padding=1 if depthwise else 0, groups=channels if depthwise else 1, bias=False)).cuda()
    sample = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    results = []
    with _convolution_test_controls(allow_tf32=False):
        controls = _convolution_control_snapshot()
        for mode in ("off", "tiled_dx"):
            x = sample.clone().requires_grad_()
            _cuda_convolution_backend_preflight(model, x, depthwise=depthwise, mode=mode, batch_size=batch_size)
            gm, _, output, report, _ = _compile(model, x, mode, batch_size=batch_size)
            seed = _cotangent(output.shape, x.device)
            scope = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) if profile and mode == "tiled_dx" else nullcontext()
            with scope as profiler:
                gradients = torch.autograd.grad(output, (x, *gm.parameters()), seed, retain_graph=True)
                repeated = torch.autograd.grad(output, (x, *gm.parameters()), seed)
                torch.cuda.synchronize()
            test.assertEqual(any(row["selected"] for row in _rows(report)), mode == "tiled_dx")
            if mode == "tiled_dx":
                test.assertTrue(all(row["dx_batch_size"] == batch_size for row in _rows(report) if row["selected"]))
            test.assertEqual(report.runtime_fallbacks, 0)
            test.assertFalse(report.as_dict()["legacy_backward_operations"])
            for actual, expected in zip(gradients, repeated):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if profiler is not None:
                partial_calls = []
                for event in profiler.events():
                    if event.name != "aten::convolution_backward":
                        continue
                    parent = event.cpu_parent
                    while parent is not None and parent.name != "deepopt.convolution.private_workspace.partial_dx":
                        parent = parent.cpu_parent
                    if parent is not None:
                        partial_calls.append(event)
                test.assertTrue(partial_calls)
                test.assertTrue(all(event.input_shapes[1][0] <= batch_size for event in partial_calls))
                destination = os.environ.get("DEEPOPT_TEST_CUDA_CONV_ARTIFACT_DIR")
                if destination:
                    folder = Path(destination)
                    folder.mkdir(parents=True, exist_ok=True)
                    family = "depthwise3x3" if depthwise else "pointwise1x1"
                    stem = family + "-" + "x".join(map(str, shape)) + f"-dx{batch_size}"
                    trace_path, summary_path = folder / (stem + "-trace.json"), folder / (stem + "-summary.json")
                    if trace_path.exists() or summary_path.exists():
                        raise FileExistsError("Convolution contract artifacts must use a fresh destination")
                    # Chrome events and scalar shapes contain no tensor data.
                    # Vendor scratch remains observable evidence, not a claim
                    # that aten's output mask alone eliminates allocations.
                    profiler.export_chrome_trace(str(trace_path))
                    summary_path.write_text(json.dumps({"schema_version": 1, "tensor_files_written": 0,
                        "actual_CUDA_execution": True, "input_shape": list(shape), "dx_batch_size": batch_size,
                        "full_batch_dw_and_db": True, "original_batch_not_microbatched": True,
                        "retained_backward_executions": 2, "vendor_workspace_elimination_claimed": False,
                        "algorithm_controls": _convolution_control_snapshot(),
                        "partial_dx_calls": [event.input_shapes for event in partial_calls]}, indent=2) + "\n")
            results.append(tuple(value.detach() for value in gradients))
        for index, (actual, expected) in enumerate(zip(results[1], results[0])):
            test.assertEqual(actual.dtype, torch.bfloat16 if index == 0 else torch.float32)
            for first in range(0, actual.numel(), 1 << 20):
                # dW keeps the exact full-batch native reduction and BF16
                # boundary. dX is alternate native arithmetic at a changed N;
                # use the existing dtype-specific spatial-kernel budget.
                options = {} if index == 0 else dict(rtol=0, atol=0)
                torch.testing.assert_close(actual.reshape(-1)[first:first + (1 << 20)],
                    expected.reshape(-1)[first:first + (1 << 20)], **options)
    print(json.dumps({"contract": "native_convolution_private_workspace_reuse", "actual_CUDA_execution": True,
        "input_shape": list(shape), "family": "depthwise3x3" if depthwise else "pointwise1x1",
        "native_conv_dx_batch_size": batch_size, "full_batch_dw_preserved": True,
        "algorithm_controls": controls,
        "parameter_gradient_dtype": "torch.float32", "profiled_partial_dx_bound_checked": profile}), flush=True)


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA BF16 convolution execution")
class NativeConvolutionReuseCudaTests(unittest.TestCase):
    def test_same_native_full_dw_and_bounded_partial_dx_for_both_geometry_families(self):
        for depthwise in (False, True):
            with self.subTest(depthwise=depthwise):
                _cuda_compare(self, depthwise, profile=True)

    def test_strided_and_expanded_upstream_keep_full_dw_and_fp32_master_grads(self):
        for depthwise in (False, True):
            for layout in ("expanded", "transposed"):
                with self.subTest(depthwise=depthwise, layout=layout):
                    model, sample = _model(depthwise).cuda(), torch.randn(5, 4, 7, 9, device="cuda", dtype=torch.bfloat16)
                    rows = []
                    with _convolution_test_controls(allow_tf32=True):
                        for mode in ("off", "tiled_dx"):
                            x = sample.clone().requires_grad_()
                            gm, _, output, report, _ = _compile(model, x, mode)
                            grads = torch.autograd.grad(output, (x, *gm.parameters()), _cotangent(output.shape, x.device, layout))
                            self.assertEqual(any(row["selected"] for row in _rows(report)), mode == "tiled_dx")
                            rows.append(grads)
                    torch.testing.assert_close(rows[1][0], rows[0][0])
                    for actual, expected in zip(rows[1][1:], rows[0][1:]):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_independent_streams_and_interleaved_retained_graphs(self):
        sample = torch.randn(5, 4, 7, 9, device="cuda", dtype=torch.bfloat16)
        model = _model(True).cuda()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        outputs = []
        with _convolution_test_controls(allow_tf32=True):
            for stream in streams:
                with torch.cuda.stream(stream):
                    x = sample.clone().requires_grad_()
                    gm, executable, first, _, _ = _compile(model, x, "tiled_dx")
                    with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                            activation_compression=dict(enabled=True, bits=2, group_size=1024)):
                        second = executable(x * .71)
                    targets = (x, *gm.parameters())
                    late = torch.autograd.grad(second, targets, _cotangent(second.shape, x.device))
                    early = torch.autograd.grad(first, targets, _cotangent(first.shape, x.device), retain_graph=True)
                    repeat = torch.autograd.grad(first, targets, _cotangent(first.shape, x.device))
                    outputs.append((late, early, repeat))
            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
            for _, first, repeated in outputs:
                for a, b in zip(first, repeated):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for first, second in zip(outputs[0][1], outputs[1][1]):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_step_checkpoint_restore_and_next_native_update(self):
        NativeConvolutionReuseTests._checkpoint_roundtrip(self, "cuda")


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_CONV_REUSE") == "1",
                     "opt-in original B128 decoder convolution capacity")
class NativeConvolutionReuseOriginalShapeCudaTests(unittest.TestCase):
    def test_original_body4_depthwise(self):
        _cuda_compare(self, True, shape=(128, 768, 224, 224), batch_size=_original_cuda_dx_batch_size(), profile=True)

    def test_original_body8_pointwise(self):
        _cuda_compare(self, False, shape=(128, 768, 224, 224), outputs=112, batch_size=_original_cuda_dx_batch_size(), profile=True)


if __name__ == "__main__":
    unittest.main()
