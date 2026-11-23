"""Labeled exact normalization fusion diagnostics; tensor payloads stay in RAM."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch


def _write_text(root, relative, text):
    data = text.encode("utf-8")
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError("Refusing to overwrite an existing fusion diagnostic artifact")
    if not path.exists():
        path.write_bytes(data)
    return dict(path=str(relative), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _compiled_code(compiled, root, prefix):
    assembly = getattr(compiled, "asm", None)
    result = dict(compiled_kernel_type=type(compiled).__name__, files=[])
    if assembly is None:
        result["unavailable_reason"] = "kernel.run returned no accessible compiled assembly"
        return result
    for kind in ("ttir", "ttgir", "llir", "ptx"):
        source = assembly.get(kind)
        if source is None:
            continue
        if isinstance(source, bytes):
            source = source.decode("utf-8")
        if not isinstance(source, str):
            raise TypeError("Fusion compiler diagnostics require textual assembly")
        data = source.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        entry = dict(kind=kind, sha256=digest, bytes=len(data))
        if root is not None:
            entry.update(_write_text(root, Path("kernel-code") / f"kernel-{digest}.{kind}", source))
            entry["invocation"] = str(prefix)
        result["files"].append(entry)
    return result


def _compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(exact=False, shape=list(actual.shape), dtype=str(actual.dtype),
                    expected_shape=list(expected.shape), expected_dtype=str(expected.dtype), reason="metadata")
    exact = bool(torch.equal(actual, expected))
    bitwise = bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))
    result = dict(exact=exact, bitwise_equal=bitwise, shape=list(actual.shape), dtype=str(actual.dtype),
                  mismatched_elements=0, max_absolute_error=0.0, relative_l2_error=0.0)
    if not exact:
        difference = actual.double() - expected.double()
        result.update(mismatched_elements=int((actual != expected).sum().item()),
            max_absolute_error=float(difference.abs().max().item()) if difference.numel() else 0.0,
            relative_l2_error=float((difference.norm() / expected.double().norm().clamp_min(1e-300)).item()))
    return result


def _metadata(info):
    missing = object()

    def scalar(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (tuple, list)):
            items = [scalar(item) for item in value]
            if all(item is not missing for item in items):
                return tuple(items) if isinstance(value, tuple) else items
        return missing

    result = {}
    for key, value in info.items():
        selected = scalar(value)
        if selected is not missing:
            result[key] = selected
    return result


@contextmanager
def _capture(root, prefix):
    from deepopt.opt.backends.triton import normalization
    from deepopt.opt.backends.triton.activation.elementwise import _activation_backward

    captured = dict(kernels=[], affine_partials=[])
    kernels = dict(partial_affine=normalization._partial_affine,
        partial_backward=normalization._partial_backward, input_gradient=normalization._input_gradient,
        merge_sums=normalization._merge_sums, activation_backward=_activation_backward,
        inline_adjusted_gradient=_inline_adjusted_gradient)
    with ExitStack() as stack:
        for name, kernel in kernels.items():
            original = kernel.run

            def launch(*args, _name=name, _original=original, **kwargs):
                compiled = _original(*args, **kwargs)
                captured["kernels"].append(dict(role=_name, **_compiled_code(compiled, root,
                    Path(prefix) / f"{_name}-{len(captured['kernels'])}")))
                if _name == "partial_affine":
                    partial = args[9]
                    if (partial.dtype != torch.float32 or tuple(partial.shape) != (2, 16, 8)):
                        raise ValueError("Unexpected affine partial geometry in bounded diagnostic")
                    # Enqueue the tiny copy on the same stream before the
                    # merger launches. Nothing is copied to disk or host.
                    captured["affine_partials"].append(partial.detach().clone())
                return compiled

            stack.enter_context(patch.object(kernel, "run", side_effect=launch))
        yield captured


if torch.cuda.is_available():
    import triton
    import triton.language as tl
    from deepopt.opt.backends.triton.normalization.activation import _adjust_spatial_gradient
    from deepopt.opt.backends.triton.normalization.common import _load_input

    @triton.jit
    def _inline_adjusted_gradient(dy, derivative, step, minimum, output,
                                  N: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                                  META: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = index < N
        n, c, s = index // (C * S), index // S % C, index % S
        value = _load_input(dy, dy, n, c, s, valid, C, S, META)
        adjusted = _adjust_spatial_gradient(value, dy, n, c, s, valid, C, S, (), "", "none",
                                            derivative, step, minimum, True)
        tl.store(output + index, adjusted.to(tl.bfloat16), valid)


class NativeNormFusionLayoutMetadataTests(unittest.TestCase):
    def test_compiler_text_hashes_and_immutable_output_paths(self):
        compiled = type("CodeOnly", (), {"asm": {"ttir": "module {}\n", "ptx": ".version 9.0\n"}})()
        with TemporaryDirectory() as directory:
            record = _compiled_code(compiled, directory, "cpu-format-only")
            self.assertEqual(len(record["files"]), 2)
            for entry in record["files"]:
                data = (Path(directory) / entry["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(len(data), entry["bytes"])
            _compiled_code(compiled, directory, "cpu-format-only")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                _write_text(directory, "cpu-format-only/report.json", "first")
                _write_text(directory, "cpu-format-only/report.json", "changed")

    def test_scalar_comparisons_label_one_ulp_without_tolerance(self):
        expected = torch.tensor([711.5, -2.0], dtype=torch.float32)
        actual = expected.clone()
        actual[0] = torch.nextafter(actual[0], torch.tensor(float("inf")))
        result = _compare(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual(result["mismatched_elements"], 1)
        self.assertEqual(result["max_absolute_error"], 6.103515625e-05)
        json.dumps(result, allow_nan=False)

    def test_metadata_preserves_scalar_strides_and_omits_tensors(self):
        result = _metadata(dict(stride=(0, 1, 0, 0), shape=[2, 16, 64, 64], tensor=torch.ones(1),
                                nested_tensor=[torch.ones(1)], source="CPU metadata helper only"))
        self.assertEqual(result["stride"], (0, 1, 0, 0))
        self.assertEqual(result["shape"], [2, 16, 64, 64])
        self.assertNotIn("tensor", result)
        self.assertNotIn("nested_tensor", result)
        json.dumps(result, allow_nan=False)


@unittest.skipUnless(torch.cuda.is_available(), "Norm layout diagnostics require actual CUDA execution")
class NativeNormFusionLayoutCudaTests(unittest.TestCase):
    def test_same_saved_state_gradient_boundary_partials_and_repeatability(self):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self._diagnose_same_saved_state()

    def _diagnose_same_saved_state(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        from tests.compiler.test_native_norm_activation_residuals import fixture, execute
        from deepopt.compiler.autodiff.norm_activation_residuals import NormActivationResidual
        from deepopt.opt.backends.triton.normalization.common import input_arguments

        torch.manual_seed(20261121)
        shape = (2, 16, 64, 64)
        graph, trace, norm, activation, _, norm_node, act_node = fixture(shape)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        weight, bias = torch.randn(16, device="cuda"), torch.randn(16, device="cuda")
        root = os.environ.get("DEEPOPT_FUSION_DIAGNOSTIC_DIR")
        directory = Path(root) / "native-norm-layout-seed-20261121" if root else None
        report = dict(schema_version=1, scope="fixed-seed CUDA operator diagnostic, not a timing or full-model qualification",
            cuda_execution_evidence=True, tensor_payload_files=0, seed=20261121, shape=list(shape),
            same_original_forward_saves=True, arithmetic_tolerance=dict(rtol=0, atol=0), cases={},
            software=dict(torch=str(torch.__version__), cuda=torch.version.cuda, triton=triton.__version__),
            device=dict(index=torch.cuda.current_device(), name=torch.cuda.get_device_name(),
                        capability=list(torch.cuda.get_device_capability())),
            algorithm_controls=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark, enable_fp_fusion=False))
        with torch.no_grad():
            _, saved = execute(graph, x, weight, bias)
            bindings = {value.name: tensor for value, tensor in zip(trace.saved_tensors, saved)}
            norm_saved = tuple(bindings[value.name] for value in norm_node.values[1:])
            derivative = tuple(bindings[value.name] for value in act_node.values[1:])
            metadata = tuple(value.meta for value in (act_node.values[0], *norm_node.values[1:], *act_node.values[1:]))
            fused = NormActivationResidual(norm, activation, metadata)
            constant = torch.tensor(.173, dtype=torch.bfloat16, device="cuda")
            logical = constant.expand(shape).clone()
            gradient_cases = dict(contiguous=logical, expanded_zero=constant.expand(shape),
                transposed=logical.transpose(-1, -2).contiguous().transpose(-1, -2),
                stride2=torch.full((*shape[:-1], shape[-1] * 2), .173, dtype=torch.bfloat16, device="cuda")[..., ::2])
            for layout, gradient in gradient_cases.items():
                case = report["cases"][layout] = dict(input_stride=list(gradient.stride()),
                    input_storage_bytes=gradient.untyped_storage().nbytes(), same_logical_gradient=bool(torch.equal(gradient, logical)),
                    comparisons={}, executions={})
                try:
                    runs = {}
                    for phase in ("baseline", "baseline_repeat", "fused", "fused_repeat", "inline_boundary"):
                        with _capture(directory, Path(layout) / phase) as capture:
                            if phase.startswith("baseline"):
                                adjusted = activation.dx(gradient, *derivative)
                                output = norm.gradients(adjusted, *norm_saved)
                                runs[phase] = (output, capture["affine_partials"], adjusted)
                            elif phase.startswith("fused"):
                                output = fused.backward(gradient, *derivative, *norm_saved)
                                runs[phase] = (output, capture["affine_partials"], None)
                            else:
                                adjusted = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
                                _inline_adjusted_gradient[((x.numel() + 1023) // 1024,)](gradient, *derivative, adjusted,
                                    x.numel(), shape[1], shape[2] * shape[3], input_arguments(gradient)[2], 1024,
                                    num_warps=4, enable_fp_fusion=False)
                                runs[phase] = adjusted
                        metadata_snapshot = _metadata(norm.module.last_execution) if phase != "inline_boundary" else {}
                        case["executions"][phase] = dict(kernels=capture["kernels"],
                            affine_partial_count=len(capture["affine_partials"]), execution_metadata=metadata_snapshot)
                    for route in ("baseline", "fused"):
                        for index, role in enumerate(("dx", "dweight", "dbias")):
                            case["comparisons"][f"{route}_repeat.{role}"] = _compare(runs[route + "_repeat"][0][index], runs[route][0][index])
                    case["comparisons"]["inline_adjusted_bf16"] = _compare(runs["inline_boundary"], runs["baseline"][2])
                    for index, role in enumerate(("dx", "dweight", "dbias")):
                        case["comparisons"][f"fused_vs_baseline.{role}"] = _compare(runs["fused"][0][index], runs["baseline"][0][index])
                    for route in ("baseline", "fused", "baseline_repeat", "fused_repeat"):
                        if len(runs[route][1]) != 1:
                            raise ValueError("Expected one affine partial tensor per GroupNorm backward")
                    case["comparisons"]["affine_partials"] = _compare(runs["fused"][1][0], runs["baseline"][1][0])
                    case["comparisons"]["baseline_repeat.affine_partials"] = _compare(runs["baseline_repeat"][1][0], runs["baseline"][1][0])
                    case["comparisons"]["fused_repeat.affine_partials"] = _compare(runs["fused_repeat"][1][0], runs["fused"][1][0])
                    expected_copy = 0 if gradient.is_contiguous() else gradient.numel() * gradient.element_size()
                    case["raw_cotangent_copy_contract"] = dict(expected_bytes=expected_copy,
                        observed_bytes=case["executions"]["fused"]["execution_metadata"].get("raw_cotangent_copy_bytes"))
                    case["raw_cotangent_copy_contract"]["passed"] = (
                        case["raw_cotangent_copy_contract"]["observed_bytes"] == expected_copy)
                    case["same_route_bitwise_stable"] = all(values["bitwise_equal"]
                        for role, values in case["comparisons"].items() if "repeat." in role)
                    case["passed"] = (case["same_logical_gradient"] and case["raw_cotangent_copy_contract"]["passed"]
                                      and case["same_route_bitwise_stable"]
                                      and all(item["exact"] for item in case["comparisons"].values()))
                except Exception as error:
                    case.update(passed=False, error=dict(type=type(error).__name__, message=str(error)))
                if directory is not None:
                    text = json.dumps(case, indent=2, allow_nan=False) + "\n"
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    _write_text(directory, Path(layout) / f"case-{digest}.json", text)
                with self.subTest(layout=layout):
                    self.assertTrue(case["passed"], json.dumps(case.get("error") or
                        {"comparisons": {role: values for role, values in case["comparisons"].items() if not values["exact"]},
                         "copy_contract": case.get("raw_cotangent_copy_contract")}, sort_keys=True))
        torch.cuda.synchronize()
        report["passed"] = all(case["passed"] for case in report["cases"].values())
        if directory is not None:
            text = json.dumps(report, indent=2, allow_nan=False) + "\n"
            digest = hashlib.sha256(text.encode()).hexdigest()
            _write_text(directory, f"report-{digest}.json", text)


if __name__ == "__main__":
    unittest.main()
