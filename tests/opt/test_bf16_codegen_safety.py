"""CPU contracts and separately gated real CUDA codegen qualification."""

import ast
import sys
from contextlib import nullcontext
from importlib.resources import files
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends.bf16 import product
from deepopt.opt.backends._shared.bf16_resources import (
    BF16ResourceError, admit_resources, bf16_scratch_requirements, launch_admitted, _admissions,
)
from deepopt.opt.backends._shared.operand_device import cuda_operand_scope
from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms, record_gemm_launch, note_gemm_launch
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


class BF16SafetyTests(unittest.TestCase):
    @staticmethod
    def interpreted_kernels():
        from tests.opt.convolution_kernel_support import KernelLanguage
        class Language(KernelLanguage):
            bfloat16, uint32 = torch.bfloat16, torch.int64
            where = staticmethod(lambda test, a, b: torch.where(torch.as_tensor(test), a, b))
            full = staticmethod(lambda shape, fill, dtype: torch.full(shape, fill, dtype=dtype))
            def dot(self, a, b, acc):
                return a.float() @ b.float() + acc
        language = Language()
        namespace = {"tl": language, "triton": SimpleNamespace(cdiv=language.cdiv)}
        for package, path, wanted in (
            ("deepopt.opt.backends.triton.gemm", "bf16.py", "_bf16_gemm"),
            ("deepopt.opt.backends.triton.gemm", "panel.py", "_decode_panel"),
            ("deepopt.opt.backends.triton.quantization.activation", "affine.py", "_affine_decode"),
        ):
            node = next(n for n in ast.parse(files(package).joinpath(path).read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == wanted)
            node.decorator_list = []
            exec(compile(ast.Module([node], type_ignores=[]), "<interpreted-kernel>", "exec"), namespace)
        return language, namespace

    def test_actual_decode_and_gemm_bodies_preserve_tails_and_fp32_panel_state(self):
        from tests.opt.convolution_kernel_support import Pointer
        language, functions = self.interpreted_kernels()
        x = torch.arange(65 * 17).reshape(65, 17).remainder(3).sub(1).bfloat16()
        a = torch.arange(19 * 65).reshape(19, 65).remainder(3).sub(1).bfloat16()
        output, accumulator = torch.empty(19, 17).bfloat16(), torch.empty(19, 17)
        for bits in (1, 2, 4, 8):
            packed = quantize_pack_activation(x, 1024, bits=bits)
            expected = a.float() @ dequantize_unpack_activation(packed).float()
            for packed_read in (True, False):
                starts = [(0, 65)] if packed_read else [(0, 32), (32, 32), (64, 1)]
                for start, count in starts:
                    decoded = torch.empty(17, 32).bfloat16()
                    if not packed_read:
                        for pid in range((count * 17 + 255) // 256):
                            language.programs = (pid, 0, 0)
                            functions["_decode_panel"](Pointer(packed.data), Pointer(packed.step), Pointer(packed.minimum),
                                Pointer(decoded), start, count, 17, 1024, packed.data.shape[1], bits, 32, 256)
                        torch.testing.assert_close(decoded[:, :count], dequantize_unpack_activation(packed)[start:start + count].T, rtol=0, atol=0)
                    arguments = dict(A=Pointer(a[:, start:start + count]), B=Pointer(packed.data if packed_read else decoded),
                        Bias=Pointer(output), Y=Pointer(output), Saved=Pointer(output), DInput=Pointer(output),
                        Steps=Pointer(packed.step), Minima=Pointer(packed.minimum), Residual=Pointer(output),
                        DSteps=Pointer(output), DMinima=Pointer(output), M=19, N=17, K=count,
                        ASM=a.stride(0), ASK=a.stride(1), BSN=32, BSK=1, BIAS=False, BIAS_STRIDE=1,
                        KIND="none", APPROX="none", SAVE=False, DERIVATIVE="none", DAPPROX="none",
                        DSM=17, DSN=1, DPACKED=False, DGROUP=1024, DWORDS=1, DBITS=bits,
                        RESIDUAL=False, RSM=17, RSN=1, PACKED=packed_read, GROUP=1024,
                        WORDS=packed.data.shape[1], BITS=bits, SOURCE_N=17, BM=16, BN=16, BK=32,
                        ORDER="column_major", GROUP_M=1, Accum=Pointer(accumulator),
                        ACCUMULATE=start > 0, FINAL=start + count == 65)
                    for pid in range(4):
                        language.programs = (pid, 0, 0)
                        functions["_bf16_gemm"](**arguments)
                torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)

    def test_widening_precedes_large_reduction_stride_product(self):
        source = files("deepopt.opt.backends.triton.gemm").joinpath("bf16.py").read_text()
        kernel = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_bf16_gemm")
        initialize = next(n for n in kernel.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "kk" for t in n.targets))
        class Language:
            int64 = torch.int64
            arange = staticmethod(torch.arange)
        namespace = {"tl": Language, "BK": 32}
        exec(compile(ast.Module([initialize], type_ignores=[]), "<wide-coordinate>", "exec"), namespace)
        self.assertEqual(namespace["kk"].dtype, torch.int64)
        for stride in (0, 1, -2**30, 2**30):
            self.assertEqual((namespace["kk"] * stride)[2].item(), 2 * stride)

    def test_actual_resource_admission_is_route_specific_and_fail_closed(self):
        for required, route in ((98304, "dense"), (98304, "materialized"), (77824, "direct")):
            self.assertEqual(admit_resources(required, 232448, role="dw", route=route, config={})["status"], "admitted")
        for required, limit in ((344064, 232448), (None, 232448), (77824, None), (-1, 232448), (0, 0)):
            with self.assertRaises(BF16ResourceError) as result:
                admit_resources(required, limit, role="dw", route="direct", config={"group_size": 1024})
            self.assertEqual(result.exception.evidence["qualification"], "unqualified")
            self.assertFalse(result.exception.evidence["fallback"])

    def test_compiler_rejection_preserves_actual_demand_without_launch_or_fallback(self):
        class OutOfResources(RuntimeError):
            required, limit, name = 344064, 232448, "shared memory"
        class Kernel:
            def warmup(self, *args, **kwargs):
                raise OutOfResources("out of resource: shared memory, Required: 344064, Hardware limit: 232448")
            def __getitem__(self, grid):
                self.fail("rejected compiler output must not launch")
        properties = SimpleNamespace(shared_memory_per_block_optin=232448, major=9, minor=0)
        with patch("torch.cuda.get_device_properties", return_value=properties), \
             patch("torch.cuda.is_current_stream_capturing", return_value=False), \
             self.assertRaises(BF16ResourceError) as result:
            launch_admitted(Kernel(), (1,), (), {}, device="cuda:0", role="dw", route="direct", config={})
        self.assertEqual(result.exception.evidence["required"], 344064)
        self.assertEqual(result.exception.evidence["resource_name"], "shared memory")

    def test_rejection_precedes_launch_and_capture_requires_warmup(self):
        class Kernel:
            calls = 0
            compilations = 0
            shared = 77824
            def warmup(self, *args, **kwargs):
                self.compilations += 1
                return SimpleNamespace(metadata=SimpleNamespace(shared=self.shared), n_regs=120, n_spills=0)
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    self.calls += 1
                return run
        kernel = Kernel()
        properties = SimpleNamespace(shared_memory_per_block_optin=232448, major=9, minor=0)
        _admissions.clear()
        with patch("torch.cuda.get_device_properties", return_value=properties), \
             patch("torch.cuda.is_current_stream_capturing", return_value=False):
            options = dict(device="cuda:0", role="dw", route="direct", config={})
            kernel.shared = 344064
            with self.assertRaises(BF16ResourceError):
                launch_admitted(kernel, (1,), (), {}, **options)
            self.assertEqual(kernel.calls, 0)
            kernel.shared = 77824
            launch_admitted(kernel, (1,), (), {}, **options)
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                launch_admitted(kernel, (1,), (), {}, **options)
                with self.assertRaisesRegex(RuntimeError, "before CUDA capture"):
                    launch_admitted(kernel, (1,), (42,), {}, **options)
        self.assertEqual(kernel.calls, 2)
        self.assertEqual(kernel.compilations, 2)
        _admissions.clear()

    def test_resource_rejections_survive_diagnostic_context(self):
        @record_gemm_launch
        def reject():
            admit_resources(344064, 232448, role="dw", route="direct", config={})
        with record_gemms() as records, self.assertRaises(BF16ResourceError):
            reject()
        self.assertEqual(records[0]["execution"], "not_launched")
        self.assertEqual(records[0]["resource_admission"]["shared_memory_bytes"], 344064)

    def test_panel_scratch_appears_in_workspace_diagnostics_without_owning_tensors(self):
        from deepopt.opt.backends._shared.gemm_diagnostics import observe_bf16_gemm
        @record_gemm_launch
        def launch(a, b):
            note_gemm_launch(resources={"shared_memory_bytes": 77824, "explicit_workspace_bytes": 0})
            note_gemm_launch(scratch_bytes=8192)
            observe_bf16_gemm(a, b, "triton", a, "dw", execution="semantic_reference")
        with record_gemms() as records:
            launch(torch.empty(2, 3), torch.empty(3, 2))
        self.assertEqual(records[0]["kernel_launch"]["resources"]["explicit_workspace_bytes"], 8192)

    def test_operand_device_scope_validates_every_descriptor_component_and_restores(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        with FakeTensorMode():
            x = torch.empty(2, device="cuda:0")
            y = torch.empty(2, device="cuda:1")
        with patch("torch.cuda.device", return_value=nullcontext()) as guard:
            with cuda_operand_scope((x, {"step": x})):
                pass
            guard.assert_called_once_with(torch.device("cuda:0"))
            with self.assertRaisesRegex(ValueError, "share one device"):
                with cuda_operand_scope(x, SimpleNamespace(data=x, step=y)):
                    self.fail("must reject before entering")
            self.assertEqual(guard.call_count, 1)

    def test_affine_jit_launches_group1024_keep_noncontiguous_broadcast_and_tail_semantics(self):
        from tests.opt.test_activation_codec_kernels import ActivationCodecKernelTests
        fixture = ActivationCodecKernelTests()
        fixture.setUp()
        source = torch.randn(3, 37, 17).bfloat16()
        for value in (source[0].T, source[:1].expand(2, -1, -1), source[:, ::2]):
            shape = (value.numel() // value.shape[-1], value.shape[-1])
            for bits in (1, 2, 4, 8):
                expected = quantize_pack_activation(value.reshape(shape), 1024, bits=bits)
                data, step, minimum = fixture.host["quantize_pack_affine"](
                    value, 1024, bits=bits, logical_shape=shape)
                for actual, reference in zip((data, step, minimum), (expected.data, expected.step, expected.minimum)):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                actual = fixture.host["dequantize_unpack_affine"](
                    data, step, minimum, shape, 1024, bits=bits, transpose=True)
                torch.testing.assert_close(actual, dequantize_unpack_activation(expected, transpose=True), rtol=0, atol=0)

    def test_panel_reference_preserves_codec_and_epilogue_semantics(self):
        torch.manual_seed(3)
        x = torch.randn(1037, 7).bfloat16()
        a = torch.randn(5, 1037).bfloat16()
        bias = torch.randn(7).bfloat16()
        for bits in (1, 2, 4, 8):
            packed = quantize_pack_activation(x, 1024, bits=bits)
            expected = product(a, dequantize_unpack_activation(packed).T, bias=bias, activation=("silu", "none"))
            actual = product(a, None, packed_b=packed, backend="triton", packed_route="panel",
                             panel_size=1024, bias=bias, activation=("silu", "none"))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "multiple"):
            product(a, None, packed_b=packed, backend="triton", packed_route="panel", panel_size=33)

    def test_panel_scratch_is_bounded_and_does_not_claim_output_or_native_workspace(self):
        rows = bf16_scratch_requirements(3072, 768, 6304, route="panel")
        self.assertEqual([x["bytes"] for x in rows], [1024 * 768 * 2, 3072 * 768 * 4])
        self.assertTrue(all(x["owner"] == "invocation" for x in rows))
        self.assertEqual(bf16_scratch_requirements(0, 768, 6304, route="panel"), ())
        self.assertEqual(bf16_scratch_requirements(2, 3, 4, route="direct"), ())

    def test_real_panel_host_loop_reuses_only_invocation_scratch_and_applies_epilogue_once(self):
        source = files("deepopt.opt.backends.triton.gemm").joinpath("panel.py").read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "panel_product")
        events, admissions = [], []
        x = torch.arange(1037 * 7).reshape(1037, 7).remainder(3).sub(1).bfloat16()
        a = torch.ones(3, 1037).bfloat16()
        bias = torch.ones(7).bfloat16()
        for bits in (1, 2, 4, 8):
            packed = quantize_pack_activation(x, 1024, bits=bits)
            dense = dequantize_unpack_activation(packed)
            events.clear()
            def decode(kernel, grid, args, kwargs, **metadata):
                count, start = kwargs["COUNT"], kwargs["START"]
                args[-1][:, :count].copy_(dense[start:start + count].T)
                events.append(("decode", start, count, tuple(args[-1].shape)))
                return None, {"status": "admitted"}
            def gemm(left, right, **options):
                events.append(("gemm", options["_accumulate"], options["_final"]))
                value = left.float() @ right.T.float()
                if options["_accumulate"]:
                    value += options["_accumulator"]
                if not options["_final"]:
                    self.assertIsNone(options["bias"])
                    options["_accumulator"].copy_(value)
                    return options["_output"]
                value = (value + options["bias"].float()).bfloat16()
                options["_output"].copy_(value)
                return options["_output"]
            namespace = dict(__package__="deepopt.opt.backends.triton.gemm", torch=torch,
                             triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
                             _decode_panel=object(), launch_admitted=decode,
                             bf16_scratch_requirements=bf16_scratch_requirements,
                             gemm_launch_is_recording=lambda: False,
                             note_gemm_launch=lambda **row: admissions.append(row))
            exec(compile(ast.Module([function], type_ignores=[]), "<panel-host-loop>", "exec"), namespace)
            with patch.dict(sys.modules, {"deepopt.opt.backends.triton.gemm.bf16": SimpleNamespace(product=gemm)}), \
                 patch("torch.cuda.device", return_value=nullcontext()):
                actual = namespace["panel_product"](a, packed, panel_size=1024, tile=64, num_stages=3,
                    role="dw", kernel_config=None, bias=bias, return_preactivation=False)
            torch.testing.assert_close(actual, product(a, dense.T, bias=bias), rtol=0, atol=0)
            self.assertEqual(events, [("decode", 0, 1024, (7, 1024)), ("gemm", False, False),
                                      ("decode", 1024, 13, (7, 1024)), ("gemm", True, True)])
            self.assertEqual(admissions[-1]["panel_count"], 2)


@unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA and Triton compilation")
class BF16CodegenCUDATests(unittest.TestCase):
    def test_archived_negative_tile_and_passing_control_use_actual_target_resources(self):
        from deepopt.opt.backends._shared.bf16_gemm import BF16GemmConfig
        rejected = BF16GemmConfig(64, 128, 64, 8, 4, "grouped_m", 4)
        control = BF16GemmConfig(128, 64, 32, 4, 3, "row_major", 1)
        for input_width, output_width in ((768, 3072), (3072, 768)):
            x = torch.randn(6304, input_width, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(output_width, 6304, device="cuda", dtype=torch.bfloat16)
            packed = quantize_pack_activation(x, 1024)
            for config in (rejected, control):
                with record_gemms() as records:
                    try:
                        actual = product(a, None, packed_b=packed, backend="triton", role="dw", kernel_config=config)
                    except BF16ResourceError as error:
                        self.assertEqual(error.evidence["qualification"], "unqualified")
                        self.assertFalse(error.evidence["fallback"])
                    else:
                        evidence = records[-1]["kernel_launch"]["resource_admission"]
                        self.assertLessEqual(evidence["shared_memory_bytes"], evidence["shared_memory_limit_bytes"])
                        self.assertEqual(tuple(actual.shape), (output_width, input_width))
                        self.assertEqual(records[-1]["execution"], "generated")
                if config == control:
                    self.assertEqual(records[-1]["execution"], "generated")

    def test_panel_graph_replay_after_admission_preserves_outputs_and_invocation_ownership(self):
        a = torch.randn(7, 1057, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(1057, 11, device="cuda", dtype=torch.bfloat16)
        packed = quantize_pack_activation(x, 1024)
        # Warm every specialization on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = product(a, None, packed_b=packed, backend="triton", packed_route="panel")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = product(a, None, packed_b=packed, backend="triton", packed_route="panel")
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        independent = product(a * 2, None, packed_b=packed, backend="triton", packed_route="panel")
        self.assertNotEqual(independent.data_ptr(), captured.data_ptr())

    def test_panel_and_direct_tails_all_bits_and_epilogues(self):
        for bits in (1, 2, 4, 8):
            x = torch.randn(1057, 17, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(19, 1057, device="cuda", dtype=torch.bfloat16)
            packed = quantize_pack_activation(x, 1024, bits=bits)
            controls = dict(backend="triton", bias=torch.randn(17, device="cuda").bfloat16(),
                            activation=("gelu", "tanh"), return_preactivation=True)
            dense = dequantize_unpack_activation(packed).T
            expected = product(a, dense, **controls)
            with record_gemms() as records:
                actual = product(a, None, packed_b=packed, packed_route="panel", **controls)
            self.assertEqual(records[-1]["execution"], "generated")
            launch = records[-1]["kernel_launch"]
            self.assertEqual(launch["route"], "panel")
            self.assertEqual(launch["selection"]["key"]["decode_route"], "panel")
            self.assertEqual(launch["kernel"], "deepopt.opt.backends.triton.gemm.bf16._bf16_gemm")
            receipt = launch["scratch_storage_receipt"]
            self.assertEqual(receipt["scope"], "one_product_invocation")
            self.assertEqual(len(receipt["ledger"]["storages"]), 2)
            self.assertEqual(sum(row["storage_bytes"] for row in receipt["ledger"]["storages"]),
                             min(1024, a.shape[1]) * packed.shape[1] * 2 + a.shape[0] * packed.shape[1] * 4)
            self.assertEqual(receipt["accounting"], "backing_storage_observation_not_total_peak")
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_large_valid_reduction_offset_and_zero_stride(self):
        stride = 2**30
        required = (2 * stride + 1) * 2
        if torch.cuda.mem_get_info()[0] < required + 2**30:
            self.skipTest("large-offset regression needs 5 GiB free")
        backing = torch.empty(2 * stride + 1, device="cuda", dtype=torch.bfloat16)
        view = backing.as_strided((1, 3), (0, stride))
        view.copy_(torch.tensor([[1., 2., 3.]], device="cuda", dtype=torch.bfloat16))
        other = torch.ones(2, 3, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(product(view, other, backend="triton"), torch.full((1, 2), 6., device="cuda", dtype=torch.bfloat16))
        torch.testing.assert_close(product(other, view, backend="triton"), torch.full((2, 1), 6., device="cuda", dtype=torch.bfloat16))

@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPUs for mismatched-current-device qualification")
class BF16CodecMultiGPUCUDA(unittest.TestCase):
    def test_codec_noncurrent_device_restores_caller_and_stream(self):
        from deepopt.opt.backends.triton.quantization.activation import indices
        from deepopt.opt.quantization.indices import empty_indices
        with torch.cuda.device(0):
            # Creating a stream context on GPU1 changes the current device to
            # GPU1. Test the genuinely mismatched device first, before entering
            # that context, so a missing operand-device guard cannot pass.
            x = torch.randn(13, 1031, device="cuda:1").bfloat16()
            reference = dequantize_unpack_activation(
                quantize_pack_activation(x.cpu(), 1024), dtype=torch.bfloat16)
            self.assertEqual(torch.cuda.current_device(), 0)
            packed = quantize_pack_activation(x, 1024)
            self.assertEqual(torch.cuda.current_device(), 0)
            decoded = dequantize_unpack_activation(packed, dtype=torch.bfloat16)
            self.assertEqual(torch.cuda.current_device(), 0)
            torch.testing.assert_close(decoded.cpu(), reference)
            index = torch.tensor([1, 2, 1], device="cuda:1")
            codes = empty_indices(tuple(index.shape), 4, index.device)
            indices.pack(index, codes)
            decoded_indices = torch.empty_like(index)
            indices.unpack(codes, decoded_indices)
            self.assertEqual(torch.cuda.current_device(), 0)
            torch.testing.assert_close(decoded_indices, index)
            stream = torch.cuda.Stream(device=1)
            with torch.cuda.stream(stream):
                self.assertEqual(torch.cuda.current_device(), 1)
                x = torch.randn(13, 7, device="cuda:1").bfloat16()
                packed = quantize_pack_activation(x, 1024)
                decoded = dequantize_unpack_activation(packed)
                index = torch.tensor([1, 2, 1], device="cuda:1")
                codes = empty_indices(tuple(index.shape), 4, index.device)
                indices.pack(index, codes)
                decoded_indices = torch.empty_like(index)
                indices.unpack(codes, decoded_indices)
                self.assertEqual(decoded_indices.device.index, 1)
                torch.testing.assert_close(decoded_indices, index, rtol=0, atol=0)
                self.assertEqual(decoded.device.index, 1)
                self.assertEqual(torch.cuda.current_stream(1), stream)
            self.assertEqual(torch.cuda.current_device(), 0)
        self.assertTrue(callable(indices.pack))


if __name__ == "__main__":
    unittest.main()
