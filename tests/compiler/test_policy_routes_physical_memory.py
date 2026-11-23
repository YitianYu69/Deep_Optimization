"""Policy-aware code and final-schedule physical evidence on a CPU host."""

from copy import deepcopy
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

from deepopt._bf16_policy import BF16Policy, bf16_policy_scope
from deepopt.compiler.codegen.triton.pointwise import generate_single_kernel_source
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import CallNode, IRGraph, InputNode, OutputNode, PointwiseNode, TensorMeta, TensorValue
from deepopt.compiler.runtime.physical_memory import InvocationReuseArena, PhysicalPlanObservation, reuse_candidates
from deepopt.compiler.scheduler import FusionGroup, Scheduler, Schedule, SinglePointwiseOp
from deepopt.opt.backends.bf16 import product
from deepopt.opt.backends._shared.bf16_routes import BF16Route, cost_recommendation, route_alternatives
from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms


class PolicyRouteTests(unittest.TestCase):
    def test_resource_rejection_keeps_frozen_policy_without_diagnostics(self):
        from deepopt.opt.backends._shared.bf16_resources import BF16ResourceError

        class DeviceMetadataTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device("cuda:0")

        # CPU payload with mocked device/compiler metadata; no CUDA execution.
        a = torch.Tensor._make_subclass(DeviceMetadataTensor, torch.ones(2, 3), False)
        b = torch.Tensor._make_subclass(DeviceMetadataTensor, torch.ones(4, 3), False)
        rejection = BF16ResourceError({"status": "rejected", "fallback": False})
        with patch("torch.cuda.get_device_capability", return_value=(9, 0)), \
                patch("deepopt.opt.backends.bf16.import_module", return_value=SimpleNamespace(
                    product=lambda *args, **kwargs: (_ for _ in ()).throw(rejection))):
            with self.assertRaises(BF16ResourceError) as result:
                product(a, b, backend="triton", bf16_policy="bf16_storage")
        self.assertEqual(result.exception.evidence["bf16_policy"], BF16Policy("bf16_storage").as_dict())
        self.assertFalse(result.exception.evidence["fallback"])

    def test_forward_descriptor_routes_do_not_reread_ambient_policy(self):
        frozen = BF16Policy("bf16_storage")
        a, b = torch.ones(2, 3), torch.ones(4, 3)
        with bf16_policy_scope("torch_amp", "cpu"), record_gemms() as records:
            actual = product(a, b, bf16_policy=frozen, role="dw")
        self.assertEqual(records[0]["bf16_policy"], frozen.as_dict())
        self.assertNotIn("kernel_launch", records[0])
        native = records[0]["route_contract"]
        self.assertEqual(native["observed_arithmetic"], "unknown")
        self.assertEqual(native["arithmetic"], "internal_accumulation_unknown")
        self.assertEqual(native["native_allow_bf16_reduced_precision_reduction"],
                         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        self.assertFalse(records[0]["route_contract"]["differentiation_fallback"])
        self.assertEqual(actual.dtype, torch.bfloat16)
        with bf16_policy_scope(frozen, "cpu"), record_gemms() as records:
            product(a, b)
        self.assertEqual(records[0]["bf16_policy"]["name"], "torch_amp")

    def test_library_is_candidate_and_storage_tilelang_fails_closed(self):
        routes = route_alternatives("dw", compressed=True, policy="bf16_storage")
        self.assertEqual([(r["backend"], r["representation"]) for r in routes],
                         [("native", "materialized"), ("triton", "materialized"),
                          ("triton", "direct"), ("triton", "panel")])
        self.assertTrue(all(not r["automatic_promotion"] for r in routes))
        with self.assertRaisesRegex(ValueError, "not qualified"):
            BF16Route("forward", "tilelang", "dense", "bf16_storage")
        from deepopt.opt.quantization.activation import quantize_pack_activation
        packed = quantize_pack_activation(torch.ones(3, 4), 1024, bits=2)
        with record_gemms() as records:
            product(torch.ones(2, 3), None, packed_b=packed, backend="native", role="dw")
        self.assertEqual(records[0]["route_contract"]["representation"], "materialized")
        self.assertNotIn("kernel_launch", records[0])
        for route in ("direct", "panel", "tiled"):
            with self.assertRaisesRegex(ValueError, "requires a packed operand"):
                product(torch.ones(2, 3), torch.ones(4, 3), operand_route=route)
        with record_gemms() as records:
            product(torch.ones(2, 3), torch.ones(4, 3), operand_route="materialized")
        self.assertEqual(records[0]["route_contract"]["representation"], "materialized")

    def test_complete_target_receipts_are_required_for_cost_recommendations(self):
        # Synthetic CUDA metadata tests the gate/ranking only, never execution.
        identity = dict(device="cuda:0", device_type="cuda", software=("torch-test",), shape=(3, 5, 7),
            strides=((5, 1), (5, 1)), dtypes=("bfloat16", "bfloat16"), layouts=("strided", "strided"),
            bf16_policy=BF16Policy().as_dict(), codec=("dense",),
            gradient_mask=(True, True, False), arithmetic="bf16_then_fp32_accumulate", schedule="final-1",
            decode_route={"native-materialized": "materialized", "panel": "panel"})
        good = dict(route="native-materialized", identity=identity, numerical_accepted=True,
                    launch_observed=True, resource_admitted=True, cost_scope="complete_pipeline",
                    execution_scope="target_cuda", evidence_kind="runtime_measurement", decode_route="materialized",
                    fresh_trials=3, latency_ms=2., peak_bytes=1024)
        slow_small = dict(good, route="panel", decode_route="panel", latency_ms=5., peak_bytes=512)
        unsupported = dict(good, route="unqualified-generated", launch_observed=False, latency_ms=.1)
        bad_identity = deepcopy(good)
        bad_identity["identity"]["bf16_policy"] = BF16Policy("bf16_storage").as_dict()
        report = cost_recommendation([good, slow_small, unsupported, bad_identity], identity=identity)
        self.assertEqual(report["selected_route"], "native-materialized")
        self.assertFalse(report["changes_execution"])
        self.assertEqual(len(report["rejected"]), 2)
        self.assertEqual(cost_recommendation([good, slow_small], identity=identity,
                         objective="capacity", capacity_bytes=600)["selected_route"], "panel")
        self.assertEqual(cost_recommendation([good], identity={})["status"], "unqualified")
        cpu_identity = dict(identity, device="cpu-diagnostic", device_type="cpu")
        self.assertEqual(cost_recommendation([dict(good, identity=cpu_identity)],
                         identity=cpu_identity)["reason"], "requires_target_cuda_execution")
        for kind in ("structural", "mock", "cpu_diagnostic"):
            self.assertEqual(cost_recommendation([dict(good, evidence_kind=kind)],
                             identity=identity)["status"], "unqualified")
        malformed = [dict(good, fresh_trials="3"), dict(slow_small, peak_bytes=-1)]
        self.assertEqual(cost_recommendation(malformed, identity=identity,
                         objective="capacity", capacity_bytes=600)["status"], "unqualified")

    def test_storage_pointwise_promotes_arithmetic_but_keeps_each_bf16_store_boundary(self):
        meta = TensorMeta((4,), torch.bfloat16, "cuda", (1,))
        x, y, z = (TensorValue(name, meta) for name in ("x", "y", "z"))
        first = PointwiseNode("mul", (x, x), {}, y, torch.mul)
        second = PointwiseNode("add", (y, x), {}, z, torch.add)
        for node in (first, second):
            node.bf16_policy = BF16Policy("bf16_storage")
        source, _ = generate_single_kernel_source(FusionGroup([first, second]), "test_policy")
        self.assertIn("(x0).to(tl.float32)", source)
        self.assertIn("(v0).to(tl.float32)", source)
        self.assertEqual(source.count(").to(tl.bfloat16)"), 2)
        self.assertIn("bf16_policy=bf16_storage", source)
        first.bf16_policy = second.bf16_policy = BF16Policy()
        baseline, _ = generate_single_kernel_source(FusionGroup([first, second]), "test_policy")
        self.assertNotIn("(x0).to(tl.float32)", baseline)
        self.assertEqual(meta.dtype, torch.bfloat16)


class PhysicalObservationTests(unittest.TestCase):
    def test_explicit_scratch_receipt_deduplicates_storage_without_payload_retention(self):
        from deepopt.opt.backends._shared.gemm_diagnostics import (
            note_gemm_launch, observe_bf16_scratch, record_gemm_launch, _launch,
        )

        @record_gemm_launch
        def diagnostic():
            base = torch.zeros(16)
            view = base[4:8]
            observe_bf16_scratch({"base": base, "view": view})
            receipt = _launch.get()["scratch_storage_receipt"]
            note_gemm_launch(kernel="diagnostic_only")
            return receipt, weakref.ref(base)

        with record_gemms():
            receipt, reference = diagnostic()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(len(receipt["ledger"]["storages"]), 1)
        self.assertEqual(receipt["ledger"]["storages"][0]["storage_bytes"], 64)
        self.assertEqual(receipt["bindings"][1]["storage_offset"], 4)
        self.assertEqual(receipt["accounting"], "backing_storage_observation_not_total_peak")

    @staticmethod
    def generated_artifact(device="cpu", *, emulate=True):
        meta = TensorMeta((4,), torch.float32, device, (1,))
        values = [TensorValue(name, meta) for name in ("x", "a", "b", "c", "d")]
        graph = IRGraph()
        graph.add(InputNode(0, values[0]))
        nodes = [PointwiseNode("neg", (source,), {}, target, torch.neg)
                 for source, target in zip(values, values[1:])]
        for node in nodes:
            graph.add(node)
        graph.add(OutputNode(values[-1]))
        schedule = Schedule([graph.nodes[0], *(SinglePointwiseOp(node) for node in nodes), graph.nodes[-1]])
        if not emulate:
            return WrapperCodegen(nn.Module(), schedule, enable_autodiff=False).run()

        def compile_partitions(_, partitions):
            return [SimpleNamespace(name="neg", kernel=None, inputs=part.node.inputs(),
                        output=part.output_value, source="cpu launch diagnostic", backend="triton")
                    for part in partitions]

        def cpu_launch(artifact, inputs, output, block_size):
            torch.neg(inputs[0], out=output)

        # Exercise wrapper allocation/retirement only; this CPU launch stub
        # deliberately bypasses the CUDA partition admission check.
        with patch("deepopt.compiler.scheduler.valid_same_shape_region", return_value=True), \
                patch("deepopt.compiler.codegen.triton.pointwise.TritonCodegen.compile_partitions", compile_partitions), \
                patch("deepopt.compiler.codegen.wrapper.launch_pointwise", cpu_launch):
            return WrapperCodegen(nn.Module(), schedule, enable_autodiff=False).run()

    def test_generated_reuse_has_physical_receipt_and_is_invocation_local(self):
        artifact = self.generated_artifact()
        x = torch.arange(4.)
        observation = PhysicalPlanObservation(artifact)
        expected = observation.run(artifact, x)
        arena = InvocationReuseArena(artifact, observation)
        actual = arena.run(x)
        torch.testing.assert_close(actual, expected)
        reused = [row for row in arena.events if row["kind"] == "reuse"]
        self.assertEqual([(row["source"], row["name"]) for row in reused], [("a", "c")])
        allocated = next(row for row in arena.events if row.get("name") == "a")
        self.assertEqual(allocated["storage_ids"], reused[0]["storage_ids"])
        self.assertGreater(reused[0]["lease_generation"], allocated["lease_generation"])
        self.assertTrue(arena.finished)
        self.assertEqual(arena.pool, {})
        self.assertEqual(arena.active, {})
        self.assertFalse(arena.as_dict()["performance_qualified"])
        self.assertNotIn("_reuse_arena", artifact.source)
        other = self.generated_artifact()
        with self.assertRaisesRegex(ValueError, "matching physical"):
            InvocationReuseArena(other, observation)
        with self.assertRaisesRegex(ValueError, "differentiable"):
            InvocationReuseArena(artifact, observation).run(x.requires_grad_())

    def test_reuse_concurrent_invocations_and_exception_cleanup(self):
        artifact = self.generated_artifact()
        observer = PhysicalPlanObservation(artifact)
        observer.run(artifact, torch.zeros(4))
        arenas = [InvocationReuseArena(artifact, observer) for _ in range(2)]
        inputs = [torch.arange(4.) + index for index in range(2)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = list(executor.map(lambda pair: pair[0].run(pair[1]), zip(arenas, inputs)))
        for output, value in zip(outputs, inputs):
            torch.testing.assert_close(output, value)
        self.assertNotEqual(outputs[0].untyped_storage()._cdata, outputs[1].untyped_storage()._cdata)
        self.assertTrue(all(arena.finished and not arena.pool for arena in arenas))
        arena = InvocationReuseArena(artifact, observer)
        with self.assertRaisesRegex(ValueError, "metadata"):
            arena.run(torch.zeros(3))
        self.assertTrue(arena.finished)
        self.assertEqual(arena.pool, {})

    def test_retired_version_change_fails_before_buffer_reassignment(self):
        artifact = self.generated_artifact()
        observer = PhysicalPlanObservation(artifact)
        observer.run(artifact, torch.zeros(4))
        arena = InvocationReuseArena(artifact, observer)
        value = arena.acquire("a", (4,), dtype=torch.float32, device="cpu")
        arena.complete("a", value)
        arena.retire("a", value)
        value.add_(1)
        with self.assertRaisesRegex(RuntimeError, "version changed"):
            arena.acquire("c", (4,), dtype=torch.float32, device="cpu")
        arena.finish()

    def test_truncated_physical_observation_never_authorizes_reuse(self):
        artifact = self.generated_artifact()
        observer = PhysicalPlanObservation(artifact, max_records=2)
        observer.run(artifact, torch.zeros(4))
        self.assertTrue(observer.bindings_truncated)
        self.assertEqual(reuse_candidates(observer)["status"], "unqualified")
        with self.assertRaisesRegex(ValueError, "untruncated physical"):
            InvocationReuseArena(artifact, observer)

    def test_capture_query_is_guarded_by_operand_device(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        value = FakeTensor(FakeTensorMode(), torch.empty(4, device="meta"), torch.device("cuda:1"))
        guarded = []

        @contextmanager
        def device_guard(device):
            guarded.append(str(device))
            try:
                yield
            finally:
                guarded.pop()

        def capturing():
            self.assertEqual(guarded, ["cuda:1"])
            return True

        artifact = self.generated_artifact()
        observer = PhysicalPlanObservation(artifact)
        observer.run(artifact, torch.zeros(4))
        arena = InvocationReuseArena(artifact, observer)
        with patch("torch.cuda.device", device_guard), \
                patch("torch.cuda.is_current_stream_capturing", capturing):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA Graph"):
                PhysicalPlanObservation(artifact).boundary(0, {"x": value})
            with self.assertRaisesRegex(RuntimeError, "not qualified for CUDA Graph"):
                arena.acquire("a", (4,), dtype=torch.float32, device="cuda:1")
        self.assertEqual(guarded, [])

    @staticmethod
    def artifact(target, *, requires_grad=False):
        meta = TensorMeta((4,), torch.float32, "cpu", (1,), requires_grad=requires_grad)
        x, y = TensorValue("x", meta), TensorValue("y", meta)
        graph = IRGraph()
        graph.add(InputNode(0, x))
        graph.add(CallNode("observed_operation", target, (x,), {}, (y,)))
        graph.add(OutputNode(y))
        return WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run()

    def test_alias_generation_version_and_offsets_are_physical_not_logical_bytes(self):
        artifact = self.artifact(lambda x: x.view_as(x))
        base = torch.arange(16.)
        x = base[4:8]
        observer = PhysicalPlanObservation(artifact)
        result = observer.run(artifact, x)
        report = observer.as_dict()
        self.assertEqual(len(report["ledger"]["storages"]), 1)
        self.assertEqual(report["ledger"]["storages"][0]["storage_bytes"], 64)
        self.assertEqual(report["alias_groups"][0]["names"], ("x", "y"))
        self.assertEqual(report["bindings"][0]["storage_offset"], 4)
        self.assertEqual(result.untyped_storage()._cdata, x.untyped_storage()._cdata)
        self.assertEqual(reuse_candidates(observer)["candidates"], ())
        self.assertNotIn("_physical_observer", artifact.source)

    def test_observation_does_not_own_payload_and_invocations_are_distinct(self):
        artifact = self.artifact(torch.neg)
        first, second = PhysicalPlanObservation(artifact), PhysicalPlanObservation(artifact)
        x = torch.ones(4)
        y = first.run(artifact, x)
        reference = weakref.ref(y)
        del y
        gc.collect()
        self.assertIsNone(reference())
        second.run(artifact, x)
        self.assertNotEqual(first.invocation, second.invocation)
        with self.assertRaisesRegex(RuntimeError, "one invocation"):
            first.run(artifact, x)

    def test_exception_and_mutation_have_no_reuse_permission(self):
        def mutate_and_fail(x):
            x.add_(1)
            raise RuntimeError("diagnostic failure")
        artifact = self.artifact(mutate_and_fail)
        observer = PhysicalPlanObservation(artifact)
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            observer.run(artifact, torch.zeros(4))
        self.assertTrue(observer.finished)
        self.assertEqual(observer.exception, "RuntimeError")
        self.assertEqual(reuse_candidates(observer)["status"], "unqualified")

    def test_retained_autograd_graph_ownership_is_preserved(self):
        artifact = self.artifact(torch.square, requires_grad=True)
        observer = PhysicalPlanObservation(artifact)
        x = torch.arange(4., requires_grad=True)
        output = observer.run(artifact, x)
        output.sum().backward(retain_graph=True)
        output.sum().backward()
        torch.testing.assert_close(x.grad, 4 * x.detach())
        self.assertEqual(reuse_candidates(observer)["candidates"], ())


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA execution")
class PhysicalMemoryCUDATests(unittest.TestCase):
    def test_generated_storage_reuse_and_stream_guard(self):
        device = str(torch.device("cuda", torch.cuda.current_device()))
        artifact = PhysicalObservationTests.generated_artifact(device, emulate=False)
        x = torch.arange(4., device=device)
        observation = PhysicalPlanObservation(artifact)
        expected = observation.run(artifact, x)
        arena = InvocationReuseArena(artifact, observation)
        actual = arena.run(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)
        self.assertEqual(sum(row["kind"] == "reuse" for row in arena.events), 1)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with self.assertRaisesRegex(RuntimeError, "stream lacks physical observation"):
                InvocationReuseArena(artifact, observation).run(x)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two actual CUDA devices")
class PhysicalMemoryMultiGPUCUDATests(unittest.TestCase):
    def test_operand_device_capture_queries_restore_current_device(self):
        original = torch.cuda.current_device()
        target = (original + 1) % torch.cuda.device_count()
        device = f"cuda:{target}"
        artifact = PhysicalObservationTests.generated_artifact(device, emulate=False)
        x = torch.arange(4., device=device)
        observer = PhysicalPlanObservation(artifact)
        observer.run(artifact, x)
        self.assertEqual(torch.cuda.current_device(), original)
        arena = InvocationReuseArena(artifact, observer)
        torch.testing.assert_close(arena.run(x), x)
        torch.cuda.synchronize(target)
        self.assertEqual(torch.cuda.current_device(), original)
