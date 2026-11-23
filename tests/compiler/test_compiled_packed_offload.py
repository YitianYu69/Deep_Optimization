"""Generated native execution owns offload without application lifecycle calls.

CPU cases inject a transport backend into real generated forward/native VJP
wrappers. They exercise ownership/code generation, not CUDA performance.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import gc
import unittest
import weakref

import torch

from deepopt._bf16_policy import resolve_bf16_policy
from deepopt._packed_offload import PackedActivationOffloadPolicy
from deepopt.compiler.autodiff import semantic
from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
from deepopt.compiler.autodiff.joint import JointProgram, JointRegionNode, _JointFunction, _OffloadedJointFunction
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import IRGraph, InputNode, TensorMeta, TensorValue, register_uses
from deepopt.compiler.offload_plan import (PackedOffloadBundle, PackedOffloadComponent,
                                          PackedOffloadConsumer, PackedOffloadPlan)
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
from deepopt.compiler.runtime.packed_offload import TestingTransportBackend
from deepopt.compiler.scheduler import Scheduler, scheduled_inputs, scheduled_outputs
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.nn.activation.elementwise_modules import adapter


def _cpu_fixture(mode="memory_first_overlap", *, device_staging_bytes=1 << 20, d2h_inflight_bytes=1 << 20):
    source = torch.randn(2, 3, 17, 19, dtype=torch.bfloat16, requires_grad=True)
    precision = resolve_bf16_policy("bf16_storage")
    module = adapter("silu", plan=dict(fp8=False, bf16_policy=precision.as_dict(),
                     activation_compression=dict(enabled=True, bits=2, group_size=1024)))
    graph, registry = IRGraph(), BackendRegistry()
    builder = GraphBuilder(graph)
    meta = TensorMeta(tuple(source.shape), source.dtype, source.device, tuple(source.stride()), True)
    x = builder.input(0, meta, name="activation_input").value
    output = TensorValue("activation_output", meta)
    node = SimpleNamespace(module=module, plan=module.plan, args=(x, None, None), kwargs={}, value=output,
                           output=lambda: output, bf16_policy=precision, module_path="activation",
                           node_name="activation", op="native_activation")
    specification = semantic.NativeActivation(node, KernelCodegenContext(kernel_policy=KernelPolicy()))
    graph.add(specification.lower(node))
    builder.output((output,))
    semantic.register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry).diff()
    for entry in trace.graph.nodes:
        if isinstance(entry, InputNode) and entry.index == 0:
            entry.value.meta = replace(entry.value.meta, stride=None)
    graph.nodes[-1].result = (graph.nodes[-1].result, trace.saved_tensors)
    register_uses(graph.nodes[-1])
    fw, bw = Scheduler(graph, registry=registry).run(), Scheduler(trace.graph, registry=registry).run()
    policy = PackedActivationOffloadPolicy(enabled=True, policy=mode, host_pinned_bytes=1 << 20,
        device_staging_bytes=device_staging_bytes, d2h_inflight_bytes=d2h_inflight_bytes,
        admission="error", max_live_forwards=2)
    components = tuple(PackedOffloadComponent(value.name, index, tuple(value.meta.shape), str(value.meta.dtype),
        tuple(torch.empty(value.meta.shape).stride()), "cpu", torch.empty((), dtype=value.meta.dtype).element_size()
        * value.meta.shape[0] * (value.meta.shape[1] if len(value.meta.shape) == 2 else 1),
        ("payload", "codec_step", "codec_minimum")[index]) for index, value in enumerate(trace.saved_tensors))
    selected = {entry.value for entry in trace.graph.nodes if isinstance(entry, InputNode) and entry.index > 0}
    consumers = tuple(PackedOffloadConsumer(f"consumer-{index}", index, ("activation",),
                      sum(component.nbytes for component in components))
                      for index, item in enumerate(bw.items) if not isinstance(item, InputNode)
                      and set(scheduled_inputs(item)) & selected)
    producer = next(index for index, item in enumerate(fw.items)
                    if all(value in scheduled_outputs(item) for value in trace.saved_tensors))
    bundle = PackedOffloadBundle("activation", components, producer, producer,
        tuple(consumer.consumer_id for consumer in consumers), consumers[0].schedule_index, consumers[-1].schedule_index)
    # Explicit CPU test descriptors; the production builder rejects this route.
    plan = PackedOffloadPlan(policy, (bundle,), consumers, signature="explicit-cpu-test-only")
    def compile_pair(offload):
        return tuple(WrapperCodegen(torch.nn.Module(), schedule, registry=registry, enable_autodiff=False,
            consume_inputs=backward, gradient_input_count=1 if backward else 0,
            diagnostic_phase="backward" if backward else "forward", packed_offload_plan=offload).run()
            for schedule, backward in ((fw, False), (bw, True)))
    region = SimpleNamespace(inputs=(x,), outputs=(output,), seed_indices=(0,), trace=trace,
        kernel_policy=KernelPolicy(), device_type="cpu", saved_state_metadata={"explicit_cpu_test": True},
        context=KernelCodegenContext())
    resident = SimpleNamespace(**vars(region))
    region.forward_artifact, region.backward_artifact = compile_pair(plan)
    resident.forward_artifact, resident.backward_artifact = compile_pair(None)
    manager = CompiledGraphRuntimeManager(policy, backend=TestingTransportBackend(), start_worker=False)
    return source, region, resident, plan, manager


class CompiledPackedOffloadTests(unittest.TestCase):
    def test_code_recorder_supports_transport_and_survives_forward_scope_exit(self):
        from deepopt.compiler.runtime.boundary_observation import native_boundary_observation
        from examples.vae.native_execution import _ArtifactRecorder

        source, region, _, plan, manager = _cpu_fixture()
        recorder = _ArtifactRecorder(torch, {})
        try:
            frame = manager.begin_forward(plan)
            with native_boundary_observation(recorder), frame.saved_token_scope():
                output, = _OffloadedJointFunction.apply(region, frame, source)
            output.sum().backward()
            manager.drain()
            self.assertEqual(recorder.calls, {"forward": 1, "backward": 1})
            self.assertTrue(recorder.payloads)
            self.assertTrue(all(isinstance(value, bytes) for value in recorder.payloads.values()))
            self.assertIsNotNone(source.grad)
        finally:
            manager.close()

    def test_numerical_observer_is_rejected_before_packed_producer(self):
        from deepopt.compiler.runtime.boundary_observation import native_boundary_observation

        source, region, _, plan, manager = _cpu_fixture()
        observer = SimpleNamespace(run_artifact=lambda *args, **kwargs: self.fail("Observer executed"))
        try:
            frame = manager.begin_forward(plan)
            with native_boundary_observation(observer), frame.saved_token_scope():
                with self.assertRaisesRegex(RuntimeError, "metadata-only"):
                    _OffloadedJointFunction.apply(region, frame, source)
            self.assertEqual(manager.snapshot()["counters"].get("d2h_bytes", 0), 0)
        finally:
            manager.close()

    def test_application_only_calls_the_compiled_graph(self):
        source, region, _, plan, manager = _cpu_fixture(device_staging_bytes=0, d2h_inflight_bytes=0)
        class TestRegion(SimpleNamespace):
            __call__ = JointProgram.__call__
        target = TestRegion(**vars(region))
        target.context = KernelCodegenContext(packed_activation_offload=plan.policy)
        target.offload_runtime, target.offload_plan, target.specifications = manager, plan, []
        outer, builder = IRGraph(), None
        builder = GraphBuilder(outer)
        value = builder.input(0, region.inputs[0].meta, name="user_input").value
        output = TensorValue("user_output", region.outputs[0].meta)
        outer.add(JointRegionNode("native_autodiff_region", target, (value,), {}, (output,), target))
        builder.output(output)
        artifact = WrapperCodegen(torch.nn.Module(), Scheduler(outer).run(), require_full_aot=True).run()
        try:
            self.assertIn("with _packed_forward_scope():", artifact.source)
            result = artifact.fn(source)
            result.sum().backward()
            self.assertIsNotNone(source.grad)
            manager.drain()
            snapshot = manager.snapshot()
            self.assertGreater(snapshot["counters"]["d2h_bytes"], 0)
            self.assertEqual(snapshot["host"]["pinned_active_bytes"], 0)
            self.assertEqual(snapshot["transport_limits"], dict(
                device_staging_bytes=plan.selected_bytes, d2h_inflight_bytes=plan.selected_bytes))
        finally:
            manager.close()

    def test_real_generated_forward_and_native_backward_match_resident(self):
        for mode in ("demand", "memory_first_overlap"):
            with self.subTest(mode=mode):
                source, region, resident, plan, manager = _cpu_fixture(mode)
                try:
                    expected_x = source.detach().clone().requires_grad_()
                    expected, = _JointFunction.apply(resident, expected_x)
                    frame = manager.begin_forward(plan)
                    refs = []
                    publish = frame.pack_ready
                    def observe_publish(bundle_id, tensors):
                        refs.extend(weakref.ref(tensor) for tensor in tensors)
                        publish(bundle_id, tensors)
                    frame.pack_ready = observe_publish
                    with frame.saved_token_scope():
                        actual, = _OffloadedJointFunction.apply(region, frame, source)
                    self.assertTrue(all(reference() is None for reference in refs))
                    seed = torch.randn_like(actual)
                    expected_gradient, = torch.autograd.grad(expected, expected_x, seed)
                    actual_gradient, = torch.autograd.grad(actual, source, seed)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                    manager.drain()
                    self.assertEqual(manager.snapshot()["host"]["pinned_active_bytes"], 0)
                finally:
                    manager.close()

    def test_retained_native_backward_owns_independent_staging(self):
        source, region, _, plan, manager = _cpu_fixture()
        try:
            frame = manager.begin_forward(plan)
            with frame.saved_token_scope():
                output, = _OffloadedJointFunction.apply(region, frame, source)
            seed = torch.tensor(0.137, dtype=output.dtype).expand_as(output)
            first, = torch.autograd.grad(output, source, seed, retain_graph=True)
            second, = torch.autograd.grad(output, source, seed)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            manager.drain()
            self.assertGreaterEqual(manager.snapshot()["counters"]["h2d_bytes"], 2 * plan.selected_bytes)
        finally:
            manager.close()

    def test_abandoned_generated_graph_releases_canonical_host_history(self):
        source, region, _, plan, manager = _cpu_fixture()
        try:
            frame = manager.begin_forward(plan)
            with frame.saved_token_scope():
                output = _OffloadedJointFunction.apply(region, frame, source)
            manager.drain()
            self.assertGreater(manager.snapshot()["host"]["pinned_active_bytes"], 0)
            del output
            gc.collect()
            manager.drain()
            self.assertEqual(manager.snapshot()["host"]["pinned_active_bytes"], 0)
        finally:
            manager.close()


class CompiledPrecisionFacadeCPUTests(unittest.TestCase):
    """Real CPU capture of the host-facade branch; CPU offload stays disabled."""

    def setUp(self):
        import deepopt

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.compiler.reset()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.compiler.reset)
        self.addCleanup(deepopt.reset_config)
        self.addCleanup(deepopt.shutdown_distributed)

    def test_same_policy_reprepare_preserves_numerical_body_and_native_backward(self):
        self._exercise_same_policy_reprepare(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU(),
            torch.nn.Linear(4, 2)))

    def test_same_policy_reprepare_preserves_root_operation_capture_and_native_backward(self):
        # Root operations add _CaptureForward around _PrecisionBody, unlike a
        # container. That known wrapper must also remain outside host scopes.
        self._exercise_same_policy_reprepare(torch.nn.Linear(4, 2))

    def _exercise_same_policy_reprepare(self, original):
        from deepopt._config_resolution import ResolvedConfig
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.engine.execution.configuration import prepare_execution_model
        from deepopt.opt.policy import prepare_fp8_model

        # A direct prepared CPU model may use BF16 autocast; CPU TrainingRuntime
        # still requires FP32. No device/config guard or transport is mocked.
        raw = {"runtime": {"precision": "bf16", "bf16_policy": "bf16_storage", "activation_compression": {
            "enabled": True, "bits": 2, "group_size": 1024}, "activation_memory": {
                "multiply_backward": "materialized", "pack_reuse": "off", "reduction_scratch_bytes": 4096},
            "compile": {"enabled": True, "backend": "deepopt", "fullgraph": True, "dynamic": False,
                "options": {"use_native_autodiff": True, "require_full_aot": True}}}}
        model, _ = prepare_fp8_model(original, config=deepcopy(raw))
        self.assertTrue(getattr(model.forward, "_deepopt_bf16_forward", False))
        parameters = tuple(model.parameters())
        owned_policy, owned_memory = deepcopy(model._deepopt_bf16_policy), model._deepopt_activation_memory_plan
        compile_model(model, CompileConfig(**raw["runtime"]["compile"]), config=deepcopy(raw))
        numerical_body, compiled_call = model.forward, model._compiled_call_impl
        self.assertTrue(getattr(compiled_call, "_deepopt_bf16_forward", False))
        # Trainer construction performs this same-policy preparation a second
        # time. Rebinding forward here puts ContextVar.set inside lazy Dynamo.
        same, _, _ = prepare_execution_model(model, ResolvedConfig(deepcopy(raw)))
        self.assertIs(same, model)
        inputs = torch.randn(2, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(inputs)
        output.float().square().mean().backward()
        self.assertIs(model.forward, numerical_body)
        self.assertIs(model._compiled_call_impl, compiled_call)
        self.assertEqual(model._deepopt_bf16_policy, owned_policy)
        self.assertEqual(model._deepopt_activation_memory_plan, owned_memory)
        self.assertTrue(getattr(numerical_body, "_deepopt_compiler_precision_body", False)
            or getattr(getattr(numerical_body, "original", None), "_deepopt_compiler_precision_body", False))
        self.assertEqual(tuple(map(id, model.parameters())), tuple(map(id, parameters)))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertIsNone(inputs.grad)
        self.assertTrue(all(value.grad is not None and value.grad.dtype == torch.float32
                            and torch.isfinite(value.grad).all() for value in parameters))
        report = compilation_report(model).as_dict()
        self.assertGreater(report["native_autodiff_operations"], 0)
        self.assertEqual(report["runtime_fallbacks"], 0)
        self.assertEqual(report["legacy_backward_operations"], [])

    def test_configured_vae_facade_checkpoint_restore_and_next_update(self):
        from contextlib import redirect_stdout
        import io
        from pathlib import Path
        import tempfile

        import deepopt
        from deepopt.checkpoint import Checkpoint
        from deepopt.compiler import compilation_report
        from deepopt.data import build_loaders
        from deepopt.engine.checkpointing.checkpoint import read_checkpoint
        from deepopt.operations import operation_recipes
        from examples.vae.mbconv_vae import MBConvVAE_V2
        from examples.vae.train_vae import make_vae_step
        from tests.checkpointing.test_compiled_operation_recipe import _SavedUpdate, _StopAfterSavedUpdate

        raw = dict(training=dict(epochs=1, batch_size=2, learning_rate=.001, checkpoint_every_steps=1, monitor=None),
            runtime=dict(device="cpu", precision="fp32", bf16_policy="torch_amp", checks=False,
                activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True),
                # A real nondefault owned plan installs _BF16PolicyForward on
                # CPU too, exercising the failed scope branch without a skip.
                activation_memory=dict(multiply_backward="materialized", pack_reuse="off", reduction_scratch_bytes=4096),
                compile=dict(enabled=True, backend="deepopt", dynamic=False, fullgraph=True,
                    options=dict(use_native_autodiff=True, require_full_aot=True, share_compatible_packs=False))),
            schedules=dict(beta=dict(type="linear_warmup", start=0., end=1e-6, duration=10000, interval="update")),
            logging=dict(level="ERROR"))
        images = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(27)) * 2 - 1

        def components():
            model = MBConvVAE_V2(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=2)
            recipe = operation_recipes(model)
            parameters = tuple(model.parameters())
            optimizer = torch.optim.SGD(parameters, lr=.001, momentum=.9)
            schedules = deepopt.ScheduleController.from_config(raw["schedules"])
            trainer = deepopt.Trainer(model, optimizer, step_fn=make_vae_step(schedules),
                                      schedules=schedules, config=deepcopy(raw))
            self.assertTrue(getattr(trainer.model._compiled_call_impl, "_deepopt_bf16_forward", False))
            self.assertFalse(trainer.model._deepopt_activation_memory_plan.is_default)
            self.assertEqual(tuple(map(id, trainer.model.parameters())), tuple(map(id, parameters)))
            self.assertTrue(trainer.model._deepopt_compile_config["fullgraph"])
            self.assertTrue(trainer.model._deepopt_compile_config["options"]["require_full_aot"])
            self.assertTrue(trainer.model._deepopt_compile_config["options"]["use_native_autodiff"])
            loader, _, _ = build_loaders(images, images[:2], batch_size=2, seed=27,
                resumable=True, num_workers=0, persistent_workers=False)
            return trainer, loader, recipe

        def assert_native(trainer, recipe):
            self.assertEqual(operation_recipes(trainer.model), recipe)
            self.assertTrue(getattr(trainer.model.forward, "_deepopt_compiler_precision_body", False))
            evidence = compilation_report(trainer.model).as_dict()
            self.assertGreater(evidence["native_autodiff_operations"], 0)
            self.assertEqual(evidence["runtime_fallbacks"], 0)
            self.assertEqual(evidence["legacy_backward_operations"], [])

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            trainer, loader, recipe = components()
            with self.assertRaises(_SavedUpdate):
                trainer.fit(loader, epochs=1, callbacks=[_StopAfterSavedUpdate(directory)], progress_bar=False)
            payload = read_checkpoint(directory / "last_step.pt")
            self.assertEqual(payload["state"]["optimizer_steps"], 1)
            self.assertEqual(payload["kind"], "step")
            assert_native(trainer, recipe)
            restored, loader, restored_recipe = components()
            self.assertEqual(restored_recipe, recipe)
            checkpoint = Checkpoint(directory)
            checkpoint.restore(restored, loader)
            self.assertEqual(restored.state.optimizer_steps, 1)
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, payload["model"][name], rtol=0, atol=0)
            restored.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False)
            final = read_checkpoint(directory / "last.pt")
            self.assertEqual(final["state"]["optimizer_steps"], 2)
            self.assertEqual(final["state"]["completed_epochs"], 1)
            self.assertTrue(all(torch.isfinite(value).all() for value in final["model"].values()))
            assert_native(restored, recipe)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA execution is required")
class CompiledPackedOffloadCudaTests(unittest.TestCase):
    def setUp(self):
        from tests.compiler.test_native_padding import _strict_algorithms

        self.enterContext(_strict_algorithms())

    def test_native_step_checkpoint_restore_and_next_update(self):
        from contextlib import redirect_stdout
        import io
        from pathlib import Path
        import tempfile

        import deepopt
        from deepopt.checkpoint import Checkpoint
        from deepopt.compiler import compilation_report
        from deepopt.compiler.runtime.manager import packed_offload_runtime_snapshot
        from deepopt.data import build_loaders
        from deepopt.engine.checkpointing.checkpoint import read_checkpoint
        from deepopt.operations import operation_recipes
        from examples.vae.mbconv_vae import MBConvVAE_V2
        from examples.vae.train_vae import make_vae_step
        from tests.checkpointing.test_compiled_operation_recipe import _SavedUpdate, _StopAfterSavedUpdate

        raw = dict(training=dict(epochs=1, batch_size=2, learning_rate=.001, checkpoint_every_steps=1, monitor=None),
            runtime=dict(device="cuda:0", precision="bf16", bf16_policy="bf16_storage", checks=False,
                activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True),
                packed_activation_offload=dict(enabled=True, host_pinned_bytes=32 << 20,
                    device_staging_bytes=16 << 20, d2h_inflight_bytes=8 << 20, admission="resident_fallback"),
                compile=dict(enabled=True, backend="deepopt", dynamic=False, fullgraph=True,
                    options=dict(use_native_autodiff=True, require_full_aot=True, share_compatible_packs=True))),
            schedules=dict(beta=dict(type="linear_warmup", start=0., end=1e-6, duration=10000, interval="update")),
            logging=dict(level="ERROR"))
        images = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(27)) * 2 - 1
        def components():
            model = MBConvVAE_V2(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=2).cuda()
            recipe = operation_recipes(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
            schedules = deepopt.ScheduleController.from_config(raw["schedules"])
            trainer = deepopt.Trainer(model, optimizer, step_fn=make_vae_step(schedules),
                                      schedules=schedules, config=deepcopy(raw))
            self.assertTrue(getattr(trainer.model._compiled_call_impl, "_deepopt_bf16_forward", False))
            self.assertTrue(trainer.model._deepopt_compile_config["fullgraph"])
            self.assertTrue(trainer.model._deepopt_compile_config["options"]["require_full_aot"])
            self.assertTrue(trainer.model._deepopt_compile_config["options"]["use_native_autodiff"])
            loader, _, _ = build_loaders(images, images[:2], batch_size=2, seed=27,
                resumable=True, num_workers=0, persistent_workers=False)
            return trainer, loader, recipe
        self.addCleanup(deepopt.reset_config)
        self.addCleanup(deepopt.shutdown_distributed)
        self.addCleanup(torch.compiler.reset)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            before = packed_offload_runtime_snapshot()["totals"]["counters"]
            trainer, loader, recipe = components()
            with self.assertRaises(_SavedUpdate):
                trainer.fit(loader, epochs=1, callbacks=[_StopAfterSavedUpdate(directory)], progress_bar=False)
            payload = read_checkpoint(directory / "last_step.pt")
            self.assertEqual(payload["state"]["optimizer_steps"], 1)
            self.assertEqual(operation_recipes(trainer.model), recipe)
            self.assertTrue(getattr(trainer.model.forward, "_deepopt_compiler_precision_body", False))
            evidence = compilation_report(trainer.model).as_dict()
            self.assertGreater(evidence["native_autodiff_operations"], 0)
            self.assertEqual(evidence["runtime_fallbacks"], 0)
            self.assertEqual(evidence["legacy_backward_operations"], [])
            after = packed_offload_runtime_snapshot()["totals"]["counters"]
            self.assertGreater(after.get("d2h_bytes", 0), before.get("d2h_bytes", 0))
            self.assertGreater(after.get("h2d_bytes", 0), before.get("h2d_bytes", 0))
            restored, loader, restored_recipe = components()
            self.assertEqual(restored_recipe, recipe)
            checkpoint = Checkpoint(directory)
            checkpoint.restore(restored, loader)
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, payload["model"][name].to(value.device), rtol=0, atol=0)
            restored.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False)
            final = read_checkpoint(directory / "last.pt")
            self.assertEqual(final["state"]["optimizer_steps"], 2)
            self.assertTrue(all(torch.isfinite(value).all() for value in final["model"].values()))
            evidence = compilation_report(restored.model).as_dict()
            self.assertGreater(evidence["native_autodiff_operations"], 0)
            self.assertEqual(evidence["runtime_fallbacks"], 0)
            self.assertEqual(evidence["legacy_backward_operations"], [])

    def test_native_compiled_training_retained_backward_and_next_update(self):
        from torch.fx.passes.shape_prop import ShapeProp
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
        from tests.compiler.test_native_swiglu_fusion import _PreparedTracer, _ProjectionSwiGLU

        torch.manual_seed(971)
        original = _ProjectionSwiGLU("bf16_storage").cuda()
        sample = torch.randn(2, 4, 13, 17, device="cuda")
        results = []
        for mode in (None, "demand", "memory_first_overlap"):
            model = deepcopy(original)
            graph = torch.fx.GraphModule(model, _PreparedTracer().trace(model))
            report = CompilationReport()
            policy = None if mode is None else dict(enabled=True, policy=mode, host_pinned_bytes=16 << 20,
                device_staging_bytes=16 << 20, d2h_inflight_bytes=16 << 20, admission="error")
            optimizer = torch.optim.AdamW(graph.parameters(), lr=0.001)
            snapshots = []
            with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                          activation_compression=dict(enabled=True, bits=2, group_size=1024)):
                ShapeProp(graph).propagate(sample)
                compiled = compile_fx(graph, [sample], config=BackendConfig(require_full_aot=True,
                    packed_activation_offload=policy, fuse_swiglu_residuals=True,
                    swiglu_fusion_stage="forward_backward"), report=report)
                for update in range(2):
                    output = compiled(sample)
                    gradient = torch.tensor(0.137, device="cuda", dtype=output.dtype).expand_as(output)
                    first = torch.autograd.grad(output, tuple(graph.parameters()), gradient, retain_graph=True)
                    second = torch.autograd.grad(output, tuple(graph.parameters()), gradient)
                    for left, right in zip(first, second):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    snapshots.append((output.detach().cpu(), tuple(value.cpu() for value in first)))
                    for parameter, value in zip(graph.parameters(), first):
                        parameter.grad = value
                        self.assertEqual(value.dtype, torch.float32)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            self.assertIsNone(sample.grad)
            evidence = report.as_dict()
            self.assertGreater(evidence["native_autodiff_operations"], 0)
            self.assertEqual(evidence["runtime_fallbacks"], 0)
            for region in compiled.backend_artifact.joint_autograd_regions:
                if region.offload_runtime is not None:
                    region.offload_runtime.drain()
                    counters = region.offload_runtime.snapshot()["counters"]
                    self.assertGreater(counters["d2h_bytes"], 0)
                    self.assertGreater(counters["h2d_bytes"], 0)
            results.append(snapshots)
        for actual in results[1:]:
            for (output, gradients), (expected, expected_gradients) in zip(actual, results[0]):
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                for value, reference in zip(gradients, expected_gradients):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
