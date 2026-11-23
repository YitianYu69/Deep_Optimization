"""Frozen activations retain operand packs without inventing VJP residuals."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler.autodiff import AOTAutoDiff
from deepopt.compiler.autodiff.joint import _clone_region
from deepopt.compiler.autodiff.optimization import forward, backward_layouts
from deepopt.compiler.autodiff.semantic import NativeActivation, BF16Linear, register_rules
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import CallNode, InputNode, GetAttrNode, OutputNode, register_uses
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.scheduler import Scheduler
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.policy import prepare_fp8_model


class FrozenProducerFusionTests(unittest.TestCase):
    def region(self, kind, compressed, mask, *, approximate="none", device="cpu", functional=False):
        activation = nn.SiLU() if kind == "silu" else nn.GELU(approximate=approximate)
        model = prepare_fp8_model(nn.Sequential(activation, nn.Linear(7, 3)).bfloat16().to(device), False,
            activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024}, config={})[0]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(mask == "all" or mask == "weight" and name.endswith("weight"))
        reference = deepcopy(model)
        if functional:
            reference[0] = deepcopy(activation)
        fx = torch.fx.Graph()
        source = fx.placeholder("x")
        activated = (fx.call_function(torch.nn.functional.silu if kind == "silu" else torch.nn.functional.gelu,
            (source,), {} if kind == "silu" else {"approximate": approximate}) if functional else
            fx.call_module("0", (source,)))
        fx.output(fx.call_module("1", (activated,)))
        graph = torch.fx.GraphModule(model, fx)
        x = torch.randn(2, 4, 7, dtype=torch.bfloat16, device=device, requires_grad=mask in ("all", "input"))
        ShapeProp(graph).propagate(x)
        registry = BackendRegistry()
        register_rules(registry)
        ir = GraphLowering(graph, [x], registry=registry).run()
        # CPU semantic simulation of CUDA selection, followed by execution on
        # the original device. CUDA tests below perform actual generated work.
        for node in ir.nodes:
            for value in node.outputs():
                value.meta = replace(value.meta, device=torch.device("cuda:0"))
        context = KernelCodegenContext(kernel_policy=KernelPolicy(), bf16_gemm_backend="generated",
                                      fuse_deepopt_activations=True)
        nodes = [n for n in ir.nodes if not isinstance(n, (InputNode, GetAttrNode, OutputNode))]
        local, inputs, _, _ = _clone_region(nodes, context, registry)
        trace = AOTAutoDiff(local, registry=registry).diff()
        output = local.nodes[-1]
        output.result = (output.result, trace.saved_tensors)
        register_uses(output)
        return graph, ir, local, inputs, trace, registry, context, x, reference

    def check_case(self, kind, compressed, mask, approximate="none", device="cpu", functional=False):
        graph, ir, local, inputs, trace, registry, context, x, reference = self.region(
            kind, compressed, mask, approximate=approximate, device=device, functional=functional)
        original_saved = tuple(trace.saved_tensors)
        candidates, patterns = forward(local, context)
        selected = [r for r in candidates if r["selected"]]
        if compressed and mask in ("all", "weight"):
            self.assertEqual(patterns, (f"bf16_{kind}_producer_pack",))
            linear = next(n for n in local.nodes if isinstance(n, CallNode) and type(n.specification) is BF16Linear)
            self.assertEqual(linear.specification.producer_derivative is not None, mask == "all" and not functional)
            self.assertEqual(len(linear.values), 7 if mask == "all" and not functional else 4)
            self.assertEqual(tuple(trace.saved_tensors), original_saved)
            self.assertEqual(tuple(local.nodes[-1].result[1]), original_saved)
            self.assertEqual(selected[0]["replacement_contract"]["effects"], "pure")
        else:
            self.assertFalse(selected)
        for ir_graph in (local, trace.graph):
            for node in ir_graph.nodes:
                for value in node.outputs():
                    value.meta = replace(value.meta, device=torch.device(device))
        backward_layouts(trace.graph)
        fwd = WrapperCodegen(graph, Scheduler(local, registry=registry).run(), registry=registry,
                             enable_autodiff=False).run().fn
        runtime = {}
        for node in ir.nodes:
            if isinstance(node, InputNode):
                runtime[node.value] = x
            elif isinstance(node, GetAttrNode):
                owner = graph
                for part in node.target.split("."):
                    owner = getattr(owner, part)
                runtime[node.value] = owner
        with torch.no_grad():
            (actual,), saved = fwd(*(runtime[value] for value in inputs))
        if device.startswith("cuda") and selected:
            from benchmarks.analysis.measurement import compiler_launch_evidence, fusion_candidate_evidence

            with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                fwd(*(runtime[value] for value in inputs))
                torch.cuda.synchronize()
            with TemporaryDirectory(prefix="frozen-producer-trace-") as directory:
                trace_path = Path(directory) / "trace.json"
                profile.export_chrome_trace(str(trace_path))
                evidence = fusion_candidate_evidence({"fusion_candidates": candidates}, {
                    "compiler_regions": compiler_launch_evidence(profile.events(), 1, trace=trace_path)})
            self.assertTrue(evidence["all_selected_observed"], evidence)
        xr = x.detach().clone().requires_grad_(x.requires_grad)
        expected = reference(xr)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if trace.grad_targets:
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            bwd = WrapperCodegen(graph, Scheduler(trace.graph, registry=registry).run(), registry=registry,
                                 enable_autodiff=False).run().fn
            with torch.no_grad():
                gradients = bwd(gradient, *saved)
            for target, actual_gradient in zip(trace.grad_targets, gradients):
                value = runtime[inputs[int(target.key)]]
                if value is x:
                    wanted = xr.grad
                else:
                    name = next(n for n, p in graph.named_parameters() if p is value)
                    wanted = reference.get_parameter(name).grad
                torch.testing.assert_close(actual_gradient, wanted, rtol=0, atol=0)

    def test_declared_residuals_and_matching_eager_gradients_for_all_masks(self):
        for kind, approximate in (("silu", "none"), ("gelu", "none"), ("gelu", "tanh")):
            for compressed in (False, True):
                for mask in ("all", "weight", "input", "none"):
                    with self.subTest(kind=kind, approximate=approximate, compressed=compressed, mask=mask):
                        self.check_case(kind, compressed, mask, approximate)

    def test_functional_producers_preserve_forward_and_backward_contract(self):
        for kind, approximate in (("silu", "none"), ("gelu", "none"), ("gelu", "tanh")):
            for compressed in (False, True):
                for mask in ("weight", "all", "input", "none"):
                    with self.subTest(kind=kind, mask=mask, approximate=approximate, compressed=compressed):
                        self.check_case(kind, compressed, mask, approximate, functional=True)

    def test_malformed_declared_residuals_reject_before_mutation(self):
        for change in ("cardinality", "dtype", "shape"):
            with self.subTest(change=change):
                _, _, local, _, _, _, context, _, _ = self.region("gelu", True, "all")
                activation = next(n for n in local.nodes if isinstance(n, CallNode) and type(n.specification) is NativeActivation)
                if change == "cardinality":
                    activation.specification.saved_components = ()
                else:
                    activation.values[1].meta = replace(activation.values[1].meta,
                        **({"dtype": torch.float32} if change == "dtype" else {"shape": (999, 999)}))
                original = tuple(local.nodes)
                decisions, patterns = forward(local, context)
                self.assertFalse(patterns)
                self.assertEqual(tuple(local.nodes), original)
                self.assertTrue(any(r["reason"].startswith("producer_residual_") for r in decisions))


@unittest.skipUnless(torch.cuda.is_available(), "Generated producer qualification requires CUDA")
class FrozenProducerFusionCudaTests(unittest.TestCase):
    def test_generated_frozen_and_trainable_producers(self):
        for kind, approximate in (("silu", "none"), ("gelu", "none"), ("gelu", "tanh")):
            for compressed in (False, True):
                for functional in (False, True):
                    for mask in ("all", "weight", "input", "none"):
                        with self.subTest(kind=kind, mask=mask, approximate=approximate,
                                          compressed=compressed, functional=functional):
                            FrozenProducerFusionTests().check_case(kind, compressed, mask, approximate,
                                                                  device="cuda:0", functional=functional)
