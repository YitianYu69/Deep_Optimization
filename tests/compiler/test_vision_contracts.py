"""Boundary, alias, ownership and recipe regressions for full vision support."""

from copy import deepcopy
import gc
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.effects import analyze_fx_effects, snapshot_stochastic_depth
from deepopt.opt.policy import prepare_fp8_model


def config(policy="bf16_storage", *, enabled=True, attention_bits=2, exclude=()):
    return dict(runtime=dict(bf16_policy=policy, fp8=dict(enabled=False, exclude_modules=list(exclude),
            intercept_linear=False, intercept_convolution=False, intercept_normalization=False, intercept_pooling=False),
        activation_compression=dict(enabled=enabled, bits=2, attention_bits=attention_bits, group_size=1024),
        attention=dict(backend="native")))


class VisionContractsTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(913)

    def test_mha_root_layouts_identity_idempotence_and_optimizer_update(self):
        from deepopt.opt.nn.attention import DOMultiheadAttention
        from deepopt.opt.quantization.attention import observe_attention_codec

        for policy, enabled, bits in (("torch_amp", False, None), ("torch_amp", True, None),
                                      ("bf16_storage", False, None), ("bf16_storage", True, 2)):
            for batch_first, bias in ((True, True), (False, False)):
                with self.subTest(policy=policy, compression=enabled, bits=bits, batch_first=batch_first, bias=bias):
                    raw = nn.MultiheadAttention(32, 4, batch_first=batch_first, bias=bias)
                    original = [(name, id(p)) for name, p in raw.named_parameters()]
                    projection, state = raw.out_proj, deepcopy(raw.state_dict())
                    optimizer = torch.optim.SGD(raw.parameters(), lr=.01)
                    cfg = config(policy, enabled=enabled, attention_bits=bits)
                    model, _ = prepare_fp8_model(raw, False, config=cfg)
                    self.assertEqual(isinstance(model, DOMultiheadAttention), enabled or policy == "bf16_storage")
                    self.assertIs(model.out_proj, projection)
                    self.assertEqual(original, [(name, id(p)) for name, p in model.named_parameters()])
                    self.assertEqual(tuple(state), tuple(model.state_dict()))
                    self.assertIs(prepare_fp8_model(model, False, config=cfg)[0], model)
                    copied = deepcopy(model)
                    copied.load_state_dict(state)
                    self.assertEqual(tuple(copied.state_dict()), tuple(state))
                    repr(model)
                    x = torch.randn((2, 17, 32) if batch_first else (17, 2, 32), dtype=torch.bfloat16)
                    outputs = []
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with observe_attention_codec() as telemetry, autocast("cpu", dtype=torch.bfloat16, fp8=False,
                                bf16_policy=policy, activation_compression=cfg["runtime"]["activation_compression"], config=cfg):
                            output, _ = model(x, x, x, need_weights=False)
                        outputs.append(output.detach().clone())
                        output.backward(torch.linspace(-1, 1, output.numel(), dtype=output.dtype).reshape(output.shape))
                        self.assertTrue(all(p.grad is not None and p.grad.dtype == torch.float32 for p in model.parameters()))
                        optimizer.step()
                        if enabled:
                            self.assertGreaterEqual(len(telemetry), 3)
                            self.assertEqual({row["bits"] for row in telemetry}, {4 if bits is None else bits})
                    self.assertFalse(torch.equal(*outputs), "Optimizer update must refresh prepared projections")

    def test_shared_attention_and_excluded_alias_preserve_scope(self):
        from deepopt.opt.nn.attention import DOMultiheadAttention

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.MultiheadAttention(32, 4, batch_first=True)
                self.second = self.first
        for exclusions in ((), ("second",)):
            raw = Shared()
            prepared, _ = prepare_fp8_model(raw, False, config=config(exclude=exclusions))
            self.assertIs(prepared.first, prepared.second)
            self.assertEqual(isinstance(prepared.first, DOMultiheadAttention), not exclusions)

    def test_storage_add_promotes_before_rounding_and_freezes_backward(self):
        class Add(nn.Module):
            def __init__(self):
                super().__init__()
                self.position = nn.Parameter(torch.tensor([[[-.999, .998, -.997]]]))
            def forward(self, x):
                return x + self.position
        model, _ = prepare_fp8_model(Add(), False, config=config(enabled=False))
        x = torch.ones(5, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        expected = (x.float() + model.position).bfloat16()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage", config={}):
            output = model(x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(output, (x + model.position.bfloat16()).bfloat16()))
        upstream = torch.linspace(-3, 4, output.numel(), dtype=torch.bfloat16).reshape(output.shape)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="torch_amp", config={}):
            output.backward(upstream)
        torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)
        torch.testing.assert_close(model.position.grad, upstream.sum((0, 1), keepdim=True, dtype=torch.float32), rtol=0, atol=0)

    def test_managed_primitive_rejects_out_duplicate_and_disabled_regions(self):
        from deepopt._bf16_policy import bf16_model_region
        from deepopt.opt.nn.primitive_state import primitive_arguments

        x, y = torch.ones(2, dtype=torch.bfloat16), torch.ones(2)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage", config={}), bf16_model_region("cpu"):
            for function, args, kwargs in ((torch.add, (x, y), {"out": y}), (torch.add, (x, y), {"input": x}),
                    (torch.cat, ((x, y), 0), {"dim": 0}), (torch.cat, ((x, 2),), {})):
                self.assertIsNone(primitive_arguments(function, args, kwargs))
            self.assertIsNotNone(primitive_arguments(torch.add, (x, y), {}))
            with autocast("cpu", enabled=False):
                self.assertIsNone(primitive_arguments(torch.add, (x, y), {}))
                self.assertEqual((x + y).dtype, torch.float32)

    def test_raw_functional_silu_write_retains_its_activation_kind(self):
        class Silu(nn.Module):
            def forward(self, x):
                return nn.functional.silu(x.clone(), True)
        gm = torch.fx.symbolic_trace(Silu())
        x = torch.linspace(-3, 3, 257, requires_grad=True)
        ShapeProp(gm).propagate(x)
        expected = gm(x)
        run = compile_fx(gm, (x,), config=BackendConfig(require_full_aot=True))
        actual = run(x)
        torch.testing.assert_close(actual, expected)
        upstream = torch.linspace(-2, 1, x.numel())
        torch.testing.assert_close(torch.autograd.grad(actual, x, upstream)[0], torch.autograd.grad(expected, x, upstream)[0])

    def test_aliases_after_write_and_external_or_duplicated_destinations_refuse_proof(self):
        cases = []
        class LaterView(nn.Module):
            def forward(self, x):
                y = x.clone()
                out = y.add_(1)
                return out, y.view(-1)
        class Duplicate(nn.Module):
            def forward(self, x):
                y = x.clone()
                return y.add_(y)
        class External(nn.Module):
            def forward(self, x):
                return x.add_(1)
        cases.extend((LaterView(), Duplicate(), External()))
        for model in cases:
            effects = analyze_fx_effects(torch.fx.symbolic_trace(model))
            self.assertTrue(effects.mutation_nodes)
            self.assertFalse(effects.write_rewrites)

    def test_stochastic_snapshot_refuses_escaping_or_later_mutated_mask(self):
        class Escaping(nn.Module):
            def forward(self, x):
                noise = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device).bernoulli_(.8)
                return x * noise, noise
        class LaterMutation(nn.Module):
            def forward(self, x):
                noise = torch.empty((x.shape[0], 1), dtype=x.dtype, device=x.device).bernoulli_(.8)
                out = x * noise
                noise.div_(.8)
                return out
        for model in (Escaping(), LaterMutation()):
            captured = []
            def backend(graph, inputs):
                captured.append(graph)
                return graph.forward
            torch.compile(model, backend=backend, fullgraph=True)(torch.ones(2, 3, dtype=torch.bfloat16))
            graph, = captured
            self.assertTrue(any(node.target == "bernoulli_" for node in graph.graph.nodes))
            self.assertFalse(snapshot_stochastic_depth(graph))

    def test_exact_mask_release_interleaved_retained_and_discarded_graphs(self):
        from benchmarks.analysis.measurement import SavedStorageLifetimes
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from tests.compiler.test_vision_patterns import StochasticDepthPattern

        cfg = config()
        model, _ = prepare_fp8_model(StochasticDepthPattern(), False, config=cfg)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options=dict(require_full_aot=True, use_native_autodiff=True)), config=cfg)
        tracker = SavedStorageLifetimes()
        x = torch.ones(4, 3, dtype=torch.bfloat16, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(tracker.pack, tracker.unpack), \
                autocast("cpu", dtype=torch.bfloat16, fp8=False, config=cfg):
            first, second, discarded = model(x), model(x), model(x)
        self.assertGreater(tracker.snapshot()["live_nonpersistent_bytes"], 0)
        del discarded
        gc.collect()
        size = tracker.snapshot()["live_nonpersistent_bytes"]
        first.backward(torch.ones_like(first), retain_graph=True)
        self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], size)
        second.backward(torch.ones_like(second))
        self.assertLess(tracker.snapshot()["live_nonpersistent_bytes"], size)
        first.backward(torch.ones_like(first))
        self.assertEqual(tracker.snapshot()["live_nonpersistent_bytes"], 0)

    def test_stale_storage_recipe_rejects_before_running_stats_or_rng(self):
        cfg = config()
        model, _ = prepare_fp8_model(nn.Sequential(nn.BatchNorm2d(4), nn.Dropout(.2)), False, config=cfg)
        model._deepopt_bf16_primitive_recipe = 1
        buffers, rng = deepcopy(model.state_dict()), torch.get_rng_state()
        with self.assertRaisesRegex(ValueError, "primitive recipe"), autocast("cpu", dtype=torch.bfloat16, fp8=False, config=cfg):
            model(torch.randn(2, 4, 4, 4, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(4)))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in buffers.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_math_attention_gqa_mask_dropout_roles_and_forged_descriptors(self):
        from deepopt.compiler import CompilationReport
        from deepopt.compiler.saved_state import classify_saved_state
        from torch.nn.attention import SDPBackend, sdpa_kernel

        class Attention(nn.Module):
            def forward(self, q, k, v, mask):
                return nn.functional.scaled_dot_product_attention(q, k, v, mask, dropout_p=.2, enable_gqa=True)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                  for shape in ((1, 4, 17, 16), (1, 2, 19, 16), (1, 2, 19, 16))]
        inputs.append(torch.ones(2, 1, 17, 19, dtype=torch.bool))
        graph = torch.fx.symbolic_trace(Attention())
        report = CompilationReport()
        with sdpa_kernel(SDPBackend.MATH), autocast("cpu", dtype=torch.bfloat16, fp8=False, config=config()):
            ShapeProp(graph).propagate(*inputs)
            run = compile_fx(graph, tuple(inputs), config=BackendConfig(require_full_aot=True,
                collect_saved_state_metadata=True, bf16_policy="bf16_storage"), report=report)
            output = run(*inputs)
        output.backward(torch.linspace(-1, 2, output.numel(), dtype=output.dtype).reshape(output.shape))
        region = report.as_dict()["graphs"][0]["native_autodiff_graphs"][0]
        classified = classify_saved_state(region, True)
        self.assertTrue(any(value["role"] == "attention.drop_mask" and value["bytes"] > 0 for value in classified))
        self.assertTrue(any(value["category"] == "packed_payload" for value in classified))
        for field in ("bits", "owner", "unknown_runtime", "absent_required"):
            changed = deepcopy(region)
            state = next(record for record in changed["saved_value_bindings"] if record["representation"] == "attention_state")
            if field == "bits":
                state["recipe"] = {**dict(state["recipe"]), "bits": 4}
            elif field == "owner":
                state["physical_components"][1]["constructor"] = "name_looks_like_attention"
            elif field == "unknown_runtime":
                changed.pop("runtime_saved_state")
            else:
                target = state["components"][1]
                actual = next(value for value in changed["runtime_saved_state"]["bindings"] if value["name"] == target)
                actual["presence"] = "absent"
            with self.subTest(field=field), self.assertRaises(AssertionError):
                classify_saved_state(changed, True)


if __name__ == "__main__":
    unittest.main()
