"""Consumer-specific residual selection and physical saved binding contracts."""

import json
import unittest

import torch

from deepopt.compiler.autodiff import SavedActivationPolicy, SavedUse
from deepopt.compiler.autodiff.activation_saving import SaveAllPolicy, capture_saved_activation_policy
from deepopt.compiler.autodiff.aot_auto_diff import BackwardTraceContext
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import GetAttrNode, IRGraph, TensorMeta, TensorValue
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


class SavedStateClassificationRegressionTests(unittest.TestCase):
    """Execute the audit geometries before invoking the production strict gate."""

    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.manual_seed(5208)

    def compile_pair(self, module, x, policy="torch_amp", *, compressed=True):
        from copy import deepcopy
        from torch.fx.passes.shape_prop import ShapeProp
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx

        if isinstance(module, torch.nn.Sequential):
            fx = torch.fx.Graph()
            value = fx.placeholder("x")
            for name, _ in module.named_children():
                value = fx.call_module(name, (value,))
            fx.output(value)
            graph = torch.fx.GraphModule(module, fx)
        else:
            graph = torch.fx.symbolic_trace(module)
        reference = deepcopy(graph)
        rx = x.detach().clone().requires_grad_(x.requires_grad)
        report, saved = CompilationReport(), []
        compression = dict(enabled=compressed, bits=2, group_size=1024, norm_xhat_recenter=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy=policy,
                      activation_compression=compression, config={}):
            ShapeProp(graph).propagate(x)
            run = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
            expected = reference(rx)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                actual = run(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        for parameter, control in zip(graph.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, control.grad, rtol=0, atol=0)
        return report.as_dict(), saved, graph

    def test_large_groupnorm_parameter_alias_is_model_state_for_both_policies(self):
        from deepopt.opt.nn.normalization.modules import DOGroupNorm
        from examples.vae.qualification import saved_state_evidence

        for policy in ("torch_amp", "bf16_storage"):
            with self.subTest(policy=policy):
                compression = dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)
                module = torch.nn.Sequential(DOGroupNorm(torch.nn.GroupNorm(32, 2048),
                    plan=dict(fp8=False, activation_compression=compression)))
                x = torch.randn(1, 2048, 2, 2, dtype=torch.bfloat16, requires_grad=True)
                report, saved, graph = self.compile_pair(module, x, policy)
                weight = graph.get_submodule("0").weight
                aliases = [t for t in saved if t.device == weight.device and t.numel() and
                           t.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()]
                self.assertTrue(aliases)
                for alias in aliases:
                    self.assertEqual((alias.storage_offset(), tuple(alias.shape), alias.stride(), alias.dtype),
                                     (weight.storage_offset(), tuple(weight.shape), weight.stride(), weight.dtype))
                self.assertTrue(saved_state_evidence(report, True)["passed"])

    def test_original_resolution_convolution_codec_metadata_is_not_dense_activation(self):
        from examples.vae.qualification import saved_state_evidence

        module = torch.nn.Sequential(torch.nn.Conv2d(32, 8, 1, bias=False))
        x = torch.randn(1, 32, 224, 224, dtype=torch.bfloat16, requires_grad=True)
        report, saved, _ = self.compile_pair(module, x)
        self.assertTrue(any(t.dtype == torch.int32 and tuple(t.shape) == (1568, 64) for t in saved))
        self.assertEqual(sum(t.dtype == torch.float32 and tuple(t.shape) == (1568,) for t in saved), 2)
        evidence = saved_state_evidence(report, True)
        self.assertEqual(evidence["saved_tensor_bytes_by_category"]["codec_metadata"], 12544)
        self.assertEqual(evidence["saved_tensor_bytes_by_category"]["packed_payload"], 401408)

    def test_amp_exp_keeps_exact_promoted_forward_coefficient(self):
        from examples.vae.qualification import saved_state_evidence

        class Exp(torch.nn.Module):
            def forward(self, x):
                return x.float().exp()

        x = (torch.randn(1, 4, 28, 28, dtype=torch.bfloat16) * .2).requires_grad_()
        report, saved, _ = self.compile_pair(Exp(), x)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].dtype, torch.float32)
        self.assertEqual(saved[0].numel() * saved[0].element_size(), 12544)
        torch.testing.assert_close(saved[0], x.float().exp(), rtol=0, atol=0)
        binding, = report["graphs"][0]["native_autodiff_graphs"][0]["saved_value_bindings"]
        self.assertEqual(binding["retention_reason"], "consumer_requires_exact")
        self.assertTrue(saved_state_evidence(report, True)["passed"])

    def test_normalization_statistics_at_dense_threshold(self):
        from deepopt.opt.nn.normalization.modules import DOGroupNorm
        from examples.vae.qualification import saved_state_evidence

        compression = dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)
        module = torch.nn.Sequential(DOGroupNorm(torch.nn.GroupNorm(8, 8),
            plan=dict(fp8=False, activation_compression=compression)))
        report, saved, _ = self.compile_pair(module, torch.randn(128, 8, 2, 2,
            dtype=torch.bfloat16, requires_grad=True))
        self.assertGreaterEqual(sum(t.dtype == torch.float32 and t.numel() == 1024 for t in saved), 2)
        evidence = saved_state_evidence(report, True)
        self.assertEqual(evidence["saved_tensor_bytes_by_category"]["normalization_auxiliary"], 8192)

    def test_normalization_alias_slots_follow_runtime_gradient_contract(self):
        from deepopt.opt.nn.normalization.modules import DOGroupNorm
        from examples.vae.qualification import saved_state_evidence

        for policy in ("torch_amp", "bf16_storage"):
            for affine, need_input, train_weight, channels_last in (
                    (True, True, False, False), (True, False, True, False),
                    (False, True, False, False), (True, True, True, True)):
                with self.subTest(policy=policy, affine=affine, input=need_input, weight=train_weight, layout=channels_last):
                    module = torch.nn.GroupNorm(32, 2048, affine=affine)
                    if affine:
                        module.weight = torch.nn.Parameter(torch.randn(4096)[::2], requires_grad=train_weight)
                        module.bias.requires_grad_(train_weight)
                    compression = dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)
                    module = torch.nn.Sequential(DOGroupNorm(module, plan=dict(fp8=False, activation_compression=compression)))
                    x = torch.randn(1, 2048, 2, 2, dtype=torch.bfloat16)
                    if channels_last:
                        x = x.to(memory_format=torch.channels_last)
                    report, _, _ = self.compile_pair(module, x.requires_grad_(need_input), policy)
                    aliases = [row for row in saved_state_evidence(report, True)["values"]
                               if row["category"] == "persistent_model_state_reference" and row["name"].endswith("__weight")]
                    self.assertEqual(bool(aliases), affine and need_input)
        # Unprepared native GroupNorm exports the original affine operand too.
        report, _, _ = self.compile_pair(torch.nn.Sequential(torch.nn.GroupNorm(32, 2048)),
            torch.randn(1, 2048, 2, 2, dtype=torch.bfloat16, requires_grad=True), compressed=False)
        self.assertIn("persistent_model_state_reference", saved_state_evidence(report, False)["categories"])

    def test_copied_model_weights_cannot_inherit_persistent_alias_exemption(self):
        from examples.vae.qualification import saved_state_evidence

        class Model(torch.nn.Module):
            def __init__(self, cast):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(1, 2048), requires_grad=False)
                self.cast = cast
            def forward(self, x):
                coefficient = self.weight.bfloat16() if self.cast else self.weight.clone()
                return x * coefficient

        for cast in (False, True):
            with self.subTest(cast=cast):
                report, saved, graph = self.compile_pair(Model(cast),
                    torch.randn(1, 2048, dtype=torch.bfloat16, requires_grad=True), compressed=False)
                self.assertTrue(any(t.numel() == 2048 and t.untyped_storage().data_ptr() !=
                                    graph.weight.untyped_storage().data_ptr() for t in saved))
                evidence = saved_state_evidence(report, False)
                self.assertIn("derived_model_state_copy", evidence["categories"])
                copies = [row for row in saved_state_evidence(report, True)["values"]
                          if row["category"] == "derived_model_state_copy"]
                self.assertTrue(copies)
                self.assertTrue(all(row["storage_source"] is None and row["bytes"] >= 4096 for row in copies))


class SavedStateClassifierTests(unittest.TestCase):
    """Metadata adversaries supplement the executed positive reproductions."""

    @staticmethod
    def value(name, shape, dtype="torch.float32"):
        return dict(name=name, shape=shape, dtype=dtype, device="cpu")

    @staticmethod
    def binding(source, components, representation="dense", roles=(), constructor="codec.save_activation", **kwargs):
        return dict(source=source, physical_source=source, provenance="activation", representation=representation, logical_device="cpu",
            components=[v["name"] for v in components],
            physical_components=[dict(v, role=roles[i] if roles else "value", constructor=constructor)
                                 for i, v in enumerate(components)], consumers=[], **kwargs)

    def region(self):
        data = self.value("data", [2048, 64], "torch.int32")
        step, minimum = self.value("step", [2048]), self.value("minimum", [2048])
        shape = [2048, 1024]
        codec = self.binding("input", [data, step, minimum], "packed_saved_operand",
            roles=("payload", "codec_step", "codec_minimum"), logical_shape=shape,
            logical_dtype="torch.bfloat16", recipe=dict(bits=2, group_size=1024,
                logical_shape=shape, logical_dtype="torch.bfloat16", matrix_shape=shape, traversal="logical_row_major"))
        codec["consumers"] = [dict(operation="mul", node=["first"], role="left")]
        return dict(saved_state_schema_version=2, bf16_policy=dict(name="torch_amp", version=1),
            saved_state=[data, step, minimum], saved_value_bindings=[
                self.binding(v["name"], [v]) for v in (data, step, minimum)] + [codec])

    def classify(self, region, compressed=True):
        from deepopt.compiler.reporting import classify_saved_state
        return classify_saved_state(region, compressed)

    def exp_region(self):
        coefficient = self.value("exp", [128, 4, 28, 28])
        policy = dict(name="torch_amp", version=1)
        exact = self.binding("exp", [coefficient], constructor="autodiff_primitive_exp",
            producer="autodiff_primitive_exp", bf16_policy=policy,
            logical_shape=coefficient["shape"], logical_dtype=coefficient["dtype"])
        exact["consumers"] = [dict(operation="autodiff_primitive_exp", node=["exp"],
            role="exp.forward_result_coefficient", allow_approximation=False,
            rounding_domain="forward_result_dtype", retention_reason="consumer_requires_exact")]
        return dict(saved_state_schema_version=2, bf16_policy=policy,
                    saved_state=[coefficient], saved_value_bindings=[exact])

    def test_every_category_and_batch128_exp_bytes(self):
        region = self.region()
        parameter = self.value("weight", [2048])
        alias = self.binding("weight", [parameter], constructor="InputNode", storage_source="weight")
        alias["provenance"] = "parameter"
        region["model_state_sources"] = [dict(source="weight", provenance="parameter", shape=[2048], dtype="torch.float32", device="cpu")]
        statistic = self.value("rstd", [1024])
        auxiliary = self.binding("rstd", [statistic], "normalization_auxiliary", roles=("rstd",),
            constructor="autodiff_spatial", recipe=dict(normalization_kind="group", normalization_groups=8,
                source_shape=[128, 8, 2, 2]))
        exp = self.exp_region()
        coefficient, exact = exp["saved_state"][0], exp["saved_value_bindings"][0]
        mask = self.value("mask", [64], "torch.int32")
        lossless = self.binding("activation", [mask], "lossless_clamp_mask", roles=("mask",),
            constructor="autodiff_scalar_clamp", logical_shape=[2048],
            recipe=dict(minimum=0, maximum=None, include_minimum=True, include_maximum=True))
        dense = self.value("small", [3])
        copy = self.value("copy", [3])
        copied = self.binding("copy", [copy], constructor="autodiff_primitive_clone")
        copied["provenance"] = "parameter"
        region["saved_state"].extend([parameter, statistic, coefficient, mask, dense, copy])
        region["saved_value_bindings"].extend([alias, auxiliary, exact, lossless,
            self.binding("small", [dense]), copied])
        rows = {row["name"]: row for row in self.classify(region)}
        expected = dict(data="packed_payload", step="codec_metadata", minimum="codec_metadata",
            weight="persistent_model_state_reference", rstd="normalization_auxiliary",
            exp="approved_exact_coefficient", mask="lossless_residual", small="unexplained_dense_activation",
            copy="derived_model_state_copy")
        self.assertEqual({name: row["category"] for name, row in rows.items()}, expected)
        self.assertEqual(rows["exp"]["bytes"], 128 * 12544)
        self.assertEqual(rows["step"]["bytes"], 8192)
        json.dumps(rows)

    def test_component_membership_is_order_independent_and_preserves_all_consumers(self):
        from copy import deepcopy
        import random

        region = self.region()
        shared = deepcopy(region["saved_value_bindings"][-1])
        shared["consumers"] = [dict(operation="mul", node=["second"], role="right")]
        region["saved_value_bindings"].append(shared)
        expected = self.classify(region)
        self.assertEqual(len(expected[1]["consumers"]), 2)
        self.assertEqual(sum(row["bytes"] for row in expected), 2048 * (64 * 4 + 8))
        for seed in range(10):
            random.Random(seed).shuffle(region["saved_value_bindings"])
            self.assertEqual(self.classify(region), expected)

    def test_malformed_codec_contracts_cannot_exempt_a_dense_tensor(self):
        from copy import deepcopy

        def modify_recipe(region, **changes):
            region["saved_value_bindings"][-1]["recipe"].update(changes)
        cases = {
            "wrong_shape": lambda r: r["saved_state"][1].update(shape=[2047]),
            "wrong_dtype": lambda r: r["saved_state"][1].update(dtype="torch.bfloat16"),
            "wrong_device": lambda r: r["saved_state"][1].update(device="cuda:0"),
            "wrong_source_device": lambda r: r["saved_value_bindings"][-1].update(logical_device="cuda:0"),
            "missing_payload": lambda r: r["saved_state"].pop(0),
            "wrong_constructor": lambda r: r["saved_value_bindings"][-1]["physical_components"][1].update(constructor="extension"),
            "wrong_role": lambda r: r["saved_value_bindings"][-1]["physical_components"][1].update(role="value"),
            "bad_bits": lambda r: modify_recipe(r, bits=3),
            "bad_group": lambda r: modify_recipe(r, group_size=0),
            "bad_block": lambda r: modify_recipe(r, block_shape=[32, 32]),
            "bad_matrix": lambda r: modify_recipe(r, matrix_shape=[2048, 1025]),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                region = deepcopy(self.region())
                mutate(region)
                with self.assertRaisesRegex(AssertionError, "Invalid native saved-state contract"):
                    self.classify(region)

    def test_conflicting_membership_and_exact_dense_source_are_not_hidden(self):
        from copy import deepcopy

        region = self.region()
        conflict = deepcopy(region["saved_value_bindings"][-1])
        conflict["components"][1:] = reversed(conflict["components"][1:])
        descriptions = conflict["physical_components"]
        descriptions[1]["name"], descriptions[2]["name"] = descriptions[2]["name"], descriptions[1]["name"]
        region["saved_value_bindings"].append(conflict)
        with self.assertRaisesRegex(AssertionError, "Conflicting"):
            self.classify(region)
        region = self.region()
        original = self.value("input", [2048, 1024], "torch.bfloat16")
        region["saved_state"].append(original)
        region["saved_value_bindings"].append(self.binding("input", [original]))
        with self.assertRaisesRegex(AssertionError, "large dense.*input"):
            self.classify(region)

    def test_names_and_retention_reasons_do_not_grant_exemptions(self):
        for representation in ("dense", "packed_fake", "lossless_fake"):
            value = self.value("packed_codec_step", [2048])
            binding = self.binding(value["name"], [value], representation, retention_reason="consumer_requires_exact")
            binding["component_of"] = ["imaginary_owner"]
            region = dict(saved_state=[value], saved_value_bindings=[binding])
            with self.subTest(representation=representation), self.assertRaisesRegex(AssertionError, "large dense"):
                self.classify(region)

    def test_model_copy_requires_a_valid_model_origin_and_transformation(self):
        from copy import deepcopy

        value = self.value("copy", [2048])
        declaration = dict(source="weight", provenance="parameter", shape=[2048], dtype="torch.float32", device="cpu")
        binding = self.binding("copy", [value], constructor="autodiff_primitive_clone",
            model_origin={key: val for key, val in declaration.items() if key != "provenance"}, model_transforms=["clone"])
        binding["provenance"] = "parameter"
        valid = dict(saved_state_schema_version=2, saved_state=[value], saved_value_bindings=[binding],
                     model_state_sources=[declaration])
        row, = self.classify(valid)
        self.assertEqual((row["category"], row["storage_source"], row["bytes"]), ("derived_model_state_copy", None, 8192))
        for field, replacement in (("model_origin", None), ("model_transforms", []),
                                   ("model_transforms", ["sin"])):
            region = deepcopy(valid)
            region["saved_value_bindings"][0][field] = replacement
            with self.subTest(field=field, replacement=replacement), self.assertRaisesRegex(AssertionError, "derived_model_state_copy"):
                self.classify(region)
        region = deepcopy(valid)
        region["model_state_sources"] = []
        with self.assertRaisesRegex(AssertionError, "derived_model_state_copy"):
            self.classify(region)
        region = deepcopy(valid)
        region["saved_value_bindings"][0]["physical_components"][0]["constructor"] = "extension"
        with self.assertRaisesRegex(AssertionError, "derived_model_state_copy"):
            self.classify(region)

    def test_normalization_auxiliary_requires_its_role_and_domain_geometry(self):
        from copy import deepcopy

        value = self.value("rstd", [1024])
        binding = self.binding("rstd", [value], "normalization_auxiliary", roles=("rstd",),
            constructor="autodiff_spatial", recipe=dict(source_shape=[128, 8, 2, 2],
                normalization_kind="group", normalization_groups=8))
        region = dict(saved_state_schema_version=2, saved_state=[value], saved_value_bindings=[binding])
        self.assertEqual(self.classify(region)[0]["category"], "normalization_auxiliary")
        cases = {
            "role": lambda b: b["physical_components"][0].update(role="activation"),
            "constructor": lambda b: b["physical_components"][0].update(constructor="extension"),
            "domains": lambda b: b["recipe"].update(normalization_groups=4),
            "dtype": lambda b: b["recipe"].update(statistics_dtype="torch.float16"),
        }
        for name, mutate in cases.items():
            bad = deepcopy(region)
            mutate(bad["saved_value_bindings"][0])
            with self.subTest(name=name), self.assertRaisesRegex(AssertionError, "normalization_auxiliary"):
                self.classify(bad)

    def test_forged_exp_contract_and_additional_exact_consumers_fail(self):
        from copy import deepcopy

        cases = {
            "operator": lambda r, b, c: b.update(producer="autodiff_primitive_sin"),
            "policy": lambda r, b, c: r.update(bf16_policy=dict(name="bf16_storage", version=1)),
            "source": lambda r, b, c: b.update(source="unrelated"),
            "shape": lambda r, b, c: b.update(logical_shape=[1]),
            "dtype": lambda r, b, c: b.update(logical_dtype="torch.bfloat16"),
            "consumer_operator": lambda r, b, c: c.update(operation="extension"),
            "consumer_source": lambda r, b, c: c.update(node=["unrelated"]),
            "consumer_role": lambda r, b, c: c.update(role="coefficient"),
            "consumer_rounding": lambda r, b, c: c.update(rounding_domain="operand_dtype"),
            "approximate": lambda r, b, c: c.update(allow_approximation=True),
            "another_exact_consumer": lambda r, b, c: b["consumers"].append(dict(c, operation="extension")),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                region = deepcopy(self.exp_region())
                binding = region["saved_value_bindings"][0]
                mutate(region, binding, binding["consumers"][0])
                with self.assertRaisesRegex(AssertionError, "large dense"):
                    self.classify(region)
        region = self.exp_region()
        other = deepcopy(region["saved_value_bindings"][0])
        other["consumers"][0]["operation"] = "extension"
        region["saved_value_bindings"].append(other)
        with self.assertRaisesRegex(AssertionError, "large dense"):
            self.classify(region)

    def test_legacy_codec_geometry_is_readable_but_missing_proof_is_rejected(self):
        region = self.region()
        region.pop("saved_state_schema_version")
        for binding in region["saved_value_bindings"]:
            binding.pop("physical_components")
        self.assertEqual(self.classify(region)[1]["category"], "codec_metadata")
        region["saved_value_bindings"][-1]["recipe"] = dict(bits=2, group_size=1024)
        with self.assertRaisesRegex(AssertionError, "Invalid native saved-state contract"):
            self.classify(region)

    def test_legacy_lossless_mask_requires_native_branch_contract(self):
        value = self.value("mask", [64], "torch.int32")
        binding = dict(source="activation", representation="lossless_clamp_mask", components=["mask"],
            logical_shape=[2048], recipe=dict(minimum=0, maximum=None, include_minimum=True, include_maximum=True),
            consumers=[dict(operation="autodiff_scalar_clamp", role="input_gradient_branch")])
        region = dict(saved_state=[value], saved_value_bindings=[binding])
        self.assertEqual(self.classify(region)[0]["category"], "lossless_residual")
        binding["consumers"] = []
        with self.assertRaisesRegex(AssertionError, "lossless_clamp_mask"):
            self.classify(region)

    def test_incompatible_schema_and_untyped_reports_fail_clearly(self):
        for version in (0, 4, "2", True):
            region = self.region()
            region["saved_state_schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(AssertionError, "Incompatible"):
                self.classify(region)
        for field in ("shape", "dtype"):
            region = self.region()
            region["saved_state"][0][field] = None
            with self.subTest(field=field), self.assertRaisesRegex(AssertionError, "Untyped"):
                self.classify(region)


class SavedActivationContractTests(unittest.TestCase):
    def make_context(self, tensor, *, policy=None, provenance="user_input"):
        graph = IRGraph()
        forward = GraphBuilder(graph)
        value = forward.input(0, TensorMeta(tuple(tensor.shape), tensor.dtype, tensor.device,
                              tuple(tensor.stride())), name="coefficient").value
        value.provenance = provenance
        backward = GraphBuilder(IRGraph(), value_prefix="b")
        context = BackwardTraceContext(backward, first_saved_input_index=0,
            activation_policy=SaveAllPolicy(), forward_graph=graph,
            saved_activation_policy=policy or SavedActivationPolicy(True, 2, 1024))
        return forward, backward, context, value

    def execute(self, tensor, forward, backward, context, results):
        forward.output((forward.graph.nodes[0].value, context.saved_forward_value))
        backward.output(results)
        forward.graph.validate()
        backward.graph.validate()
        def compile_graph(graph):
            return WrapperCodegen(torch.nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run().fn
        output, physical = compile_graph(forward.graph)(tensor)
        restored = compile_graph(backward.graph)(*physical)
        torch.testing.assert_close(output, tensor, rtol=0, atol=0)
        return restored

    def test_omitted_request_is_exact_even_with_legacy_value_annotation(self):
        tensor = torch.randn(65, 17, dtype=torch.bfloat16)
        forward, backward, context, value = self.make_context(tensor)
        value.save_codec = {"bits": 2, "group_size": 1024}
        saved = context.saved(value)
        self.assertEqual(context.saved_forward_value, (value,))
        self.assertFalse(any(getattr(node, "op", "") == "codec.save_activation" for node in forward.graph.nodes))
        restored, = self.execute(tensor, forward, backward, context, (saved,))
        torch.testing.assert_close(restored, tensor, rtol=0, atol=0)
        self.assertEqual(context.saved_value_specs[0]["retention_reason"], "consumer_requires_exact")

    def test_exact_and_approximate_consumers_have_distinct_bindings(self):
        tensor = torch.randn(65, 17, dtype=torch.bfloat16)
        forward, backward, context, value = self.make_context(tensor)
        exact = context.saved(value)
        approximate = context.saved(value, use=SavedUse("mul.grad_input", True, "mul"))
        self.assertEqual(len(context.saved_forward_value), 4)
        self.assertEqual(len(context.saved_value_specs), 2)
        dense, decoded = self.execute(tensor, forward, backward, context, (exact, approximate))
        oracle = dequantize_unpack_activation(quantize_pack_activation(tensor, 1024), dtype=tensor.dtype)
        torch.testing.assert_close(dense, tensor, rtol=0, atol=0)
        torch.testing.assert_close(decoded, oracle, rtol=0, atol=0)
        self.assertEqual(len({name for spec in context.saved_value_specs for name in spec["components"]}), 4)
        json.dumps(context.saved_value_specs)

    def test_binding_cache_keeps_every_consumer_and_node_dependency(self):
        tensor = torch.randn(65, 17, dtype=torch.bfloat16)
        forward, backward, context, value = self.make_context(tensor)
        first = forward.emit("mul", value, value).value.producer
        second = forward.emit("mul", value, value).value.producer
        aliases = []
        for node in (first, second):
            context.begin_node(node)
            aliases.append(context.saved(value, use=SavedUse("grad_left", True, "mul")))
            context.saved(value, use=SavedUse("grad_right", True, "mul"))
            self.assertEqual(len(context.finish_node()), 3)
        self.assertIs(aliases[0], aliases[1])
        self.assertEqual(len(context.saved_forward_value), 3)
        self.assertEqual(len(context.saved_value_specs), 1)
        self.assertEqual(len(context.saved_value_specs[0]["consumers"]), 4)
        self.execute(tensor, forward, backward, context, (aliases[0],))

    def test_codec_geometry_identity_is_immutable_and_recipe_specific(self):
        tensor = torch.randn(65, 17, dtype=torch.bfloat16)
        forward, _, context, value = self.make_context(tensor)
        first = context.saved(value, use=SavedUse("first", True, codec=(("group_size", 1024), ("bits", 2))))
        second = context.saved(value, use=SavedUse("second", True, codec=(("bits", 4), ("group_size", 1024))))
        self.assertIsNot(first, second)
        self.assertEqual(len(context.saved_forward_value), 6)
        self.assertEqual(len({value.name for value in context.saved_forward_value}), 6)
        packs = [node for node in forward.graph.nodes if getattr(node, "op", "") == "codec.save_activation"]
        self.assertEqual(len(packs), 2)
        with self.assertRaises(TypeError):
            packs[0].specification["bits"] = 8

    def test_model_owned_values_stay_exact_after_joint_input_lifting(self):
        from deepopt.compiler.autodiff.joint import _clone_region

        graph = IRGraph()
        builder = GraphBuilder(graph)
        meta = TensorMeta((4, 17), torch.bfloat16, "cpu", (17, 1), True)
        parameter = TensorValue("weight", meta, provenance="parameter")
        graph.add(GetAttrNode("weight", parameter))
        output = builder.mul(parameter, parameter)
        builder.output(output)
        local, _, _, _ = _clone_region([output.value.producer], KernelCodegenContext(), BackendRegistry())
        lifted = local.nodes[0].value
        self.assertEqual(lifted.provenance, "parameter")
        recipe, reason = SavedActivationPolicy(True, 2, 1024).select(lifted, SavedUse("mul", True))
        self.assertIsNone(recipe)
        self.assertEqual(reason, "model_owned_state")

    def test_scalar_rank_one_spatial_tails_offsets_and_layout_restoration(self):
        tensors = [torch.tensor(1.25, dtype=torch.float32),
            torch.randn(37, dtype=torch.bfloat16),
            torch.randn(2, 3, 4, 7, dtype=torch.bfloat16).to(memory_format=torch.channels_last),
            torch.randn(3, 9, dtype=torch.bfloat16).t(),
            torch.randn(5, 13, dtype=torch.bfloat16)[1:4, 2:11:2]]
        for tensor in tensors:
            with self.subTest(shape=tuple(tensor.shape), strides=tensor.stride()):
                forward, backward, context, value = self.make_context(tensor)
                use = SavedUse("explicit", True, codec=(("bits", 2), ("group_size", 1024)))
                restored = context.saved(value, use=use)
                decoded, = self.execute(tensor, forward, backward, context, (restored,))
                matrix = tensor.reshape((-1, tensor.shape[-1]) if tensor.ndim else (1, 1))
                oracle = dequantize_unpack_activation(quantize_pack_activation(matrix, 1024), dtype=tensor.dtype).reshape(tensor.shape)
                torch.testing.assert_close(decoded, oracle, rtol=0, atol=0)
                self.assertEqual(tuple(decoded.stride()), restored.value.meta.stride)
                if tensor.ndim == 4:
                    self.assertTrue(decoded.is_contiguous(memory_format=torch.channels_last))

    def test_empty_and_expanded_storage_do_not_allocate_larger_packs(self):
        for tensor in (torch.empty(0, 3, dtype=torch.bfloat16),
                       torch.randn(1, 4, dtype=torch.bfloat16).expand(512, 4)):
            forward, backward, context, value = self.make_context(tensor)
            restored = context.saved(value, use=SavedUse("mul", True))
            self.assertEqual(context.saved_forward_value, (value,))
            self.execute(tensor, forward, backward, context, (restored,))

    def test_explicit_disabled_scope_overrides_inherited_saved_policy(self):
        from deepopt.opt import autocast

        with autocast("cpu", activation_compression={"enabled": True, "bits": 2, "group_size": 1024}):
            outer = capture_saved_activation_policy("cpu")
            with autocast("cpu", activation_compression=None):
                self.assertEqual(capture_saved_activation_policy("cpu"), outer)
            with autocast("cpu", activation_compression=False):
                self.assertFalse(capture_saved_activation_policy("cpu").enabled)
            self.assertEqual(capture_saved_activation_policy("cpu"), outer)
        self.assertTrue(outer.enabled)
        self.assertFalse(capture_saved_activation_policy("cpu").enabled)

    def test_fp8_binding_cache_preserves_physical_components_and_consumers(self):
        graph, backward = IRGraph(), GraphBuilder(IRGraph())
        forward = GraphBuilder(graph)
        value = forward.input(0, TensorMeta((2, 8), torch.bfloat16, "cpu", (8, 1),
            representation="fp8_scaled", storage_dtype=torch.float8_e4m3fn)).value
        context = BackwardTraceContext(backward, first_saved_input_index=0,
                                      activation_policy=SaveAllPolicy(), forward_graph=graph)
        for role in ("first", "second"):
            context.begin_node()
            context.saved_activation_policy = SavedActivationPolicy(True, 2, 1024)
            context.saved(value, use=SavedUse(role, allow_approximation=True))
            self.assertEqual(len(context.finish_node()), 2)
        self.assertEqual(len(context.saved_forward_value), 2)
        self.assertEqual(len(context.saved_value_specs), 1)
        self.assertEqual(len(context.saved_value_specs[0]["consumers"]), 2)
        self.assertEqual(context.saved_value_specs[0]["representation"], "fp8_scaled")

    def test_replay_reuse_records_leaves_for_every_consumer(self):
        from deepopt.compiler.autodiff.activation_saving import RecomputePointwisePolicy

        tensor = torch.randn(4, 17)
        forward, _, context, value = self.make_context(tensor)
        context.activation_policy = RecomputePointwisePolicy()
        intermediate = forward.emit("sigmoid", value).value
        consumers = (forward.emit("mul", intermediate, 2).value.producer,
                     forward.emit("mul", intermediate, 3).value.producer)
        aliases = []
        for node in consumers:
            context.begin_node(node)
            aliases.append(context.saved(intermediate))
            self.assertEqual(context.finish_node(), (value,))
        self.assertIs(aliases[0], aliases[1])
        leaf = next(spec for spec in context.saved_value_specs if spec["source"] == value.name)
        self.assertEqual(len(leaf["consumers"]), 2)

    def test_custom_policy_cannot_replay_a_stateful_pointwise_target(self):
        class AlwaysReplay:
            name = "always"
            def should_recompute(self, producer, value):
                return True

        tensor = torch.randn(4, 17)
        forward, _, context, value = self.make_context(tensor)
        intermediate = forward.emit("sigmoid", value).value
        intermediate.producer.target = torch.rand_like
        context.activation_policy = AlwaysReplay()
        context.saved(intermediate)
        self.assertEqual(context.saved_forward_value, (intermediate,))

    def test_model_owned_view_retains_exact_provenance(self):
        from deepopt.compiler import BackendConfig, compile_fx
        from torch.fx.passes.shape_prop import ShapeProp

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 17, dtype=torch.bfloat16))
            def forward(self, x):
                return x * self.weight.view(1, 4, 17)

        model = torch.fx.symbolic_trace(Model())
        x = torch.randn(2, 4, 17, dtype=torch.bfloat16, requires_grad=True)
        ShapeProp(model).propagate(x)
        from deepopt.opt import autocast
        with autocast("cpu", activation_compression={"enabled": True, "bits": 2, "group_size": 1024}):
            run = compile_fx(model, [x], config=BackendConfig(require_full_aot=True))
            output = run(x)
        bindings = run.compilation_report.native_autodiff_graphs[0]["saved_value_bindings"]
        parameter_bindings = [item for item in bindings if item["provenance"] == "parameter"]
        self.assertEqual(len(parameter_bindings), 1)
        self.assertEqual(parameter_bindings[0]["representation"], "dense")
        self.assertEqual(parameter_bindings[0]["retention_reason"], "model_owned_state")
        output.sum().backward()

    def test_host_scalar_does_not_override_cuda_activation_policy_device(self):
        from deepopt.compiler.autodiff.activation_saving import activation_device_type

        host = TensorValue("scale", TensorMeta((), torch.float32, "cpu"))
        activation = TensorValue("feature", TensorMeta((8, 16), torch.bfloat16, "cuda"))
        self.assertEqual(activation_device_type((host, activation)), "cuda")


if __name__ == "__main__":
    unittest.main()
