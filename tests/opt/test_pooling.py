"""Installed pooling APIs, represented-value math and compressed saved-state contracts."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import deepopt
from deepopt.opt.nn import functional as D
from deepopt.opt.nn.pooling.spec import SPECS, MODULE_SPECS
from deepopt.opt.policy import pooling_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.quantization.indices import pack_indices, unpack_indices


def policy(compressed=True, fp8=False, bits=2, group=17, **settings):
    return pooling_plan({"enabled": True, "backend": "reference", **settings} if fp8 else False,
                        activation_compression=dict(enabled=compressed, bits=bits, group_size=group))


def case(kind, *, dtype=torch.float32, unbatched=False, device="cpu"):
    spec = SPECS[kind]
    shape = ((3,) if unbatched else (2, 3)) + (5,) * spec.rank
    x = torch.randn(shape, device=device, dtype=dtype)
    if spec.family == "unpool":
        y, indices = getattr(F, f"max_pool{spec.rank}d")(x.float(), 2, 2, return_indices=True)
        x = y.to(dtype)
        options = dict(indices=indices, kernel_size=2, output_size=shape[-spec.rank:])
    elif spec.family.startswith("adaptive"):
        options = dict(output_size=(3,) * spec.rank)
    elif spec.family == "power":
        options = dict(norm_type=2, kernel_size=2, ceil_mode=True)
    elif spec.family == "fractional_max":
        options = dict(kernel_size=2, output_size=(3,) * spec.rank,
                       _random_samples=torch.rand(1 if unbatched else 2, 3, spec.rank, device=device, dtype=dtype))
    else:
        options = dict(kernel_size=2, stride=2, ceil_mode=True)
    return x.requires_grad_(), options


class PoolingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8501)
        # Native unpool oracles reject deterministic mode even with unique
        # indices. Isolate these value tests from Trainer's global setting.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        torch.use_deterministic_algorithms(False)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_all_twenty_families_values_and_backward(self):
        self.assertEqual(len(SPECS), 20)
        for kind, spec in SPECS.items():
            for unbatched in (False, True):
                for compressed in (False, True):
                    with self.subTest(kind=kind, unbatched=unbatched, compressed=compressed):
                        x, options = case(kind, unbatched=unbatched)
                        z = x.detach().clone().requires_grad_()
                        actual = getattr(D, "do_" + kind)(x, **options, plan=policy(compressed, fp8=not compressed))
                        expected = spec.native(z, **options)
                        dy = torch.randn_like(actual)
                        actual.backward(dy); expected.backward(dy)
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
                        if spec.family != "power" or not compressed:
                            torch.testing.assert_close(x.grad, z.grad, rtol=3e-6, atol=3e-6)
                        self.assertTrue(torch.isfinite(x.grad).all())

    def test_bf16_fp8_inputs_outputs_all_families(self):
        for kind, spec in SPECS.items():
            for encoded in (False, True):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    with self.subTest(kind=kind, encoded=encoded, dtype=dtype):
                        x, options = case(kind, dtype=torch.bfloat16)
                        source = to_fp8_activation(x, layout="channels", block_shape=(2, 4)) if encoded else x
                        represented = source.dequantize(torch.float32) if encoded else x.float()
                        native_options = dict(options)
                        if "_random_samples" in native_options:
                            native_options["_random_samples"] = native_options["_random_samples"].float()
                        expected = spec.native(represented, **native_options)
                        y = getattr(D, "do_" + kind)(source, **options, plan=policy(True, True),
                                                    output_dtype=dtype, output_layout="channels")
                        self.assertEqual(isinstance(y, FP8Activation), dtype != torch.bfloat16)
                        torch.testing.assert_close(dequantize_activation(y).float(), expected.float(),
                                                   rtol=0.15 if dtype != torch.bfloat16 else 0.006,
                                                   atol=0.035 if dtype != torch.bfloat16 else 0.016)
                        y.backward(torch.randn_like(x, dtype=torch.bfloat16).new_empty(y.shape).normal_())
                        self.assertTrue(torch.isfinite(x.grad).all())

    def test_lp_compressed_derivatives_match_decoded_factor_oracle(self):
        for rank in (1, 2, 3):
            kind = f"lp_pool{rank}d"
            for power in (1., 2., 3., 2.5, -1.):
                for bits in (1, 2, 4, 8):
                    x = (torch.rand((2, 3) + (5,) * rank) + 0.5).requires_grad_()
                    y = getattr(D, "do_" + kind)(x, power, 2, stride=1, ceil_mode=True,
                                                plan=policy(True, bits=bits))
                    _, data, step, minimum, factor, *_ = y.grad_fn.saved_tensors
                    shape = (x.numel() // x.shape[-1], x.shape[-1])
                    packed = PackedActivation(data, step, minimum, shape, 17, None, bits)
                    a = dequantize_unpack_activation(packed, dtype=torch.float32).reshape_as(x)
                    powered = x.detach().pow(power).requires_grad_()
                    z = getattr(F, f"avg_pool{rank}d")(powered, 2, 1, 0, True)
                    native = (z.sign() * F.relu(z.abs()) * 2**rank).pow(1 / power)
                    dy = torch.randn_like(y)
                    reduced = torch.autograd.grad(native, powered, dy)[0]
                    y.backward(dy)
                    torch.testing.assert_close(x.grad, a * reduced, rtol=3e-6, atol=3e-6)
                    self.assertEqual(factor.numel(), y.numel())

    def test_private_routing_public_indices_guards_and_unpool(self):
        for kind, spec in SPECS.items():
            if "max" not in spec.family:
                continue
            x, options = case(kind)
            options["return_indices"] = True
            y, indices = getattr(D, "do_" + kind)(x, **options, plan=policy())
            expected, expected_indices = spec.native(x.detach(), **options)
            torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
            torch.testing.assert_close(y, expected)
            self.assertEqual(indices.dtype, torch.int64)
            self.assertFalse(indices.requires_grad)
            saved = y.grad_fn.saved_tensors
            self.assertEqual(saved[0].dtype, torch.int32)
            self.assertIs(saved[5], indices)
            indices.add_(0)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                y.sum().backward()
        x, options = case("max_unpool2d")
        y = D.do_max_unpool2d(x, **options, plan=policy())
        self.assertEqual(y.grad_fn.saved_tensors[0].dtype, torch.int32)
        options["indices"].add_(0)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            y.sum().backward()

    def test_average_state_elimination_and_inference_for_every_family(self):
        for kind, spec in SPECS.items():
            x, options = case(kind)
            info = {}
            y = getattr(D, "do_" + kind)(x, **options, plan=policy(), _execution_info=info)
            self.assertEqual(info["saved_raw_input_bytes"], 0)
            self.assertTrue(info["compression_active"])
            if "average" in spec.family:
                self.assertTrue(all(t is None for t in y.grad_fn.saved_tensors))
                self.assertEqual(info["saved_state_bytes"], 0)
                self.assertEqual(info["retained_storage_bytes"], 0)
            with torch.no_grad():
                getattr(D, "do_" + kind)(x, **options, plan=policy(), _execution_info=info)
            self.assertEqual(info["saved_state_bytes"], 0)
            self.assertFalse(info["compression_active"])

    def test_module_conversion_shared_identity_scope_and_policy(self):
        for native, spec in MODULE_SPECS.items():
            _, options = case(spec.name, dtype=torch.bfloat16)
            constructor = {k: v for k, v in options.items() if k not in ("indices", "output_size") or spec.family != "unpool"}
            module = native(**constructor)
            model, _ = prepare_fp8_model(nn.Sequential(module, module), False, activation_compression=True)
            self.assertIs(model[0], model[1])
            self.assertEqual(model[0].kind, spec.name)
            x, opts = case(spec.name, dtype=torch.bfloat16)
            if spec.family == "unpool":
                y = model[0](x, opts["indices"], opts["output_size"])
            else:
                y = model[0](x)
            y.sum().backward()
            self.assertTrue(model[0].execution_info["compression_active"])
            with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=True):
                scoped = spec.native(x, **opts)
            self.assertEqual(scoped.shape, y.shape)

    def test_planned_pool_and_global_flatten_linear_edges(self):
        for kind in (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d):
            source = nn.Sequential(nn.Conv2d(3, 4, 1), kind(1), nn.Flatten(1), nn.Linear(4, 2))
            cfg = deepopt.FP8Config(enabled=True, backward_precision="auto")
            model, _ = prepare_fp8_model(source, cfg, activation_compression=True)
            result = model(torch.randn(2, 3, 4, 4))
            result.sum().backward()
            report = precision_report(model)
            self.assertFalse(report["fallbacks"], report)
            self.assertEqual(report["analyses"], 1)
            self.assertEqual(model[1].execution_info["output"], "fp8")
            model(torch.randn(2, 3, 4, 4)).sum().backward()
            self.assertEqual(precision_report(model)["analyses"], 1)

    def test_fake_meta_and_eager_compile_boundaries(self):
        for kind in SPECS:
            with FakeTensorMode():
                x, options = case(kind)
                y = getattr(D, "do_" + kind)(x, **options, plan=policy())
                y.sum().backward()
        x = torch.randn(2, 3, 5, 5, requires_grad=True)
        function = torch.compile(lambda a: D.do_adaptive_avg_pool2d(a, (2, 2), plan=policy()), backend="eager")
        torch.testing.assert_close(function(x), D.do_adaptive_avg_pool2d(x, (2, 2), plan=policy()))

    def test_native_cpu_amp_output_dtypes_for_every_family(self):
        for kind, spec in SPECS.items():
            x, options = case(kind, dtype=torch.bfloat16)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected = spec.native(x, **options)
                actual = getattr(D, "do_" + kind)(x, **options, plan=policy())
            self.assertEqual(actual.dtype, expected.dtype, kind)
            actual.sum().backward()

    def test_unpool_duplicate_destinations_and_invalid_bounds(self):
        x = torch.tensor([[[1., 4.]]], requires_grad=True)
        indices = torch.tensor([[[1, 1]]])
        y = D.do_max_unpool1d(x, indices, 2, plan=policy())
        self.assertIn(y[0, 0, 1].item(), (1., 4.))
        self.assertNotEqual(y[0, 0, 1].item(), 5.)
        y.backward(torch.arange(4).float().reshape(1, 1, 4))
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        with self.assertRaisesRegex(RuntimeError, "outside output bounds"):
            D.do_max_unpool1d(x, torch.tensor([[[4, 0]]]), 2, plan=policy())


class PackedIndexTests(unittest.TestCase):
    def test_every_width_strides_tails_empty_and_metadata(self):
        for bits in (0, 1, 2, 3, 5, 7, 9, 16, 31, 32, 40, 63):
            count = 1 << bits
            values = torch.randint(min(count, 2**62), (5, 74))[:, ::2]
            values[-1, -1] = count - 1
            packed = pack_indices(values, count)
            self.assertEqual(packed.bits, bits)
            torch.testing.assert_close(unpack_indices(packed), values, rtol=0, atol=0)
            self.assertFalse(packed.data.untyped_storage()._cdata == values.untyped_storage()._cdata)
            torch.testing.assert_close(unpack_indices(pack_indices(torch.empty(0, 3, dtype=torch.int64), count)), torch.empty(0, 3, dtype=torch.int64))
        with self.assertRaises(ValueError):
            pack_indices(torch.tensor([-1]), 4)
        with self.assertRaises(ValueError):
            pack_indices(torch.tensor([4]), 4)
        values = torch.tensor([0, 2**31 - 1], dtype=torch.int32)
        torch.testing.assert_close(unpack_indices(pack_indices(values, 2**32)), values.long(), rtol=0, atol=0)
        tiny = pack_indices(torch.tensor([7]), 8)
        self.assertEqual(tiny.data.numel(), 1)


if __name__ == "__main__":
    unittest.main()
