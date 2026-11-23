"""Measured dispatch, exact saved-state variants and per-forward producer ownership."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends._shared.activation_codec import activation_codec_scope, spatial_row_alignment, select_pack_warps
from deepopt.opt.backends._shared.activation_codec import narrow_source_indices
from deepopt.opt.backends._shared.normalization import measured_layer_variant
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.convolution.geometry import ConvGeometry
from deepopt.opt.policy import activation_plan, normalization_plan, linear_plan
from deepopt.opt.quantization.activation import quantize_pack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation


def policy(factory, compressed=True):
    return factory(fp8={"enabled": True, "backend": "reference"},
                   activation_compression={"enabled": compressed, "group_size": 32, "bits": 4})


class RecoveryV5Tests(unittest.TestCase):
    def test_diagnostic_module_hooks_do_not_mix_concurrent_forward_contexts(self):
        import threading
        from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
        entered,ready,done=threading.Event(),threading.Event(),threading.Event()
        failures=[]
        class ConcurrentLinear(torch.nn.Module):
            def forward(self,x):
                if threading.current_thread() is threading.main_thread():
                    entered.set()
                    if not ready.wait(5):raise RuntimeError("worker did not enter")
                else:
                    ready.set()
                    if not done.wait(5):raise RuntimeError("main did not return")
                return x
        model=ConcurrentLinear()
        def worker():
            try:
                if not entered.wait(5):raise RuntimeError("main did not enter")
                model(torch.ones(1))
            except Exception as error:failures.append(error)
        with record_gemms(model):
            thread=threading.Thread(target=worker);thread.start()
            try:
                model(torch.ones(1))
            finally:
                done.set();thread.join(6)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures,[])

    def test_producer_pack_policy_is_owned_by_checkpoint_recomputation(self):
        from deepopt.opt.scaling.delayed import fp8_checkpoint
        torch.manual_seed(765)
        raw=torch.randn(3,17,dtype=torch.bfloat16)
        weight=torch.randn(8,17)
        observed=[]
        def forward(x,w):
            y=doF.gelu(x,plan=policy(activation_plan),output_dtype=torch.float8_e4m3fn)
            observed.append(y.linear_packed(group_size=32,block_shape=None,bits=4) is not None)
            return doF.linear(y,w,plan=policy(linear_plan))
        results=[]
        for checkpoint in (False,True):
            x,w=raw.clone().requires_grad_(),weight.clone().requires_grad_()
            with activation_codec_scope(producer_pack="all"):
                out=fp8_checkpoint(forward,x,w) if checkpoint else forward(x,w)
            out.float().square().mean().backward()
            results.append((out.detach(),x.grad,w.grad))
        self.assertTrue(all(observed))
        for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_native_preparation_preserves_metadata_and_role_scope_restores(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from deepopt.opt.backends._shared.gemm_policy import prepare_native_operands,linear_backward_scope,linear_backward_policy
        with linear_backward_scope(dw_backend="native_precise",native_layout="prepare"):
            self.assertEqual(linear_backward_policy(),("native_precise","prepare"))
        self.assertEqual(linear_backward_policy(),("inherit","preserve"))
        with FakeTensorMode():
            a=QuantizedTensor(torch.empty(96,128,device="cuda",dtype=torch.float8_e4m3fn),torch.ones(1,1,device="cuda"))
            b=QuantizedTensor(torch.empty(128,64,device="cuda",dtype=torch.float8_e4m3fn).T,torch.ones(1,1,device="cuda"))
            pa,pb,info=prepare_native_operands(a,b)
            self.assertEqual(info["kind"],"row_major_copy")
            self.assertIs(pa.scale_inv,a.scale_inv)
            self.assertIs(pb.scale_inv,b.scale_inv)
            self.assertEqual(pb.data.stride(),(128,1))

    def test_bounded_source_indices_retain_wide_indexing_for_large_views(self):
        shape=(512,12,197,64); strides=(453888,64,2304,1)
        self.assertFalse(narrow_source_indices(shape,strides))
        with activation_codec_scope(strided_index="bounded"):
            self.assertTrue(narrow_source_indices(shape,strides))
            self.assertFalse(narrow_source_indices((2**31,1),(1,1)))
            self.assertFalse(narrow_source_indices((2,2),(2**31,1)))

    def test_fp32_norm_rollout_uses_realized_groups_and_preserves_guards(self):
        dev = torch.device("cuda:0")
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=True):
            for packed, recenter in [(None, False), (SimpleNamespace(bits=2, group_size=768), True)]:
                self.assertEqual(measured_layer_variant(100864, 768, torch.float32, packed, recenter, dev), "layer_tiled")
            self.assertIsNone(measured_layer_variant(100864, 768, torch.float32, SimpleNamespace(bits=2, group_size=1024), True, dev))
            self.assertIsNone(measured_layer_variant(100864, 768, torch.float32, None, True, dev))
            self.assertIsNone(measured_layer_variant(6304, 768, torch.float32, None, False, dev))

    def test_spatial_auto_is_limited_to_the_measured_recipe_and_explicit_override(self):
        g = ConvGeometry((512, 3, 224, 224), (768, 3, 16, 16), (16, 16), (0, 0), (1, 1), 1)
        comp = {"enabled": True, "bits": 2, "group_size": 1024}
        args = (g, torch.float8_e4m3fn, None, comp, torch.device("cuda:0"))
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=True):
            self.assertEqual(spatial_row_alignment(*args), 1)
            self.assertEqual(spatial_row_alignment(*args, delayed=True), 16)
            with activation_codec_scope(spatial_layout="aligned"):
                self.assertEqual(spatial_row_alignment(*args), 16)
            with activation_codec_scope(spatial_layout="compact"):
                self.assertEqual(spatial_row_alignment(*args), 1)
            self.assertEqual(spatial_row_alignment(g, torch.float8_e4m3fn, None, {**comp, "bits": 4}, args[-1]), 16)

    def test_pack_warp_auto_keeps_explicit_and_unmeasured_policies(self):
        args = ((100864, 3072), torch.bfloat16, 2, 512, None, torch.device("cuda:0"))
        with patch("deepopt.opt.backends._shared.normalization.measured_norm_device", return_value=True):
            self.assertEqual(select_pack_warps("auto", *args, contiguous=True), 1)
            self.assertEqual(select_pack_warps(4, *args, contiguous=True), 4)
            self.assertEqual(select_pack_warps("auto", *args, contiguous=True, strided=True), 4)
            self.assertEqual(select_pack_warps("auto", args[0], args[1], 2, 1024, None, args[-1], contiguous=True), 2)
        with self.assertRaises(ValueError):
            select_pack_warps(True, *args, contiguous=True)

    def test_gelu_recompute_saves_bf16_input_and_preserves_outputs_and_gradients(self):
        torch.manual_seed(643)
        for approximate in ("none", "tanh"):
            raw = torch.randn(3, 37, dtype=torch.bfloat16)
            upstream = torch.randn_like(raw)
            results, sizes = [], []
            for mode in ("derivative", "recompute"):
                x = raw.clone().requires_grad_()
                saved = []
                with activation_codec_scope(gelu_state=mode), torch.autograd.graph.saved_tensors_hooks(
                        lambda t: saved.append(t) or t, lambda t: t):
                    y = doF.gelu(x, approximate=approximate, plan=policy(activation_plan, False))
                y.backward(upstream)
                results.append((y.detach(), x.grad))
                sizes.append(sum(t.numel()*t.element_size() for t in saved))
                self.assertEqual(saved[0].dtype, torch.bfloat16 if mode == "recompute" else torch.float32)
            self.assertEqual(sizes[0], sizes[1]*2)
            for a,b in zip(*results):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_producer_payload_roundtrip_and_linear_reuses_without_repacking(self):
        for kind in ("gelu", "layer_norm"):
            torch.manual_seed(61)
            x = torch.randn(3,17,dtype=torch.bfloat16,requires_grad=True)
            with activation_codec_scope(producer_pack="all"):
                if kind == "gelu":
                    y = doF.gelu(x,plan=policy(activation_plan),output_dtype=torch.float8_e4m3fn)
                else:
                    y = doF.layer_norm(x,(17,),plan=policy(normalization_plan),output_dtype=torch.float8_e4m3fn)
            self.assertIsInstance(y,FP8Activation)
            packed = y.linear_packed(group_size=32,block_shape=None,bits=4)
            self.assertIsNotNone(packed)
            expected = quantize_pack_activation(y._data,32,bits=4)
            for a,b in zip((packed.data,packed.step,packed.minimum),(expected.data,expected.step,expected.minimum)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            fields,meta = y.__tensor_flatten__()
            restored = FP8Activation.__tensor_unflatten__({f:getattr(y,f) for f in fields},meta,None,None)
            self.assertIsNotNone(restored.linear_packed(group_size=32,block_shape=None,bits=4))
            self.assertIsNone(y.linear_packed(group_size=64,block_shape=None,bits=4))
            w = torch.randn(8,17,requires_grad=True)
            with patch("deepopt.opt.quantization.activation.quantize_pack_activation", side_effect=AssertionError("repacked")):
                z = doF.linear(y,w,plan=policy(linear_plan))
            z.float().sum().backward()
            self.assertTrue(torch.isfinite(w.grad).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            with torch.no_grad():
                packed.minimum.add_(1)
            self.assertIsNone(y.linear_packed(group_size=32,block_shape=None,bits=4))
            with torch.no_grad():
                y._data.copy_(torch.zeros_like(y._data))
            self.assertIsNone(y.linear_packed(group_size=32,block_shape=None,bits=4))


if __name__ == "__main__":
    unittest.main()
