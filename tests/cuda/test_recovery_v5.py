"""Real CUDA compilation, consumer packing and random-init CNN acceptance."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import activation_plan, normalization_plan
from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
from deepopt.opt.quantization.activation import quantize_pack_activation


CUDA = torch.cuda.is_available() and torch.version.hip is None


@unittest.skipUnless(CUDA, "NVIDIA CUDA is required")
class RecoveryV5CudaTests(unittest.TestCase):
    def test_precise_dw_and_preparation_follow_the_forward_owner(self):
        if torch.cuda.get_device_capability()[0]<9:self.skipTest("Native FP8 needs SM90+")
        from deepopt.opt.policy import linear_plan
        from deepopt.opt.quantization.activation_tensor import to_fp8_activation
        from deepopt.opt.backends._shared.gemm_policy import linear_backward_scope
        from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
        x=torch.randn(128,64,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        weight=torch.randn(96,64,device="cuda",requires_grad=True)
        plan=linear_plan(fp8={"enabled":True,"backend":"triton","num_stages":3,"backward_backend":"native_fast"})
        with record_gemms() as records:
            with linear_backward_scope(dw_backend="native_precise",native_layout="prepare"):
                out=doF.linear(to_fp8_activation(x,layout="last"),weight,plan=plan)
            out.float().square().mean().backward()
        dw,=[r for r in records if r["role"]=="dw"]
        self.assertEqual(dw["effective_backend"],"native_precise")
        self.assertEqual(dw["preparation"]["kind"],"row_major_copy")
        self.assertTrue(torch.isfinite(weight.grad).all())

    def policy(self,factory,compressed=True):
        return factory(fp8={"enabled":True,"backend":"triton"},
                       activation_compression={"enabled":compressed,"bits":2,"group_size":32,"norm_xhat_recenter":True})

    def test_gelu_and_norm_output_pack_matches_separate_packing_and_replays_changed_inputs(self):
        for kind in ("gelu","layer_norm"):
            x=torch.randn(7,33,device="cuda",dtype=torch.bfloat16,requires_grad=True)
            def call():
                with deepopt.autocast("cuda",enabled=False,check_finite=False), activation_codec_scope(producer_pack="all"):
                    return (doF.gelu(x,plan=self.policy(activation_plan),output_dtype=torch.float8_e4m3fn) if kind=="gelu" else
                            doF.layer_norm(x,(33,),plan=self.policy(normalization_plan),output_dtype=torch.float8_e4m3fn))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                out=call()
            for _ in range(2):
                with torch.no_grad():x.copy_(torch.randn_like(x))
                graph.replay()
                packed=out.linear_packed(group_size=32,block_shape=None,bits=2)
                self.assertIsNotNone(packed)
                expected=quantize_pack_activation(out._data,32,bits=2)
                for a,b in [(packed.data,expected.data),(packed.step,expected.step),(packed.minimum,expected.minimum)]:
                    torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_gelu_recomputation_matches_the_existing_cuda_derivative(self):
        raw=torch.randn(17,257,device="cuda",dtype=torch.bfloat16)
        dy=torch.randn_like(raw)
        grads=[]
        for state in ("derivative","recompute"):
            x=raw.detach().clone().requires_grad_()
            with activation_codec_scope(gelu_state=state),deepopt.autocast("cuda",enabled=False,check_finite=False):
                y=doF.gelu(x,plan=self.policy(activation_plan,False),output_dtype=torch.float8_e4m3fn)
            y.backward(dy)
            grads.append(x.grad)
        torch.testing.assert_close(grads[0],grads[1],rtol=0,atol=0)

    @unittest.skipUnless(importlib.util.find_spec("torchvision"),"torchvision is required")
    def test_random_init_cnn_models_execute_all_five_feature_cases(self):
        from tests.benchmarks.test_do_analysis_benchmark import cli
        from analysis.engine import run_task
        import json
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name in ("resnet18","efficientnet_b0"):
                config={"settings":{"fp8":{"backend":"triton","num_stages":3,"backward_backend":"native_precise"},
                        "runtime":{"binding":"prepared","scope_lifetime":"reused","execution":"graph",
                                   "timing_fill_uninitialized_memory":False,"optimizer":"adamw","lr":1e-4}},
                        "workloads":[{"name":name,"op":"model","shape":[2,3,64,64],"input_requires_grad":False,
                        "factory":"analysis.example_model:build_torchvision_cnn","factory_kwargs":{"model_name":name,
                        "batch_size":2,"image_size":64,"classes":5,"ema_decay":None}}]}
                path=directory/(name+".json");path.write_text(json.dumps(config))
                args=cli.parser().parse_args(["--config",str(path),"--warmup","1","--iterations","1","--repeats","1",
                    "--phase-steps","0","--no-memory","--deterministic","--fill-uninitialized-memory",
                    "--max-numerical-elements","1024","--require-fp8-plan"])
                plan=cli.make_plan(args)
                hashes=set()
                for task in plan["tasks"]:
                    target=directory/name/task["id"];target.mkdir(parents=True)
                    result=run_task(task,plan["options"],target)
                    self.assertTrue(result["finite_after_timing"])
                    self.assertTrue(result["numerical_snapshot"]["full_tensor_finite"])
                    hashes.add((result["initial_state_sha256"],result["rng_baseline_sha256"]))
                self.assertEqual(len(hashes),1)


if __name__ == "__main__":
    unittest.main()
