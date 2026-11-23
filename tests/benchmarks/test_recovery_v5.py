"""Result persistence, real cache hits and download-free CNN constructor contracts."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tests.benchmarks import test_do_analysis_benchmark as support
from analysis.engine import run_task
from analysis.engine import Session
from analysis.measurement import snapshot
from analysis.artifacts import snapshot_valid, profiles_valid, profile_detail
from analysis.example_model import build_torchvision_cnn


class RecoveryV5BenchmarkTests(unittest.TestCase):
    def test_stochastic_model_rng_baseline_survives_preparation_consuming_rng(self):
        from deepopt.opt.policy import prepare_fp8_model
        def prepare(*args,**kwargs):
            result=prepare_fp8_model(*args,**kwargs)
            torch.rand(39)
            return result
        values=[]
        for case in ("torch_bf16","do_bf16"):
            task,options=support.AnalysisExecutionTests().task(case=case,update={"runtime":{"binding":"prepared"}})
            task["workload"].update(op="model",factory="unused:factory")
            def source(*_):
                return {"model":torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(4,3)),
                        "inputs":(torch.randn(2,4),),"kwargs":{}}
            with patch("analysis.engine.build_source",side_effect=source),patch("deepopt.opt.policy.prepare_fp8_model",side_effect=prepare):
                session=Session(task,options)
            try:
                values.append((session.rng_fingerprint,snapshot(session)["output.0"]))
            finally:
                session.close()
        self.assertEqual(values[0][0],values[1][0])
        torch.testing.assert_close(values[0][1],values[1][1],rtol=0,atol=0)

    def test_cnn_constructors_explicitly_disable_weights_and_preserve_training_state(self):
        def constructor(**kwargs):
            self.assertIsNone(kwargs["weights"])
            return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*8*8,kwargs["num_classes"]))
        resnet,efficient = Mock(side_effect=constructor),Mock(side_effect=constructor)
        fake = SimpleNamespace(models=SimpleNamespace(resnet18=resnet,efficientnet_b0=efficient))
        with patch.dict("sys.modules", {"torchvision":fake}):
            for name,spy in [("resnet18",resnet),("efficientnet_b0",efficient)]:
                torch.manual_seed(24)
                package = build_torchvision_cnn(torch.device("cpu"),name,batch_size=2,image_size=8,classes=3,ema_decay=None)
                spy.assert_called_once_with(weights=None,num_classes=3)
                model = package["model"]
                self.assertTrue(model.training)
                output = model(*package["inputs"])
                package["loss_fn"](output).backward()
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
        with self.assertRaises(ValueError):
            build_torchvision_cnn(torch.device("cpu"),"unsupported")

    def test_optional_trace_failure_preserves_snapshot_and_can_resume_without_retiming(self):
        task,options = support.AnalysisExecutionTests().task(case="do_bf16")
        options.update(profile_steps=1,profile_detail="trace",memory=False)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch("torch.profiler.profile.export_chrome_trace",side_effect=OSError("quota exceeded")):
                result = run_task(task,options,root)
            self.assertEqual(result["status"],"ok")
            self.assertTrue(result["artifact_errors"])
            self.assertTrue(snapshot_valid(result))
            self.assertTrue((root/"core-result.json").is_file())
            self.assertFalse(profiles_valid(result,task,options))
            stamp=(root/"snapshot.pt").stat().st_mtime_ns
            with patch("analysis.engine.measure",side_effect=AssertionError("retimed")):
                resumed=run_task(task,{**options,"_resume_core":str(root/"core-result.json")},root)
            self.assertEqual(resumed["timing"],result["timing"])
            self.assertTrue(resumed["diagnostics_resumed"])
            self.assertTrue(profiles_valid(resumed,task,options))
            self.assertEqual((root/"snapshot.pt").stat().st_mtime_ns,stamp)
            with (root/"snapshot.pt").open("ab") as f:
                f.write(b"corrupt")
            self.assertFalse(snapshot_valid(resumed))

    def test_summary_profiles_and_selected_trace_cases_have_distinct_resume_requirements(self):
        task,options = support.AnalysisExecutionTests().task(case="do_bf16")
        options.update(profile_steps=1,profile_detail="stacks",trace_cases=["fp8_compression"])
        self.assertEqual(profile_detail(task,options),"summary")
        row={"profile":{"steps":1,"detail":"summary","trace":None}}
        self.assertTrue(profiles_valid(row,task,options))
        options["trace_cases"]=["do_bf16"]
        self.assertFalse(profiles_valid(row,task,options))

    def test_codec_cached_results_reuse_only_complete_matching_rows(self):
        path=Path(__file__).resolve().parents[2]/"benchmarks/activation_compression/benchmark_activation_pack.py"
        spec=importlib.util.spec_from_file_location("v5_pack_cache",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        helper=module.helpers()
        args=SimpleNamespace(resume=True,repeats=2,execution="both")
        timing={"wall_ms_median":1.,"wall_ms_samples":[1.,1.]}
        complete={"case_id":"done","exact":True,"timing":{"eager":timing,"graph":timing}}
        incomplete={"case_id":"missing","exact":True,"timing":{"eager":timing}}
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"result.json"
            helper.write_checkpoint(target,{"resume_key":"hash","identity":{"build":"source"},"results":[complete,incomplete]})
            self.assertEqual(set(module.cached_results(target,"hash",args,helper)),{"done"})
            self.assertFalse(module.cached_results(target,"different",args,helper))


if __name__ == "__main__":
    unittest.main()
