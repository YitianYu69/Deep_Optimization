"""Admission, cancellation, aliases and fresh-process benchmark integration."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from tests.benchmarks.test_do_analysis_benchmark import cli, ROOT
from analysis.scheduler import Broker, Device, GiB, effective_task, run_command, profile_options,discover_devices,ensure_lease,demand_key


class SchedulerTests(unittest.TestCase):
    def test_device_visibility_order_and_standalone_logical_rebasing(self):
        rows=[["0","GPU-zero","test","81920"],["1","GPU-one","test","81920"]]
        with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":"1,0"}),patch("analysis.scheduler.smi",return_value=rows):
            device,=discover_devices(None,"cuda:0")
            self.assertEqual(device.uuid,"GPU-one")
        with patch("analysis.scheduler.inherited_lease",return_value=False),\
             patch("analysis.scheduler.discover_devices",return_value=[Device("GPU-one","1",80*GiB)]),\
             patch("analysis.scheduler.run_external",return_value=0) as launch:
            self.assertEqual(ensure_lease(["--device","cuda:1","--output","test.json"],"program.py","cuda:1"),0)
            self.assertEqual(launch.call_args.args[0][-4:],["--device","cuda:0","--output","test.json"])

    def test_static_phase_complete_memory_calibration_does_not_alias_unknown_factories(self):
        config=ROOT/"benchmarks/analysis/configs/hopper_v6_models.json"
        plan=cli.make_plan(cli.parser().parse_args(["--config",str(config)]))
        task=plan["tasks"][0]
        first={**plan["options"],"warmup":1,"iterations":1,"repeats":1,"phase_steps":1}
        second={**first,"warmup":10,"iterations":30,"repeats":5,"phase_steps":5}
        self.assertEqual(demand_key(task,first,"build","code"),demand_key(task,second,"build","code"))
        task=deepcopy(task);task["workload"]["factory"]="custom:factory"
        self.assertNotEqual(demand_key(task,first,"build","code"),demand_key(task,second,"build","code"))

    def test_suite_dry_run_is_gpu_free_and_preserves_model_coverage(self):
        from analysis.benchmark_suite import main
        original=__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ("torch","deepopt","triton"):raise AssertionError("GPU-free planning imported "+name)
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch("builtins.__import__",new=guarded):
            self.assertEqual(main([temp,"--recovery-v6","--dry-run"]),0)
            plan=json.loads((Path(temp)/"suite-plan.json").read_text())
            self.assertEqual(len(plan["stages"]),29)
        config=ROOT/"benchmarks/analysis/configs/hopper_v6_models.json"
        plan=cli.make_plan(cli.parser().parse_args(["--config",str(config),"--sweep","cartesian"]))
        self.assertEqual(len(plan["tasks"]),50)
        for case in cli.CASES:
            self.assertEqual(sum(t["case"]==case for t in plan["tasks"]),10)

    def test_preflight_blocks_only_the_matching_model_shape_mode_and_execution(self):
        from analysis.preflight import blocking_reason,model_failure_rules
        config=ROOT/"benchmarks/analysis/configs/hopper_v6_models.json"
        tasks=cli.make_plan(cli.parser().parse_args(["--config",str(config),"--sweep","cartesian"]))["tasks"]
        task=next(t for t in tasks if t["workload"]["name"]=="efficientnet_b0_b512" and t["case"]=="fp8"
                  and t["settings"]["runtime"]["execution"]=="graph")
        rules=model_failure_rules([{"task":task,"status":"error","reason":"compiler failure"}],full_size=True)
        self.assertIn("compiler failure",blocking_reason(task,{"blocked":rules}))
        self.assertEqual(sum(blocking_reason(t,{"blocked":rules}) is not None for t in tasks),1)

    def test_attention_sweep_records_failures_and_completes_the_other_cases(self):
        import importlib.util
        import torch
        from analysis.reporting import write_json
        from deepopt.opt.backends._shared.activation_codec import strided_indexing
        from deepopt.opt.quantization.activation import _quantize_pack_strided_activation
        path=ROOT/"benchmarks/activation_compression/benchmark_attention_pack.py"
        spec=importlib.util.spec_from_file_location("v6_attention",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        h=SimpleNamespace(__file__=__file__,write_checkpoint=write_json,timings=lambda *_:{"eager":{"wall_ms_median":1}})
        helper=SimpleNamespace(helpers=lambda:h,resume_identity=lambda *_,**kwargs:{},cached_results=lambda *_:{})
        def quantize(*args,**kwargs):
            if strided_indexing()=="bounded":raise RuntimeError("simulated compiler failure")
            return _quantize_pack_strided_activation(*args,**kwargs)
        randn=torch.randn
        def cpu_randn(*args,**kwargs):
            kwargs.pop("device",None);return randn(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch.dict("sys.modules",{"benchmark_activation_pack":helper}),\
             patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.set_device"),\
             patch("torch.cuda.get_device_properties",return_value="test"),patch("torch.randn",side_effect=cpu_randn),\
             patch("deepopt.opt.quantization.activation._quantize_pack_strided_activation",side_effect=quantize):
            output=Path(temp)/"attention.json"
            code=module.main(["--output",str(output),"--batch-size","2","--heads","3","--sequence-length","53",
                "--head-dim","7","--bits","4","--group-sizes","1024","--num-warps","4","--indexing","wide","bounded"])
            report=json.loads(output.read_text())
            self.assertEqual(code,1)
            self.assertEqual(report["manifest"],{"expected":48,"completed":48,"ok":33,"failed":15,"missing":0})
            self.assertEqual({row["source_layout"] for row in report["results"]}, {"bhsd", "bshd", "interleaved"})
            self.assertEqual({row["implementation"] for row in report["results"] if row["source"] == "qkv"},
                             {"separate", "joint"})

    def test_atomic_memory_admission_and_exclusion_across_brokers(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"broker.sqlite"
            a,b=Broker(path),Broker(path)
            device=Device("GPU-test","0",80*GiB)
            with ThreadPoolExecutor(max_workers=8) as executor:
                tokens=list(executor.map(lambda _:Broker(path).acquire(device,30*GiB,False,4,4*GiB,free=80*GiB),range(8)))
            held=[token for token in tokens if token]
            self.assertEqual(len(held),2)
            self.assertIsNone(b.acquire(device,GiB,True,4,4*GiB,free=80*GiB))
            for token in held:a.release(token)
            token=b.acquire(device,None,True,4,4*GiB,free=80*GiB)
            self.assertIsNotNone(token)
            self.assertIsNone(a.acquire(device,GiB,False,4,4*GiB,free=80*GiB))
            b.release(token)
            self.assertIsNone(a.acquire(device,GiB,True,4,0,free=80*GiB,foreign=True))

    def test_dead_coordinator_with_live_child_keeps_lease(self):
        with tempfile.TemporaryDirectory() as temp:
            broker=Broker(Path(temp)/"broker.sqlite")
            device=Device("GPU-test","0",80*GiB)
            token=broker.acquire(device,GiB,True,1,0)
            broker.attach(token,os.getpid())
            with broker.connect() as db:db.execute("UPDATE leases SET owner=?",(999999999,))
            self.assertIsNone(broker.acquire(device,GiB,True,1,0))
            with broker.connect() as db:db.execute("UPDATE leases SET child=?",(999999999,))
            self.assertIsNotNone(broker.acquire(device,GiB,True,1,0))

    def test_timeout_waits_for_owned_child_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            pid=[]
            with open(os.devnull,"w") as log,self.assertRaises(TimeoutError):
                run_command([sys.executable,"-c","import time;time.sleep(30)"],env=os.environ.copy(),log=log,
                    timeout=.1,cancel_file=Path(temp)/"cancel",on_spawn=pid.append)
            with self.assertRaises(ProcessLookupError):os.kill(pid[0],0)

    def test_dedup_keeps_modes_active_settings_and_runtime_policies(self):
        plan=cli.make_plan(cli.parser().parse_args(["--device","cpu","--ops","linear","--small"]))
        native=next(t for t in plan["tasks"] if t["case"]=="torch_bf16")
        other=deepcopy(native);other["settings"]["runtime"]["producer_pack"]="layer_norm"
        self.assertEqual(effective_task(native),effective_task(other))
        other["settings"]["runtime"]["timing_fill_uninitialized_memory"]=True
        self.assertNotEqual(effective_task(native),effective_task(other))
        compressed=next(t for t in plan["tasks"] if t["case"]=="compression")
        other=deepcopy(compressed);other["settings"]["compression"]["bits"]=4
        self.assertNotEqual(effective_task(compressed),effective_task(other))
        other=deepcopy(native);other["case"]="do_bf16"
        self.assertNotEqual(effective_task(native),effective_task(other))

    def test_selected_profiles_are_disabled_outside_selection(self):
        task={"case":"fp8","workload":{"name":"other"}}
        selected=profile_options(task,{"profile_steps":3,"profile_replay_steps":1,
            "trace_workloads":["vit_*"],"trace_cases":["fp8_compression"]},"selected")
        self.assertEqual((selected["profile_steps"],selected["profile_replay_steps"]),(0,0))

    def test_local_workers_resume_all_five_cases_and_canonical_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            config=root/"config.json"
            config.write_text(json.dumps({"grid":{"compression.bits":[2,4]}}))
            command=[sys.executable,str(ROOT/"benchmarks/analysis/benchmark_do_ops.py"),"--device","cpu",
                "--backend","reference","--ops","linear","--small","--config",str(config),"--scheduler","local",
                "--cpu-workers","2","--warmup","0","--iterations","1","--repeats","1","--phase-steps","0",
                "--no-memory","--deduplicate-effective-settings","--resume","--output",str(root/"results")]
            env={**os.environ,"PYTHONPATH":str(ROOT/"src")}
            first=None
            for attempt in range(2):
                result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                report=json.loads((root/"results/summary.json").read_text())
                self.assertEqual(report["counts"],{"ok":10,"skipped":0,"error":0})
                self.assertEqual({r["task"]["case"] for r in report["results"]},set(cli.CASES))
                aliases=[r for r in report["results"] if r.get("canonical_task_id")]
                self.assertEqual(len(aliases),3)
                stamps={r["task"]["id"]:Path(r["snapshot"]).stat().st_mtime_ns for r in report["results"]}
                if attempt:self.assertEqual(stamps,first)
                else:first=stamps


if __name__=="__main__":unittest.main()
