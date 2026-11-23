"""V6 suite coordinator: preflight, parallel stages and shared physical GPU admission."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import time
import uuid

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from analysis.reporting import write_json
from analysis.scheduler import discover_devices,default_broker_path,output_lock,run_command,run_external
from analysis.preflight import model_failure_rules
from analysis.example_model import DEFAULT_CIFAR10_ROOT


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",nargs="?",type=Path,default=Path("full-comparison-results-v6"))
    p.add_argument("--recovery-v6",action="store_true")
    p.add_argument("--scheduler",choices=("auto","serial","local","dask"),default="auto")
    p.add_argument("--devices",default="auto")
    p.add_argument("--cpu-workers",type=int,default=4)
    p.add_argument("--gpu-concurrency",default="auto")
    p.add_argument("--timing-isolation",choices=("exclusive","shared"),default="exclusive")
    p.add_argument("--memory-headroom-gib",type=float,default=4)
    p.add_argument("--deduplicate-effective-settings",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--profile-policy",choices=("all","selected"),default="selected")
    p.add_argument("--profile-steps",type=int,default=int(os.environ.get("BENCHMARK_PROFILE_STEPS","3")))
    p.add_argument("--trace-directory",type=Path,default=os.environ.get("BENCHMARK_TRACE_DIRECTORY"))
    p.add_argument("--max-numerical-elements",type=int,default=int(os.environ.get("BENCHMARK_MAX_NUMERICAL_ELEMENTS","16384")))
    p.add_argument("--lease-db",type=Path,default=default_broker_path())
    p.add_argument("--stage-timeout",type=float,default=21600)
    p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--dry-run",action="store_true",help="Write stage descriptors without Torch, CUDA, or nvidia-smi")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--full-coverage",action="store_true",help="Also run every operator/gradient request and declared setting axis")
    p.add_argument("--cifar10-root",type=Path,nargs="?",const=Path(DEFAULT_CIFAR10_ROOT),
                   default=os.environ.get("CIFAR10_ROOT") or None,
                   help=f"Enable real eager CIFAR10 epochs; optional path defaults to {DEFAULT_CIFAR10_ROOT}. "
                        "CIFAR10_ROOT can also supply the path; never download")
    return p


def stages(args,root):
    config_root=root/"configs" if (root/"configs").is_dir() else root
    repo=root.parent if root.name=="analysis" else root
    def script(name):
        direct=root/name
        if direct.is_file():return str(direct)
        for folder in (repo/"activation_compression",repo/"fp8",repo/"normalization"):
            if (folder/name).is_file():return str(folder/name)
        return str(direct)
    matrix=[]
    for name,sweep in (("hopper_v6_models","cartesian"),("hopper_v6_vit_hot","one-at-a-time"),
                       ("hopper_v6_cnn_hot","one-at-a-time"),("hopper_v6_resnet_bridge","cartesian"),
                       ("hopper_v6_precision_2","cartesian"),("hopper_v6_precision_4","cartesian"),
                       ("hopper_v6_precision_8","cartesian")):
        matrix.append({"name":name,"kind":"matrix","config":str(config_root/(name+".json")),"sweep":sweep})
    if args.full_coverage:
        for name,sweep in (("all_operators","one-at-a-time"),("all_settings","one-at-a-time"),("precision_boundaries","cartesian")):
            matrix.append({"name":name,"kind":"matrix","config":str(config_root/(name+".json")),"sweep":sweep,
                           "extra":["--gradient-modes","all","dx","dw","db","none"] if name=="all_operators" else []})
    external=[
        ("attention-pack","benchmark_attention_pack.py",["--bits","4","8","--group-sizes","1024",
            "--num-warps","1","2","4","--indexing","wide","bounded"],"json"),
        ("convolution-codec","benchmark_convolution_codec.py",["--group-sizes","1024"],"json"),
        ("spatial-layout","benchmark_spatial_layout.py",["--channels","1","3","4","16"],"json"),
        ("activation-pack","benchmark_activation_pack.py",["--bits","2","--group-sizes","1024",
            "--variants","batched","serial","--groups-per-program","2","4","--num-warps","1","2","4","auto"],"json"),
        ("fp8-accumulation","benchmark_fp8_accumulation.py",["--backends","triton","native_fast","native_precise"],"json"),
        ("allocator","benchmark_allocator_compare.py",["--config",str(config_root/"hopper_v5_allocator.json"),
            "--variable-config",str(config_root/"hopper_v4_allocator_variable.json"),"--profile-steps","0",
            "--max-numerical-elements",str(args.max_numerical_elements)],"directory"),
    ]
    if args.cifar10_root:
        external.append(("cifar10-epochs","benchmark_cifar10_epoch.py",["--dataset-root",str(args.cifar10_root.expanduser().absolute())],"json"))
    for dtype in ("bf16","e4m3fn","e5m2"):
        external.append(("unpack-"+dtype,"benchmark_activation_unpack.py",["--source-dtype",dtype,
            "--dtypes","bf16","e4m3fn","e5m2","--bits","1","2","4","8","--group-sizes","1024",
            "--execution","both"],"json"))
    # Former group-grid cells are independent named repetitions of the single
    # required codec recipe, not coverage of the superseded group sizes.
    repetitions = {"attention-pack": 3, "activation-pack": 2, "convolution-codec": 2,
                   "unpack-bf16": 4, "unpack-e4m3fn": 4, "unpack-e5m2": 4}
    return matrix+[{"name":name if slot == 0 else f"{name}-group1024-slot-{slot}",
                   "kind":"standalone","script":script(program),"arguments":list(options),"output_kind":kind,
                   "qualification_case":f"group1024-slot-{slot}",
                   "group_coverage":"single group_size=1024; independent repetition"}
                  for name,program,options,kind in external for slot in range(repetitions.get(name, 1))]


def main(argv=None, *, benchmark_dir=None):
    p=parser();args=p.parse_args(argv)
    if args.cpu_workers<1 or args.profile_steps<0 or args.stage_timeout<=0 or args.memory_headroom_gib<0:
        p.error("worker count/timeout must be positive; profile steps/headroom must be nonnegative")
    root=Path(benchmark_dir or Path(__file__).resolve().parent)
    from analysis.scheduler import cancellation_signals
    with cancellation_signals(),output_lock(args.output):
        return run_suite(args,root)


def run_suite(args,root):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=stages(args,root)
    write_json(output/"suite-plan.json",{"stages":jobs,"configuration":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "notes":["Every requested feature mode remains visible. Qualified timings use exclusive physical GPU leases.",
                 "Resident synthetic updates and optional real CIFAR10 epochs are different measurements.",
                 "Unknown VRAM demand calibrates exclusively. Shared screening does not establish standalone speedups."]})
    if args.dry_run:
        print(json.dumps({"stages":len(jobs),"plan":str(output/"suite-plan.json")},indent=2));return 0
    devices=discover_devices(args.devices)
    affinity=output/"device-affinity.json"
    # Preserve each architecture's GPU across preflight, main and ablation
    # matrices; assigning by each matrix's sorted list would move controls.
    architectures=("vit_base_patch16_224","efficientnet_b0","resnet18")
    names=sorted({w["name"] for stage in jobs if stage["kind"]=="matrix"
                  for w in json.loads(Path(stage["config"]).read_text()).get("workloads",[])})
    write_json(affinity,{"by_model":{name:devices[i%len(devices)].uuid for i,name in enumerate(architectures)},
        "by_workload":{name:devices[i%len(devices)].uuid for i,name in enumerate(names)}})
    run_id=uuid.uuid4().hex
    cancel=output/("cancel-"+run_id)
    logs=output/"logs";logs.mkdir(exist_ok=True)
    state={"run_id":run_id,"stages":[],"blocked":[],"status":"running"}
    stage_workers=1 if args.scheduler=="serial" else min(2,args.cpu_workers)
    child_workers=max(1,args.cpu_workers//stage_workers)
    manifest=output/"preflight-manifest.json"
    write_json(manifest,{"blocked":[]})
    def execute(stage,index=0, *, preflight=False):
        began=time.time()
        try:
            if stage["kind"]=="matrix":
                command=[sys.executable,str(root/"benchmark_do_ops.py"),"--config",stage["config"],"--sweep",stage.get("sweep","cartesian"),
                    "--output",str(output/stage["name"]),"--scheduler",args.scheduler,"--devices",args.devices,
                    "--cpu-workers",str(child_workers),"--gpu-concurrency",str(args.gpu_concurrency),
                    "--device-affinity-file",str(affinity),
                    "--lease-db",str(args.lease_db),"--memory-headroom-gib",str(args.memory_headroom_gib),
                    "--timing-isolation","exclusive" if preflight else args.timing_isolation,
                    "--max-experiments","100000","--deterministic","--fill-uninitialized-memory",
                    "--warmup","1" if preflight else "10","--iterations","1" if preflight else "30",
                    "--repeats","1" if preflight else "5","--phase-steps","1" if preflight else "5",
                    "--profile-steps","0" if preflight else str(args.profile_steps),
                    "--profile-replay-steps","0" if preflight or not args.profile_steps else "1",
                    "--max-numerical-elements",str(args.max_numerical_elements),"--profile-policy",args.profile_policy,
                    "--profile-detail","trace","--trace-variants","baseline","--compress-traces",
                    "--trace-workloads","vit_b512","resnet18_b512","efficientnet_b0_b512",
                    "--trace-cases","torch_bf16","fp8_compression","--preflight-manifest",str(manifest)]
                if preflight or stage["name"].startswith("hopper_v6"):command.append("--require-fp8-plan")
                if args.trace_directory:command.extend(["--trace-directory",str(args.trace_directory)])
                if args.deduplicate_effective_settings and not preflight:command.append("--deduplicate-effective-settings")
                command+=stage.get("extra",[])
            else:
                target=output/(stage["name"]+(".json" if stage.get("output_kind","json")=="json" else ""))
                command=[sys.executable,stage["script"],*stage.get("arguments",[]),"--output",str(target)]
            if args.resume and stage.get("resume",True):command.append("--resume")
            log=logs/(stage["name"]+".log")
            if stage["kind"]=="matrix":
                with log.open("w") as stream:
                    code=run_command(command,env=os.environ.copy(),log=stream,timeout=args.stage_timeout,cancel_file=cancel)
            else:
                code=run_external(command,device=devices[index%len(devices)],broker_path=args.lease_db,
                    timeout=args.stage_timeout,cancel=cancel,log_path=log)
            result={"name":stage["name"],"status":"ok" if code==0 else "error","exit_code":code,
                    "elapsed_seconds":time.time()-began,"log":str(log),"command":command}
        except Exception as error:
            result={"name":stage["name"],"status":"error","reason":f"{type(error).__name__}: {error}",
                    "elapsed_seconds":time.time()-began}
        print(json.dumps(result),flush=True)
        return result
    def record(row):
        state["stages"].append(row);write_json(output/"suite-progress.json",state)
    try:
        kernel_script=root/"gpu_preflight.py"
        command=[sys.executable,str(kernel_script),"--output",str(output/"kernel-preflight.json")]
        export=root/"manifest.json"
        if export.exists():
            exported=json.loads(export.read_text())
            expected=exported.get("recommended_library_source_sha256")
            if expected:command.extend(["--expected-source-sha256",expected])
        code=run_external(command,device=devices[0],broker_path=args.lease_db,timeout=args.stage_timeout,
                          cancel=cancel,log_path=logs/"kernel-preflight.log")
        record({"name":"kernel-preflight","status":"ok" if code==0 else "error","exit_code":code})
        kernel=json.loads((output/"kernel-preflight.json").read_text())
        if kernel.get("reason"):
            state.update(status="error",reason=kernel["reason"])
            write_json(output/"suite-progress.json",state);return 1
        for check in kernel.get("checks",[]):
            if check["status"]=="ok":continue
            reason=check["name"]+": "+check.get("reason","failed")
            if "fp8-gradient" in check["name"]:
                for case in ("fp8","fp8_compression"):
                    state["blocked"].append({"case":case,"fp8":{"backward_precision":"auto"},"reason":reason})
            elif "bounded" in check["name"]:
                state["blocked"].append({"runtime":{"strided_indexing":"bounded"},"reason":reason})
            elif "shared-gradient" in check["name"]:
                for key,value in (("linear_gradient_reduction","shared"),("linear_forward_backend","native_fast"),
                                  ("linear_forward_backend","native_precise")):
                    state["blocked"].append({"runtime":{key:value},"reason":reason})
            elif "fp8-only-gelu-auto" in check["name"]:
                for selected in ("auto","recompute"):
                    state["blocked"].append({"model":"vit_base_patch16_224","case":"fp8",
                        "runtime":{"gelu_saved_state":selected},"reason":reason})
            elif "max-pool3x3" in check["name"]:
                for case in ("compression","fp8","fp8_compression"):
                    state["blocked"].append({"model":"resnet18","case":case,
                        "runtime":{"pool_kernel_variant":"auto"},"reason":reason})
            elif "singleton-conv" in check["name"]:
                for case in ("compression","fp8","fp8_compression"):
                    state["blocked"].append({"model":"efficientnet_b0","case":case,"reason":reason})
            elif "masked-tail" in check["name"]:
                affected={"relu":["resnet18"],"silu":["efficientnet_b0"],"gelu":["vit_base_patch16_224"],
                          "sigmoid":["efficientnet_b0"],"tanh":[],"relu6":[]}
                for model in affected.get(check["name"].split("-")[0],[]):
                    for case in (("fp8","fp8_compression") if "float8" in check["name"] else ("compression","fp8","fp8_compression")):
                        state["blocked"].append({"model":model,"case":case,"reason":reason})
            else:
                for case in ("compression","fp8","fp8_compression"):
                    state["blocked"].append({"case":case,"reason":reason})
        write_json(manifest,{"blocked":state["blocked"]})
        config_root=root/"configs" if (root/"configs").is_dir() else root
        for name,config,full in (("small-model-preflight","hopper_v6_preflight",False),
                                 ("full-size-model-preflight","hopper_v6_models",True)):
            record(execute({"name":name,"kind":"matrix","config":str(config_root/(config+".json"))},preflight=True))
            summary=output/name/"summary.json"
            if summary.exists():
                state["blocked"]+=model_failure_rules(json.loads(summary.read_text())["results"],full_size=full)
            else:
                state["blocked"].append({"reason":name+" did not produce a model preflight summary"})
            write_json(manifest,{"blocked":state["blocked"]})
        if not args.preflight_only:
            with ThreadPoolExecutor(max_workers=stage_workers) as executor:
                futures=[executor.submit(execute,stage,index) for index,stage in enumerate(jobs)]
                try:
                    for future in as_completed(futures):record(future.result())
                except BaseException:
                    cancel.touch();raise
        state["status"]="error" if state["blocked"] or any(r["status"]!="ok" for r in state["stages"]) else "ok"
        write_json(output/"suite-progress.json",state)
    except BaseException:
        cancel.touch();raise
    return int(state["status"]!="ok")


if __name__=="__main__":raise SystemExit(main())
