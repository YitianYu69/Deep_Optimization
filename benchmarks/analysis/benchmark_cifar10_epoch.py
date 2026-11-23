"""Real CIFAR10 eager epochs, all five modes, one fresh CUDA process per case."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from analysis.reporting import write_json
from analysis.example_model import DEFAULT_CIFAR10_ROOT


def epoch_worker(request):
    import torch
    from analysis.engine import Session,environment
    from analysis.measurement import memory_fill_scope,dense
    task,options=request["task"],request["options"]
    session=Session(task,options)
    try:
        if session.fp8:
            from deepopt.opt.precision import precision_report
            report=precision_report(session.model)
            if report is None or report["fallbacks"] or not any(e["precision"]=="fp8" for e in report["edges"]):
                raise RuntimeError("The epoch workload has no valid required FP8 plan")
        if not session.epoch_updates:raise RuntimeError("Factory must declare the real epoch update count")
        with memory_fill_scope(False):
            for _ in range(request["warmup_updates"]):session.step()
            remainder=session.benchmark_metadata["dataset_examples"]%session.benchmark_metadata["batch_size"]
            if remainder and not session.benchmark_metadata["drop_last"]:
                fetch=session.fetch
                images=session.raw_inputs[0][:remainder]
                session.epoch_state["target"]=session.epoch_state["target"][:remainder]
                session.fetch=lambda:((images,),{})
                try:session.step()
                finally:session.fetch=fetch
            samples=[]
            for _ in range(request["repeats"]):
                session.reset();torch.cuda.synchronize(session.device)
                torch.cuda.reset_peak_memory_stats(session.device)
                losses=[]
                began=time.perf_counter()
                for _ in range(session.epoch_updates):
                    _,loss=session.step(return_loss=True);losses.append(loss)
                torch.cuda.synchronize(session.device)
                elapsed=time.perf_counter()-began
                count=session.epoch_state["examples"]
                finite=all(bool(torch.isfinite(dense(v)).all()) for v in (*session.last_output,*session.model.parameters()))
                if not finite:raise RuntimeError("Nonfinite model/output after the measured epoch")
                samples.append({"seconds":elapsed,"examples":count,"updates":session.epoch_updates,
                    "images_per_second":count/elapsed,"training_loss":float(torch.stack(losses).mean()),
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated(session.device),
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved(session.device)})
        return {"status":"ok","task":task,"epochs":samples,"seconds_median":statistics.median(r["seconds"] for r in samples),
            "initial_state_sha256":session.initial_fingerprint,"rng_baseline_sha256":session.rng_fingerprint,
            "initial_component_sha256":session.initial_component_fingerprints,"environment":environment(session.device),
            "benchmark_metadata":session.benchmark_metadata,
            "timing_scope":"Full eager training epoch: loader/H2D/forward/loss/backward/AdamW/EMA; worker and kernel warmup excluded",
            "numerical_qualification":"Finiteness is checked. Matched gradients/updates are checked separately by the resident-input matrix."}
    finally:session.close()


def _lease_cli(args, argv=None):
    if __name__ != "__main__" or not getattr(args, "device", "cuda:0").startswith("cuda"):
        return None
    import runpy
    bootstrap = Path(__file__).with_name("_gpu_lease.py")
    if not bootstrap.is_file():
        bootstrap = Path(__file__).resolve().parents[1] / "_gpu_lease.py"
    if not bootstrap.is_file():
        raise SystemExit("Keep _gpu_lease.py and analysis/ beside GPU benchmark programs")
    return runpy.run_path(str(bootstrap))["ensure"](__file__, argv, getattr(args, "device", "cuda:0"))


def main(argv=None, *, benchmark_dir=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,default=Path(DEFAULT_CIFAR10_ROOT),
                   help="Local CIFAR10 root (default: %(default)s); never download")
    p.add_argument("--models",nargs="+",choices=("resnet18","efficientnet_b0","vit_base_patch16_224"),
                   default=["resnet18","efficientnet_b0","vit_base_patch16_224"])
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--drop-last",action="store_true")
    p.add_argument("--deterministic",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--warmup-updates",type=int,default=3)
    p.add_argument("--memory-format",choices=("contiguous","channels_last"),default="contiguous")
    p.add_argument("--output",type=Path,default=Path("cifar10-epochs.json"))
    p.add_argument("--resume",action="store_true")
    p.add_argument("--_request",type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args._request:
        request=json.loads(args._request.read_text())
        try:row=epoch_worker(request)
        except Exception as error:
            row={"status":"error","task":request["task"],"reason":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()}
        row["resume_key"]=request["resume_key"]
        write_json(args._request.parent/"result.json",row)
        return int(row["status"]!="ok")
    if not args.dataset_root or not args.dataset_root.is_dir():p.error("--dataset-root must name an existing local CIFAR10 directory")
    if min(args.batch_size,args.image_size,args.repeats,args.warmup_updates)<1 or args.workers<0:
        p.error("sizes/repeats/warmup-updates must be positive; workers must be nonnegative")
    lease_code = _lease_cli(args, argv)
    if lease_code is not None:
        return lease_code
    from analysis.benchmark_do_ops import parser as matrix_parser,make_plan,code_hash
    from analysis.engine import installed_build_fingerprint,resume_environment_fingerprint
    root=Path(benchmark_dir or Path(__file__).resolve().parent)
    config_root=root/"configs" if (root/"configs").is_dir() else root
    base=json.loads((config_root/"hopper_v6_models.json").read_text())
    base.pop("grid",None)
    base["settings"]["runtime"]["execution"]="eager"
    base["workloads"]=[]
    for model in args.models:
        base["workloads"].append({"name":model,"op":"model","shape":[args.batch_size,3,args.image_size,args.image_size],
            "input_requires_grad":False,"factory":"analysis.example_model:build_local_cifar10","factory_kwargs":{
                "dataset_root":str(args.dataset_root.resolve()),"model_name":model,"batch_size":args.batch_size,
                "image_size":args.image_size,"workers":args.workers,"drop_last":args.drop_last,"memory_format":args.memory_format}})
    directory=args.output.parent/(args.output.stem+"-cases");directory.mkdir(parents=True,exist_ok=True)
    config=directory/"config.json";write_json(config,base)
    plan=make_plan(matrix_parser().parse_args(["--config",str(config),"--deterministic" if args.deterministic else "--no-deterministic"]))
    dataset=sha256()
    for path in sorted(args.dataset_root.rglob("data_batch*")):
        dataset.update(path.name.encode())
        with path.open("rb") as stream:
            while chunk:=stream.read(2**20):dataset.update(chunk)
    key=sha256(json.dumps({"options":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("resume","output","_request")},
        "dataset_sha256":dataset.hexdigest(),"source":installed_build_fingerprint(),"helpers":code_hash(),
        "environment":resume_environment_fingerprint("cuda:0")},sort_keys=True).encode()).hexdigest()
    rows=[]
    for task in plan["tasks"]:
        folder=directory/task["id"];folder.mkdir(parents=True,exist_ok=True)
        result=folder/"result.json"
        row=json.loads(result.read_text()) if args.resume and result.exists() else {}
        if row.get("status")!="ok" or row.get("resume_key")!=key:
            request=folder/"request.json"
            write_json(request,{"task":task,"options":plan["options"],"repeats":args.repeats,
                "warmup_updates":args.warmup_updates,"resume_key":key})
            with (folder/"worker.log").open("w") as log:
                code=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--_request",str(request)],
                                    stdout=log,stderr=subprocess.STDOUT).returncode
            row=json.loads(result.read_text()) if result.exists() else {"status":"error","task":task,"reason":f"Epoch worker exited {code}"}
        rows.append(row)
        write_json(args.output,{"resume_key":key,"dataset_sha256":dataset.hexdigest(),"results":rows,
            "manifest":{"expected":len(plan["tasks"]),"completed":len(rows),"missing":len(plan["tasks"])-len(rows),
                        "failed":sum(r["status"]!="ok" for r in rows)}})
        print(json.dumps({"model":task["workload"]["name"],"case":task["case"],"status":row["status"],
                          "seconds_median":row.get("seconds_median")}),flush=True)
    return int(any(r["status"]!="ok" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
