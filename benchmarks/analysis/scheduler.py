"""Isolated benchmark workers, physical-device leases and conservative admission.

This module imports no CUDA libraries. Local threads only supervise fresh child
processes; they never execute a Torch workload. Dask is an optional local-host
adapter over the same process and lease protocol.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from contextlib import contextmanager,nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from .matrix import CASES, fingerprint
from .reporting import write_json


GiB = 2**30


@dataclass(frozen=True)
class Device:
    uuid: str
    index: str
    total: int
    name: str = ""


def smi(query):
    result = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))


def discover_devices(selection, device="cuda:0"):
    if device == "cpu":
        return [Device("cpu", "cpu", 0, "CPU validation")]
    rows = smi(["--query-gpu=index,uuid,name,memory.total"])
    devices = [Device(row[1].strip(), row[0].strip(), int(row[3]) * 2**20, row[2].strip()) for row in rows]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = visible.split(",")
        devices = [d for token in tokens for d in devices if token in (d.index,d.uuid)]
    if selection in (None, ""):
        ordinal = int(device.split(":")[1]) if ":" in device else 0
        devices = devices[ordinal:ordinal+1]
    elif selection != "auto":
        tokens = selection.split(",")
        selected = [next((d for d in devices if t in (d.index,d.uuid)), None) for t in tokens]
        if any(d is None for d in selected):
            raise ValueError("--devices must name available physical UUIDs/indices within CUDA_VISIBLE_DEVICES")
        devices = list(dict.fromkeys(selected))
    if not devices:
        raise ValueError("No admitted NVIDIA device is visible (MIG instances require an independent dedicated run)")
    return devices


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Broker:
    """SQLite serializes admission across unrelated launchers on this host."""
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS leases (
                    token TEXT PRIMARY KEY, gpu TEXT, owner INTEGER, child INTEGER,
                    demand INTEGER, exclusive INTEGER, created REAL);
                CREATE TABLE IF NOT EXISTS demand (key TEXT PRIMARY KEY, bytes INTEGER);
                CREATE TABLE IF NOT EXISTS quarantined_leases (
                    token TEXT PRIMARY KEY, reason TEXT, created REAL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def estimate(self, key):
        with self.connect() as db:
            row = db.execute("SELECT bytes FROM demand WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def may_admit(self,device,exclusive,limit):
        if device.uuid=="cpu":return True
        with self.connect() as db:
            rows=db.execute("SELECT owner,child,exclusive,token IN (SELECT token FROM quarantined_leases) FROM leases WHERE gpu=?",(device.uuid,)).fetchall()
        live=[row for row in rows if row[3] or alive(row[0]) or alive(row[1])]
        return len(live)<limit and not any(row[2] for row in live) and (not exclusive or not live)

    def learn(self, key, size):
        if size is not None and size > 0:
            with self.connect() as db:
                db.execute("INSERT INTO demand VALUES (?,?) ON CONFLICT(key) DO UPDATE SET bytes=MAX(bytes,excluded.bytes)",
                           (key, int(size)))

    def acquire(self, device, demand, exclusive, limit, headroom, *, free=None, foreign=False):
        if device.uuid == "cpu":
            return "cpu-" + uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A crashed coordinator may leave a live CUDA child. Retain its
            # reservation until BOTH processes have exited.
            for token, owner, child in db.execute("SELECT token,owner,child FROM leases WHERE token NOT IN (SELECT token FROM quarantined_leases)").fetchall():
                if not alive(owner) and not alive(child):
                    db.execute("DELETE FROM leases WHERE token=?", (token,))
            running = db.execute("SELECT demand,exclusive FROM leases WHERE gpu=?", (device.uuid,)).fetchall()
            if foreign and exclusive or len(running) >= limit or running and (exclusive or any(r[1] for r in running)):
                return None
            size = demand if demand is not None else device.total
            if not exclusive:
                if size + sum(r[0] for r in running) + headroom > device.total:
                    return None
                # Free memory already excludes resident children. Requiring
                # the new job's full demand is conservative during ramps.
                if free is None or size + headroom > free:
                    return None
            token = uuid.uuid4().hex
            db.execute("INSERT INTO leases VALUES (?,?,?,?,?,?,?)",
                       (token,device.uuid,os.getpid(),None,size,int(exclusive),time.time()))
            return token

    def attach(self, token, pid):
        with self.connect() as db:
            db.execute("UPDATE leases SET child=? WHERE token=?", (pid,token))

    def release(self, token):
        with self.connect() as db:
            db.execute("DELETE FROM leases WHERE token=? AND token NOT IN (SELECT token FROM quarantined_leases)", (token,))

    def quarantine(self, token, reason):
        # A descendant can outlive both recorded PIDs. Uncertain cleanup is a
        # durable stop, even after the coordinator exits, until verified safe.
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO quarantined_leases VALUES (?,?,?)",(token,reason,time.time()))
            db.execute("UPDATE leases SET exclusive=1 WHERE token=?",(token,))

    def members(self,device):
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT token FROM leases WHERE gpu=? ORDER BY created",(device.uuid,))]

    def telemetry(self, device):
        rows = smi([f"--id={device.uuid}", "--query-gpu=memory.free"])
        processes = smi(["--query-compute-apps=gpu_uuid,pid"])
        with self.connect() as db:
            owned = {p[0] for p in db.execute("SELECT child FROM leases WHERE gpu=?", (device.uuid,)) if p[0]}
        foreign = any(row[0].strip() == device.uuid and int(row[1]) not in owned for row in processes if len(row)>1)
        return int(rows[0][0]) * 2**20, foreign


def default_broker_path():
    return Path(tempfile.gettempdir()) / f"deepopt-benchmark-gpus-{os.getuid()}.sqlite3"


@contextmanager
def output_lock(directory):
    import fcntl
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/".coordinator.lock").open("a") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another coordinator owns {directory}; use a different output directory") from None
        yield


def run_external(command, *, device, broker_path, timeout, cancel, log_path=None):
    """Lease a standalone codec/allocator program through the same broker."""
    broker=Broker(broker_path)
    deadline=time.monotonic()+timeout
    token=None
    while token is None:
        if Path(cancel).exists():raise InterruptedError("Suite cancelled")
        if time.monotonic()>deadline:raise TimeoutError("Standalone stage timed out waiting for the GPU")
        if not broker.may_admit(device,True,1):
            time.sleep(.25);continue
        free,foreign=(None,False) if device.uuid=="cpu" else broker.telemetry(device)
        token=broker.acquire(device,None,True,1,0,free=free,foreign=foreign)
        if token is None:time.sleep(.25)
    env=os.environ.copy()
    if device.uuid!="cpu":env["CUDA_VISIBLE_DEVICES"]=device.uuid
    env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
               DEEPOPT_BENCHMARK_LEASE=token,DEEPOPT_BENCHMARK_LEASE_DB=str(broker_path))
    release=True
    try:
        with Path(log_path).open("w") if log_path else nullcontext(None) as log:
            return run_command(command,env=env,log=log,timeout=timeout,cancel_file=cancel,
                               on_spawn=lambda pid:broker.attach(token,pid))
    except BaseException as error:
        if isinstance(error,ProcessCleanupError) or getattr(error,"_deepopt_cleanup_failed",False):
            release=False
            broker.quarantine(token,str(error))
        raise
    finally:
        if release:broker.release(token)


def inherited_lease():
    token=os.environ.get("DEEPOPT_BENCHMARK_LEASE")
    if not token:return False
    with Broker(os.environ.get("DEEPOPT_BENCHMARK_LEASE_DB") or default_broker_path()).connect() as db:
        if not db.execute("SELECT 1 FROM leases WHERE token=?",(token,)).fetchone():
            raise RuntimeError("Inherited benchmark GPU lease is no longer active")
    return True


def guarded_main(main,argv,script):
    """CLI-only guard, before importing Torch or creating a CUDA context."""
    import inspect
    def call():
        return main(argv) if "argv" in inspect.signature(main).parameters else main()
    if (any(a in argv for a in ("--help","-h","--dry-run","--_worker","--_request"))
            or inherited_lease()):
        return call()
    device="cuda:0"
    if "--device" in argv:device=argv[argv.index("--device")+1]
    if device=="cpu":return call()
    code=ensure_lease(argv,script,device)
    return call() if code is None else code


def ensure_lease(argv,script,device="cuda:0"):
    if device=="cpu" or inherited_lease() or any(a in argv for a in ("--dry-run","--_worker","--_request")):
        return None
    devices=discover_devices(None,device)
    logical=[]
    for index,argument in enumerate(argv):
        if index and argv[index-1]=="--device":logical.append("cuda:0")
        elif argument.startswith("--device="):logical.append("--device=cuda:0")
        else:logical.append(argument)
    with tempfile.TemporaryDirectory(prefix="deepopt-standalone-lease-") as temporary:
        return run_external([sys.executable,str(script),*logical],device=devices[0],broker_path=default_broker_path(),
            timeout=float(os.environ.get("DEEPOPT_BENCHMARK_STAGE_TIMEOUT","21600")),cancel=Path(temporary)/"cancel")


@contextmanager
def cancellation_signals():
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield;return
    previous=signal.getsignal(signal.SIGTERM)
    def interrupted(signum,frame):
        raise KeyboardInterrupt("Benchmark coordinator received SIGTERM")
    signal.signal(signal.SIGTERM,interrupted)
    try:yield
    finally:signal.signal(signal.SIGTERM,previous)


IGNORED_NATIVE_RUNTIME = (
    "norm_kernel_variant", "norm_row_chunk", "norm_max_programs", "codec_pack_variant", "codec_groups_per_program",
    "spatial_activation_layout", "codec_num_warps", "activation_pointwise_tile", "producer_pack", "gelu_saved_state", "silu_saved_state", "activation_layout",
    "strided_indexing", "convolution_codec", "convolution_kernel_variant", "pool_kernel_variant", "linear_dw_backend", "native_operand_layout", "linear_forward_backend", "linear_gradient_reduction", "compression_families",
    "compression_bits_by_family", "compression_exclude_modules", "check_finite",
    "attention_backend", "attention_pack", "attention_grad_layout",
)


def effective_task(task):
    """Only explicit inactive rules are removed; feature case identity is kept."""
    value = deepcopy(task)
    for key in ("id", "variant_id", "baseline_variant"):
        value.pop(key,None)
    if task["case"] == "codec" or task.get("skip_reason"):
        return value
    fp8, compressed = CASES[task["case"]]
    settings = value["settings"]
    if not fp8 or not compressed:
        # Producer output packing requires both FP8 output and compressed saves.
        settings["runtime"].pop("producer_pack", None)
    if not compressed:
        settings["compression"] = {"enabled":False}
        settings["runtime"].pop("attention_pack", None)
    if not fp8:
        settings["fp8"] = {"enabled":False, "exclude_modules":settings["fp8"]["exclude_modules"]}
        settings["runtime"].pop("convolution_kernel_variant", None)
    if task["case"] == "torch_bf16":
        settings["fp8"] = {"enabled":False}
        for key in IGNORED_NATIVE_RUNTIME:
            settings["runtime"].pop(key,None)
    return value


def control_key(task):
    value=deepcopy(task)
    value["case"]="torch_bf16"
    return fingerprint(effective_task(value))


def demand_key(task,options,build,code):
    value=effective_task(task)
    policy=dict(options)
    # Only these built-in resident factories have a fixed current-scaling
    # allocation graph. Preflight executes every phase at least once; repeated
    # event samples do not change their device storage requirement. Arbitrary
    # factories, delayed recipes, variable shapes and quality runs keep the
    # exact counts in their calibration key.
    static=(task["workload"].get("factory") in (
        "analysis.example_model:build_timm_vit","analysis.example_model:build_torchvision_cnn")
        and task["settings"]["fp8"]["scaling_recipe"]=="current"
        and not options.get("quality_steps") and not options.get("latency_steps"))
    if static:
        for name in ("warmup","iterations","repeats"):policy.pop(name,None)
        policy["phase_steps"]=bool(policy.get("phase_steps"))
    return fingerprint({"task":value,"options":policy,"build":build,"code":code})


def profile_options(task, options, policy):
    selected = dict(options)
    if policy == "selected":
        from fnmatch import fnmatchcase
        names = options.get("trace_workloads")
        cases = options.get("trace_cases")
        if ((options.get("trace_variants")=="baseline" and not task.get("baseline_variant",True))
                or (names and not any(fnmatchcase(task["workload"]["name"], pattern) for pattern in names))
                or (cases and task["case"] not in cases)):
            selected["profile_steps"] = selected["profile_replay_steps"] = 0
    return selected


def peak_bytes(row):
    def peaks(value):
        if not isinstance(value,dict):
            return []
        found = [v for k,v in value.items() if "reserved" in k and "bytes" in k
                 and isinstance(v,(float,int)) and v>0]
        return found + [p for v in value.values() for p in peaks(v)]
    values = peaks(row)
    return max(values) if values else None


class ProcessCleanupError(RuntimeError):
    """An owned group could not be verified dead; retain its GPU reservation."""


def child_exited(child):
    """Observe exit without reaping: the leader's PID pins the owned group ID."""
    if hasattr(os,"waitid"):
        return os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None
    # macOS has no os.waitid. ps observes the zombie without releasing its PID.
    result=subprocess.run(["ps","-p",str(child.pid),"-o","stat="],capture_output=True,text=True,check=True,timeout=5)
    return result.stdout.strip().startswith("Z")


def process_group_active(group):
    result=subprocess.run(["ps","-e","-o","pid=,pgid=,stat="],capture_output=True,text=True,check=True,timeout=5)
    for line in result.stdout.splitlines():
        _,pgid,state=line.split()
        # Zombies have exited and no longer retain device contexts. Keep the
        # unreaped leader until all live descendants are gone to prevent reuse.
        if int(pgid)==group and not state.startswith("Z"):
            return True
    return False


def terminate_child(child, *, grace=5):
    """Finish the whole owned process group before reaping its pinned leader."""
    try:
        if child.returncode is not None:
            raise RuntimeError("Owned process-group leader was reaped before cleanup")
        if process_group_active(child.pid):
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            deadline=time.monotonic()+grace
            killed=False
            kill_deadline=None
            while process_group_active(child.pid):
                if not killed and time.monotonic()>=deadline:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    killed=True
                    kill_deadline=time.monotonic()+max(1,grace)
                elif killed and time.monotonic()>=kill_deadline:
                    raise RuntimeError("Live descendants remain after SIGKILL; retaining the GPU reservation")
                time.sleep(.05)
        child.wait()
    except Exception as error:
        raise ProcessCleanupError(f"Could not confirm exit of owned process group {child.pid}: {error}") from error


def run_command(command, *, env, log, timeout, cancel_file, on_spawn=None):
    """Wait for owned process exit before a caller can release its lease."""
    child = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        if on_spawn:
            on_spawn(child.pid)
        deadline = time.monotonic()+timeout
        while not child_exited(child):
            if Path(cancel_file).exists():
                raise InterruptedError("Benchmark run was cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Worker timed out after {timeout}s")
            time.sleep(.1)
    finally:
        try:
            terminate_child(child)
        except BaseException as error:
            error._deepopt_cleanup_failed=True
            raise
    return child.returncode


def run_isolated(job):
    """Dask payload consists solely of descriptors/paths, never live tensors."""
    try:
        return _run_isolated(job)
    except InterruptedError:
        raise
    except Exception as error:
        return scheduler_failure(job,error)


def scheduler_failure(job,error):
    folder=Path(job["folder"]);folder.mkdir(parents=True,exist_ok=True)
    exception=folder/"scheduler-exception.txt"
    exception.write_text("".join(traceback.format_exception(type(error),error,error.__traceback__)))
    row={"status":"error","reason":f"Scheduler {type(error).__name__}: {error}",
         "task":job["task"],"run_id":job["run_id"],"scheduler_exception":str(exception),
         "scheduling":{"gpu_uuid":job["device"]["uuid"],"requested_isolation":job["isolation"]}}
    if isinstance(error,ProcessCleanupError):row["scheduling"]["reservation_retained"]=True
    write_json(folder/"result.json",row)
    return row


def _run_isolated(job):
    from .artifacts import snapshot_valid, profiles_valid
    task,options = job["task"],job["options"]
    folder = Path(job["folder"]); folder.mkdir(parents=True,exist_ok=True)
    result_path = folder/"result.json"
    if job["resume"] and result_path.exists():
        try:
            row = json.loads(result_path.read_text())
            if (row.get("status") == "ok" and row.get("task") == task
                    and row.get("environment",{}).get("deepopt_source_sha256") == job["build"]
                    and row.get("environment",{}).get("resume_environment_sha256") == options["resume_environment_sha256"]
                    and row.get("scheduling",{}).get("requested_isolation") == job["isolation"]
                    and (task["case"] == "codec" or snapshot_valid(row)) and profiles_valid(row,task,options)):
                return row
        except (OSError,ValueError):
            pass
    if task.get("skip_reason"):
        row={"status":"skipped","reason":task["skip_reason"],"task":task,"run_id":job["run_id"]}
        write_json(result_path,row)
        return row
    broker=Broker(job["broker"])
    device=Device(**job["device"])
    key=job["demand_key"]
    estimate=broker.estimate(key)
    # Profiles and allocator trials are always exclusive. Unknown footprints
    # require calibration; no batch128 -> batch512 extrapolation is accepted.
    forced=(job["isolation"]=="exclusive" or estimate is None or
            options.get("profile_steps",0)>0 or options.get("profile_replay_steps",0)>0)
    if estimate and estimate+job["headroom"] > device.total:
        forced=True
    retry=0
    while True:
        token=None
        started=time.time()
        deadline=time.monotonic()+job["timeout"]
        while token is None:
            if Path(job["cancel"]).exists():
                raise InterruptedError("Cancelled before GPU admission")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for a physical GPU lease (foreign jobs may be active)")
            if not broker.may_admit(device,forced,job["concurrency"]):
                time.sleep(.25);continue
            free,foreign=(None,False) if device.uuid=="cpu" else broker.telemetry(device)
            token=broker.acquire(device,estimate,forced,job["concurrency"],job["headroom"],free=free,foreign=foreign)
            if token is None:
                time.sleep(.25)
        scheduling={"gpu_uuid":device.uuid,"device_name":device.name,"requested_isolation":job["isolation"],
                    "isolation":"cpu_validation" if device.uuid=="cpu" else "exclusive" if forced else "shared_screening","cohort_id":token,
                    "calibration_required":estimate is None,"estimated_device_bytes":estimate,
                    "headroom_bytes":job["headroom"],"admission_wait_seconds":time.time()-started,
                    "oom_retry":retry,"admitted_utc_unix":time.time()}
        row=None
        release=True
        try:
            scheduling["foreign_gpu_processes_at_admission"]=foreign
            members=broker.members(device)
            scheduling["overlapping_lease_ids"]=[member for member in members if member!=token]
            scheduling["concurrency_at_admission"]=max(1,len(members))
            scheduling["cohort_id"]=members[0] if members else token
            request=folder/"request.json"
            env=os.environ.copy()
            if device.uuid!="cpu":
                env["CUDA_VISIBLE_DEVICES"]=device.uuid
            env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            worker_options=dict(options)
            core=folder/"core-result.json"
            if job["resume"] and core.exists() and task["case"]!="codec":
                try:
                    cached=json.loads(core.read_text())
                    if (cached.get("task")==task and cached.get("status")=="ok" and snapshot_valid(cached)
                            and cached.get("environment",{}).get("deepopt_source_sha256")==job["build"]
                            and cached.get("environment",{}).get("resume_environment_sha256")==options["resume_environment_sha256"]
                            and cached.get("measurement_policy",{}).get("timing_isolation")==scheduling["isolation"]):
                        worker_options["_resume_core"]=str(core)
                except (OSError,ValueError):
                    pass
            worker_options["scheduling"]=scheduling
            attempt_id=uuid.uuid4().hex
            scheduling["attempt_id"]=attempt_id
            write_json(request,{"task":task,"options":worker_options,"run_id":attempt_id})
            with (folder/("worker-exclusive-retry.log" if retry else "worker.log")).open("w") as log:
                returncode=run_command([sys.executable,job["script"],"--_worker",str(request)],env=env,log=log,
                    timeout=job["timeout"],cancel_file=job["cancel"],on_spawn=lambda pid:broker.attach(token,pid))
            if result_path.exists():
                candidate=json.loads(result_path.read_text())
                if candidate.get("run_id")==attempt_id and candidate.get("task",{}).get("id")==task["id"]:
                    row=candidate
            if returncode and row is not None:
                row.update(status="error", reason=f"Worker exited {returncode} after writing its result; see worker.log")
            if row is None:
                row={"status":"error","reason":f"Worker exited {returncode} without a fresh result; see worker.log","task":task}
        except TimeoutError as error:
            row={"status":"error","reason":str(error),"task":task}
        except BaseException as error:
            if isinstance(error,ProcessCleanupError) or getattr(error,"_deepopt_cleanup_failed",False):
                release=False
                broker.quarantine(token,str(error))
            raise
        finally:
            if release:
                scheduling["released_utc_unix"]=time.time()
                broker.release(token)
        row.update(run_id=job["run_id"],scheduling=scheduling)
        observed=peak_bytes(row)
        scheduling["observed_peak_reserved_bytes"]=observed
        if row["status"]=="ok" and observed:
            # Context/library memory is outside Torch's allocator counters.
            broker.learn(key,int(observed*1.05)+GiB)
        if (row["status"]=="error" and not forced and retry==0
                and "out of memory" in row.get("reason","").lower()):
            write_json(folder/"shared-oom.json",row)
            forced=True;retry=1
            continue
        write_json(result_path,row)
        return row


def run_plan(args, plan):
    from .engine import installed_build_fingerprint,resume_environment_fingerprint
    devices=discover_devices(args.devices,args.device)
    build=installed_build_fingerprint()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    broker_path=args.lease_db or default_broker_path()
    Broker(broker_path)
    run_id=uuid.uuid4().hex
    cancel=output/("cancel-"+run_id)
    workloads=sorted({t["workload"]["name"] for t in plan["tasks"]})
    affinity={name:devices[i%len(devices)] for i,name in enumerate(workloads)}
    if args.device_affinity_file:
        from .preflight import model_name
        declared=json.loads(args.device_affinity_file.read_text())
        available={d.uuid:d for d in devices}
        for task in plan["tasks"]:
            name=task["workload"]["name"]
            requested=declared.get("by_model",{}).get(model_name(task["workload"])) or declared.get("by_workload",{}).get(name)
            if requested:
                if requested not in available:raise ValueError("Device affinity names a GPU outside --devices/CUDA_VISIBLE_DEVICES")
                affinity[name]=available[requested]
    hashes={d.uuid:resume_environment_fingerprint("cpu" if d.uuid=="cpu" else "cuda:0",
                                                visible_devices=None if d.uuid=="cpu" else d.uuid) for d in devices}
    canonical,aliases,jobs={},{},[]
    for task in plan["tasks"]:
        device=affinity[task["workload"]["name"]]
        options=profile_options(task,plan["options"],args.profile_policy)
        options.update(device="cpu" if device.uuid=="cpu" else "cuda:0",resume_environment_sha256=hashes[device.uuid])
        effective=effective_task(task) if args.deduplicate_effective_settings else task
        key=fingerprint({"task":effective,"options":options,"code":plan["code_sha256"],"build":build})
        if key in canonical:
            aliases[task["id"]]=canonical[key]
            continue
        canonical[key]=task["id"]
        jobs.append({"task":task,"options":options,"folder":str(output/"cases"/task["id"]),
                     "run_id":run_id,"resume":args.resume,"build":build,"device":asdict(device),
                     "demand_key":demand_key(task,options,build,plan["code_sha256"]),"broker":str(broker_path),"isolation":args.timing_isolation,
                     "concurrency":args.gpu_concurrency,"headroom":int(args.memory_headroom_gib*GiB),
                     "timeout":args.timeout,"cancel":str(cancel),"script":str(Path(__file__).with_name("benchmark_do_ops.py"))})
    plan.update(installed_build_sha256=build,scheduler={"backend":args.scheduler,"devices":[asdict(d) for d in devices],
        "isolation":args.timing_isolation,"requested_tasks":len(plan["tasks"]),"unique_tasks":len(jobs),"aliases":aliases,
        "workload_device_affinity":{name:device.uuid for name,device in affinity.items()}})
    write_json(output/"plan.json",plan)
    # Interleave device queues so waiting jobs for GPU0 cannot occupy every
    # supervisor and leave GPU1 idle.
    def interleave(items):
        queues=[deque(j for j in items if j["device"]["uuid"]==d.uuid) for d in devices]
        return [q.popleft() for _ in range(max(map(len,queues),default=0)) for q in queues if q]
    controls=[j for j in jobs if j["task"]["case"]=="torch_bf16"]
    anchors=[]
    anchored=args.deduplicate_effective_settings and args.device!="cpu" and args.timing_isolation=="exclusive"
    if anchored:
        middle,end=[],[]
        for original in controls:
            for label,target in (("middle",middle),("end",end)):
                job=deepcopy(original)
                job["task"]["id"]=fingerprint({"control":original["task"]["id"],"anchor":label})
                job.update(anchor_for=original["task"]["id"],resume=False,
                    folder=str(output/"controls"/original["task"]["id"]/label))
                target.append(job);anchors.append(job)
        rest=[j for j in jobs if j not in controls]
        phases=[controls,rest[:len(rest)//2],middle,rest[len(rest)//2:],end]
    else:
        phases=[jobs]
    rows={}
    cluster=client=None
    executor=None
    try:
        if args.scheduler=="dask":
            try:
                from distributed import LocalCluster,Client,as_completed as dask_completed
            except ImportError as error:
                raise ValueError("--scheduler dask requires dask[distributed]; --scheduler local uses the standard library") from error
            cluster=LocalCluster(n_workers=args.cpu_workers,threads_per_worker=1,processes=True,dashboard_address=None)
            client=Client(cluster)
        else:
            executor=ThreadPoolExecutor(max_workers=args.cpu_workers)
        futures=[]
        for phase in phases:
            submitted=interleave(phase)
            futures=([client.submit(run_isolated,job,pure=False) for job in submitted] if client else
                     [executor.submit(run_isolated,job) for job in submitted])
            jobs_by_future=dict(zip(futures,submitted))
            for future in dask_completed(futures) if client else as_completed(futures):
                try:
                    row=future.result()
                except InterruptedError:
                    raise
                except Exception as error:
                    row=scheduler_failure(jobs_by_future[future],error)
                rows[row["task"]["id"]]=row
                print(f"[{len(rows)}/{len(jobs)+len(anchors)} unique/anchor] {row['task']['workload']['name']} {row['task']['case']}: {row['status']}",
                      file=sys.stderr,flush=True)
                write_json(output/"progress.json",{"run_id":run_id,"completed_unique_and_anchors":len(rows),
                    "total_unique":len(jobs),"total_anchors":len(anchors),"total_requested":len(plan["tasks"]),
                    "aliases":aliases,"completed_ids":list(rows)})
    except BaseException:
        cancel.touch()
        raise
    finally:
        if executor:
            executor.shutdown(wait=True,cancel_futures=True)
        if client:
            if cancel.exists():
                client.gather(futures,errors="skip")
            client.close()
        if cluster:
            cluster.close()
    results=[]
    drift={}
    for original in controls if anchored else []:
        samples=[rows[original["task"]["id"]]]+[rows[j["task"]["id"]] for j in anchors if j["anchor_for"]==original["task"]["id"]]
        times=[r["timing"]["wall_ms"]["median"] for r in samples if r["status"]=="ok"]
        ratio=(max(times)-min(times))/sorted(times)[len(times)//2] if len(times)==3 else None
        drift[control_key(original["task"]) ]={"anchor_ids":[r["task"]["id"] for r in samples],
            "wall_ms":times,"relative_range":ratio,"passed":ratio is not None and ratio<=.03,
            "threshold":.03,"note":"A failed drift gate requires fresh exclusive qualification; it never qualifies aliased timings."}
    plan["scheduler"]["control_drift"]=drift
    write_json(output/"plan.json",plan)
    for task in plan["tasks"]:
        source=aliases.get(task["id"],task["id"])
        row=deepcopy(rows[source])
        if source!=task["id"]:
            row.update(task=task,canonical_task_id=source,reuse_reason="Explicit inactive-setting equivalence",
                       canonical_effective_task=effective_task(task))
        if control_key(task) in drift:
            row["control_drift"]=drift[control_key(task)]
        write_json(output/"cases"/task["id"]/"result.json",row)
        results.append(row)
    return results,run_id
