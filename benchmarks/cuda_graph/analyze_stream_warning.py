"""Diagnose AccumulateGrad stream warnings using the installed libraries.

Each case and diagnostic/timing/profile pass runs in a fresh process. Copy this
file anywhere; no repository imports or checkout path injection are required.
"""

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import timedelta
import gc
import importlib.metadata
import importlib.util
import json
from math import isfinite
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
import warnings


CASES = ("eager", "torch_graph", "deepopt_graph", "runtime", "trainer", "fp8", "fp8_compressed")
DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
            "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
            "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE")
WARNING_TEXT = "AccumulateGrad node's stream does not match"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    result.add_argument("--device", default="cuda")
    result.add_argument("--model", choices=("tiny", "vit"), default="tiny")
    result.add_argument("--timm-model", default="vit_base_patch16_224")
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--features", type=int, default=64)
    result.add_argument("--classes", type=int, default=10)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--steps", type=int, default=3, help="diagnostic optimizer updates")
    result.add_argument("--accumulation", type=int, help="inherits training.accumulation_steps; default 1")
    result.add_argument("--warmup", type=int, help="inherits runtime.cuda_graph.warmup; default 10 without config")
    result.add_argument("--allow-unused-input", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--timing-steps", type=int, default=20, help="0 disables the separate timing process")
    result.add_argument("--timing-warmup", type=int, default=5)
    result.add_argument("--profile", action="store_true", help="run another fresh process exporting a Chrome trace")
    result.add_argument("--profile-steps", type=int, default=2)
    result.add_argument("--launcher", choices=("python", "torchrun"), default="python")
    result.add_argument("--ranks", type=int, default=1)
    result.add_argument("--execution-stream", choices=("default", "side"), default="default")
    result.add_argument("--input-grad", action="store_true")
    result.add_argument("--validation-transition", action="store_true")
    result.add_argument("--retain-outputs", action="store_true", help="intentional graph-lifetime control")
    result.add_argument("--retain-graph", action="store_true", help="intentional backward-lifetime control; manual cases only")
    result.add_argument("--zero-grad", choices=("none", "zero"), default="none")
    result.add_argument("--node-trace", action="store_true", help="record AccumulateGrad creation metadata when supported")
    result.add_argument("--gradient-clones", choices=("on", "off"), default="on",
                        help="off is an unsafe diagnostic control for DeepOpt graph ownership")
    result.add_argument("--config", type=Path, help="optional existing DeepOpt TOML configuration")
    result.add_argument("--execution-mode", choices=("legacy", "max_speed", "balanced", "analyze"))
    result.add_argument("--checks", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--backward-backend", choices=("inherit", "native_fast", "native_precise"), default="native_fast")
    result.add_argument("--backward-precision", choices=("native", "auto"), default="native")
    result.add_argument("--fp8-orchestrator", choices=("manual", "runtime", "trainer"), default="trainer",
                        help="FP8 cases use Trainer by default to reproduce the reported setup")
    result.add_argument("--bits", type=int, choices=(1, 2, 4, 8), help="inherits compression config; default 2")
    result.add_argument("--group-size", type=int, help="inherits compression config; default 512")
    result.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--lr", type=float, help="overrides training.learning_rate; default 1e-4")
    result.add_argument("--rtol", type=float, default=0.02)
    result.add_argument("--atol", type=float, default=0.002)
    result.add_argument("--seed", type=int, help="inherits config seed; default 42")
    result.add_argument("--timeout", type=int, default=900, help="timeout for each fresh-process job")
    result.add_argument("--output-dir", type=Path, default=Path("work/stream-warning-report"))
    result.add_argument("--_worker-config", help=argparse.SUPPRESS)
    return result


def validate(args):
    for name in ("batch_size", "features", "classes", "image_size", "steps", "accumulation", "warmup",
                 "profile_steps", "ranks", "timeout", "group_size"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(name.replace("_", "-") + " must be positive")
    if args.timing_steps < 0 or args.timing_warmup < 0 or not isfinite(args.atol) or not isfinite(args.rtol) or args.atol < 0 or args.rtol < 0:
        raise ValueError("timing counts and tolerances must be nonnegative")
    if args.lr is not None and (not isfinite(args.lr) or args.lr <= 0):
        raise ValueError("lr must be positive")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if args.bits not in (1, 2, 4, 8) or type(args.bits) is not int:
        raise ValueError("bits must be 1, 2, 4 or 8")
    if type(args.recenter) is not bool or type(args.allow_unused_input) is not bool:
        raise ValueError("recenter and allow-unused-input must be booleans")
    if args.launcher == "python" and args.ranks != 1:
        raise ValueError("multiple ranks require --launcher torchrun")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("run the controller once with python; use its --launcher torchrun --ranks N options")
    if args.device.startswith("cpu") and (args.cases != ["eager"] or args.execution_stream == "side"):
        raise ValueError("graph/FP8/side-stream cases require CUDA; CPU smoke mode is --device cpu --cases eager")
    orchestrated = any(case in ("runtime", "trainer") or case in ("fp8", "fp8_compressed")
                       and args.fp8_orchestrator != "manual" for case in args.cases)
    if args.retain_graph and orchestrated:
        raise ValueError("--retain-graph is a manual-loop control; select manual cases explicitly")
    if args.zero_grad != "none" and orchestrated:
        raise ValueError("Runtime/Trainer own gradient clearing; select manual cases for --zero-grad zero")


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Recorder:
    """Metadata only: never retain a live autograd Node, output, or loss."""

    def __init__(self, torch, device, enabled=True):
        self.torch, self.device, self.enabled = torch, device, enabled
        self.current = "startup"
        self.events, self.warnings, self.nodes = [], [], []
        self.limit = 10000
        self.update_index = self.microbatch_index = 0

    def stream(self):
        if self.device.type != "cuda":
            return {"stream": None, "capturing": False}
        return {"stream": int(self.torch.cuda.current_stream(self.device).cuda_stream),
                "capturing": bool(self.torch.cuda.is_current_stream_capturing())}

    def event(self, name, **details):
        if self.enabled and len(self.events) < self.limit:
            self.events.append({"phase": name, "time_ns": time.perf_counter_ns(),
                                "update": self.update_index, "microbatch": self.microbatch_index,
                                **self.stream(), **details})

    @contextmanager
    def phase(self, name):
        previous, self.current = self.current, name
        self.event(name + ".enter")
        try:
            yield
        finally:
            self.event(name + ".exit")
            self.current = previous

    @contextmanager
    def warning_capture(self):
        original = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            self.warnings.append({"phase": self.current, "update": self.update_index,
                                  "microbatch": self.microbatch_index, "message": str(message), "category": category.__name__,
                                  "filename": filename, "lineno": lineno, **self.stream(),
                                  "stack": traceback.format_stack(limit=18) if WARNING_TEXT in str(message) else []})
            original(message, category, filename, lineno, file=file, line=line)

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = show
            try:
                yield
            finally:
                warnings.showwarning = original

    def node_created(self, node):
        if node.name() == "torch::autograd::AccumulateGrad" or node.name() == "AccumulateGrad":
            if len(self.nodes) < self.limit:
                variable = getattr(node, "variable", None)
                self.nodes.append({"phase": self.current, "update": self.update_index, "microbatch": self.microbatch_index,
                                   "node_id": id(node), "name": node.name(),
                                   "variable_id": id(variable) if variable is not None else None,
                                   "shape": list(variable.shape) if variable is not None else None, **self.stream()})

    @contextmanager
    def setup_instrumentation(self, model):
        """Observe the helper's forward/grad/stream scopes; never change streams."""
        from unittest.mock import patch

        torch = self.torch
        original_forward, original_grad, original_stream = model.forward, torch.autograd.grad, torch.cuda.stream
        active = True

        def forward(*args, **kwargs):
            if not active:
                return original_forward(*args, **kwargs)
            kind = "record" if torch.cuda.is_current_stream_capturing() else "warmup"
            with self.phase(f"capture.{kind}.forward"):
                return original_forward(*args, **kwargs)

        def grad(*args, **kwargs):
            kind = "record" if torch.cuda.is_current_stream_capturing() else "warmup"
            with self.phase(f"capture.{kind}.backward"):
                return original_grad(*args, **kwargs)

        @contextmanager
        def stream(value):
            with original_stream(value):
                self.event("capture.internal_stream.enter")
                try:
                    yield
                finally:
                    self.event("capture.internal_stream.exit")

        model.forward = forward
        try:
            with patch.object(torch.autograd, "grad", grad), patch.object(torch.cuda, "stream", stream):
                yield
        finally:
            active = False
            # The graph helper may have installed a new forward. That wrapper
            # must remain installed; the observed original stays in its closure.
            if model.forward is forward:
                model.forward = original_forward


def error_summary(torch, actual, expected, atol, rtol):
    if set(actual) != set(expected):
        return {"passed": False, "reason": "tensor keys differ"}
    maximum, squared_error, squared_reference, mismatches, count = 0.0, 0.0, 0.0, 0, 0
    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            if value is not reference:
                return {"passed": False, "reason": "missing tensor: " + name}
            continue
        if value.shape != reference.shape:
            return {"passed": False, "reason": "shape mismatch: " + name}
        a, b = value.double(), reference.double()
        finite = torch.isfinite(a) & torch.isfinite(b)
        difference = (a[finite] - b[finite]).abs()
        maximum = max(maximum, difference.max().item() if difference.numel() else 0.0)
        squared_error += difference.square().sum().item()
        squared_reference += b[finite].square().sum().item()
        # Nonfinite gradients/updates are reported as failures even if equal.
        mismatches += int((~finite).sum()) + int((difference > atol + rtol * b[finite].abs()).sum())
        count += value.numel()
    return {"passed": mismatches == 0, "max_abs": maximum,
            "relative_l2": (squared_error / max(squared_reference, 1e-60)) ** 0.5,
            "mismatches": mismatches, "elements": count}


def summarize_profile(torch, profile):
    sync_names = ("cudaStreamWaitEvent", "cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize")
    synchronization, graphs, kernels = {}, {}, 0
    for event in profile.events():
        for name in sync_names:
            if name in event.name:
                synchronization[name] = synchronization.get(name, 0) + 1
        if "cudaGraphLaunch" in event.name:
            graphs[event.name] = graphs.get(event.name, 0) + 1
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels += 1
    return {"synchronization_calls": synchronization, "graph_launch_calls": graphs, "cuda_activities": kernels,
            "note": "Synchronization counts include expected ordering and diagnostics. Inspect named forward/backward ranges in the trace; counts alone do not attribute cost to the warning."}


def environment(torch, device):
    spec = importlib.util.find_spec("deepopt")
    driver = None
    if device.type == "cuda":
        try:
            driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    text=True, capture_output=True, timeout=5, check=False).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        direct = json.loads(importlib.metadata.distribution("deep-optimization").read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        commit = None
    return {"python": sys.version, "pid": os.getpid(), "executable": sys.executable,
            "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "driver": driver,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "triton": version("triton"), "timm": version("timm"), "deepopt": version("deep-optimization"),
            "deepopt_path": spec.origin if spec is not None else None, "deepopt_commit": commit,
            "rank": int(os.environ.get("RANK", "0")), "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "torchrun": "RANK" in os.environ,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "warning_control_api_available": hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"),
            "node_creation_hook_available": hasattr(torch.autograd.graph, "node_creation_hook")}


def make_model(torch, args):
    torch.manual_seed(args["seed"])
    if args["model"] == "vit":
        import timm

        return timm.create_model(args["timm_model"], pretrained=False, num_classes=args["classes"],
                                 img_size=args["image_size"])
    return torch.nn.Sequential(torch.nn.Linear(args["features"], args["features"]),
                               torch.nn.LayerNorm(args["features"]), torch.nn.GELU(),
                               torch.nn.Linear(args["features"], args["classes"]))


def case_config(args, case):
    config = deepcopy(args.get("loaded_config", {}))
    runtime = config.setdefault("runtime", {})
    runtime.update(device=args["device"], precision="bf16" if args["device"].startswith("cuda") else "fp32",
                   compile={"enabled": False})
    if args["checks"] is not None:
        runtime["checks"] = args["checks"]
    runtime.setdefault("checks", True)
    mode = args["execution_mode"]
    if mode is not None:
        runtime.pop("diagnostics", None)
        runtime["execution_mode"] = None if mode == "legacy" else mode
    runtime.pop("mode_schedule", None)
    runtime["activation_compression"] = {
        "enabled": case == "fp8_compressed", "group_size": args["group_size"],
        "bits": args["bits"], "norm_xhat_recenter": args["recenter"],
    }
    runtime["fp8"] = {"enabled": case in ("fp8", "fp8_compressed"), "backend": "triton",
                      "scaling_recipe": "current", "trace_mode": "modules", "intercept_linear": False,
                      "activation_dtype": "e4m3fn", "weight_dtype": "e4m3fn", "grad_dtype": "e4m3fn",
                      "num_stages": 3, "backward_backend": args["backward_backend"]
                      if case in ("fp8", "fp8_compressed") else "inherit",
                      "backward_precision": args["backward_precision"]}
    training = config.setdefault("training", {})
    training.setdefault("ema", {"enabled": False})
    training.update(batch_size=args["batch_size"], accumulation_steps=args["accumulation"])
    training["learning_rate"] = args["lr"] if args["lr"] is not None else training.get("learning_rate", 1e-4)
    return config


class Run:
    """One candidate or matched eager pass; snapshots are detached CPU data."""

    def __init__(self, torch, args, case, rec, device, prefix, initial=None):
        self.torch, self.args, self.case, self.rec, self.device, self.prefix = torch, args, case, rec, device, prefix
        self.capture = self.runtime = self.trainer = self.raw_original = None
        self.retained = []
        self.outputs, self.gradients, self.parameters, self.input_gradients = {}, {}, {}, {}
        self.update = 0
        self.current_input = None
        self.validation_delta = {}
        self.deepopt = None
        self.handles = []
        self.diagnostic = args["pass"] == "diagnostic"
        self.config = case_config(args, case)
        self.checks = self.config["runtime"]["checks"]
        self.config_mode = self.config["runtime"].get("execution_mode")
        self.finite = self.checks if self.config_mode != "max_speed" else False
        self.record_snapshots = self.diagnostic
        self.microbatch = 0
        self.reference = prefix == "reference"
        self.orchestrator = (args["fp8_orchestrator"] if case in ("fp8", "fp8_compressed") else
                             case if case in ("runtime", "trainer") else "manual")
        self.rec.update_index = self.rec.microbatch_index = 0
        with rec.phase(prefix + ".model_prepare"):
            self.model = make_model(torch, args).to(device)
            if initial is not None:
                self.model.load_state_dict(initial)
            self.initial = ({name: value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
                             for name, value in self.model.state_dict().items()} if not self.reference and self.diagnostic else None)
            self.settings = False
            if case not in ("eager", "torch_graph"):
                import deepopt

                self.deepopt = deepopt
                self.settings = deepopt.FP8Config(enabled=case in ("fp8", "fp8_compressed"), backend="triton",
                                                 trace_mode="modules", intercept_linear=False, num_stages=3,
                                                 activation_dtype="e4m3fn", weight_dtype="e4m3fn", grad_dtype="e4m3fn",
                                                 scaling_recipe="current", backward_backend=args["backward_backend"]
                                                 if case in ("fp8", "fp8_compressed") else "inherit",
                                                 backward_precision=args["backward_precision"])
                from deepopt.opt.policy import prepare_fp8_model

                self.model, _ = prepare_fp8_model(self.model, self.settings, config=self.config,
                                                 activation_compression=self.config["runtime"]["activation_compression"])
            self.model.train()
        self.graph_enabled = case != "eager" and not self.reference
        self.sample = self.batch(0)[0]
        if self.graph_enabled:
            self.capture_model()
        self.active = self.model
        training = self.config["training"]
        optimizer_name = training.get("optimizer", "adamw")
        optimizers = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        if optimizer_name not in optimizers:
            raise ValueError("diagnostic supports adamw, adam and sgd optimizers")
        optimizer_options = dict(lr=training["learning_rate"], weight_decay=training.get("weight_decay", 0.0))
        if optimizer_name == "sgd":
            optimizer_options["momentum"] = training.get("momentum", 0.0)
        with self.phase("optimizer_prepare"):
            self.optimizer = optimizers[optimizer_name](self.model.parameters(), **optimizer_options)
        with rec.phase(prefix + ".orchestrator_prepare"):
            if self.orchestrator == "runtime":
                from deepopt import TrainingRuntime

                self.runtime = TrainingRuntime(self.model, self.optimizer, config=self.config,
                                               fp8=self.settings if self.settings.enabled else False)
            elif self.orchestrator == "trainer":
                from deepopt import Trainer

                self.trainer = Trainer(self.model, self.optimizer, loss_fn=self.loss, config=self.config,
                                       fp8=self.settings if self.settings.enabled else False,
                                       accumulation_steps=args["accumulation"])
                self.runtime = self.trainer.runtime
            elif torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                from torch.nn.parallel import DistributedDataParallel

                self.active = DistributedDataParallel(self.model, device_ids=[device.index] if device.type == "cuda" else None)
                for module in self.model.modules():
                    if hasattr(module, "clear_fp8_cache"):
                        module.clear_fp8_cache()
        if self.runtime is not None:
            self.instrument_runtime()
        self.handles.extend((self.optimizer.register_step_pre_hook(self.before_step),
                             self.optimizer.register_step_post_hook(self.after_step)))

    @contextmanager
    def phase(self, name):
        label = self.prefix + "." + name
        with self.rec.phase(label):
            with self.torch.profiler.record_function(label) if self.args["pass"] == "profile" else nullcontext():
                yield

    def autocast(self, capture=False):
        if self.deepopt is not None:
            # Runtime/Trainer inherit the ambient FP8 policy when disabled.
            # Explicit False here would capture a different operation policy
            # and make every BF16 training forward fall back to eager.
            fp8 = self.settings if self.settings.enabled or self.orchestrator == "manual" else None
            return self.deepopt.autocast(self.device.type, enabled=self.device.type == "cuda",
                                         dtype=self.torch.bfloat16, fp8=fp8,
                                         activation_compression=self.config["runtime"]["activation_compression"],
                                         cache_enabled=False, check_finite=False if capture else self.finite)
        return self.torch.autocast(self.device.type, enabled=self.device.type == "cuda",
                                   dtype=self.torch.bfloat16, cache_enabled=False)

    def batch(self, index):
        # Generation is deterministic and independent from model/dropout RNG.
        generator = self.torch.Generator().manual_seed(self.args["seed"] + 1000 + index +
                                                       int(os.environ.get("RANK", "0")) * 100000)
        shape = ((self.args["batch_size"], 3, self.args["image_size"], self.args["image_size"])
                 if self.args["model"] == "vit" else (self.args["batch_size"], self.args["features"]))
        x = self.torch.randn(shape, generator=generator).to(self.device).requires_grad_(self.args["input_grad"])
        target = self.torch.randint(self.args["classes"], (self.args["batch_size"],), generator=generator).to(self.device)
        return x, target

    def capture_model(self):
        torch = self.torch
        buffers = [(buffer, buffer.detach().clone()) for buffer in self.model.buffers()]
        instrument = self.rec.setup_instrumentation(self.model) if self.diagnostic else nullcontext()
        try:
            with self.phase("capture_setup"), instrument, torch.random.fork_rng(devices=[self.device.index]), self.autocast(capture=True):
                if self.case == "torch_graph":
                    self.raw_original = self.model.forward
                    self.active = torch.cuda.make_graphed_callables(self.model, (self.sample,), num_warmup_iters=self.args["warmup"],
                                                                    allow_unused_input=self.args["allow_unused_input"])
                else:
                    self.capture = self.deepopt.CUDAGraphCore(self.device).capture_model(
                        self.model, (self.sample,), warmup=self.args["warmup"], autograd=True,
                        allow_unused_input=self.args["allow_unused_input"])
        finally:
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)

    def instrument_runtime(self):
        compute, backward = self.runtime.compute, self.runtime.backward

        def observed_compute(batch, step):
            self.current_input = batch[0]
            self.rec.update_index, self.rec.microbatch_index = self.update + 1, self.microbatch + 1
            with self.phase("train.forward"):
                return compute(batch, step)

        def observed_backward(loss, **kwargs):
            with self.phase("train.backward"):
                result = backward(loss, **kwargs)
            self.save_input_gradient()
            return result

        self.runtime.compute, self.runtime.backward = observed_compute, observed_backward

    def loss(self, output, target):
        with self.phase("train.loss"):
            if self.record_snapshots:
                self.outputs[str(self.microbatch)] = output.detach().float().cpu().clone()
            self.microbatch += 1
            loss = self.torch.nn.functional.cross_entropy(output.float(), target)
            if self.args["retain_outputs"]:
                self.retained.append((output, loss))
            return loss

    def save_input_gradient(self):
        if self.record_snapshots and self.args["input_grad"] and self.current_input is not None:
            grad = self.current_input.grad
            self.input_gradients[str(self.microbatch - 1)] = None if grad is None else grad.detach().cpu().clone()
        self.current_input = None

    def before_step(self, optimizer, args, kwargs):
        self.optimizer_phase = self.rec.current
        self.rec.current = self.prefix + ".optimizer.step"
        self.rec.event(self.rec.current + ".enter")
        # Runtime/Trainer have already normalized parameter gradients here.
        if self.record_snapshots and self.update == self.args["steps"] - 1:
            self.gradients = {name: None if p.grad is None else p.grad.detach().float().cpu().clone()
                              for name, p in self.model.named_parameters()}

    def after_step(self, optimizer, args, kwargs):
        self.update += 1
        if self.record_snapshots and self.update == self.args["steps"]:
            self.parameters = {name: p.detach().float().cpu().clone() for name, p in self.model.named_parameters()}
        if self.diagnostic and self.args["validation_transition"] and self.update == 1:
            self.validate_once()
        self.rec.event(self.prefix + ".optimizer.step.exit")
        self.rec.current = self.optimizer_phase

    def statistics(self):
        return dict(self.capture.statistics) if self.capture is not None else None

    def validate_once(self):
        before = self.statistics()
        with self.phase("validation"), self.torch.no_grad(), self.autocast():
            self.model.eval()
            self.active.eval()
            value, _ = self.batch(999)
            output = self.active(value)
            if self.record_snapshots:
                self.outputs["validation"] = output.detach().float().cpu().clone()
            del output, value
            self.model.train()
            self.active.train()
        after = self.statistics()
        if before is not None:
            self.validation_delta = {key: after[key] - before[key] for key in before}

    def batches(self, updates, fixed=False):
        fixed_batch = self.batch(0) if fixed else None
        for index in range(updates * self.args["accumulation"]):
            value = fixed_batch or self.batch(index)
            value[0].grad = None
            yield value

    def train(self, updates, fixed=False):
        batches = self.batches(updates, fixed)
        # The native AMP cache policy remains identical inside nested runtime scopes.
        with self.torch.autocast(self.device.type, enabled=self.device.type == "cuda",
                                 dtype=self.torch.bfloat16, cache_enabled=False):
            if self.trainer is not None:
                with self.phase("trainer.train_epoch"):
                    self.trainer.train_epoch(batches)
                return
            self.optimizer.zero_grad(set_to_none=self.args["zero_grad"] == "none")
            for index, (x, target) in enumerate(batches):
                self.current_input = x
                self.rec.update_index, self.rec.microbatch_index = self.update + 1, self.microbatch + 1
                if self.runtime is not None:
                    loss = self.runtime.compute((x, target), lambda batch: self.loss(self.model(batch[0]), batch[1]))
                    self.runtime.backward(loss)
                else:
                    with self.phase("train.forward"), self.autocast():
                        output = self.active(x)
                        loss = self.loss(output, target)
                    with self.phase("train.backward"):
                        loss.backward(retain_graph=self.args["retain_graph"])
                    self.save_input_gradient()
                    del output
                if (index + 1) % self.args["accumulation"] == 0:
                    with self.phase("optimizer.step"):
                        if self.runtime is not None:
                            self.runtime.step(gradient_divisor=self.args["accumulation"])
                        else:
                            for parameter in self.model.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.div_(self.args["accumulation"])
                            self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=self.args["zero_grad"] == "none")
                del x, target, loss

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.retained.clear()
        if self.capture is not None:
            self.capture.close()
        elif self.raw_original is not None:
            self.model.forward = self.raw_original


def execute_pass(torch, args, case, recorder, device, prefix, initial=None):
    run = None
    result = {}
    try:
        run = Run(torch, args, case, recorder, device, prefix, initial)
        torch.manual_seed(args["seed"] + 1)
        cloning = nullcontext()
        if args["gradient_clones"] == "off" and run.capture is not None:
            from unittest.mock import patch

            cloning = patch("deepopt.cuda_graph.model._own_backward_gradients", lambda outputs: outputs)
        with cloning:
            if args["pass"] == "diagnostic":
                run.train(args["steps"])
                result["snapshots"] = {"outputs": run.outputs, "gradients_last_update": run.gradients,
                                       "parameters_last_update": run.parameters, "input_gradients": run.input_gradients}
                result["initial"] = run.initial
            else:
                run.train(args["timing_warmup"], fixed=True)
                before = run.statistics()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                profile = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] +
                                                  ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []))
                           if args["pass"] == "profile" else nullcontext(None))
                count = args["profile_steps"] if args["pass"] == "profile" else args["timing_steps"]
                # Preallocate the timing batch before starting timers.
                fixed = run.batch(0)
                original_batch = run.batch
                run.batch = lambda index: fixed
                try:
                    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    with profile as captured_profile:
                        began = time.perf_counter()
                        if start is not None:
                            start.record()
                        run.train(count, fixed=True)
                        if end is not None:
                            end.record()
                            end.synchronize()
                        wall = time.perf_counter() - began
                    result["timing"] = {"updates": count, "wall_ms_per_update": wall * 1000 / count,
                                        "cuda_event_ms_per_update": start.elapsed_time(end) / count if start is not None else None,
                                        "synthetic_batch_reused": True, "profiler_enabled": captured_profile is not None,
                                        "note": "Includes optimizer and orchestration; event intervals include submission gaps. No per-step device barrier. Compare the separate unprofiled timing pass."}
                    if captured_profile is not None:
                        trace = Path(args["report_dir"]) / f"{case}.rank{os.environ.get('RANK', '0')}.trace.json"
                        captured_profile.export_chrome_trace(str(trace))
                        result["profile"] = {"trace": str(trace), **summarize_profile(torch, captured_profile)}
                finally:
                    run.batch = original_batch
                after = run.statistics()
                if before is not None:
                    result["measured_replay_delta"] = {key: after[key] - before[key] for key in before}
        result.update(statistics=run.statistics(), validation_statistics_delta=run.validation_delta,
                      optimizer_updates=run.update, effective_config=run.runtime.config if run.runtime is not None else run.config,
                      orchestrator=run.orchestrator, gradient_clones_control=args["gradient_clones"],
                      node_trace_requested=args["node_trace"])
        if run.deepopt is not None:
            from deepopt.opt.precision import precision_report

            result["precision_plan"] = precision_report(run.model)
        return result
    finally:
        if run is not None:
            run.close()


def worker(job):
    args, case = job["args"], job["case"]
    rank = int(os.environ.get("RANK", "0"))
    destination = Path(args["report_dir"]) / f"rank{rank}.json"
    report = {"case": case, "pass": args["pass"], "rank": rank, "status": "error", "configuration": args}
    recorder = None
    initialized = False
    try:
        deterministic = args.get("loaded_config", {}).get("runtime", {}).get("deterministic", True)
        if deterministic:
            # Native cases need this before CUDA/BLAS initialization too.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        device = torch.device(args["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for stream/capture analysis; CPU smoke mode is --device cpu --cases eager")
            index = int(os.environ.get("LOCAL_RANK", str(device.index or 0)))
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
            args["device"] = str(device)
            if case in ("fp8", "fp8_compressed") and torch.cuda.get_device_capability(device)[0] < 9:
                raise RuntimeError("This diagnostic's Triton FP8 cases require an NVIDIA GPU with compute capability >= 9")
        report["environment"] = environment(torch, device)
        if "RANK" in os.environ:
            torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                                 timeout=timedelta(seconds=min(args["timeout"], 300)))
            initialized = True
        torch.use_deterministic_algorithms(deterministic)
        recorder = Recorder(torch, device, enabled=args["pass"] == "diagnostic")
        hook = getattr(torch.autograd.graph, "node_creation_hook", None)
        nodes = hook(recorder.node_created) if args["node_trace"] and hook is not None and args["pass"] == "diagnostic" else nullcontext()
        current = torch.cuda.current_stream(device) if device.type == "cuda" else None
        side = torch.cuda.Stream(device=device) if args["execution_stream"] == "side" else None
        if side is not None:
            side.wait_stream(current)
        try:
            with recorder.warning_capture(), nodes, torch.cuda.stream(side) if side is not None else nullcontext():
                candidate = execute_pass(torch, args, case, recorder, device, "candidate")
                if args["pass"] == "diagnostic":
                    initial = candidate.pop("initial")
                    snapshots = candidate.pop("snapshots")
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    reference = execute_pass(torch, args, case, recorder, device, "reference", initial=initial)
                    reference.pop("initial")
                    reference_snapshots = reference.pop("snapshots")
                    comparisons = {name: error_summary(torch, snapshots[name], reference_snapshots[name], args["atol"], args["rtol"])
                                   for name in snapshots}
                    report["correctness"] = {"passed": all(row["passed"] for row in comparisons.values()),
                                             "comparisons": comparisons,
                                             "scope": "all diagnostic outputs/input gradients; normalized parameter gradients and parameters at the final update",
                                             "reference": "same precision/compression/orchestrator, eager, new model initialized from the same state",
                                             "input_gradient_note": "Trainer input gradients are measured before optimizer gradient normalization, in both passes"}
                    del initial, snapshots, reference_snapshots
                    report["reference"] = reference
                report["candidate"] = candidate
        finally:
            if side is not None:
                current.wait_stream(side)
        if recorder is not None:
            report.update(events=recorder.events, warnings=recorder.warnings, accumulator_creations=recorder.nodes,
                          metadata_limit=recorder.limit,
                          metadata_may_be_truncated=len(recorder.events) >= recorder.limit or len(recorder.nodes) >= recorder.limit,
                          stream_note="Handles describe the calling stream, not a direct read of an existing accumulator's internal stream. Node IDs may be reused after destruction.",
                          reference_warning_note="Reference runs after the candidate in this process; absence of a reference warning is not independent evidence because the C++ warning is emitted once.")
        if case in ("eager", "torch_graph"):
            report["native_case_imported_deepopt"] = "deepopt" in sys.modules
            if report["native_case_imported_deepopt"]:
                raise RuntimeError("Native baseline unexpectedly imported DeepOpt")
        stats = candidate.get("statistics")
        if stats is not None:
            if args["pass"] == "diagnostic":
                validation = candidate.get("validation_statistics_delta", {})
                training = {key: value - validation.get(key, 0) for key, value in stats.items()}
                count = args["steps"]
            else:
                training = candidate["measured_replay_delta"]
                count = args["profile_steps"] if args["pass"] == "profile" else args["timing_steps"]
            expected = count * args["accumulation"]
            valid = training["forward_replays"] == expected and not training["eager_mode_fallbacks"] and not training["eager_autocast_fallbacks"]
            report["training_replay"] = {"passed": valid, "expected_forward_replays": expected, "statistics": training,
                                         "backward_replay": "requires profiler evidence; forward counter alone is insufficient"}
        success = report.get("correctness", {}).get("passed", True) and report.get("training_replay", {}).get("passed", True)
        report["status"] = "ok" if success else "mismatch"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if recorder is not None:
            report.update(events=recorder.events, warnings=recorder.warnings, accumulator_creations=recorder.nodes)
    finally:
        if initialized:
            try:
                torch.distributed.destroy_process_group()
            except Exception as error:
                report["distributed_cleanup_error"] = str(error)
        write_json(destination, report)
    return 0 if report["status"] == "ok" else 1


def child_environment():
    environment = dict(os.environ)
    # Do not inherit a checkout injected by the controller's shell or a parent
    # torchrun rendezvous. Each job owns its independent launcher/process group.
    environment.pop("PYTHONPATH", None)
    for name in DIST_ENV:
        environment.pop(name, None)
    return environment


def launch_job(script, job, directory):
    if job["args"]["launcher"] == "python":
        command = [sys.executable, "-u", str(script)]
    else:
        # A loopback rendezvous avoids hostname/socket-path issues and keeps
        # each local job separate from the shell's existing process group.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--rdzv-backend=static",
                   "--master-addr=127.0.0.1", f"--master-port={port}",
                   f"--nproc-per-node={job['args']['ranks']}", str(script)]
    command.extend(["--_worker-config", json.dumps(job)])
    log = directory / "process.log"
    timed_out = False
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=child_environment(), start_new_session=os.name == "posix")
        try:
            code = process.wait(timeout=job["args"]["timeout"])
        except subprocess.TimeoutExpired:
            timed_out = True

            def stop(force=False):
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                    elif force:
                        process.kill()
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass

            stop()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop(force=True)
                code = process.wait()
    ranks = []
    for rank in range(job["args"]["ranks"]):
        path = directory / f"rank{rank}.json"
        if path.exists():
            try:
                ranks.append(json.loads(path.read_text()))
            except (ValueError, OSError) as error:
                ranks.append({"rank": rank, "status": "error", "error": str(error)})
        else:
            ranks.append({"rank": rank, "status": "error", "error": "worker report missing; inspect process.log"})
    return {"case": job["case"], "pass": job["args"]["pass"], "exit_code": code,
            "timed_out": timed_out, "log": str(log), "ranks": ranks}


def case_summary(job):
    ranks = []
    for row in job["ranks"]:
        mismatch = next((warning for warning in row.get("warnings", [])
                         if WARNING_TEXT in warning["message"] and not warning["phase"].startswith("reference.")), None)
        ranks.append({"rank": row["rank"], "status": row["status"],
                      "first_mismatch_phase": mismatch["phase"] if mismatch else None,
                      "correctness": row.get("correctness", {}).get("passed"),
                      "training_replay": row.get("training_replay", {}).get("passed"),
                      "timing": row.get("candidate", {}).get("timing"), "error": row.get("error")})
    return {"case": job["case"], "pass": job["pass"], "exit_code": job["exit_code"], "ranks": ranks}


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args._worker_config:
        return worker(json.loads(args._worker_config))
    try:
        loaded_config = {}
        if args.config is not None:
            import tomllib

            with args.config.open("rb") as source:
                loaded_config = tomllib.load(source)
            # Application-specific tables are unrelated to the probe and may
            # contain credentials. Only consume/report these core sections.
            loaded_config = {key: value for key, value in loaded_config.items()
                             if key in ("name", "seed", "training", "runtime")}
        json.dumps(loaded_config, allow_nan=False)
        runtime = loaded_config.get("runtime", {})
        compression = runtime.get("activation_compression", {})
        inherited = {"seed": loaded_config.get("seed", 42),
                     "accumulation": loaded_config.get("training", {}).get("accumulation_steps", 1),
                     "warmup": runtime.get("cuda_graph", {}).get("warmup", 10),
                     "allow_unused_input": runtime.get("cuda_graph", {}).get("allow_unused_input", False),
                     "bits": compression.get("bits", 2), "group_size": compression.get("group_size", 512),
                     "recenter": compression.get("norm_xhat_recenter", True)}
        for key, value in inherited.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        validate(args)
    except (ValueError, OSError) as error:
        argument_parser.error(str(error))
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Distinct invocation directories prevent stale rank reports from looking
    # like successful output from a new run that exited before reporting.
    import tempfile

    invocation = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
    options = vars(args).copy()
    options.pop("_worker_config")
    options["config"] = str(args.config) if args.config else None
    options["output_dir"] = str(directory)
    options["loaded_config"] = loaded_config
    report = {"schema_version": 1, "directory": str(invocation), "configuration": options,
              "notes": ["Each case/pass is a fresh process; warning absence after its first occurrence cannot establish warning frequency.",
                        "Correctness snapshots and setup instrumentation are excluded from separate timing/profile processes.",
                        "Only final-update parameter/gradient snapshots are retained on CPU; ViT validation still needs substantial host memory.",
                        "Use profiler ranges to verify backward graph replay and synchronization; forward counters alone are insufficient.",
                        "CPU smoke mode validates orchestration/reporting only, not CUDA stream ownership.",
                        "The reference warning is not an independent warning-frequency experiment: it runs after the candidate. Compare independent cases for attribution.",
                        "gradient-clones off and retained graphs/outputs are intentional diagnostic controls, never proposed fixes."],
              "jobs": []}
    passes = ["diagnostic"] + (["timing"] if args.timing_steps else []) + (["profile"] if args.profile else [])
    for case in args.cases:
        for kind in passes:
            destination = invocation / case / kind
            destination.mkdir(parents=True)
            job_args = {**options, "pass": kind, "report_dir": str(destination)}
            print(f"Running {case}/{kind} ({args.ranks} rank(s))", file=sys.stderr, flush=True)
            job = launch_job(Path(__file__).resolve(), {"args": job_args, "case": case}, destination)
            report["jobs"].append(job)
            write_json(invocation / "report.json", report)
    report["summary"] = [case_summary(job) for job in report["jobs"]]
    report["status"] = "ok" if all(job["exit_code"] == 0 and all(row["status"] == "ok" for row in job["ranks"])
                                   for job in report["jobs"]) else "attention_required"
    write_json(invocation / "report.json", report)
    write_json(directory / "latest.json", {"report": str(invocation / "report.json"), "status": report["status"]})
    print(json.dumps({"report": str(invocation / "report.json"), "status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
