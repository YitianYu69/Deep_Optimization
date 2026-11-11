"""One fresh experiment session per worker: preparation, timing and diagnostics."""

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from hashlib import sha256
from pathlib import Path
import platform
import sys
import subprocess

from .catalog import family
from .matrix import CASES, incompatibility
from .measurement import (cpu_state, dense, measure, memory_diagnostics, phase_timings, profile,
                          snapshot, state_fingerprint, tensor_leaves)
from .workloads import (UnsupportedWorkload, adapter_state, build_source, functional_call)


def installed_build_fingerprint():
    import deepopt

    root = Path(deepopt.__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment(device):
    import torch
    import deepopt

    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "deepopt_path": str(Path(deepopt.__file__).resolve()), "device": str(device),
              "deepopt_source_sha256": installed_build_fingerprint()}
    for name in ("deep-optimization", "triton", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": properties.name, "capability": [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory}
        try:
            telemetry = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.sm,clocks.mem,power.limit",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            result["nvidia_smi_after_timing"] = telemetry.stdout.strip() if telemetry.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_smi_after_timing"] = None
    else:
        result["warning"] = "CPU/reference results validate behavior and reporting; they do not measure CUDA performance."
    return result


def module_family(module):
    name = type(module).__name__.lower()
    if "multiheadattention" in name:
        return "attention"
    if "linear" in name or "hfconv1d" in name:
        return "linear"
    if "conv" in name:
        return "convolution"
    if "norm" in name:
        return "normalization"
    if "pool" in name:
        return "pooling"
    if any(word in name for word in ("relu", "silu", "gelu", "sigmoid", "tanh")):
        return "activation"
    return None


class Session:
    def __init__(self, task, options):
        import torch
        import deepopt
        from deepopt.config import fp8_config_from_dict, activation_compression_config_from_dict
        from deepopt.opt.policy import linear_plan, prepare_fp8_model

        self.torch, self.deepopt, self.task = torch, deepopt, task
        self.device = torch.device(options["device"])
        self.spec, self.settings = task["workload"], deepcopy(task["settings"])
        self.runtime = self.settings["runtime"]
        self.fp8, self.compressed = CASES[task["case"]]
        self.native = task["case"] == "torch_bf16"
        self.seed = options["seed"]
        self.max_numerical_elements = options["max_numerical_elements"]
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(options["deterministic"])
        torch.backends.cudnn.benchmark = options["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = options["deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.utils, "deterministic"):
            torch.utils.deterministic.fill_uninitialized_memory = options["fill_uninitialized_memory"]
        deepopt.reset_config()
        reason = incompatibility(self.settings, task["case"], self.spec, options["device"])
        if reason:
            raise UnsupportedWorkload(reason)
        if self.device.type == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
            raise UnsupportedWorkload("An NVIDIA CUDA device is required")
        package = build_source(self.spec, self.device)
        self.model, self.inputs, self.kwargs = package["model"], package["inputs"], package["kwargs"]
        self.loss_fn = package.get("loss_fn")
        self.fetch, self.reset_source = package.get("fetch"), package.get("reset")
        if self.fetch is not None and self.reset_source is None:
            raise ValueError("A factory with fetch must also provide reset() for repeatable diagnostic passes")
        mode = self.spec["gradient_mode"]
        if mode != "all":
            for value in tensor_leaves(self.inputs):
                if value.is_floating_point():
                    value.requires_grad_(mode == "dx")
            count = 0
            for name, parameter in self.model.named_parameters():
                enabled = (mode == "dw" and "weight" in name) or (mode == "db" and "bias" in name)
                parameter.requires_grad_(enabled)
                count += enabled
            if mode in ("dw", "db") and not count:
                raise UnsupportedWorkload(f"{self.spec['op']} has no trainable targets for gradient mode {mode}")
        self.initial_fingerprint = state_fingerprint(self.model, self.inputs, (self.kwargs, package.get("fingerprint_data", ())))
        self.input_metadata = [{"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
                                "requires_grad": x.requires_grad, "device": str(x.device)}
                               for x in tensor_leaves(self.inputs)]
        fp8_options = {**self.settings["fp8"], "enabled": self.fp8}
        if not self.fp8:
            # Disabled FP8 knobs must not invalidate native controls in a Cartesian sweep.
            fp8_options = {"enabled": False, "backend": "reference", "intercept_linear": False,
                           "intercept_convolution": False, "intercept_normalization": False,
                           "intercept_pooling": False, "exclude_modules": self.settings["fp8"]["exclude_modules"]}
        intercept = self.runtime["binding"] == "intercept"
        for name in ("intercept_linear", "intercept_convolution", "intercept_normalization", "intercept_pooling"):
            if intercept and self.fp8:
                fp8_options[name] = True
        self.fp8_config = fp8_config_from_dict(fp8_options)
        self.compression = activation_compression_config_from_dict({**self.settings["compression"], "enabled": self.compressed})
        self.plan = linear_plan(self.fp8_config, activation_compression=self.compression)
        self.info, self.owner_state, self.weight_cache, self.handles = {}, None, None, []
        self.report, self.notes = None, []
        pipeline = self.spec["op"] in ("model", "linear_pipeline", "conv_pipeline", "attention_pipeline", "multihead_attention")
        self.binding = self.runtime["binding"]
        if pipeline and self.binding == "functional":
            self.binding = "prepared"
            self.notes.append("Model/pipeline/MHA uses prepared binding; standalone functional controls are not applied.")
            if self.runtime["output_dtype"] != "auto":
                raise UnsupportedWorkload("Pipeline output precision is owned by the planner; select output_dtype=auto")
        if not self.native and self.binding == "prepared":
            self.model, self.report = prepare_fp8_model(self.model, self.fp8_config, activation_compression=self.compression)
            # Custom functional pool/SDPA wrappers require interception for their forward body.
            if self.spec["op"] == "scaled_dot_product_attention" or family(self.spec["op"]) == "pooling":
                self.binding = "functional"
        if not self.native and self.binding == "functional":
            self.owner_state, self.weight_cache, _ = adapter_state(
                self.model, self.spec["op"], self.fp8_config, self.compression, self.device)
            self.call = self._functional
        else:
            self.call = self.model
        self.ablation = self.runtime["compression_families"]
        self.scope_compression = self.compression
        if self.ablation is not None:
            enabled = self.compressed and self.binding == "functional" and family(self.spec["op"]) in self.ablation
            self.scope_compression = activation_compression_config_from_dict({**asdict(self.compression), "enabled": enabled})
            if self.binding == "functional":
                self.plan = linear_plan(self.fp8_config, activation_compression=self.scope_compression)
        if self.ablation is not None and self.compressed and not self.native:
            self._install_ablation()
        self.raw_inputs = self.inputs
        self.precomputed = None
        if self.runtime["input_dtype"] == "fp8" and self.fp8:
            if self.spec["op"] in ("scaled_dot_product_attention", "multihead_attention"):
                raise UnsupportedWorkload("Native attention requires BF16 Q/K/V; use attention_pipeline")
            if self.runtime["input_scaling"] == "precomputed":
                self.precomputed = self._fp8_inputs(self.raw_inputs, detach=True)
                self.inputs = self.precomputed
        if self.runtime["output_dtype"] == "fp8" and self.fp8 and self.spec["op"] in (
                "linear", "hf_conv1d", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                "scaled_dot_product_attention"):
            raise UnsupportedWorkload("Explicit FP8 output is supported by activation/norm/pool functions; projection outputs use fp8.out_dtype")
        if self.runtime["output_scaling"] == "provided" and self.fp8:
            self._make_provided_scale()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = None
        if self.runtime["optimizer"] != "none":
            if not trainable:
                raise UnsupportedWorkload("Optimizer timing requires trainable parameters")
            kind = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[self.runtime["optimizer"]]
            self.optimizer = kind(trainable, lr=self.runtime["lr"], **self.runtime["optimizer_options"])
        if self.runtime["execution"] == "compile":
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            self.call = torch.compile(self.call, **self.runtime["compile"])
        # CPU restore copies stay outside measured GPU memory and timed regions.
        self.initial_state = cpu_state(self.model.state_dict())
        self.owner_initial = cpu_state(self.owner_state.state_dict()) if self.owner_state is not None else None
        self.optimizer_initial = deepcopy(self.optimizer.state_dict()) if self.optimizer is not None else None
        self.rng_cpu = torch.random.get_rng_state()
        self.rng_cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        self.last_output = None
        self.graph, self.graph_output = None, None

    def _fp8_inputs(self, inputs, detach=False):
        from deepopt.opt.quantization.activation_tensor import to_fp8_activation

        dtype = {"e4m3fn": self.torch.float8_e4m3fn, "e5m2": self.torch.float8_e5m2}[self.fp8_config.activation_dtype]
        result = []
        layout = "channels" if family(self.spec["op"]) in ("convolution", "normalization", "pooling") and self.spec["op"] != "layer_norm" else "last"
        for value in inputs:
            if not isinstance(value, self.torch.Tensor) or not value.is_floating_point():
                result.append(value)
                continue
            wrapped = to_fp8_activation(value, layout=layout, fp8_dtype=dtype,
                                        block_shape=self.fp8_config.block_shapes["activation"])
            result.append(wrapped.detach().requires_grad_(value.requires_grad) if detach else wrapped)
        return tuple(result)

    def _make_provided_scale(self):
        shape, op = self.spec["shape"], self.spec["op"]
        if family(op) == "pooling":
            from deepopt.opt.nn.pooling.geometry import geometry
            from deepopt.opt.nn.pooling.spec import SPECS
            options = dict(self.model.options)
            if self.model.indices is not None:
                options["indices"] = self.model.indices
            if self.model.samples is not None:
                options["_random_samples"] = self.model.samples
            base = op.removesuffix("_with_indices")
            options = SPECS[base].arguments((), options)
            raw = self.raw_inputs[0]
            channels = self.runtime["input_dtype"] == "fp8" or (
                raw.ndim in (4, 5) and raw.is_contiguous(memory_format=self.torch.channels_last if raw.ndim == 4 else self.torch.channels_last_3d))
            g = geometry(base, raw, options, "channels" if channels else "last")
            rows, cols = g.transport.matrix_shape
        else:
            rows, cols = __import__("math").prod(shape[:-1]), shape[-1]
        block = self.fp8_config.block_shapes["activation"]
        grid = (1, 1) if block is None else ((rows + block[0] - 1) // block[0], (cols + block[1] - 1) // block[1])
        self.runtime["_provided_scale"] = self.torch.ones(grid, device=self.device, dtype=self.torch.float32)

    def _functional(self, *inputs, **kwargs):
        if kwargs:
            raise ValueError("Standalone functional workloads do not accept runtime keyword inputs")
        return functional_call(self.model, self.spec["op"], inputs, self.plan, self.runtime,
                               self.owner_state, self.info, self.weight_cache)

    def _install_ablation(self):
        if self.binding == "functional":
            return
        stacks = {}
        for module in self.model.modules():
            kind = module_family(module)
            if kind is None:
                continue
            compression = {**asdict(self.compression), "enabled": kind in self.ablation}
            owned = getattr(module, "plan", None)
            if isinstance(owned, dict):
                # Explicit native plans own compression and intentionally ignore scope overrides.
                from deepopt.opt.nn.functional import _prepare_linear_plan
                if "fp8_training" in owned:
                    selected = {**owned, "fp8_training": {**owned["fp8_training"], "activation_compression": compression}}
                else:
                    selected = {**owned, "activation_compression": compression}
                module.plan = _prepare_linear_plan(selected)

            def before(current, args, compression=compression):
                context = self.deepopt.autocast(self.device.type, enabled=True, dtype=self.torch.bfloat16,
                                               fp8=None, activation_compression=compression,
                                               check_finite=self.runtime["check_finite"],
                                               cache_enabled=self.runtime["cache_enabled"])
                context.__enter__()
                stacks.setdefault(id(current), []).append(context)

            def after(current, args, output):
                contexts = stacks.get(id(current), [])
                if contexts:
                    contexts.pop().__exit__(None, None, None)

            self.handles.extend((module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after, always_call=True)))
        self.notes.append("Family ablations use module plans/scopes (MHA includes its internal projections); hooks add submission overhead. "
                          "Functional calls outside recognized module scopes are disabled during ablations. "
                          "Compare family selections with an all-family selection to control for this overhead.")

    @contextmanager
    def context(self):
        torch = self.torch
        with ExitStack() as stack:
            if self.runtime["sdpa_backend"] != "auto":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                backend = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                           "efficient": SDPBackend.EFFICIENT_ATTENTION}[self.runtime["sdpa_backend"]]
                stack.enter_context(sdpa_kernel(backend))
            if self.native:
                stack.enter_context(torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                   cache_enabled=self.runtime["cache_enabled"]))
            else:
                stack.enter_context(self.deepopt.autocast(
                    self.device.type, dtype=torch.bfloat16, fp8=self.fp8_config,
                    activation_compression=self.scope_compression, check_finite=self.runtime["check_finite"],
                    cache_enabled=self.runtime["cache_enabled"]))
                stack.enter_context(self.deepopt.fp8_scaling_scope())
            yield

    def clear(self):
        # Zero allocated buffers consistently; graph replay must retain stable addresses.
        for value in (*tensor_leaves(self.inputs), *tensor_leaves(self.raw_inputs), *self.model.parameters()):
            if value.grad is not None:
                value.grad.zero_()

    def reset(self):
        self.last_output = None
        self.model.load_state_dict(self.initial_state)
        if self.owner_state is not None:
            self.owner_state.load_state_dict(self.owner_initial)
        if self.weight_cache is not None:
            self.weight_cache.clear()
        if self.optimizer is not None:
            self.optimizer.load_state_dict(deepcopy(self.optimizer_initial))
        self.torch.random.set_rng_state(self.rng_cpu)
        if self.rng_cuda is not None:
            self.torch.cuda.set_rng_state(self.rng_cuda, self.device)
        if self.reset_source is not None:
            self.reset_source()
        self.clear()

    def forward(self):
        if self.fetch is not None:
            self.inputs, self.kwargs = self.fetch()
            self.inputs = tuple(self.inputs)
            self.raw_inputs = self.inputs
        inputs = self.inputs
        if self.runtime["input_dtype"] == "fp8" and self.fp8 and self.precomputed is None:
            inputs = self._fp8_inputs(inputs)
        with self.context():
            if self.runtime["checkpoint"]:
                if self.native:
                    from torch.utils.checkpoint import checkpoint
                    output = checkpoint(self.call, *inputs, **self.kwargs, use_reentrant=False)
                else:
                    output = self.deepopt.fp8_checkpoint(self.call, *inputs, **self.kwargs, use_reentrant=False)
            else:
                output = self.call(*inputs, **self.kwargs)
            if self.loss_fn is not None:
                loss = self.loss_fn(output)
            else:
                values = [dense(value).float() for value in tensor_leaves(output) if value.is_floating_point()]
                if not values:
                    raise ValueError("Workload produced no floating-point outputs")
                loss = sum(value.square().mean() for value in values)
        return output, loss

    def backward(self, loss):
        if loss.requires_grad:
            loss.backward()

    def step(self, annotate=False, optimizer=True):
        self.last_output = None
        self.clear()
        mark = self.torch.profiler.record_function if annotate else lambda _: nullcontext()
        accumulation = self.runtime["accumulation_steps"]
        for _ in range(accumulation):
            with mark("analysis.forward_loss"):
                output, loss = self.forward()
            with mark("analysis.backward"):
                self.backward(loss / accumulation)
        if optimizer and self.optimizer is not None:
            with mark("analysis.optimizer"):
                self.optimizer.step()
        # Retain output buffers, not the entire autograd graph between updates.
        self.last_output = tuple(value.detach() for value in tensor_leaves(output))
        return self.last_output

    def graph_step(self):
        if self.graph is None:
            if self.fetch is not None:
                raise UnsupportedWorkload("CUDA Graph replay needs static resident inputs; factory fetch is dynamic")
            if self.ablation is not None:
                raise UnsupportedWorkload("Python module ablation scopes require eager execution")
            stream = self.torch.cuda.Stream(device=self.device)
            stream.wait_stream(self.torch.cuda.current_stream(self.device))
            with self.torch.cuda.stream(stream):
                for _ in range(3):
                    self.step(optimizer=False)
            self.torch.cuda.current_stream(self.device).wait_stream(stream)
            self.torch.cuda.synchronize(self.device)
            self.graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(self.graph, stream=stream):
                self.graph_output = self.step(optimizer=False)
            self.notes.append("Graph replay captures gradient clearing/forward/backward; optimizer runs outside capture. "
                              "Phase, correctness, memory and annotated profiles exercise the eager callable separately.")
        self.graph.replay()
        if self.optimizer is not None:
            self.optimizer.step()
        return self.graph_output

    def execution_info(self):
        from deepopt.opt.precision import precision_report

        modules = {}
        for name, module in self.model.named_modules():
            info = getattr(module, "execution_info", None)
            if info is not None:
                modules[name or "<root>"] = info
        counters = {}
        if self.runtime["execution"] == "compile":
            counters = {str(key): dict(value) for key, value in self.torch._dynamo.utils.counters.items()}
        inactive = []
        if not self.fp8:
            inactive.append("FP8 compute knobs are disabled; native control policy is normalized independently")
        if not self.compressed:
            inactive.append("compression bits/group/attention/recenter knobs (compression disabled)")
        if self.binding == "functional":
            inactive.append("module tracing/exclusion settings (standalone functional call)")
            if self.fp8_config.backward_precision == "auto":
                inactive.append("planned gradient transport requires prepared consumer edges; this call has no model planner")
        if self.fp8_config.scaling_recipe != "delayed":
            inactive.append("delayed history/amax/margin settings (current scaling)")
        if family(self.spec["op"]) == "pooling" and "lp_pool" not in self.spec["op"]:
            inactive.append("lossy compression bits/group (routing is lossless; average pooling stores geometry)")
        if self.spec["op"] == "scaled_dot_product_attention":
            inactive.append("FP8 GEMM settings (standalone SDPA uses native Q/K/V computation)")
        return {"binding": self.binding, "functional": dict(self.info), "modules": modules,
                "not_applicable": inactive,
                "precision_boundary": {"requested_input": self.runtime["input_dtype"],
                                       "fp8_input_enabled": self.fp8 and self.runtime["input_dtype"] == "fp8",
                                       "input_scaling": self.runtime["input_scaling"],
                                       "requested_output": self.runtime["output_dtype"],
                                       "effective_scope_compression": asdict(self.scope_compression)},
                "effective_fp8_policy": asdict(self.fp8_config),
                "transformation": asdict(self.report) if self.report is not None else None,
                "precision_plan": precision_report(self.model) if not self.native else None,
                "compile_counters": counters,
                "effective_attention_bits": self.compression.attention_bits or max(4, self.compression.bits),
                "activation_compression_block_override": self.fp8_config.block_shapes["activation"] if self.fp8 else None,
                "notes": self.notes}

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_task(task, options, directory):
    import torch

    session = Session(task, options)
    try:
        numerical = snapshot(session)
        recorded = []
        if options["capture_workloads"] and session.native:
            from .capture import record_workloads
            recorded = record_workloads(session)
        # Complete diagnostics before graph capture creates private allocator pools.
        phases = phase_timings(session, options["phase_steps"]) if options["phase_steps"] else None
        memory = memory_diagnostics(session) if options["memory"] else None
        tracing = profile(session, directory, options["profile_steps"]) if options["profile_steps"] else None
        session.reset()
        timing_call = session.graph_step if session.runtime["execution"] == "graph" else session.step
        timing = measure(timing_call, torch, session.device, options["warmup"], options["iterations"], options["repeats"])
        finite_after_timing = (all(bool(torch.isfinite(dense(value)).all()) for value in (session.last_output or ()) if value.is_floating_point())
                               and all(bool(torch.isfinite(parameter).all()) for parameter in session.model.parameters()))
        # Capture effective metadata before reset discards cache observations.
        execution = session.execution_info()
        snapshot_path = directory / "snapshot.pt"
        torch.save(numerical, snapshot_path)
        return {"status": "ok", "initial_state_sha256": session.initial_fingerprint,
                "environment": environment(session.device), "inputs": session.input_metadata,
                "timing": timing, "phases": phases, "memory": memory, "profile": tracing,
                "finite_after_timing": finite_after_timing,
                "numerical_snapshot": session.snapshot_metadata,
                "execution": execution, "snapshot": str(snapshot_path),
                "recorded_workloads": recorded,
                "timing_unit": "optimizer update" if session.optimizer is not None else "forward/backward update",
                "phase_unit": "single microbatch forward/loss and backward; excludes optimizer and accumulation loop",
                "accumulation_steps": session.runtime["accumulation_steps"],
                "diagnostic_note": "Numerical snapshot uses one fresh forward/backward (one microbatch), before timing. "
                                   "Phase/memory/profile passes are separate and do not contribute to throughput timing."}
    finally:
        session.close()
