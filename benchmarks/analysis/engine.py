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
import os
import json
from fnmatch import fnmatchcase
from time import perf_counter

from .catalog import family
from .matrix import CASES, incompatibility, bf16_policy_identity, numerical_recipe_identity
from .measurement import (cpu_state, dense, measure, memory_diagnostics, phase_timings, profile,
                          snapshot, state_fingerprint, tensor_leaves, memory_fill_scope, latency_diagnostics, quality_diagnostics)
from .workloads import (UnsupportedWorkload, adapter_state, build_source, functional_call)


def installed_build_fingerprint():
    import deepopt

    root = Path(deepopt.__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resume_environment_fingerprint(device, *, visible_devices=...):
    identity = {"device": device, "python": sys.version, "platform": platform.platform(),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is ... else visible_devices,
                "allocator_environment": {key: os.environ.get(key) for key in
                                          ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}
    identity["math_environment"] = {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "TRITON_DEFAULT_FP_FUSION",
                                  "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_LAUNCH_BLOCKING")}
    for name in ("torch", "triton", "timm", "torchvision", "transformers", "flash-attn"):
        try:
            identity[name] = version(name)
        except PackageNotFoundError:
            identity[name] = None
    if device.startswith("cuda"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5)
            identity["gpu_driver"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            identity["gpu_driver"] = "unavailable"
        if not identity["gpu_driver"] or identity["gpu_driver"] == "unavailable":
            # Do not mix CUDA timings across unidentified devices/drivers.
            identity["unverified_run_nonce"] = os.urandom(16).hex()
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def environment(device):
    import torch
    import deepopt

    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "deepopt_path": str(Path(deepopt.__file__).resolve()), "device": str(device),
              "deepopt_source_sha256": installed_build_fingerprint(),
              "allocator_environment": {key: os.environ.get(key) for key in
                                        ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}}
    result["mps_environment"]={key:os.environ.get(key) for key in
        ("CUDA_MPS_PIPE_DIRECTORY","CUDA_MPS_LOG_DIRECTORY","CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}
    for name in ("deep-optimization", "triton", "transformers", "torchvision", "timm", "flash-attn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
        result["gpu"] = {"name": properties.name, "capability": [properties.major, properties.minor],
                         "total_memory_bytes": properties.total_memory}
        try:
            telemetry = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.sm,clocks.mem,power.limit",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
            result["nvidia_smi_after_timing"] = telemetry.stdout.strip() if telemetry.returncode == 0 else None
            modes=subprocess.run(["nvidia-smi","--query-gpu=uuid,compute_mode,mig.mode.current","--format=csv,noheader"],
                                 capture_output=True,text=True,timeout=5)
            result["gpu_compute_and_mig_modes"]=modes.stdout.strip() if modes.returncode==0 else None
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
        from deepopt._bf16_policy import resolve_bf16_policy

        self.requested_bf16_policy = resolve_bf16_policy(self.runtime.get("bf16_policy", "torch_amp"))
        self.bf16_policy = resolve_bf16_policy(bf16_policy_identity(self.settings, task["case"]))
        self.numerical_recipe = numerical_recipe_identity(self.settings, task["case"])
        if "numerical_recipe" in task and task["numerical_recipe"] != self.numerical_recipe:
            raise ValueError("Benchmark numerical recipe changed before model allocation")
        self.policy_config = {"runtime": {"bf16_policy": self.bf16_policy.name}}
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
        self.benchmark_metadata=package.get("benchmark_metadata",{})
        self.epoch_updates,self.epoch_state=package.get("epoch_updates"),package.get("epoch_state")
        self.loss_fn = package.get("loss_fn")
        self.validate = package.get("validate")
        if self.validate is not None and not callable(self.validate):
            raise ValueError("factory validate must be callable")
        self.fetch, self.reset_source = package.get("fetch"), package.get("reset")
        self.after_step = package.get("after_step")
        if (self.fetch is not None or self.after_step is not None) and self.reset_source is None:
            raise ValueError("A factory with fetch/after_step must provide reset() for repeatable diagnostic passes")
        self.model.train(self.spec["training"])
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
        warmed_states = self._warm_native_optimizer()
        self.optimizer_baseline_fingerprint = (state_fingerprint(self.model, self.inputs,
            {name: warmed_states[id(p)] for name, p in self.model.named_parameters() if id(p) in warmed_states})
            if warmed_states else None)
        self.initial_fingerprint = state_fingerprint(self.model, self.inputs, (self.kwargs, package.get("fingerprint_data", ())))
        self.initial_component_fingerprints={
            "parameters":state_fingerprint(torch.nn.Identity(),tuple(self.model.parameters()),()),
            "buffers":state_fingerprint(torch.nn.Identity(),tuple(self.model.buffers()),()),
            "inputs":state_fingerprint(torch.nn.Identity(),self.inputs,()),
            "factory_data":state_fingerprint(torch.nn.Identity(),(),(self.kwargs,package.get("fingerprint_data",()))) }
        # FX/preparation may execute stochastic Python. Every feature case must
        # probe dropout/stochastic depth from the same pre-preparation state.
        baseline_rng_cpu = torch.random.get_rng_state()
        baseline_rng_cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
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
        self.plan = linear_plan(self.fp8_config, activation_compression=self.compression, config=self.policy_config)
        self.info, self.owner_state, self.weight_cache, self.handles = {}, None, None, []
        self.report, self.notes = None, []
        pipeline = self.spec["op"] in ("model", "linear_pipeline", "conv_pipeline", "attention_pipeline", "multihead_attention")
        self.binding = self.runtime["binding"]
        if not self.native and self.spec["op"] in ("mul", "pad"):
            # These wrappers own explicit functional boundaries; an unconverted
            # generic module must not silently substitute the Torch control.
            if self.binding != "functional":
                self.notes.append(f"{self.spec['op']} uses its explicit functional binding.")
            self.binding = "functional"
        if pipeline and self.binding == "functional":
            self.binding = "prepared"
            self.notes.append("Model/pipeline/MHA uses prepared binding; standalone functional controls are not applied.")
            if self.runtime["output_dtype"] != "auto":
                raise UnsupportedWorkload("Pipeline output precision is owned by the planner; select output_dtype=auto")
        if not self.native and self.binding == "prepared":
            self.model, self.report = prepare_fp8_model(self.model, self.fp8_config, activation_compression=self.compression,
                config={"runtime": {**self.policy_config["runtime"],
                                    "attention": {"backend": self.runtime["attention_backend"]}}})
            # Custom functional pool/SDPA wrappers require interception for their forward body.
            if self.spec["op"] == "scaled_dot_product_attention" or family(self.spec["op"]) == "pooling":
                self.binding = "functional"
        if not self.native and self.binding == "functional":
            self.owner_state, self.weight_cache, _ = adapter_state(
                self.model, self.spec["op"], self.fp8_config, self.compression, self.device)
            self.call = self._functional
        else:
            self.call = self.model
        self.selective_compression = (self.runtime["compression_bits_by_family"] is not None
                                      or bool(self.runtime["compression_exclude_modules"]))
        self.ablation = self.runtime["compression_families"]
        if self.selective_compression and self.ablation is None:
            self.ablation = ["linear", "convolution", "normalization", "activation", "pooling", "attention", "binary", "padding"]
        self.module_compression_policies = {}
        self.scope_compression = self.compression
        if self.ablation is not None:
            enabled = self.compressed and self.binding == "functional" and family(self.spec["op"]) in self.ablation
            policy = self._family_compression(family(self.spec["op"]), "<root>")
            self.scope_compression = activation_compression_config_from_dict({**policy, "enabled": enabled and policy["enabled"]})
            if self.binding == "functional":
                self.plan = linear_plan(self.fp8_config, activation_compression=self.scope_compression,
                                        config=self.policy_config)
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
            if warmed_states:
                owned = self.optimizer.state_dict()
                for actual, saved in zip(self.optimizer.param_groups, owned["param_groups"]):
                    for parameter, index in zip(actual["params"], saved["params"]):
                        if id(parameter) not in warmed_states:
                            raise UnsupportedWorkload("Prepared parameters cannot be matched to the shared native optimizer baseline")
                        owned["state"][index] = warmed_states[id(parameter)]
                self.optimizer.load_state_dict(owned)
        if self.runtime["execution"] == "compile":
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            if self.runtime["compile"]["backend"] == "deepopt" and self.call is self.model:
                from deepopt.engine.execution.compilation import compile_model

                compile_model(self.model, {"enabled": True, **self.runtime["compile"]}, config={})
            else:
                compile_options = dict(self.runtime["compile"])
                if not compile_options.get("options"):
                    compile_options.pop("options", None)
                self.call = torch.compile(self.call, **compile_options)
        # CPU restore copies stay outside measured GPU memory and timed regions.
        self.initial_state = cpu_state(self.model.state_dict())
        self.owner_initial = cpu_state(self.owner_state.state_dict()) if self.owner_state is not None else None
        self.optimizer_initial = cpu_state(self.optimizer.state_dict()) if self.optimizer is not None else None
        self.rng_cpu, self.rng_cuda = baseline_rng_cpu, baseline_rng_cuda
        rng_digest = sha256(bytes(baseline_rng_cpu.tolist()))
        if baseline_rng_cuda is not None:
            rng_digest.update(bytes(baseline_rng_cuda.tolist()))
        self.rng_fingerprint = rng_digest.hexdigest()
        self.last_output = None
        self.graph, self.graph_output = None, None
        self.graph_capture_count, self.graph_replay_count = 0, 0
        self._amp_context = self._make_amp_context() if self.runtime["scope_lifetime"] == "reused" else None

    def _warm_native_optimizer(self):
        """Every case starts from the same BF16 model/moments, before preparation."""
        count = self.runtime["optimizer_warmup_steps"]
        if not count:
            return {}
        if self.fetch is not None:
            raise UnsupportedWorkload("Shared optimizer warmup requires static resident inputs")
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise UnsupportedWorkload("Shared optimizer warmup requires trainable parameters")
        kind = {"sgd": self.torch.optim.SGD, "adamw": self.torch.optim.AdamW}[self.runtime["optimizer"]]
        optimizer = kind(parameters, lr=self.runtime["lr"], **self.runtime["optimizer_options"])
        for _ in range(count):
            optimizer.zero_grad(set_to_none=True)
            with self._native_amp_context(cache_enabled=False):
                output = self.model(*self.inputs, **self.kwargs)
                loss = (self.loss_fn(output) if self.loss_fn is not None else
                        sum(value.float().square().mean() for value in tensor_leaves(output) if value.is_floating_point()))
            loss.backward()
            optimizer.step()
        states = {id(p): cpu_state(optimizer.state[p]) for p in parameters}
        optimizer.zero_grad(set_to_none=True)
        for value in tensor_leaves(self.inputs):
            value.grad = None
        return states

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

    def _family_compression(self, kind, name):
        policy = {**asdict(self.compression), "enabled": self.compressed and (self.ablation is None or kind in self.ablation)}
        overrides = self.runtime["compression_bits_by_family"] or {}
        if kind in overrides:
            bits = overrides[kind]
            policy["enabled"] = policy["enabled"] and bits is not None
            if bits is not None:
                policy["bits"] = bits
                if kind == "attention":
                    policy["attention_bits"] = bits
        if any(fnmatchcase(name, pattern) for pattern in self.runtime["compression_exclude_modules"]):
            policy["enabled"] = False
        self.module_compression_policies[name] = dict(policy)
        return policy

    def _install_ablation(self):
        if self.binding == "functional":
            return
        stacks = {}
        for name, module in self.model.named_modules():
            kind = module_family(module)
            if kind is None:
                continue
            compression = self._family_compression(kind, name or "<root>")
            owned = getattr(module, "plan", None)
            if isinstance(owned, dict):
                # Explicit native plans own compression and intentionally ignore scope overrides.
                from deepopt.opt.nn.functional import _prepare_linear_plan
                if "fp8_training" in owned:
                    selected = {**owned, "fp8_training": {**owned["fp8_training"], "activation_compression": compression}}
                else:
                    selected = {**owned, "activation_compression": compression}
                module.plan = _prepare_linear_plan(selected)

            context = self.deepopt.autocast(self.device.type, enabled=True, dtype=self.torch.bfloat16,
                                           fp8=self.fp8_config if self.fp8 else None, activation_compression=compression,
                                           check_finite=self.runtime["check_finite"],
                                           cache_enabled=self.runtime["cache_enabled"],
                                           bf16_policy=self.bf16_policy.name)
            def before(current, args, context=context):
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
    def _native_amp_context(self, *, cache_enabled=None):
        from deepopt._bf16_policy import bf16_policy_scope, suspend_bf16_rewrites

        with bf16_policy_scope("torch_amp", self.device.type), suspend_bf16_rewrites(self.device.type), \
                self.torch.autocast(self.device.type, dtype=self.torch.bfloat16,
                    cache_enabled=self.runtime["cache_enabled"] if cache_enabled is None else cache_enabled):
            yield

    def _make_amp_context(self):
        if self.native:
            return self._native_amp_context()
        return self.deepopt.autocast(
            self.device.type, dtype=self.torch.bfloat16, fp8=self.fp8_config,
            activation_compression=self.scope_compression, check_finite=self.runtime["check_finite"],
            cache_enabled=self.runtime["cache_enabled"], attention_backend=self.runtime["attention_backend"],
            bf16_policy=self.bf16_policy.name)

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            if not self.native and self.runtime["convolution_kernel_variant"] != "auto":
                from deepopt.opt.backends._shared.convolution import convolution_kernel_scope
                stack.enter_context(convolution_kernel_scope(self.runtime["convolution_kernel_variant"]))
            if not self.native and self.runtime["norm_kernel_variant"] != "auto":
                from deepopt.opt.backends._shared.normalization import normalization_kernel_scope
                stack.enter_context(normalization_kernel_scope(self.runtime["norm_kernel_variant"],
                    row_chunk=self.runtime["norm_row_chunk"], max_programs=self.runtime["norm_max_programs"]))
            if not self.native and self.runtime["pool_kernel_variant"]!="auto":
                from deepopt.opt.backends._shared.pooling import pooling_kernel_scope
                stack.enter_context(pooling_kernel_scope(self.runtime["pool_kernel_variant"]))
            if not self.native and (self.runtime["codec_pack_variant"] != "auto"
                                    or self.runtime["spatial_activation_layout"] != "auto"
                                    or self.runtime["codec_num_warps"] != 4
                                    or self.runtime["activation_pointwise_tile"] != 512
                                    or self.runtime["producer_pack"] != "off"
                                    or self.runtime["activation_layout"] != "auto"
                                    or self.runtime["strided_indexing"] != "wide"
                                    or self.runtime["convolution_codec"] != "auto"
                                    or self.runtime["attention_pack"] != "separate"
                                    or self.runtime["attention_grad_layout"] != "auto"
                                    or self.runtime["gelu_saved_state"] != "auto"
                                    or self.runtime["silu_saved_state"] != "derivative"):
                from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
                stack.enter_context(activation_codec_scope(self.runtime["codec_pack_variant"],
                    groups_per_program=self.runtime["codec_groups_per_program"],
                    spatial_layout=self.runtime["spatial_activation_layout"],
                    num_warps=self.runtime["codec_num_warps"], pointwise_tile=self.runtime["activation_pointwise_tile"],
                    producer_pack=self.runtime["producer_pack"], gelu_state=self.runtime["gelu_saved_state"],
                    strided_index=self.runtime["strided_indexing"],convolution_codec=self.runtime["convolution_codec"],
                    attention_pack=self.runtime["attention_pack"], attention_grad_layout=self.runtime["attention_grad_layout"],
                    **({"silu_state": self.runtime["silu_saved_state"]} if self.runtime["silu_saved_state"] != "derivative" else {}),
                    **({"activation_layout":self.runtime["activation_layout"]} if self.runtime["activation_layout"] != "auto" else {})))
            if not self.native and (self.runtime["linear_dw_backend"] != "inherit" or self.runtime["native_operand_layout"] != "preserve"
                                    or self.runtime["linear_forward_backend"] != "inherit" or self.runtime["linear_gradient_reduction"] != "separate"):
                from deepopt.opt.backends._shared.gemm_policy import linear_backward_scope
                stack.enter_context(linear_backward_scope(dw_backend=self.runtime["linear_dw_backend"],
                    native_layout=self.runtime["native_operand_layout"],forward_backend=self.runtime["linear_forward_backend"],
                    gradient_reduction=self.runtime["linear_gradient_reduction"]))
            if self.runtime["sdpa_backend"] != "auto":
                from torch.nn.attention import SDPBackend, sdpa_kernel
                backend = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                           "efficient": SDPBackend.EFFICIENT_ATTENTION}[self.runtime["sdpa_backend"]]
                stack.enter_context(sdpa_kernel(backend))
            stack.enter_context(self._amp_context if self._amp_context is not None else self._make_amp_context())
            if self.fp8 and self.fp8_config.scaling_recipe == "delayed":
                # Histories need a fresh batch lifetime. Current scaling and
                # disabled controls have no delayed work to collect/commit.
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

    def step(self, annotate=False, optimizer=True, *, return_loss=False):
        self.last_output = None
        self.clear()
        mark = self.torch.profiler.record_function if annotate else lambda _: nullcontext()
        accumulation = self.runtime["accumulation_steps"]
        losses = [] if return_loss else None
        for _ in range(accumulation):
            with mark("analysis.forward_loss"):
                output, loss = self.forward()
                if return_loss:
                    losses.append(loss.detach())
            with mark("analysis.backward"):
                self.backward(loss / accumulation)
        if optimizer and self.optimizer is not None:
            with mark("analysis.optimizer"):
                self.optimizer.step()
        if optimizer and self.after_step is not None:
            with mark("analysis.after_step"):
                self.after_step()
        # Retain output buffers, not the entire autograd graph between updates.
        self.last_output = tuple(value.detach() for value in tensor_leaves(output))
        return (self.last_output, self.torch.stack(losses).mean()) if return_loss else self.last_output

    def graph_step(self):
        initializing = self.graph is None
        if initializing:
            capture_started = perf_counter()
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
            self.graph_capture_count = getattr(self, "graph_capture_count", 0) + 1
            self.notes.append("Graph replay captures gradient clearing/forward/backward; optimizer runs outside capture. "
                              "Phase, correctness, memory and annotated profiles exercise the eager callable separately.")
        self.graph.replay()
        self.graph_replay_count = getattr(self, "graph_replay_count", 0) + 1
        if initializing:
            self.torch.cuda.synchronize(self.device)
            self.capture_setup_ms = (perf_counter() - capture_started) * 1000
        if self.optimizer is not None:
            self.optimizer.step()
        if self.after_step is not None:
            self.after_step()
        self.last_output = self.graph_output
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
        convolutions = [v for v in modules.values() if v.get("fp8_requested")]
        convolution_coverage = {"requested_modules": len(convolutions),
            "fp8_compute_modules": sum(v.get("compute_precision") == "fp8" for v in convolutions),
            "native_compute_modules": sum(v.get("compute_precision") == "native" for v in convolutions),
            "fp8_storage_fp32_modules": sum(v.get("compute_precision") == "fp8_storage_fp32" for v in convolutions),
            "estimated_forward_flops": sum(v.get("estimated_forward_flops", 0) for v in convolutions),
            "estimated_fp8_forward_flops": sum(v.get("estimated_forward_flops", 0) for v in convolutions
                                                if v.get("compute_precision") == "fp8"),
            "note": "Convolution-only coverage of observed module dispatch, not whole-model FP8 coverage."}
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
        if self.spec["op"] in ("mul", "pad"):
            inactive.append("FP8 GEMM settings (standalone mul/pad preserve native arithmetic)")
        if self.spec["op"] == "pad":
            inactive.append("lossy compression bits/group (padding retains geometry without tensor values)")
        compiler_report = getattr(self.model, "_deepopt_compile_report", None)
        return {"binding": self.binding, "functional": dict(self.info), "modules": modules,
                "bf16_policy": self.bf16_policy.as_dict(),
                "numerical_recipe": self.numerical_recipe,
                "requested_bf16_policy": self.requested_bf16_policy.as_dict(),
                "native_torch_ops_only": self.native,
                "compiler_report": compiler_report.as_dict() if compiler_report is not None else None,
                "cuda_graph": {"captures": self.graph_capture_count, "replays": self.graph_replay_count,
                               "optimizer_captured": False, "ema_captured": False},
                "convolution_coverage": convolution_coverage,
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
                "factory_after_step": self.after_step is not None,
                "scope_lifetime": self.runtime["scope_lifetime"],
                "numerical_probe": self.runtime["numerical_probe"],
                "norm_kernel_variant": self.runtime["norm_kernel_variant"],
                "codec_pack_variant": self.runtime["codec_pack_variant"],
                "codec_groups_per_program": self.runtime["codec_groups_per_program"],
                "spatial_activation_layout": self.runtime["spatial_activation_layout"],
                "codec_num_warps": self.runtime["codec_num_warps"],
                "activation_pointwise_tile": self.runtime["activation_pointwise_tile"],
                "producer_pack": self.runtime["producer_pack"], "gelu_saved_state": self.runtime["gelu_saved_state"],
                "silu_saved_state": self.runtime["silu_saved_state"],
                "activation_layout": self.runtime["activation_layout"],
                "strided_indexing": self.runtime["strided_indexing"],
                "convolution_codec":self.runtime["convolution_codec"],
                "convolution_kernel_variant":self.runtime["convolution_kernel_variant"],
                "attention_backend":self.runtime["attention_backend"], "attention_pack":self.runtime["attention_pack"],
                "attention_grad_layout": self.runtime["attention_grad_layout"],
                "attention_preparation":getattr(self.model, "_deepopt_attention_report", None),
                "pool_kernel_variant":self.runtime["pool_kernel_variant"],
                "linear_dw_backend": self.runtime["linear_dw_backend"], "native_operand_layout": self.runtime["native_operand_layout"],
                "linear_forward_backend":self.runtime["linear_forward_backend"], "linear_gradient_reduction":self.runtime["linear_gradient_reduction"],
                "module_compression_policies": self.module_compression_policies,
                "effective_attention_bits": self.compression.attention_bits or max(4, self.compression.bits),
                "activation_compression_block_override": self.fp8_config.block_shapes["activation"] if self.fp8 else None,
                "notes": self.notes}

    def close(self):
        for handle in self.handles:
            handle.remove()


def run_task(task, options, directory):
    # Fresh compiler workers save the code that produced this row's timings.
    # Artifact copying/hashing runs after all measurement/profile passes.
    try:
        from compiler.inductor_benchmark_artifacts import InductorBenchmarkArtifacts
    except ModuleNotFoundError as error:
        if error.name != "compiler":
            raise
        from benchmarks.compiler.inductor_benchmark_artifacts import InductorBenchmarkArtifacts

    with InductorBenchmarkArtifacts(directory, task) as artifacts:
        try:
            row = _run_task(task, options, directory)
        except BaseException as error:
            artifacts.finish(error=error)
            raise
        saved = artifacts.finish(row=row)
        if saved is not None:
            row["inductor_artifacts"] = saved
            row.setdefault("execution", {})["inductor_artifacts"] = {
                key: saved[key] for key in ("status", "reference_valid", "manifest", "triton_generation") if key in saved}
            # Update the resumable scalar core after its code inventory exists.
            from .reporting import write_json
            write_json(Path(directory) / "core-result.json", {**row, "task": task})
        return row


def _run_task(task, options, directory):
    import torch

    session = Session(task, options)
    try:
        if options.get("require_fp8_plan") and session.fp8:
            from deepopt.opt.precision import precision_report

            plan = precision_report(session.model)
            if (plan is None or plan["fallbacks"] or not any(edge["precision"] == "fp8" for edge in plan["edges"])):
                raise RuntimeError(f"Required FP8 edge plan is missing or failed: {plan}")
        if options.get("_resume_core"):
            row = json.loads(Path(options["_resume_core"]).read_text())
            if (row["initial_state_sha256"] != session.initial_fingerprint
                    or row["shared_optimizer_baseline_sha256"] != session.optimizer_baseline_fingerprint
                    or row.get("rng_baseline_sha256") != session.rng_fingerprint):
                raise RuntimeError("Diagnostic resume does not match the reconstructed model/optimizer baseline")
            row["diagnostics_resumed"] = True
            return _complete_profiles(row, session, task, options, directory)
        try:
            from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
            import inspect
            gemm_observation = record_gemms(session.model) if "model" in inspect.signature(record_gemms).parameters else record_gemms()
            gemm_diagnostics_available = True
        except ModuleNotFoundError as error:
            if error.name != "deepopt.opt.backends._shared.gemm_diagnostics":
                raise
            # Keep older compatible builds usable as native/FP8 controls.
            # New kernel experiment APIs still fail visibly if unavailable.
            gemm_observation = nullcontext([])
            gemm_diagnostics_available = False
        try:
            from deepopt.opt.backends._shared.activation_codec import record_codec_dispatches
            codec_observation = record_codec_dispatches()
        except ImportError as error:
            if error.name != "deepopt.opt.backends._shared.activation_codec":
                raise
            codec_observation = nullcontext([])
        try:
            from deepopt.opt.backends._shared.convolution import record_convolution_launches
            convolution_observation = record_convolution_launches()
        except ModuleNotFoundError as error:
            if error.name != "deepopt.opt.backends._shared.convolution":
                raise
            convolution_observation = nullcontext([])
        with gemm_observation as gemm_executions, codec_observation as codec_executions, convolution_observation as convolution_launches:
            if options.get("numerical_comparison", True):
                numerical = snapshot(session)
                session.snapshot_metadata["bf16_policy"] = session.bf16_policy.as_dict()
                session.snapshot_metadata["numerical_recipe"] = session.numerical_recipe
            else:
                # Populate actual dispatch evidence without copying the model's
                # gradients/updates to the host or exporting numerical tensors.
                session.reset()
                session.step()
                numerical = None
                session.snapshot_metadata = {"status": "not_requested"}
                session.reset()
        quality = quality_diagnostics(session, options["quality_steps"], options.get("quality_interval", 20)) if options.get("quality_steps") else None
        recorded = []
        if options["capture_workloads"] and session.native:
            from .capture import record_workloads
            recorded = record_workloads(session)
        with memory_fill_scope(session.runtime["timing_fill_uninitialized_memory"]) as performance_fill:
            # Correctness always uses the requested safety policy; all performance
            # passes share the separately recorded fill policy below.
            phases = phase_timings(session, options["phase_steps"]) if options["phase_steps"] else None
            memory = memory_diagnostics(session) if options["memory"] else None
            tracing = None
            latency = None
            session.reset()
            graph_mode = session.runtime["execution"] == "graph"
            before_capture = torch.cuda.memory_allocated(session.device) if graph_mode else None
            if graph_mode:
                torch.cuda.reset_peak_memory_stats(session.device)
            replay_trace = None
            if graph_mode:
                # Compilation/capture is never part of throughput, even with
                # --warmup 0. Optimizer and EMA remain outside the captured graph.
                if session.graph is None:
                    session.graph_step()
                    torch.cuda.synchronize(session.device)
                    session.reset()
                capture_peak = {"capture_and_replay_peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
                                "capture_and_replay_peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device)}
                torch.cuda.reset_peak_memory_stats(session.device)
            timing_call = session.graph_step if graph_mode else session.step
            timing = measure(timing_call, torch, session.device, options["warmup"], options["iterations"], options["repeats"])
            graph_memory = ({**capture_peak, "allocated_before_capture_bytes": before_capture,
                             "warmup_capture_and_first_replay_ms": session.capture_setup_ms,
                             "allocated_with_graph_bytes": torch.cuda.memory_allocated(session.device),
                             "reserved_with_graph_bytes": torch.cuda.memory_reserved(session.device),
                             "timing_peak_allocated_bytes": torch.cuda.max_memory_allocated(session.device),
                             "timing_peak_reserved_bytes": torch.cuda.max_memory_reserved(session.device),
                             "note": "Capture-inclusive peaks include warmup/setup and diagnostic replay. Postcapture tensor peaks exclude freed intermediates retained in graph pools; use reserved totals for retained footprint. Eager peaks are separate."}
                            if graph_mode else None)
        integer_payload = sum(item["storage_bytes"] for item in (memory or {}).get("saved_tensors", [])
                              if item["dtype"] in ("torch.int32", "torch.int64", "torch.int8", "torch.uint8"))
        codec_kernels = [item["name"] for source in (tracing, replay_trace) for item in (source or {}).get("top_kernels", [])
                         if any(token in item["name"] for token in ("quantize_pack", "dequantize_unpack", "pack_boolean", "_pack_codes"))]
        coverage = {"status": "not_requested" if not session.compressed else "observed" if integer_payload or codec_kernels else "unverified",
                    "observed_integer_saved_storage_bytes": integer_payload, "codec_kernels": codec_kernels,
                    "note": "Integer saves/codec kernels are evidence; nested hooks and fused kernels can hide payloads. Unverified does not mean disabled."}
        if session.compressed and session.spec["gradient_mode"] == "none":
            coverage["status"] = "not_required_forward_only"
        observed_values = list(session.last_output or ()) + list(session.model.parameters())
        observed_values.extend(value.grad for value in tensor_leaves((session.inputs, session.raw_inputs, tuple(session.model.parameters())))
                               if value.grad is not None)
        finite_after_timing = all(bool(torch.isfinite(dense(value)).all())
                                  for value in observed_values if value.is_floating_point())
        if options.get("latency_steps"):
            with memory_fill_scope(performance_fill):
                latency = latency_diagnostics(session, options["latency_steps"])
        # Capture effective metadata before reset discards cache observations.
        execution = session.execution_info()
        execution["gemm_executions"] = gemm_executions
        execution["convolution_launches"] = convolution_launches
        execution["codec_dispatches"] = codec_executions
        producer_consumers = [item for item in codec_executions if item.get("operation") == "producer_pack_consume"]
        execution["producer_pack_reuse"] = {"observations": len(producer_consumers),
            "hits": sum(bool(item.get("reused")) for item in producer_consumers),
            "misses": sum(not item.get("reused") for item in producer_consumers),
            "note": "Separate diagnostic update; counts do not measure kernel duration."}
        execution["gemm_diagnostics_available"] = gemm_diagnostics_available
        snapshot_path, snapshot_file, comparison_input = None, None, None
        if numerical is not None:
            mode = options.get("numerical_snapshot_mode", "disk")
            if mode == "memory":
                from .numerical_ipc import send_snapshot

                context = json.loads(os.environ.get("DEEPOPT_NUMERICAL_CONTEXT", "null"))
                if not isinstance(context, dict) or not context or not options.get("_numerical_worker_run_id"):
                    raise ValueError("Memory numerical comparison requires a fresh suite and worker identity")
                identity = {**context, "task": task, "worker_run_id": options["_numerical_worker_run_id"]}
                comparison_input = send_snapshot(identity, numerical, session.snapshot_metadata)
            elif mode == "disk":
                from .artifacts import save_snapshot

                snapshot_path = directory / "snapshot.pt"
                snapshot_file = save_snapshot(torch, numerical, snapshot_path)
            else:
                raise ValueError("Unknown numerical snapshot mode")
            del numerical
        row = {"status": "ok", "initial_state_sha256": session.initial_fingerprint,
                "bf16_policy": session.bf16_policy.as_dict(),
                "numerical_recipe": session.numerical_recipe,
                "requested_bf16_policy": session.requested_bf16_policy.as_dict(),
                "initial_component_sha256":session.initial_component_fingerprints,"benchmark_metadata":session.benchmark_metadata,
                "rng_baseline_sha256": session.rng_fingerprint,
                "shared_optimizer_baseline_sha256": session.optimizer_baseline_fingerprint,
                "environment": environment(session.device), "inputs": session.input_metadata,
                "timing": timing, "phases": phases, "memory": memory, "profile": tracing,
                "replay_profile": replay_trace, "graph_memory": graph_memory, "latency_diagnostics": latency,
                "quality_diagnostics": quality,
                "measurement_policy": {"correctness_fill_uninitialized_memory": options["fill_uninitialized_memory"],
                    "bf16_policy": session.bf16_policy.as_dict(),
                    "timing_isolation": options.get("scheduling",{}).get("isolation","exclusive"),
                    "gpu_uuid": options.get("scheduling",{}).get("gpu_uuid"),
                    "performance_fill_uninitialized_memory": performance_fill, "deterministic_algorithms": options["deterministic"],
                    "performance_allocation_filling_active": bool(performance_fill and options["deterministic"]),
                    "correctness_allocation_filling_active": bool(options["fill_uninitialized_memory"] and options["deterministic"]),
                    "numerical_comparison": options.get("numerical_comparison", True),
                    "numerical_snapshot_mode": options.get("numerical_snapshot_mode", "disk"),
                    "shared_native_optimizer_warmup_steps": session.runtime["optimizer_warmup_steps"]},
                "compression_coverage": coverage,
                "finite_after_timing": finite_after_timing,
                "numerical_snapshot": session.snapshot_metadata,
                "execution": execution, "snapshot": str(snapshot_path) if snapshot_path is not None else None, "snapshot_file": snapshot_file,
                "comparison_input": comparison_input,
                "numerical_worker_run_id": options.get("_numerical_worker_run_id"),
                "recorded_workloads": recorded,
                "timing_unit": "optimizer update" if session.optimizer is not None else "forward/backward update",
                "phase_unit": "single microbatch forward/loss and backward; excludes optimizer and accumulation loop",
                "accumulation_steps": session.runtime["accumulation_steps"],
                "diagnostic_note": "Numerical snapshots are optional and explicitly reported; when enabled they use one fresh microbatch before timing. "
                                   "Phase/memory/profile passes are separate and do not contribute to throughput timing."}
        row["environment"]["resume_environment_sha256"] = options.get("resume_environment_sha256")
        from .reporting import write_json
        # Persist essential measurements before optional profiler artifacts.
        write_json(directory / "core-result.json", {**row, "task": task})
        return _complete_profiles(row, session, task, options, directory)
    finally:
        session.close()


def _complete_profiles(row, session, task, options, directory):
    from .artifacts import profile_detail,trace_directory
    directory = trace_directory(directory,task,options)
    detail = profile_detail(task, options)
    with memory_fill_scope(row["measurement_policy"]["performance_fill_uninitialized_memory"]):
        for name, count, replay in [("profile", options.get("profile_steps", 0), False),
                                    ("replay_profile", options.get("profile_replay_steps", 0)
                                     if session.runtime["execution"] == "graph" else 0, True)]:
            row[name] = (profile(session, directory, count, replay=replay, detail=detail,
                                 compress=options.get("compress_traces", False)) if count else None)
    row["artifact_errors"] = [{"phase": name, "reason": row[name]["trace_export_error"]}
                              for name in ("profile", "replay_profile")
                              if row.get(name) and row[name].get("trace_export_error")]
    kernels = [v["name"] for name in ("profile", "replay_profile") for v in (row.get(name) or {}).get("top_kernels", [])
               if any(token in v["name"] for token in ("quantize_pack", "dequantize_unpack", "pack_boolean", "_pack_codes"))]
    row["compression_coverage"]["codec_kernels"] = kernels
    if kernels and session.compressed:
        row["compression_coverage"]["status"] = "observed"
    return row
