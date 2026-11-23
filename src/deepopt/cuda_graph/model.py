"""Model/function graph capture; training and DDP remain eager orchestrators."""

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from deepopt._config_resolution import _UNSET
from deepopt._memory_policy import capture_memory_scope, fill_spec

from ._configuration import _capture_settings
from ._validation import (arguments, autocast_spec, capture_guard, check_arguments,
                          cuda_device, operation_policy_spec, static_arguments, tensor_spec, warmup_count)
from .capture import CapturedFunction


def _storage_signature(model):
    # Hold actual tensors elsewhere too: removed storage must not be reused and
    # accidentally pass an address check.
    from deepopt._bf16_policy import resolve_bf16_policy

    policies = []
    for name, module in model.named_modules():
        owned = getattr(module, "_deepopt_bf16_policy", None)
        plan = getattr(module, "plan", None)
        if owned is not None:
            policy = resolve_bf16_policy(owned)
            policies.append((name, "owned", policy))
        if isinstance(plan, dict) and "bf16_policy" in plan:
            policies.append((name, "plan", resolve_bf16_policy(plan["bf16_policy"])))
        for marker in ("_deepopt_bf16_primitive_recipe", "_deepopt_attention_recipe_version", "_deepopt_activation_memory_plan"):
            if hasattr(module, marker):
                policies.append((name, marker, getattr(module, marker)))
        if isinstance(plan, dict):
            # Prepared plans are recursively immutable; replacing a plan is
            # an explicit numerical-policy change, even at the same shape.
            policies.append((name, "numerical_plan_identity", id(plan)))
    tensors = tuple((name, id(t), t.data_ptr(), tensor_spec(t))
                    for name, t in (*model.named_parameters(), *model.named_buffers()))
    return tensors + (("bf16_policy", tuple(policies)),)


def _modes(model):
    return tuple((id(module), module.training) for module in model.modules())


def _policy_description(amp, operation, fill):
    """Host metadata only: diagnostics must never read/synchronize CUDA tensors."""
    names = ("fp8", "activation_compression", "linear_interception", "convolution_interception",
             "normalization_interception", "pooling_interception", "attention_backend", "kernel_policy", "activation_memory", "bf16_recipe", "bf16_policy")
    def portable(value):
        if hasattr(value, "as_dict"):
            return portable(value.as_dict())
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {str(k): portable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [portable(v) for v in value]
        if isinstance(value, torch.Tensor):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device)}
        return str(value)
    return {"autocast": portable(amp), **dict(zip(names, map(portable, operation))),
            "fill_uninitialized_memory": portable(fill)}


def _own_backward_gradients(outputs):
    """Keep native graph scratch buffers out of accumulators and callers.

    make_graphed_callables returns detached aliases of static grad buffers on
    supported PyTorch versions (pytorch/pytorch#181723). AccumulateGrad can steal
    these aliases when .grad is None. Clone at the public autograd-node boundary,
    outside CUDA capture, before either AccumulateGrad or DDP sees the gradients.
    This also gives torch.autograd.grad callers independent result storage.
    """
    nodes = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            node = value.grad_fn
            if node is not None and node not in nodes:
                nodes.add(node)
                node.register_hook(lambda inputs, outputs: tuple(
                    grad.clone() if grad is not None else None for grad in inputs))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(outputs)
    return outputs


class CapturedCallable:
    """Callable capture handle for a module or function.

    Modules get an installed forward, preserving model/parameter identity.
    Plain functions are not modified; call this handle to replay. Autograd mode
    uses PyTorch's forward/backward graph pair. Inference mode exposes a native
    graph through .graph. close() releases the capture and restores a module's
    original forward; close only after outstanding backward work has completed.
    Disabled handles call eagerly without changing the module's forward.
    """

    def __init__(self, function, sample_args, *, device=_UNSET, warmup=_UNSET, allow_unused_input=_UNSET,
                 pool=None, autograd=_UNSET, config=None):
        self.config, options = _capture_settings(config, device=device, warmup=warmup,
                                                allow_unused_input=allow_unused_input, autograd=autograd)
        device, warmup = options["device"], options["warmup"]
        allow_unused_input, autograd = options["allow_unused_input"], options["autograd"]
        if not callable(function):
            raise TypeError("capture_model/capture_callable requires a module or callable")
        warmup_count(warmup)
        if type(allow_unused_input) is not bool or type(autograd) is not bool:
            raise ValueError("allow_unused_input and autograd must be booleans")
        self.enabled = options["enabled"]
        model = function if isinstance(function, nn.Module) else None
        modules = tuple(model.modules()) if model is not None else ()
        if any(getattr(m.forward, "_deepopt_cuda_capture", False) for m in modules):
            raise ValueError("Model is already captured; close the existing capture first")
        self.function, self.model, self.autograd = function, model, autograd
        self._closed = False
        self._graphed = None
        self._statistics = {"forward_replays": 0, "eager_mode_fallbacks": 0, "eager_autocast_fallbacks": 0,
                            "eager_memory_policy_fallbacks": 0}
        self._fallback_diagnostics = {}
        if not self.enabled:
            self.device = device
            return
        if autograd and torch.is_inference_mode_enabled():
            raise ValueError("Autograd capture cannot run under inference_mode; use autograd=False")
        from deepopt.opt.scaling.delayed import DelayedScalingState

        if any(isinstance(module, DelayedScalingState) for module in modules):
            raise ValueError("Delayed scaling requires eager preprocessing; full training CUDA capture is unsupported")
        if any(isinstance(m, nn.parallel.DistributedDataParallel) for m in modules):
            raise ValueError("Capture the model before constructing DDPCore or DistributedDataParallel")
        self.device = cuda_device(device)
        self._specs = arguments(sample_args, self.device)
        self._tensors = tuple(model.parameters()) + tuple(model.buffers()) if model is not None else ()
        if any(t.device != self.device or t.layout != torch.strided for t in self._tensors):
            raise ValueError("Move all model parameters and buffers to the graph's CUDA device before capture")
        if not autograd and any(t.requires_grad for t in sample_args):
            raise ValueError("Inference capture requires inputs with requires_grad=False")
        if model is not None:
            if any(t.requires_grad for t in model.buffers()):
                raise ValueError("Captured model buffers cannot require gradients")
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Capture at an update boundary with parameter gradients set to None")
            if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in modules):
                raise ValueError("Remove module hooks before capture; attach external observers after capture")
        if autograd and not any(t.requires_grad for t in (*sample_args, *self._tensors)):
            raise ValueError("Autograd capture needs differentiable inputs or parameters; use autograd=False")
        self._original = model.forward if model is not None else function
        self._had_forward = model is not None and "forward" in model.__dict__
        self._signature = _storage_signature(model) if model is not None else None
        self._modes = _modes(model) if model is not None else None
        self._autocast = autocast_spec()
        self._operation_policy = operation_policy_spec()
        try:
            with capture_memory_scope(self.config), torch.cuda.device(self.device), \
                    torch.enable_grad() if autograd else torch.no_grad():
                self._fill_policy = fill_spec()
                # Native warmup runs real forwards. Preserve module buffers
                # (e.g. BatchNorm counters) and default RNG state across setup.
                buffers = [(b, b.detach().clone()) for b in model.buffers()] if model is not None else []
                try:
                    with torch.random.fork_rng(devices=[self.device.index]):
                        with torch.autocast("cuda", enabled=self._autocast[0],
                                            dtype=self._autocast[1], cache_enabled=False):
                            if autograd:
                                self._inputs = static_arguments(sample_args)
                                with capture_guard(), torch.cuda.stream(torch.cuda.current_stream(self.device)):
                                    graphed = torch.cuda.make_graphed_callables(
                                        function, self._inputs, num_warmup_iters=warmup,
                                        allow_unused_input=allow_unused_input, pool=pool)
                                self._graphed = graphed.forward if model is not None else graphed
                            else:
                                self._inputs = None
                                self._graphed = CapturedFunction(self._original, device=self.device,
                                                                warmup=warmup, pool=pool, config=self.config)
                                self._graphed(*sample_args)
                finally:
                    with torch.no_grad():
                        for buffer, value in buffers:
                            buffer.copy_(value)
                if model is not None and (_storage_signature(model) != self._signature or _modes(model) != self._modes):
                    raise RuntimeError("Model structure, storage or modes changed during capture")
        except BaseException:
            self._restore_forward()
            if isinstance(self._graphed, CapturedFunction):
                self._graphed.close()
            raise

        def replay_forward(*args, **kwargs):
            if self._closed:
                raise RuntimeError("Callable capture is closed")
            if model is not None and _storage_signature(model) != self._signature:
                raise RuntimeError("Model parameters, buffers or BF16 policy changed storage/metadata; close and recapture")
            # A graph must not silently serve a different train/eval/precision
            # path. Such calls run the original callable eagerly instead.
            mode_changed = model is not None and _modes(model) != self._modes
            fallback = None
            if mode_changed:
                fallback = "eager_mode_fallbacks"
            elif autocast_spec() != self._autocast or operation_policy_spec() != self._operation_policy:
                fallback = "eager_autocast_fallbacks"
            elif fill_spec() != self._fill_policy:
                fallback = "eager_memory_policy_fallbacks"
            if fallback is not None:
                if fallback not in self._fallback_diagnostics:
                    expected = _policy_description(self._autocast, self._operation_policy, self._fill_policy)
                    actual = _policy_description(autocast_spec(), operation_policy_spec(), fill_spec())
                    if mode_changed:
                        expected["module_modes"] = list(self._modes)
                        actual["module_modes"] = list(_modes(model))
                    changed = {key: {"captured": expected[key], "current": actual[key]}
                               for key in expected if expected[key] != actual[key]}
                    self._fallback_diagnostics[fallback] = changed
                with nullcontext() if autograd else torch.no_grad():
                    outputs = self._original(*args, **kwargs)
                self._statistics[fallback] += 1
                return outputs
            check_arguments(args, kwargs, self._specs)
            with torch.cuda.device(self.device):
                outputs = self._graphed(*args)
                outputs = _own_backward_gradients(outputs) if autograd else outputs
            self._statistics["forward_replays"] += 1
            return outputs

        def forward(*args, **kwargs):
            with capture_memory_scope(self.config):
                return replay_forward(*args, **kwargs)

        forward._deepopt_cuda_capture = True
        self._installed = forward
        if model is not None:
            model.forward = forward

    def _restore_forward(self):
        if self.model is not None:
            if self._had_forward:
                self.model.forward = self._original
            else:
                self.model.__dict__.pop("forward", None)

    @property
    def captured(self):
        return self.enabled and not self._closed

    @property
    def fallback_diagnostics(self):
        """First host policy mismatch per fallback category, without GPU reads.

        Details are collected once per reason, so repeated evaluation fallbacks
        do not add serialization work to the training loop. The returned value
        is independently owned; existing statistics counters remain unchanged.
        """
        return deepcopy(self._fallback_diagnostics)

    @property
    def capture_policy(self):
        """Portable capture policy without tensor reads, or None when disabled."""
        if not self.enabled:
            return None
        return _policy_description(self._autocast, self._operation_policy, self._fill_policy)

    @property
    def statistics(self):
        """Cumulative successful forward routes, without synchronizing CUDA.

        Warmup/capture and rejected or failed forwards are excluded. Mode
        fallback takes precedence if both mode and precision policy changed.
        Autocast fallbacks include changed FP8 and activation compression
        scopes as well as native AMP. Counts
        describe dispatch, not completed GPU execution or backward replays.
        The returned snapshot stays available after close().
        Disabled eager handles leave all capture counters at zero.
        """
        return self._statistics.copy()

    @property
    def graph(self):
        """Native inference graph, or None for PyTorch's opaque autograd pair."""
        return self._graphed.graph if isinstance(self._graphed, CapturedFunction) else None

    def __call__(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Callable capture is closed")
        if not self.enabled:
            with capture_memory_scope(self.config), nullcontext() if self.autograd else torch.no_grad():
                return self.function(*args, **kwargs)
        if self.model is not None:
            return self.model(*args, **kwargs)
        return self._installed(*args, **kwargs)

    replay = __call__

    def close(self):
        if self._closed:
            return
        if not self.enabled:
            self._closed = True
            return
        if self.model is not None and self.model.forward is not self._installed:
            raise RuntimeError("Model forward was replaced after capture; refusing to overwrite it")
        self._restore_forward()
        if isinstance(self._graphed, CapturedFunction):
            self._graphed.close()
        self._closed = True
        self._graphed = self._installed = self._inputs = self._tensors = None
