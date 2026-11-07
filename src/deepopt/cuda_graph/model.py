"""Model/function graph capture; training and DDP remain eager orchestrators."""

from contextlib import nullcontext

import torch
from torch import nn

from deepopt._config_resolution import _UNSET

from ._configuration import _capture_settings
from ._validation import (arguments, autocast_spec, capture_guard, check_arguments,
                          cuda_device, static_arguments, tensor_spec, warmup_count)
from .capture import CapturedFunction


def _storage_signature(model):
    # Hold actual tensors elsewhere too: removed storage must not be reused and
    # accidentally pass an address check.
    return tuple((name, id(t), t.data_ptr(), tensor_spec(t))
                 for name, t in (*model.named_parameters(), *model.named_buffers()))


def _modes(model):
    return tuple((id(module), module.training) for module in model.modules())


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
        if autograd and torch.is_inference_mode_enabled():
            raise ValueError("Autograd capture cannot run under inference_mode; use autograd=False")
        model = function if isinstance(function, nn.Module) else None
        modules = tuple(model.modules()) if model is not None else ()
        if any(isinstance(m, nn.parallel.DistributedDataParallel) for m in modules):
            raise ValueError("Capture the model before constructing DDPCore or DistributedDataParallel")
        if any(getattr(m.forward, "_deepopt_cuda_capture", False) for m in modules):
            raise ValueError("Model is already captured; close the existing capture first")
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
        self.function, self.model, self.autograd = function, model, autograd
        self._original = model.forward if model is not None else function
        self._had_forward = model is not None and "forward" in model.__dict__
        self._signature = _storage_signature(model) if model is not None else None
        self._modes = _modes(model) if model is not None else None
        self._autocast = autocast_spec()
        self._closed = False
        self._graphed = None
        try:
            with torch.cuda.device(self.device), torch.enable_grad() if autograd else torch.no_grad():
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

        def forward(*args, **kwargs):
            if self._closed:
                raise RuntimeError("Callable capture is closed")
            if model is not None and _storage_signature(model) != self._signature:
                raise RuntimeError("Model parameters or buffers changed storage/metadata; close and recapture")
            # A graph must not silently serve a different train/eval/precision
            # path. Such calls run the original callable eagerly instead.
            mode_changed = model is not None and _modes(model) != self._modes
            if mode_changed or autocast_spec() != self._autocast:
                with nullcontext() if autograd else torch.no_grad():
                    return self._original(*args, **kwargs)
            check_arguments(args, kwargs, self._specs)
            with torch.cuda.device(self.device):
                outputs = self._graphed(*args)
                return _own_backward_gradients(outputs) if autograd else outputs

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
        return not self._closed

    @property
    def graph(self):
        """Native inference graph, or None for PyTorch's opaque autograd pair."""
        return self._graphed.graph if isinstance(self._graphed, CapturedFunction) else None

    def __call__(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("Callable capture is closed")
        if self.model is not None:
            return self.model(*args, **kwargs)
        return self._installed(*args, **kwargs)

    replay = __call__

    def close(self):
        if self._closed:
            return
        if self.model is not None and self.model.forward is not self._installed:
            raise RuntimeError("Model forward was replaced after capture; refusing to overwrite it")
        self._restore_forward()
        if isinstance(self._graphed, CapturedFunction):
            self._graphed.close()
        self._closed = True
        self._graphed = self._installed = self._inputs = self._tensors = None
