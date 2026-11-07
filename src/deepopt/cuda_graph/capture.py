"""Raw capture lifetime and lazy fixed-signature inference functions."""

from contextlib import ExitStack
from functools import update_wrapper

import torch

from deepopt._config_resolution import _UNSET

from ._configuration import _capture_settings
from ._validation import (arguments, autocast_spec, capture_guard, check_arguments,
                          cuda_device, static_arguments, warmup_count)
from ._warmup import warmup_on_stream


class CapturedGraph:
    """Single-use capture context. Owns a native graph, but not caller tensors.

    Warm up code on a side stream before entering. Keep input/output tensors
    alive, copy new inputs into the same storage, then call replay(). The block
    records work; it does not run the recorded CUDA kernels until replay.
    """

    def __init__(self, *, device=_UNSET, pool=None, stream=None, capture_error_mode=_UNSET,
                 warmup=_UNSET, config=None):
        self.config, options = _capture_settings(config, device=device, warmup=warmup,
                                                capture_error_mode=capture_error_mode)
        device, warmup = options["device"], options["warmup"]
        capture_error_mode = options["capture_error_mode"]
        if capture_error_mode not in ("global", "thread_local", "relaxed"):
            raise ValueError("capture_error_mode must be global, thread_local or relaxed")
        warmup_count(warmup)
        self.device, self.pool, self.stream = device, pool, stream
        self.capture_error_mode, self.warmup = capture_error_mode, warmup
        self.graph = None
        self._state = "new"

    @property
    def captured(self):
        return self._state == "ready"

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("Capture contexts are single-use; create a new capture")
        self._state = "capturing"
        self._stack = ExitStack()
        try:
            self.device = cuda_device(self.device)
            self._stack.enter_context(torch.cuda.device(self.device))
            self._stack.enter_context(capture_guard())
            if self.stream is not None and self.stream.device != self.device:
                raise ValueError("Capture stream must belong to the graph's device")
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=self.device)
            self.graph = torch.cuda.CUDAGraph()
            # Some supported PyTorch versions do not restore the stream when
            # their graph context raises while entering/exiting capture.
            self._stack.enter_context(torch.cuda.stream(self.stream))
            self._stack.enter_context(torch.cuda.graph(self.graph, pool=self.pool, stream=self.stream,
                                                       capture_error_mode=self.capture_error_mode))
        except BaseException:
            self._state = "failed"
            self._stack.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._stack.__exit__(exc_type, exc, tb)
        except BaseException:
            self._state = "failed"
            raise
        self._state = "ready" if exc_type is None else "failed"
        return False

    def replay(self):
        if not self.captured:
            raise RuntimeError("Replay requires a successful, open capture")
        with torch.cuda.device(self.device):
            self.graph.replay()

    def close(self):
        if self._state == "capturing":
            raise RuntimeError("Cannot close a graph during capture")
        if self.graph is not None:
            with torch.cuda.device(self.device):
                self.graph.reset()
            self.graph = None
        self._state = "closed"

    def __call__(self, function):
        if self._state != "new":
            raise RuntimeError("Use a fresh capture configuration for a decorator")
        return CapturedFunction(function, device=self.device, pool=self.pool, stream=self.stream,
                                capture_error_mode=self.capture_error_mode, warmup=self.warmup, config=self.config)


class CapturedFunction:
    """Lazy inference-only decorator; outputs borrow reusable graph storage.

    First call warms up, captures, and replays. Subsequent calls copy positional
    tensor inputs and replay, without calling Python. Warmup has real effects:
    use pure inference code here, and the raw context for stateful/training code.
    """

    def __init__(self, function, **options):
        if not callable(function):
            raise TypeError("capture decorates a callable")
        update_wrapper(self, function)
        self.function = function
        self._capture = CapturedGraph(**options)
        self.config = self._capture.config
        self._inputs = self._outputs = self._specs = self._autocast = None

    @property
    def graph(self):
        return self._capture.graph

    @property
    def captured(self):
        return self._capture.captured

    def _initialize(self, args, kwargs):
        if kwargs:
            raise TypeError("Captured calls accept positional tensors only")
        device = cuda_device(self._capture.device)
        self._specs = arguments(args, device)
        if any(arg.requires_grad for arg in args):
            raise ValueError("The capture decorator is inference-only; use capture_model for autograd")
        self._autocast = autocast_spec()
        with torch.cuda.device(device), torch.no_grad():
            self._inputs = static_arguments(args)
            stream = self._capture.stream or torch.cuda.Stream(device=device)
            if stream.device != device:
                raise ValueError("Capture stream must belong to the graph's device")
            self._capture.stream = stream
            warmup_on_stream(self.function, self._inputs, {}, iterations=self._capture.warmup,
                             stream=stream, device=device)
            # Warmup may have modified inputs; start capture from caller values.
            for target, source in zip(self._inputs, args):
                target.copy_(source)
            with self._capture:
                self._outputs = self.function(*self._inputs)
            outputs = self._outputs if isinstance(self._outputs, tuple) else (self._outputs,)
            if any(o is not None and (not isinstance(o, torch.Tensor) or o.device != device) for o in outputs):
                self.close()
                raise ValueError("Captured functions must return a CUDA tensor, a tuple of CUDA tensors, or None")

    def __call__(self, *args, **kwargs):
        if self._capture._state == "new":
            try:
                self._initialize(args, kwargs)
            except BaseException:
                self._capture._state = "failed"
                raise
        if not self.captured:
            raise RuntimeError("Captured function is failed or closed; create a new decorated function")
        check_arguments(args, kwargs, self._specs)
        if autocast_spec() != self._autocast:
            raise ValueError("Autocast settings changed; recapture the function")
        with torch.cuda.device(self._capture.device), torch.no_grad():
            for target, source in zip(self._inputs, args):
                target.copy_(source)
            self._capture.replay()
        return self._outputs

    replay = __call__

    def close(self):
        self._capture.close()
        self._inputs = self._outputs = None
