"""Capture/replay policy only; no optimizers, process groups or epoch state."""

import torch

from ._validation import cuda_device, warmup_count
from .capture import CapturedGraph
from .model import CapturedCallable


class CUDAGraphCore:
    """Independent graph service, composable with TrainingCore and DDPCore.

    Constructing the core does not initialize CUDA. Each returned capture owns
    its own buffers/lifetime; the core keeps no global graph cache.
    """

    def __init__(self, device=None):
        if device is not None and torch.device(device).type != "cuda":
            raise ValueError("CUDAGraphCore requires a CUDA device")
        self.device = device

    def capture_model(self, model, sample_args, *, warmup=3, allow_unused_input=False, pool=None, autograd=True):
        """Capture a module or callable and return a replayable handle.

        Capture before DDP construction. Parameters and state_dict keys retain
        their identity. Plain functions are left unchanged: call the returned
        handle. Pass autograd=False for inference and a native .graph handle.
        Call close() to restore ordinary model execution.
        """
        return CapturedCallable(model, sample_args, device=self.device, warmup=warmup,
                                allow_unused_input=allow_unused_input, pool=pool, autograd=autograd)

    capture_callable = capture_model

    def capture(self, function=None, *, warmup=3, pool=None, stream=None, capture_error_mode="global"):
        """Use as a raw context or an inference decorator, with or without ().

        warmup controls decorator setup only. For a with-block, warm up the
        workload explicitly with core.warmup() before entering the context.
        """
        capture = CapturedGraph(device=self.device, warmup=warmup, pool=pool, stream=stream,
                                capture_error_mode=capture_error_mode)
        return capture if function is None else capture(function)

    def warmup(self, function, *args, iterations=3, stream=None, **kwargs):
        """Run real work on a side stream, then join the caller's stream.

        This intentionally preserves side effects, including optimizer updates.
        The caller chooses grad/autocast mode and owns any state restoration.
        """
        warmup_count(iterations)
        device = cuda_device(self.device)
        with torch.cuda.device(device):
            stream = stream or torch.cuda.Stream(device=device)
            if stream.device != device:
                raise ValueError("Warmup stream must belong to the graph's device")
            current = torch.cuda.current_stream(device)
            stream.wait_stream(current)
            try:
                with torch.cuda.stream(stream):
                    for _ in range(iterations):
                        function(*args, **kwargs)
            finally:
                current.wait_stream(stream)
