"""Capture/replay policy only; no optimizers, process groups or epoch state."""

import torch

from deepopt._config_resolution import _UNSET, resolve_config

from ._validation import cuda_device, warmup_count
from .capture import CapturedGraph
from .model import CapturedCallable


def _capture_settings(config=None, *, base=_UNSET, device=_UNSET, warmup=_UNSET,
                      allow_unused_input=_UNSET, autograd=_UNSET, capture_error_mode=_UNSET):
    """Resolve capture-only defaults without querying or initializing CUDA."""
    explicit = {name: value for name, value in {
        "device": device, "warmup": warmup, "allow_unused_input": allow_unused_input,
        "autograd": autograd, "capture_error_mode": capture_error_mode,
    }.items() if value is not _UNSET}
    if explicit.get("device") is not None:
        explicit["device"] = str(explicit["device"])
    snapshot = resolve_config(config, base=base,
                              overrides={"runtime": {"cuda_graph": explicit}} if explicit else None)
    runtime = snapshot.get("runtime", {})
    configured = runtime.get("cuda_graph", {})
    if "device" in configured:
        selected_device = configured["device"]
    else:
        runtime_device = runtime.get("device")
        selected_device = runtime_device if isinstance(runtime_device, str) and runtime_device.startswith("cuda") else None
    options = {"device": selected_device, "warmup": 3, "allow_unused_input": False,
               "autograd": True, "capture_error_mode": "global", **configured}
    return snapshot, options


class CUDAGraphCore:
    """Independent graph service, composable with TrainingCore and DDPCore.

    Constructing the core does not initialize CUDA. Each returned capture owns
    its own buffers/lifetime; the core keeps no global graph cache.
    """

    def __init__(self, device=_UNSET, *, config=None):
        self._config, options = _capture_settings(config, device=device)
        self._device_override = device
        self.device = options["device"]

    @classmethod
    def _from_global_config(cls):
        """The shared convenience core reads defaults when creating a handle."""
        core = cls.__new__(cls)
        core._config = core._device_override = _UNSET
        core.device = None
        return core

    @property
    def config(self):
        """Return an owned copy; ordinary cores keep their construction snapshot."""
        return resolve_config(base=self._config)

    def capture_model(self, model, sample_args, *, warmup=_UNSET, allow_unused_input=_UNSET,
                      pool=None, autograd=_UNSET, config=None):
        """Capture a module or callable and return a replayable handle.

        Capture before DDP construction. Parameters and state_dict keys retain
        their identity. Plain functions are left unchanged: call the returned
        handle. Pass autograd=False for inference and a native .graph handle.
        Call close() to restore ordinary model execution.
        """
        snapshot, options = _capture_settings(config, base=self._config, device=self._device_override,
                                             warmup=warmup, allow_unused_input=allow_unused_input, autograd=autograd)
        return CapturedCallable(model, sample_args, device=options["device"], warmup=options["warmup"],
                                allow_unused_input=options["allow_unused_input"], pool=pool,
                                autograd=options["autograd"], config=snapshot)

    capture_callable = capture_model

    def capture(self, function=None, *, warmup=_UNSET, pool=None, stream=None,
                capture_error_mode=_UNSET, config=None):
        """Use as a raw context or an inference decorator, with or without ().

        warmup controls decorator setup only. For a with-block, warm up the
        workload explicitly with core.warmup() before entering the context.
        """
        snapshot, options = _capture_settings(config, base=self._config, device=self._device_override,
                                             warmup=warmup, capture_error_mode=capture_error_mode)
        capture = CapturedGraph(device=options["device"], warmup=options["warmup"], pool=pool, stream=stream,
                                capture_error_mode=options["capture_error_mode"], config=snapshot)
        return capture if function is None else capture(function)

    def warmup(self, function, *args, iterations=_UNSET, stream=None, config=None, **kwargs):
        """Run real work on a side stream, then join the caller's stream.

        This intentionally preserves side effects, including optimizer updates.
        The caller chooses grad/autocast mode and owns any state restoration.
        """
        _, options = _capture_settings(config, base=self._config, device=self._device_override, warmup=iterations)
        iterations = options["warmup"]
        warmup_count(iterations)
        device = cuda_device(options["device"])
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
