"""Capture defaults shared by graph handles and their orchestration service."""

from deepopt._config_resolution import _UNSET, resolve_config


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
