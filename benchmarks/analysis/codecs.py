"""Storage controls and codec phase diagnostics, separate from training comparisons."""

from types import SimpleNamespace

from .measurement import measure, profile
from .workloads import UnsupportedWorkload


def run_codec(task, options, directory):
    import torch
    import deepopt
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
    from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
    from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask
    from deepopt.opt.quantization.indices import pack_indices, unpack_indices
    from .engine import environment

    device, spec, settings = torch.device(options["device"]), task["workload"], task["settings"]
    if settings["runtime"]["execution"] != "eager":
        raise UnsupportedWorkload("Codec diagnostics use eager timing; training graph/compile settings do not apply")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise UnsupportedWorkload("CUDA is unavailable")
        torch.cuda.set_device(device)
    torch.manual_seed(options["seed"])
    op, shape, config = spec["op"], spec["shape"], settings["compression"]
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    if op == "boolean_codec":
        value = value > 0
        pack, unpack = pack_boolean_mask, unpack_boolean_mask
    elif op == "index_codec":
        value_count = spec["kwargs"].get("value_count", 49)
        value = torch.randint(value_count, shape, device=device, dtype=torch.int64)
        pack = lambda x: pack_indices(x, value_count)
        unpack = unpack_indices
    elif op == "activation_codec":
        pack = lambda x: quantize_pack_activation(x, config["group_size"], bits=config["bits"])
        unpack = lambda p: dequantize_unpack_activation(p, dtype=value.dtype)
    else:
        fp8_dtype = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[settings["fp8"]["activation_dtype"]]
        block = settings["fp8"]["block_shapes"]["activation"]
        pack = lambda x: quantize_pack_fp8_activation(x, fp8_dtype=fp8_dtype, block_shape=block,
                                                     group_size=config["group_size"], bits=config["bits"],
                                                     check_finite=settings["runtime"]["check_finite"])
        unpack = lambda pair: dequantize_unpack_activation(pair[1], dtype=fp8_dtype).float() * pair[0].scale_inv.repeat_interleave(
            block[0], 0).repeat_interleave(block[1], 1)[:shape[0], :shape[1]] if block else (
                dequantize_unpack_activation(pair[1], dtype=fp8_dtype).float() * pair[0].scale_inv)
    if spec["layout"] == "strided":
        backing = torch.empty((*shape[:-1], shape[-1] * 2), device=device, dtype=value.dtype)
        backing[..., ::2].copy_(value)
        value = backing[..., ::2]
    elif spec["layout"] == "channels_last":
        if len(shape) not in (4, 5):
            raise UnsupportedWorkload("channels_last requires rank 4 or 5")
        value = value.contiguous(memory_format=torch.channels_last if len(shape) == 4 else torch.channels_last_3d)
    with deepopt.autocast(device.type, enabled=False, fp8=False, check_finite=settings["runtime"]["check_finite"]):
        packed = pack(value)
        decoded = unpack(packed)
        payload = packed[1] if isinstance(packed, tuple) else packed
        retained = [getattr(payload, key) for key in ("data", "step", "minimum") if hasattr(payload, key)]
        if isinstance(packed, tuple):
            retained.append(packed[0].scale_inv)
        methods = {"pack": lambda: pack(value), "unpack": lambda: unpack(packed),
                   "roundtrip": lambda: unpack(pack(value)), "native_copy": lambda: value.clone()}
        timings = {key: measure(fn, torch, device, options["warmup"], options["iterations"], options["repeats"])
                   for key, fn in methods.items()}
        trace = None
        if options["profile_steps"]:
            session = SimpleNamespace(device=device, model=torch.nn.Identity(), reset=lambda: None,
                                      step=lambda **_: methods["roundtrip"]())
            trace = profile(session, directory, options["profile_steps"])
    expected, restored = value.float(), decoded.float()
    finite = bool(torch.isfinite(restored).all())
    exact = bool(torch.equal(value, decoded)) if op in ("boolean_codec", "index_codec") else None
    return {"status": "ok", "kind": "codec_diagnostic", "timing": timings, "profile": trace, "environment": environment(device),
            "logical_input_bytes": value.numel() * value.element_size(),
            "physical_input_storage_bytes": value.untyped_storage().nbytes(),
            "packed_payload_and_metadata_bytes": sum(x.numel() * x.element_size() for x in retained),
            "finite": finite, "exact": exact,
            "max_abs_error": (restored - expected).abs().max().item() if finite else None,
            "relative_l2_error": ((restored - expected).double().norm() / expected.double().norm().clamp_min(1e-30)).item() if finite else None,
            "note": "Codec phases are independent diagnostics. A storage-copy control is not a BF16 training baseline. "
                    "FP8 fused packing also emits a forward operand, which is excluded from packed saved-payload bytes."}
