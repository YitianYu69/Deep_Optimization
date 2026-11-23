"""Replay one real training snapshot against native and codec-oracle gradients.

FP8 is deliberately disabled: this diagnoses the reported BF16+compression
failure. Run each case serially on one GPU. No datasets or weights are fetched.
"""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from fnmatch import fnmatchcase
from functools import wraps
from hashlib import sha256
from importlib import import_module
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import traceback


DIAGNOSTIC_REVISION = 2


def _diagnostic_sources():
    oracle = (import_module(".stability_oracle", __package__) if __package__ else
              import_module("stability_oracle"))
    runner_path = Path(__file__).resolve()
    oracle_path = Path(oracle.__file__).resolve()
    if (getattr(oracle, "ORACLE_REVISION", None) != DIAGNOSTIC_REVISION
            or oracle_path != runner_path.with_name("stability_oracle.py")):
        raise ValueError(f"Mixed diagnostic files: runner revision {DIAGNOSTIC_REVISION} at {runner_path} "
                         f"loaded oracle revision {getattr(oracle, 'ORACLE_REVISION', 'missing')} at {oracle_path}. "
                         "Use both Python files from the same diagnostic bundle.")
    return {"runner": {"path": str(runner_path), "sha256": sha256(runner_path.read_bytes()).hexdigest()},
            "oracle": {"path": str(oracle_path), "sha256": sha256(oracle_path.read_bytes()).hexdigest()}}


def _torch():
    import torch
    return torch


def _freeze(value, parameters=None):
    torch = _torch()
    if isinstance(value, torch.nn.Parameter) and parameters is not None:
        if id(value) not in parameters:
            raise TypeError("optimizer holds a parameter outside the model")
        return {"__snapshot_parameter__": parameters[id(value)]}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_freeze(v, parameters) for v in value)
    if isinstance(value, dict) and all(type(k) in (str, int, float, bool) for k in value):
        return {k: _freeze(v, parameters) for k, v in value.items()}
    raise TypeError(f"unsupported extra state: {type(value).__module__}.{type(value).__qualname__}")


def _move(value, device, parameters=None):
    torch = _torch()
    if isinstance(value, dict) and set(value) == {"__snapshot_parameter__"}:
        return parameters[value["__snapshot_parameter__"]]
    if isinstance(value, torch.Tensor):
        return value.to(device).clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_move(v, device, parameters) for v in value)
    if isinstance(value, dict):
        return {k: _move(v, device, parameters) for k, v in value.items()}
    return deepcopy(value)


def _source_hash(obj):
    try:
        path = inspect.getsourcefile(obj)
        return sha256(Path(path).read_bytes()).hexdigest() if path else None
    except (OSError, TypeError):
        return None


def _library_hash():
    import deepopt
    root = Path(deepopt.__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rng(device):
    torch = _torch()
    import numpy as np
    return {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(state, device):
    torch = _torch()
    import numpy as np
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if device.type == "cuda":
        if state["cuda"] is None:
            raise ValueError("A CPU snapshot has no matched CUDA RNG state")
        torch.cuda.set_rng_state(state["cuda"], device)


@contextmanager
def _deterministic_scope(enabled):
    torch = _torch()
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _capture_contract(trainer):
    torch = _torch()
    from deepopt.engine.strategies.loss import ExplicitObjective
    if trainer.runtime.distributed or trainer.runtime.fp8.enabled:
        raise ValueError("This diagnostic captures one GPU with FP8 disabled")
    if trainer.accumulation_steps != 1:
        raise ValueError("This diagnostic requires accumulation_steps=1")
    if type(trainer._steps.loss_fn) is not torch.nn.CrossEntropyLoss:
        raise ValueError("Supply the same plain CrossEntropyLoss used by the reported training recipe")
    if (trainer.runtime.gradient_strategy is not None
            or type(trainer._epochs.loss_strategy) is not ExplicitObjective):
        raise ValueError("Custom loss/gradient processing must be reproduced before update comparisons can qualify")
    if trainer.config.get("runtime", {}).get("compile", {}).get("enabled", False):
        raise ValueError("This diagnostic reproduces eager/CUDA Graph execution, not torch.compile")


def save_snapshot(trainer, batch, path, *, dense_modules=()):
    """Capture before an update; preserve custom counters omitted by state_dict.

    The packet is a diagnostic input, not an exact-resume training checkpoint.
    Unsupported optimizer attributes are recorded and prevent qualified updates.
    """
    torch = _torch()
    from deepopt._memory_policy import resolve_fill
    _capture_contract(trainer)
    if any(p.grad is not None for p in trainer.model.parameters()):
        raise ValueError("Capture at an update boundary with accumulation_steps=1 and gradients=None")
    if not isinstance(batch, (list, tuple)) or len(batch) != 2 or not all(isinstance(v, torch.Tensor) for v in batch):
        raise ValueError("The supported batch is (images, class_targets)")
    loss = trainer._steps.loss_fn
    optimizer, model = trainer.optimizer, trainer.model
    fill = resolve_fill(trainer.runtime.execution_mode,
                        trainer.config.get("runtime", {}).get("fill_uninitialized_memory"))
    names = {id(p): name for name, p in model.named_parameters()}
    base = torch.optim.Optimizer([torch.nn.Parameter(torch.empty(0))], {})
    excluded = set(vars(base)) | {"state", "param_groups", "defaults"}
    extra, unsupported = {}, {}
    for key, value in vars(optimizer).items():
        if key in excluded:
            if "hook" in key and value:
                unsupported[key] = "Nonempty optimizer hooks cannot be reconstructed"
            continue
        try:
            extra[key] = _freeze(value, names)
        except TypeError as error:
            unsupported[key] = str(error)
    patterns = tuple(dense_modules)
    overrides = [name for name, module in model.named_modules()
                 if "forward" in vars(module) and not getattr(module.forward, "_deepopt_cuda_capture", False)
                 and not any(fnmatchcase(name, p) for p in patterns)]
    packet = {
        "format": "deepopt-stability-v1", "config": trainer.config,
        "completed_updates": trainer.state.optimizer_steps,
        "model": _freeze(model.state_dict()), "parameter_order": list(names.values()),
        "requires_grad": {name: p.requires_grad for name, p in model.named_parameters()},
        "module_modes": {name: module.training for name, module in model.named_modules()},
        "model_type": type(model).__module__ + ":" + type(model).__qualname__,
        "model_source_sha256": _source_hash(type(model)), "unreconstructed_forward_overrides": overrides,
        "batch": _freeze(batch), "rng": _rng(trainer.device), "dense_modules": list(patterns),
        "was_captured": bool(getattr(model.forward, "_deepopt_cuda_capture", False)),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory if fill is None else fill,
        "loss": {"reduction": loss.reduction, "ignore_index": loss.ignore_index,
                 "label_smoothing": loss.label_smoothing, "weight": _freeze(loss.weight)},
        "optimizer": {"factory": type(optimizer).__module__ + ":" + type(optimizer).__qualname__,
                      "source_sha256": _source_hash(type(optimizer)), "defaults": _freeze(optimizer.defaults),
                      "state": _freeze(optimizer.state_dict()), "extra": extra, "unsupported": unsupported,
                      "groups": [{**_freeze({k: v for k, v in group.items() if k != "params"}),
                                  "params": [names[id(p)] for p in group["params"]]}
                                 for group in optimizer.param_groups]},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "device": str(trainer.device), "deepopt_source_sha256": _library_hash()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(packet, temporary)
    temporary.replace(path)
    return {"path": str(path), "completed_updates": packet["completed_updates"],
            "optimizer_extra_fields": sorted(extra), "unsupported_optimizer_fields": unsupported,
            "unreconstructed_forward_overrides": overrides}


class SnapshotLoader:
    """Delegate the existing loader and capture the next batch without refetching.

    Install after Trainer construction; pass this wrapper only as train_loader.
    Its iteration does not consume an extra batch or alter transform RNG state.
    """
    def __init__(self, loader, trainer, path, *, before_update=1100, dense_modules=()):
        _capture_contract(trainer)
        if type(before_update) is not int or before_update < 1:
            raise ValueError("before_update must be a positive integer")
        self.loader, self.trainer, self.path = loader, trainer, Path(path)
        self.before_update, self.dense_modules, self.captured = before_update, dense_modules, False

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if not self.captured and self.trainer.state.optimizer_steps + 1 >= self.before_update:
                summary = save_snapshot(self.trainer, batch, self.path, dense_modules=self.dense_modules)
                self.captured = True
                print("Stability snapshot: " + json.dumps(summary), flush=True)
            yield batch


def _resolve(spec):
    module, name = spec.split(":", 1)
    value = import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def _optimizer(model, packet, override=None):
    saved = packet["optimizer"]
    if saved["unsupported"]:
        raise ValueError("Unreconstructed optimizer attributes: " + json.dumps(saved["unsupported"]))
    constructor = _resolve(override or saved["factory"])
    if (_source_hash(constructor) != saved["source_sha256"]
            or constructor.__qualname__ != saved["factory"].split(":", 1)[1]):
        raise ValueError("Optimizer source differs from the captured training optimizer")
    signature = inspect.signature(constructor)
    constructor_values = {**saved["extra"], **saved["defaults"]}
    keywords = {k: v for k, v in constructor_values.items() if k in signature.parameters and k != "params"}
    parameters = dict(model.named_parameters())
    groups = [{**_move({k: v for k, v in group.items() if k != "params"}, next(model.parameters()).device),
               "params": [parameters[name] for name in group["params"]]} for group in saved["groups"]]
    optimizer = constructor(groups, **keywords)
    optimizer.load_state_dict(deepcopy(saved["state"]))
    for key, value in saved["extra"].items():
        setattr(optimizer, key, _move(value, next(model.parameters()).device, parameters))
    return optimizer


def gradient_alignment(actual, reference, *, sign_rms_fraction=1e-3, head_axis=None):
    """Direction, scale and orthogonal error with bounded FP64 reductions.

    Zero vector directions are undefined. Sign statistics ignore reference
    coordinates below a reported fraction of reference RMS, not a fixed cutoff
    that would discard small but meaningful training gradients.
    """
    torch = _torch()
    if actual.shape != reference.shape:
        return {"finite": False, "reason": "shape mismatch"}
    a, b = actual.detach().cpu(), reference.detach().cpu()
    x, y = a.reshape(-1), b.reshape(-1)
    aa = bb = ab = 0.
    for start in range(0, x.numel(), 1 << 18):
        u, v = x[start:start + (1 << 18)].double(), y[start:start + (1 << 18)].double()
        if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):
            return {"finite": False, "reason": "nonfinite gradient"}
        aa += float(u.square().sum()); bb += float(v.square().sum()); ab += float((u * v).sum())
    defined = aa > 0 and bb > 0
    cosine = max(-1., min(1., ab / math.sqrt(aa * bb))) if defined else None
    cutoff = sign_rms_fraction * math.sqrt(bb / max(1, y.numel()))
    eligible = flipped = 0
    for start in range(0, x.numel(), 1 << 18):
        u, v = x[start:start + (1 << 18)].double(), y[start:start + (1 << 18)].double()
        mask = v.abs() > cutoff
        eligible += int(mask.sum())
        flipped += int(((u * v < 0) & mask).sum())
    result = {"finite": True, "direction_defined": defined, "cosine": cosine,
              "angle_degrees": math.degrees(math.acos(cosine)) if defined else None,
              "norm_ratio": math.sqrt(aa / bb) if bb else None,
              "parallel_scale": ab / bb if bb else None,
              "orthogonal_relative_l2": math.sqrt(max(0., aa - ab * ab / bb) / bb) if bb else None,
              "sign_disagreement_fraction": flipped / eligible if eligible else None,
              "sign_eligible_coordinates": eligible, "sign_threshold": cutoff,
              "zero_reference": bb == 0, "zero_actual": aa == 0}
    if head_axis is not None and a.ndim > 1:
        result["heads"] = [gradient_alignment(a.select(head_axis, i), b.select(head_axis, i),
                                                sign_rms_fraction=sign_rms_fraction)
                           for i in range(a.shape[head_axis])]
        head_cos = [row["cosine"] for row in result["heads"] if row.get("direction_defined")]
        result["negative_head_fraction"] = sum(v < 0 for v in head_cos) / len(head_cos) if head_cos else None
        # Each logical last-dimension vector is a token/head direction. Chunk
        # rows so an image-batch-512 diagnostic does not allocate full FP64 QKV.
        rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        cosines, undefined = [], 0
        for start in range(0, rows_a.shape[0], max(1, (1 << 18) // a.shape[-1])):
            count = max(1, (1 << 18) // a.shape[-1])
            u, v = rows_a[start:start + count].double(), rows_b[start:start + count].double()
            denom = (u.square().sum(-1) * v.square().sum(-1)).sqrt()
            valid = denom > 0
            undefined += int((~valid).sum())
            cosines.append(((u * v).sum(-1)[valid] / denom[valid]).clamp(-1, 1))
        values = torch.cat(cosines) if cosines else torch.empty(0)
        result["token_directions"] = {"defined": values.numel(), "undefined": undefined,
            "negative_fraction": float((values < 0).double().mean()) if values.numel() else None,
            "cosine_percentiles": {str(p): float(torch.quantile(values, p / 100))
                                   for p in (0, 1, 5, 50, 95, 100)} if values.numel() else {}}
    return result


def tensor_error(actual, reference, *, rtol=.03, atol=1e-5):
    """Full tensors, chunked FP64 accumulation; do not hide zero/NaN gradients."""
    torch = _torch()
    if actual.shape != reference.shape:
        return {"passed": False, "reason": "shape mismatch"}
    a, b = actual.detach().cpu().reshape(-1), reference.detach().cpu().reshape(-1)
    a2 = b2 = difference2 = dot = maximum = 0.
    finite = passed = True
    for start in range(0, a.numel(), 1 << 18):
        x, y = a[start:start + (1 << 18)].double(), b[start:start + (1 << 18)].double()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            finite = passed = False
            continue
        d = x - y
        a2 += float(x.square().sum()); b2 += float(y.square().sum())
        difference2 += float(d.square().sum()); dot += float((x * y).sum())
        maximum = max(maximum, float(d.abs().max()) if d.numel() else 0.)
        passed &= bool((d.abs() <= atol + rtol * y.abs()).all())
    return {"finite": finite, "passed": bool(passed), "max_abs": maximum if finite else None,
            "actual_l2": a2 ** .5 if finite else None, "reference_l2": b2 ** .5 if finite else None,
            "relative_l2": ((difference2 / b2) ** .5 if b2 else 0. if not difference2 else None) if finite else None,
            "cosine": (dot / (a2 * b2) ** .5 if a2 and b2 else 1. if a2 == b2 else None) if finite else None,
            "zero_reference": b2 == 0, "rtol": rtol, "atol": atol,
            "alignment": gradient_alignment(actual, reference)}


def _compare(a, b, *, rtol, atol):
    rows = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            rows[name] = {"passed": False, "reason": "gradient/update missing on one path"}
        else:
            rows[name] = tensor_error(a[name], b[name], rtol=rtol, atol=atol)
    return {"passed": all(row["passed"] for row in rows.values()), "parameters": rows,
            "largest_errors": sorted(rows, key=lambda k: rows[k].get("relative_l2") or 0., reverse=True)[:20]}


def _skip_compression(module, device):
    import deepopt
    original = module.forward

    @wraps(original)
    def forward(*args, **kwargs):
        with deepopt.autocast(device.type, enabled=_torch().is_autocast_enabled(device.type),
                             activation_compression=False):
            return original(*args, **kwargs)
    module.forward = forward


@contextmanager
def _probes(model, selected, records, budget):
    """Record leaf inputs and upstream gradients, only during eager diagnosis."""
    torch = _torch()
    from deepopt.opt.nn.linear import DOLinear

    linear_types = (torch.nn.Linear, DOLinear)
    used = 0
    handles = []
    attached = set()
    matched = set()
    def attach(name, module):
        def after(module, args, output):
            nonlocal used
            if name in records or not args or not isinstance(output, torch.Tensor) or not output.requires_grad:
                return
            x = args[0]
            size = x.numel() * x.element_size() + output.numel() * output.element_size()
            if used + size > budget:
                records[name] = {"status": "skipped", "reason": "probe byte budget exceeded"}
                return
            used += size
            record = {"status": "captured", "kind": "linear" if isinstance(module, linear_types) else
                      "layer_norm" if isinstance(module, torch.nn.LayerNorm) else "gelu",
                      "input": _freeze(x), "stride": tuple(x.stride()), "requires_grad": x.requires_grad,
                      "weight": _freeze(getattr(module, "weight", None)),
                      "bias": _freeze(getattr(module, "bias", None)),
                      "weight_requires_grad": bool(getattr(getattr(module, "weight", None), "requires_grad", False)),
                      "bias_requires_grad": bool(getattr(getattr(module, "bias", None), "requires_grad", False)),
                      "normalized_shape": getattr(module, "normalized_shape", None),
                      "eps": getattr(module, "eps", None), "approximate": getattr(module, "approximate", "none")}
            records[name] = record
            output.register_hook(lambda grad: record.update(dy=_freeze(grad)))
        handles.append(module.register_forward_hook(after))
        attached.add(name)
    try:
        for name, module in model.named_modules():
            patterns = [pattern for pattern in selected if fnmatchcase(name, pattern)]
            if patterns:
                matched.update(patterns)
                if isinstance(module, (*linear_types, torch.nn.LayerNorm, torch.nn.GELU)):
                    attach(name, module)
                else:
                    records[name] = {"status": "skipped", "reason": "probe supports Linear, LayerNorm and GELU"}
        for pattern in set(selected) - matched:
            records["pattern:" + pattern] = {"status": "skipped", "reason": "no module matched this pattern"}
        yield
    finally:
        for handle in handles:
            handle.remove()
        for name in attached:
            records.setdefault(name, {"status": "skipped", "reason": "no differentiable forward was observed"})


def _factory_model(packet, factory, kwargs, device):
    torch = _torch()
    constructor = _resolve(factory)
    options = deepcopy(kwargs)
    if "device" in inspect.signature(constructor).parameters and "device" not in options:
        options["device"] = torch.device("cpu")
    result = constructor(**options)
    model = result["model"] if isinstance(result, dict) else result
    if hasattr(model, "_deepopt_fp8_config"):
        raise ValueError("Model factory must return the original unprepared model")
    if _source_hash(type(model)) != packet["model_source_sha256"]:
        raise ValueError("Model implementation differs from the captured training model")
    model.load_state_dict(deepcopy(packet["model"]), strict=True, assign=True)
    if list(dict(model.named_parameters())) != packet["parameter_order"]:
        raise ValueError("Model parameter order differs; positional optimizer state would be invalid")
    for name, p in model.named_parameters():
        p.requires_grad_(packet["requires_grad"][name])
    return model.to(device)


def _case(name, packet, args, folder, probes, audits):
    torch = _torch()
    import deepopt
    from deepopt._config_resolution import ResolvedConfig
    from deepopt._memory_policy import allocation_fill_scope
    from deepopt.opt.nn.linear import DOLinear
    from deepopt.opt.policy import prepare_fp8_model
    try:
        from .stability_oracle import oracle_scope, codec_audit_scope
    except ImportError:
        from stability_oracle import oracle_scope, codec_audit_scope

    device = torch.device(args.device)
    oracle, compressed = name == "codec_oracle", name not in ("native", "native_repeat")
    config = deepcopy(packet["config"])
    config["runtime"]["fp8"]["enabled"] = False
    config["runtime"]["activation_compression"]["enabled"] = compressed
    config["runtime"]["execution_mode"] = "max_speed"
    config["runtime"]["compile"] = {"enabled": False}
    config = ResolvedConfig(config)
    model = _factory_model(packet, args.model_factory, args.model_kwargs, device)
    if compressed:
        model, _ = prepare_fp8_model(model, False, config=config)
        matched = []
        for module_name, module in model.named_modules():
            if any(fnmatchcase(module_name, p) for p in packet["dense_modules"]):
                if not isinstance(module, (torch.nn.Linear, DOLinear)):
                    raise ValueError("dense_modules currently supports only Linear modules")
                _skip_compression(module, device)
                matched.append(module_name)
        if packet["dense_modules"] and not matched:
            raise ValueError("No dense-module exclusion matched this model")
    for module_name, module in model.named_modules():
        module.training = packet["module_modes"].get(module_name, True)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = None
    optimizer_error = None
    try:
        optimizer = _optimizer(model, packet, args.optimizer)
    except Exception as error:
        optimizer_error = f"{type(error).__name__}: {error}"
    batch = _move(packet["batch"], device)
    loss_options = deepcopy(packet["loss"])
    if loss_options["weight"] is not None:
        loss_options["weight"] = loss_options["weight"].to(device)
    criterion = torch.nn.CrossEntropyLoss(**loss_options)
    precision = config["runtime"].get("precision", "bf16")
    fill = {"production_fill": True, "production_no_fill": False}.get(name, packet["fill_uninitialized_memory"])
    context = (lambda: deepopt.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16",
               cache_enabled=False, fp8=False, config=config, check_finite=False)) if compressed else (
               lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16", cache_enabled=False))
    captured = None
    with ExitStack() as stack:
        if name in ("production_fill", "production_no_fill"):
            # Both sides must use the same deterministic compute policy.
            # Merely setting fill=True while determinism=False fills nothing.
            stack.enter_context(_deterministic_scope(True))
        stack.enter_context(allocation_fill_scope(fill))
        if oracle:
            stack.enter_context(oracle_scope())
        if name == "production":
            stack.enter_context(_probes(model, args.probe_modules, probes, args.max_probe_bytes))
            if getattr(args, "audit_codec", True):
                stack.enter_context(codec_audit_scope(audits))
        if name == "production_graph":
            if device.type != "cuda":
                raise ValueError("Graph diagnosis requires CUDA")
            with context():
                captured = deepopt.CUDAGraphCore(device, config=config).capture_model(
                    model, (batch[0],), warmup=3, autograd=True)
            stack.callback(captured.close)
        _restore_rng(packet["rng"], device)
        with context():
            logits = model(batch[0])
            loss = criterion(logits, batch[1])
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        updates, ratios = {}, {}
        gradients_finite = all(bool(torch.isfinite(v).all()) for v in gradients.values())
        if optimizer is not None and gradients_finite:
            optimizer.step()
            for n, p in model.named_parameters():
                update = p.detach().cpu().float() - initial[n].float()
                updates[n] = update
                weight_norm, update_norm = float(initial[n].double().norm()), float(update.double().norm())
                finite = math.isfinite(weight_norm) and math.isfinite(update_norm)
                ratios[n] = {"finite": finite, "weight_l2": weight_norm if finite else None,
                             "update_l2": update_norm if finite else None,
                             "update_to_weight": update_norm / weight_norm if finite and weight_norm else None}
        output = {"logits": logits.detach().cpu(), "gradients": gradients, "updates": updates}
        torch.save(output, folder / (name + ".pt"))
        loss_value = float(loss.detach())
        return {"status": "ok", "loss": loss_value if math.isfinite(loss_value) else None,
                "loss_finite": math.isfinite(loss_value), "gradients_finite": gradients_finite, "compressed": compressed,
                "optimizer_update": ("matched_captured_state" if gradients_finite else "skipped_nonfinite_gradients")
                    if optimizer is not None else "unavailable",
                "optimizer_error": optimizer_error, "update_ratios": ratios,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "fill": fill, "effective_fill": fill and torch.are_deterministic_algorithms_enabled(),
                "graph_routes": captured.statistics if captured else None,
                "graph_replay_verified": captured.statistics["forward_replays"] > 0 if captured else None}


def _local_checks(records, packet, args):
    torch = _torch()
    import deepopt
    from deepopt.opt.nn import functional as DF
    from deepopt.opt.policy import linear_plan
    try:
        from .stability_oracle import oracle_scope
    except ImportError:
        from stability_oracle import oracle_scope
    results = {}
    device = torch.device(args.device)
    for name, record in records.items():
        if record["status"] != "captured" or "dy" not in record:
            results[name] = {"status": "skipped", "reason": record.get("reason", "no upstream gradient")}
            continue
        values = []
        for implementation in ("production", "oracle", "native"):
            x = torch.empty_strided(record["input"].shape, record["stride"], device=device,
                                   dtype=record["input"].dtype)
            x.copy_(record["input"].to(device)); x.requires_grad_(record["requires_grad"])
            w = record["weight"].to(device).clone().requires_grad_(record["weight_requires_grad"]) if record["weight"] is not None else None
            b = record["bias"].to(device).clone().requires_grad_(record["bias_requires_grad"]) if record["bias"] is not None else None
            compression = deepcopy(packet["config"]["runtime"]["activation_compression"])
            if any(fnmatchcase(name, p) for p in packet["dense_modules"]):
                compression["enabled"] = False
            plan = linear_plan(False, activation_compression=compression)
            with oracle_scope() if implementation == "oracle" else nullcontext():
                context = (torch.autocast(device.type, dtype=torch.bfloat16,
                               enabled=packet["config"]["runtime"].get("precision") == "bf16", cache_enabled=False)
                           if implementation == "native" else deepopt.autocast(device.type, dtype=torch.bfloat16,
                        enabled=packet["config"]["runtime"].get("precision") == "bf16",
                        fp8=False, activation_compression=compression, check_finite=False))
                with context:
                    if implementation == "native":
                        from torch.nn import functional as F
                        if record["kind"] == "linear":
                            y = F.linear(x, w, b)
                        elif record["kind"] == "layer_norm":
                            y = F.layer_norm(x, record["normalized_shape"], w, b, eps=record["eps"])
                        else:
                            y = F.gelu(x, approximate=record["approximate"])
                    elif record["kind"] == "linear":
                        y = DF.do_linear(x, w, b, plan=plan)
                    elif record["kind"] == "layer_norm":
                        y = DF.layer_norm(x, record["normalized_shape"], w, b, eps=record["eps"], plan=plan)
                    else:
                        y = DF.do_gelu(x, approximate=record["approximate"], plan=plan)
                y.backward(record["dy"].to(device))
                values.append({key: value.detach().cpu().clone() for key, value in (
                    ("output", y), ("dx", x.grad), ("dw", w.grad if w is not None else None),
                    ("db", b.grad if b is not None else None)) if value is not None})
        results[name] = {"status": "ok", "kind": record["kind"], "shape": list(record["input"].shape),
                         "comparison": _compare(values[0], values[1], rtol=args.rtol, atol=args.atol),
                         "production_vs_native": _compare(values[0], values[2], rtol=args.rtol, atol=args.atol),
                         "oracle_vs_native": _compare(values[1], values[2], rtol=args.rtol, atol=args.atol)}
    return results


def _run(packet, args):
    torch = _torch()
    diagnostic_sources = _diagnostic_sources()
    if packet.get("format") != "deepopt-stability-v1":
        raise ValueError("Use a captured stability snapshot, not a raw training checkpoint")
    if packet["unreconstructed_forward_overrides"]:
        raise ValueError("Declare/reconstruct forward overrides: " + str(packet["unreconstructed_forward_overrides"]))
    if not packet["config"]["runtime"]["activation_compression"]["enabled"]:
        raise ValueError("Capture the failing compression-enabled recipe")
    if packet["config"]["runtime"].get("fp8", {}).get("enabled", False):
        raise ValueError("This diagnostic isolates BF16 activation compression; FP8 must be disabled")
    if _library_hash() != packet["environment"]["deepopt_source_sha256"]:
        raise ValueError("DeepOpt source changed since capture; use the same installed build for this diagnosis")
    torch.use_deterministic_algorithms(packet["deterministic"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "matched-batch BF16 compression diagnostic, not a convergence benchmark",
              "diagnostic_revision": DIAGNOSTIC_REVISION, "diagnostic_sources": diagnostic_sources,
              "captured_deterministic": packet["deterministic"],
              "oracle_scale_division": "FP32 tensor/tensor division with on-device divisor",
              "completed_updates": packet["completed_updates"], "batch_shape": list(packet["batch"][0].shape),
              "compression": packet["config"]["runtime"]["activation_compression"],
              "dense_modules": packet["dense_modules"], "optimizer": packet["optimizer"]["factory"],
              "optimizer_extra_fields": sorted(packet["optimizer"]["extra"]), "cases": {}, "comparisons": {},
              "captured_environment": packet["environment"],
              "diagnostic_environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
              "device": str(args.device), "model_factory": args.model_factory, "model_kwargs": args.model_kwargs,
              "rtol": args.rtol, "atol": args.atol, "probe_patterns": args.probe_modules}
    def write():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    probes, audits = {}, []
    report["codec_audits"] = audits
    cases = ["native", "native_repeat", "codec_oracle", "production"]
    if args.check_fill:
        cases.extend(("production_no_fill", "production_fill"))
    if args.graph:
        cases.append("production_graph")
    with tempfile.TemporaryDirectory(prefix="case-tensors-", dir=output) as temporary:
        folder = Path(temporary)
        for name in cases:
            try:
                report["cases"][name] = _case(name, packet, args, folder, probes, audits)
            except Exception as error:
                report["cases"][name] = {"status": "error", "reason": f"{type(error).__name__}: {error}",
                                          "traceback": traceback.format_exc()}
            write()
            print(json.dumps({"case": name, "status": report["cases"][name]["status"]}), flush=True)
            if torch.device(args.device).type == "cuda":
                import gc
                gc.collect(); torch.cuda.empty_cache()
        for name, actual, reference in (("native_repeatability", "native_repeat", "native"),
            ("expected_approximation", "codec_oracle", "native"), ("production_vs_oracle", "production", "codec_oracle"),
            ("production_vs_native", "production", "native"),
            ("fill_sensitivity", "production_fill", "production_no_fill"),
            ("determinism_sensitivity", "production_no_fill", "production"),
            ("graph_vs_eager", "production_graph", "production")):
            if all(report["cases"].get(key, {}).get("status") == "ok" for key in (actual, reference)):
                a, b = (torch.load(folder / (key + ".pt"), map_location="cpu", weights_only=False) for key in (actual, reference))
                report["comparisons"][name] = {"logits": tensor_error(a["logits"], b["logits"], rtol=args.rtol, atol=args.atol),
                    "gradients": _compare(a["gradients"], b["gradients"], rtol=args.rtol, atol=args.atol),
                    "updates": _compare(a["updates"], b["updates"], rtol=args.rtol, atol=args.atol)
                        if a["updates"] and b["updates"] else {"status": "unavailable"}}
                del a, b
                write()
        try:
            from deepopt._memory_policy import allocation_fill_scope
            with allocation_fill_scope(packet["fill_uninitialized_memory"]):
                report["local_checks"] = _local_checks(probes, packet, args)
        except Exception as error:
            report["local_checks"] = {"error": {"status": "error", "reason": f"{type(error).__name__}: {error}",
                                               "traceback": traceback.format_exc()}}
    report["status"] = "complete" if (all(case["status"] == "ok" and case["optimizer_update"] == "matched_captured_state"
        and case["graph_replay_verified"] is not False
        for case in report["cases"].values()) and report["local_checks"]
        and all(row["status"] == "ok" for row in report["local_checks"].values())) else "partial"
    report["interpretation"] = [
        "expected_approximation measures the intended lossy policy against native BF16.",
        "production_vs_oracle also includes amplified forward rounding; use local_checks before calling it a kernel defect.",
        "local_checks compare production/oracle and each path versus native using identical inputs and upstream gradients.",
        "codec_audits compare public pack operands and exact-payload unpacking; fused LayerNorm/GELU use local_checks.",
        "fill_sensitivity holds deterministic algorithms enabled on both sides; determinism_sensitivity isolates that policy change.",
        "graph_vs_eager isolates replay effects under the captured deterministic and fill policies.",
        "Updates use the actual optimizer class, captured state and extra counters. Missing state never substitutes another optimizer.",
        "A small one-step difference does not prove long-run convergence."]
    write()
    return report


def run(packet, args):
    """Keep diagnostic calls from changing the caller's RNG and fill policy."""
    torch = _torch()
    device = torch.device(args.device)
    original_rng = _rng(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        return _run(packet, args)
    finally:
        _restore_rng(original_rng, device)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version",
                        version=f"DeepOpt stability diagnostic revision {DIAGNOSTIC_REVISION}\nRunner: {Path(__file__).resolve()}")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("stability-diagnostic"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-factory", default="timm:create_model")
    parser.add_argument("--model-kwargs", type=json.loads,
                        default={"model_name": "vit_base_patch16_224", "pretrained": False, "num_classes": 10})
    parser.add_argument("--optimizer", help="module:Class, if the captured module name is not importable")
    parser.add_argument("--probe-modules", nargs="+", default=["blocks.0.norm1", "blocks.5.mlp.act", "blocks.11.mlp.fc2"])
    parser.add_argument("--max-probe-bytes", type=int, default=4 * 2**30)
    parser.add_argument("--check-fill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--audit-codec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtol", type=float, default=.03)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.model_kwargs.get("pretrained"):
        parser.error("Use pretrained=False; snapshot weights supply the model")
    if (not math.isfinite(args.rtol) or not math.isfinite(args.atol)
            or args.rtol < 0 or args.atol < 0 or args.max_probe_bytes < 0):
        parser.error("tolerances and probe byte budget must be nonnegative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Custom optimizer modules live in the launch directory. The bundled
    # oracle must take precedence over any older helper left there.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(1, str(Path.cwd()))
    sources = _diagnostic_sources()
    print(json.dumps({"diagnostic_revision": DIAGNOSTIC_REVISION, "diagnostic_sources": sources}), flush=True)
    packet = _torch().load(args.snapshot, map_location="cpu", weights_only=False)
    report = run(packet, args)
    print("Report: " + str(args.output / "report.json"))
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
