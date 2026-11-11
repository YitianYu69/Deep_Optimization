"""Owned delayed-scaling state and explicit, nestable update boundaries."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from threading import RLock

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.config import delayed_scaling_config_from_dict
from .._execution import finite_checks_enabled
from ..quantization.tensor import QuantizedTensor


ROLES = ("activation", "weight", "grad")
_batch = ContextVar("deepopt_delayed_batch", default=None)
_recomputation = ContextVar("deepopt_delayed_recomputation", default=None)


class DelayedScalingState(nn.Module):
    """Numerical buffers belong to one projection, independently of its weight."""

    def __init__(self, options, dtypes, *, device):
        super().__init__()
        self.options = delayed_scaling_config_from_dict(options)
        self.dtypes = dict(dtypes)
        self.recipe_signature = (tuple(self.dtypes[role] for role in ROLES),
                                 (self.options.history_length, self.options.amax_algorithm,
                                  self.options.margin, self.options.reduce_amax))
        with torch.inference_mode(False):
            self.register_buffer("scales", torch.ones(3, dtype=torch.float32, device=device))
            self.register_buffer("history", torch.zeros((3, self.options.history_length), dtype=torch.float32, device=device))
            self.register_buffer("updates", torch.zeros(3, dtype=torch.int64, device=device))
            self.register_buffer("generations", torch.zeros(3, dtype=torch.int64, device=device))
        self._epoch = 0
        self._pending_owner = None
        self._stream = self._current_stream()
        self._lock = RLock()
        self._storage_epoch = 0
        self._metadata_cache = None

    def __getstate__(self):
        if self._pending_owner is not None:
            raise RuntimeError("Finish the delayed-scaling boundary before copying state")
        result = super().__getstate__()
        result.pop("_lock", None)
        result["_stream"] = None
        result["_metadata_cache"] = None
        return result

    def __setstate__(self, values):
        super().__setstate__(values)
        self._lock = RLock()
        self._stream = self._current_stream()

    def _current_stream(self):
        if self.scales.device.type == "cuda" and not isinstance(self.scales, FakeTensor):
            return torch.cuda.current_stream(self.scales.device)
        return None

    def get_extra_state(self):
        if self._pending_owner is not None:
            raise RuntimeError("Finish the delayed-scaling boundary before checkpointing")
        return {"version": 1, "options": asdict(self.options),
                "dtypes": {role: str(dtype) for role, dtype in self.dtypes.items()}}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Delayed-scaling recipe differs from saved state")
        self._epoch += 1
        self._storage_epoch += 1
        self._stream = self._current_stream()
        if self._metadata_cache is not None:
            self._metadata_cache.clear()

    def _apply(self, fn, recurse=True):
        if self._pending_owner is not None:
            raise RuntimeError("Cannot move delayed-scaling state during an active boundary")
        # Infer the destination device without first rounding FP32 metadata to
        # a parameter dtype (calling .float() afterward would lose precision).
        device = fn(torch.empty(0, device=self.scales.device)).device
        for name, value in self._buffers.items():
            self._buffers[name] = value.to(device=device)
        self._epoch += 1
        self._storage_epoch += 1
        self._metadata_cache = None
        self._stream = self._current_stream()
        return self

    def ordered(self):
        if isinstance(self.scales, FakeTensor):
            return
        if self.scales.device.type != "cuda":
            return
        current = torch.cuda.current_stream(self.scales.device)
        if self._stream is not None and self._stream != current:
            event = torch.cuda.Event()
            event.record(self._stream)
            current.wait_event(event)
            for value in self._buffers.values():
                value.record_stream(current)
        self._stream = current

    @property
    def cache_token(self):
        return ("delayed", id(self), self._epoch)

    @property
    def metadata_cache(self):
        if self._metadata_cache is None:
            from deepopt.opt.backends.triton.quantization.launch import MetadataCache

            self._metadata_cache = MetadataCache()
        return self._metadata_cache

    @torch.no_grad()
    def quantize(self, x, role, *, normal=True, transposed=False, bias=False, training=True, check_finite=None):
        return self._quantize(x, role, normal=normal, transposed=transposed, bias=bias,
                              training=training, check_finite=check_finite)[:3]

    @torch.no_grad()
    def quantize_pack_activation(self, x, *, group_size=512, training=True, check_finite=None, bits=2):
        """Use the same owned scale snapshot/observations with fused affine saving."""
        normal, _, _, packed = self._quantize(x, "activation", training=training,
                                              check_finite=check_finite, compression_group_size=group_size,
                                              compression_bits=bits)
        return normal, packed

    @torch.no_grad()
    def relu_forward(self, x, geometry, compute, arguments, *, training=True):
        return self.activation_forward(x, geometry, compute, arguments, training=training)

    @torch.no_grad()
    def activation_forward(self, x, geometry, compute, arguments, *, training=True):
        """Use owned delayed scales in fused activation execution without dense intermediates."""
        if x.device != self.scales.device:
            raise ValueError("Move the prepared adapter and delayed scales to the input device")
        from ..quantization.activation_tensor import FP8Activation

        storage = x._data if isinstance(x, FP8Activation) else x
        if isinstance(storage, FakeTensor) or not x.numel():
            return compute.forward(x, geometry, **{**arguments, "scale": self.scales[:1].view(1, 1)})
        if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Delayed scaling requires eager preprocessing; full training CUDA capture is unsupported")
        replay = _recomputation.get()
        if torch._C._current_graph_task_id() >= 0 and replay is None:
            raise RuntimeError("Delayed scaling with activation checkpointing requires deepopt.fp8_checkpoint")
        with fp8_scaling_scope(), self._lock:
            self.ordered()
            if self._pending_owner is not None and self._pending_owner is not _batch.get():
                raise RuntimeError("Overlapping delayed-scaling boundaries cannot share an output scale owner")
            scale = self.scales[:1]
            observe = training and not (replay is not None and replay[0] == "replay")
            if replay is not None and replay[0] == "replay":
                tape, position = replay[1], replay[2]
                if position >= len(tape) or tape[position][:2] != (self, "activation"):
                    raise RuntimeError("Activation checkpoint recomputation changed delayed output order")
                scale = tape[position][2]
                replay[2] += 1
            result = compute.forward(x, geometry, **{**arguments, "scale": scale.view(1, 1)},
                                     observe=observe, capture_scale=True)
            _, _, _, snapshot, observations, info = result
            info["scale_mode"] = "delayed"
            if replay is not None and replay[0] == "record":
                replay[1].append((self, "activation", snapshot[:1]))
            if observe:
                _batch.get().record(self, 0, observations)
            return result

    def _quantize(self, x, role, *, normal=True, transposed=False, bias=False, training=True,
                  check_finite=None, compression_group_size=None, compression_bits=2):
        from .amax import AmaxScaling

        checked = finite_checks_enabled() if check_finite is None else check_finite
        if compression_group_size is not None:
            from ..quantization.activation import _validate_bits

            _validate_bits(compression_bits)
        dtype = self.dtypes[role]
        if not normal and not transposed:
            raise ValueError("Delayed conversion must emit at least one FP8 orientation")
        AmaxScaling()._validate_input(x, dtype, checked)
        if x.device != self.scales.device:
            raise ValueError("Move the prepared adapter and delayed-scaling buffers to the input device")
        if isinstance(x, FakeTensor):
            m, n = x.shape
            inverse = torch.empty((1, 1), device=x.device, dtype=torch.float32)
            values = (torch.empty_strided((m, n), (((n + 15) // 16) * 16, 1), device=x.device, dtype=dtype)
                      if normal else None)
            transpose = (torch.empty_strided((n, m), (((m + 15) // 16) * 16, 1), device=x.device, dtype=dtype)
                         if transposed else None)
            packed = None
            if compression_group_size is not None:
                from ..quantization.activation import quantize_pack_activation

                packed = quantize_pack_activation(values.float(), group_size=compression_group_size, bits=compression_bits)
            return (QuantizedTensor(values, inverse) if normal else None,
                    QuantizedTensor(transpose, inverse) if transposed else None,
                    torch.empty(n, device=x.device, dtype=torch.float32) if bias else None, packed)
        replay = _recomputation.get()
        if torch._C._current_graph_task_id() >= 0 and role != "grad" and replay is None:
            raise RuntimeError("Delayed scaling with activation checkpointing requires deepopt.fp8_checkpoint")
        index = ROLES.index(role)
        with fp8_scaling_scope(), self._lock:
            self.ordered()
            if self._pending_owner is not None and self._pending_owner is not _batch.get():
                raise RuntimeError("Overlapping delayed-scaling boundaries cannot share a projection")
            scale = self.scales[index:index + 1]
            observe = training and not (replay is not None and replay[0] == "replay")
            if replay is not None and replay[0] == "replay":
                tape, position = replay[1], replay[2]
                if position >= len(tape) or tape[position][:2] != (self, role):
                    raise RuntimeError("Activation checkpoint recomputation changed delayed projection order")
                scale = tape[position][2]
                replay[2] += 1
            packed = None
            if x.device.type == "cuda":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Delayed scaling requires eager preprocessing; full training CUDA capture is unsupported")
                if compression_group_size is not None:
                    from deepopt.opt.backends.triton.quantization.activation.fp8 import quantize_pack_delayed_fp8_activation

                    values, packed, snapshot, observations = quantize_pack_delayed_fp8_activation(
                        x, scale, dtype, compression_group_size, observe=observe, check_finite=checked,
                        bits=compression_bits)
                    transpose, db = None, None
                else:
                    from deepopt.opt.backends.triton.quantization.delayed import convert

                    values, transpose, snapshot, observations, db = convert(
                        x, scale, dtype, normal=normal, transposed=transposed, bias=bias,
                        observe=observe, check_finite=checked)
            else:
                values32 = x.float()
                if checked and not torch.isfinite(values32).all():
                    raise ValueError("Delayed scaling requires finite inputs")
                snapshot = torch.stack((scale[0], torch.div(torch.ones_like(scale[0]), scale[0])))
                maximum = torch.finfo(dtype).max
                quantized = (values32 * snapshot[0]).clamp(-maximum, maximum).to(dtype)
                values = quantized.contiguous() if normal else None
                transpose = quantized.T.contiguous() if transposed else None
                observations = values32.abs().amax().reshape(1) if observe else None
                db = values32.sum(0) if bias else None
                if compression_group_size is not None:
                    from ..quantization.activation import quantize_pack_activation

                    packed = quantize_pack_activation(quantized.float(), group_size=compression_group_size,
                                             bits=compression_bits)
            if replay is not None and replay[0] == "record":
                replay[1].append((self, role, snapshot[:1]))
            if observe:
                _batch.get().record(self, index, observations)
            inverse = snapshot[1:].view(1, 1)
            return (QuantizedTensor(values, inverse) if normal else None,
                    QuantizedTensor(transpose, inverse) if transposed else None, db, packed)


class _UpdateBatch:
    def __init__(self, states=()):
        self.states = list(dict.fromkeys(states))
        if any(state._pending_owner is not None for state in self.states):
            raise RuntimeError("Overlapping delayed-scaling boundaries cannot share a projection")
        self.observations = []
        self.bytes = 0
        self.claimed = set()
        self._storages = set()
        self.active = True
        self._lock = RLock()

    def record(self, state, role, value):
        # Device-specific autograd workers can append for different projections
        # concurrently; compaction must not race another append.
        with self._lock:
            if not self.active:
                raise RuntimeError("Delayed-scaling boundary has already finished")
            if state not in self.states:
                self.states.append(state)
            state._pending_owner = self
            self.claimed.add(state)
            self.observations.append((state, role, value))
            from deepopt.opt.backends.triton.quantization.launch import storage_key

            storage = storage_key(value)
            if storage not in self._storages:
                self._storages.add(storage)
                self.bytes += storage[2]
            if self.bytes > 32 * 1024 * 1024 or len(self.observations) > max(512, len(self.states) * 6):
                self.compact()

    def reduced(self, states):
        groups = {(state, role): index for index, (state, role) in
                  enumerate((state, role) for state in states for role in range(3))}
        values = [(groups[state, role], value) for state, role, value in self.observations if state in states]
        device = states[0].scales.device
        if device.type == "cuda":
            for state in states:
                state.ordered()
            from deepopt.opt.backends.triton.quantization.delayed import reduce_observations

            with torch.cuda.device(device):
                packed = reduce_observations(values, len(groups), device, states[0].metadata_cache)
        else:
            packed = torch.zeros((3, len(groups)), dtype=torch.float32, device=device)
            for group, value in values:
                observed = value.amax()
                packed[0, group] = torch.maximum(packed[0, group], observed.nan_to_num(nan=0, posinf=0))
                packed[1, group] = 1
                kind = torch.where(observed.isnan(), 2., torch.where(observed.isinf(), 1., 0.))
                packed[2, group] = torch.maximum(packed[2, group], kind)
        return packed

    def devices(self):
        return {device: [state for state in self.states if state.scales.device == device]
                for device in dict.fromkeys(state.scales.device for state in self.states)}

    def compact(self):
        observations = []
        for states in self.devices().values():
            packed = self.reduced(states)
            values = torch.where(packed[2] == 2, float("nan"),
                                 torch.where(packed[2] == 1, float("inf"), packed[0]))
            for state in states:
                for role in range(3):
                    if any(item[0] is state and item[1] == role for item in self.observations):
                        index = states.index(state) * 3 + role
                        observations.append((state, role, values[index:index + 1]))
        self.observations = observations
        from deepopt.opt.backends.triton.quantization.launch import storage_key

        self._storages = {storage_key(value) for _, _, value in observations}
        self.bytes = sum(key[2] for key in self._storages)

    @torch.no_grad()
    def finish(self):
        committed = False
        try:
            if any(state._pending_owner is not None and state._pending_owner is not self for state in self.states):
                raise RuntimeError("Overlapping delayed-scaling boundaries cannot share a projection")
            synchronized = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
            if not self.observations and not (synchronized and any(state.options.reduce_amax for state in self.states)):
                return
            for states in self.devices().values():
                packed = self.reduced(states)
                if synchronized and all(state.options.reduce_amax for state in states):
                    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.MAX)
                elif synchronized and any(state.options.reduce_amax for state in states):
                    raise ValueError("A delayed boundary cannot mix local and synchronized amax policies")
                if packed.device.type == "cuda":
                    from deepopt.opt.backends.triton.quantization.delayed import update_states

                    with torch.cuda.device(packed.device):
                        update_states(states, packed, states[0].metadata_cache)
                    for state in states:
                        state._epoch += 1
                else:
                    for index, state in enumerate(states):
                        for role in range(3):
                            group = index * 3 + role
                            if not packed[1, group]:
                                continue
                            observed = (float("nan") if packed[2, group] == 2 else
                                        float("inf") if packed[2, group] == 1 else packed[0, group])
                            slot = int(state.updates[role]) % state.options.history_length
                            state.history[role, slot] = observed
                            amax = state.history[role].amax() if state.options.amax_algorithm == "max" else state.history[role, slot]
                            maximum = torch.full_like(amax, torch.finfo(state.dtypes[ROLES[role]]).max)
                            candidate = torch.div(maximum, amax) / (2. ** state.options.margin)
                            valid = torch.isfinite(amax) & (amax > 0) & torch.isfinite(candidate) & (candidate > 0)
                            state.scales[role] = torch.where(valid, candidate, state.scales[role])
                            state.updates[role] += 1
                            state.generations[role] += 1
                        state._epoch += 1
            committed = True
        finally:
            self.release(aborted=not committed)

    def release(self, *, aborted=True):
        self.active = False
        for state in self.claimed:
            if aborted:
                # A cached weight from an abandoned window must be observed
                # again, even though its parameter and old scale are unchanged.
                state._epoch += 1
            state._pending_owner = None
        self.observations.clear()
        self.claimed.clear()
        self._storages.clear()
        self.bytes = 0


@contextmanager
def fp8_scaling_scope(*, states=()):
    """Freeze scales within this boundary; combine observations once on exit.

    An explicit outer boundary can span a gradient-accumulation window. Nested
    FP8/autocast scopes inherit it. Evaluation never records observations.
    """
    existing = _batch.get()
    if existing is not None:
        for state in states:
            if state._pending_owner is not None and state._pending_owner is not existing:
                raise RuntimeError("Overlapping delayed-scaling boundaries cannot share a projection")
            if state not in existing.states:
                existing.states.append(state)
        yield
        return
    batch = _UpdateBatch(states)
    token = _batch.set(batch)
    try:
        yield
    except BaseException:
        batch.release()
        raise
    else:
        batch.finish()
    finally:
        _batch.reset(token)


def delayed_states(model):
    return tuple(module for module in model.modules() if isinstance(module, DelayedScalingState))


def current_scaling_batch():
    return _batch.get()


@contextmanager
def autograd_scaling_scope(batch):
    """Rebind a still-open forward owner on an autograd worker thread.

    Closed standalone/aborted forward scopes are never reopened. A backward
    after such a scope uses its current boundary or ordinary standalone policy.
    """
    if batch is None or not batch.active:
        yield
        return
    current = _batch.get()
    if current is not None and current is not batch:
        raise RuntimeError("Overlapping delayed-scaling boundaries cannot share a projection")
    token = _batch.set(batch)
    try:
        yield
    finally:
        _batch.reset(token)


def checkpoint_recomputing():
    return _recomputation.get() is not None


@contextmanager
def runtime_scaling_scope(runtime):
    """Keep optimizer-owned scales frozen across microbatches until step()."""
    if getattr(runtime, "_delayed_state_owner", None) is not runtime.model:
        runtime._delayed_state_owner = runtime.model
        runtime._delayed_state_cache = delayed_states(runtime.model)
    states = runtime._delayed_state_cache
    if _batch.get() is not None or runtime.optimizer is None:
        with fp8_scaling_scope(states=states if runtime.model.training else ()):
            yield
        return
    window = getattr(runtime, "_delayed_window", None)
    if window is None:
        if not runtime.model.training or not torch.is_grad_enabled():
            yield
            return
        window = runtime._delayed_window = _UpdateBatch(states)
    token = _batch.set(window)
    try:
        yield
    except BaseException:
        window.release()
        runtime._delayed_window = None
        raise
    finally:
        _batch.reset(token)


def finish_runtime_scaling(runtime, *, commit):
    window = getattr(runtime, "_delayed_window", None)
    if window is not None:
        runtime._delayed_window = None
        if commit:
            window.finish()
        else:
            window.release()


def fp8_checkpoint(function, *args, **kwargs):
    """Non-reentrant activation checkpointing with immutable delayed scales."""
    from torch.utils.checkpoint import checkpoint

    if _recomputation.get() is not None:
        raise ValueError("Nested fp8_checkpoint is unsupported; checkpoint one enclosing region instead")
    if kwargs.pop("use_reentrant", False) or "context_fn" in kwargs:
        raise ValueError("fp8_checkpoint owns non-reentrant context_fn")
    from .._autocast import (_active_fp8, _active_interception, _active_convolution_interception,
                             _active_normalization_interception, _active_compression, _FP8LinearMode)
    from ..precision.planner import _frame, _Frame
    from .._execution import finite_check_scope
    from contextlib import nullcontext

    plans, interception = dict(_active_fp8.get()), dict(_active_interception.get())
    convolution, normalization = dict(_active_convolution_interception.get()), dict(_active_normalization_interception.get())
    compression = dict(_active_compression.get())
    frame = _frame.get()
    frame_positions = dict(frame.positions) if frame is not None else None
    checked = finite_checks_enabled()
    tape = []

    @contextmanager
    def context(mode):
        token = _recomputation.set([mode, tape, 0])
        plan_token = _active_fp8.set(plans)
        interception_token = _active_interception.set(interception)
        convolution_token = _active_convolution_interception.set(convolution)
        normalization_token = _active_normalization_interception.set(normalization)
        compression_token = _active_compression.set(compression)
        frame_token = (_frame.set(_Frame(frame.root, frame.training, dict(frame_positions)))
                       if mode == "replay" and frame is not None else None)
        try:
            intercept = (mode == "replay" and (
                ((any(interception.values()) or any(convolution.values()) or any(normalization.values()))
                 and any(plan is not False for plan in plans.values()))
                or any(policy["enabled"] for policy in compression.values())
                or frame is not None and bool(frame.root._deepopt_precision_functional.get(frame.training))))
            with finite_check_scope(checked) if mode == "replay" else nullcontext(), \
                 _FP8LinearMode() if intercept else nullcontext():
                yield
        finally:
            if frame_token is not None:
                _frame.reset(frame_token)
            _active_compression.reset(compression_token)
            _active_normalization_interception.reset(normalization_token)
            _active_convolution_interception.reset(convolution_token)
            _active_interception.reset(interception_token)
            _active_fp8.reset(plan_token)
            _recomputation.reset(token)

    return checkpoint(function, *args, use_reentrant=False,
                      context_fn=lambda: (context("record"), context("replay")), **kwargs)
