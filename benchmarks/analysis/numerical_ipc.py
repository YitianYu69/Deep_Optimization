"""Bounded, authenticated CPU tensor transfer without checkpoint files.

Only descriptors enter JSON. Payloads travel as raw bytes over a private
loopback socket, in bounded chunks; no pickle, tensor files or shared-memory
files are used. Importing this helper does not import Torch.
"""

import hashlib
import hmac
import json
import math
import os
import secrets
import socket
import struct
import threading
from time import monotonic


CHANNEL_ENV = "DEEPOPT_NUMERICAL_CHANNEL"
CHUNK_BYTES = 1024 * 1024
HEADER_BYTES = 1024 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8,
                "torch.int8": 1, "torch.uint8": 1, "torch.int16": 2, "torch.int32": 4,
                "torch.int64": 8, "torch.bool": 1}


class IdleTimeout(TimeoutError):
    """No worker connected during the receiver's polling interval."""


class NumericalTransferError(ValueError):
    """A connected worker did not complete its declared numerical payload."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _raw(tensor):
    import torch

    return memoryview(tensor.reshape(-1).view(torch.uint8).numpy())


def _digest(data):
    digest = hashlib.sha256()
    for offset in range(0, len(data), CHUNK_BYTES):
        digest.update(data[offset:offset + CHUNK_BYTES])
    return digest.hexdigest()


def snapshot_descriptor(identity, tensors, metadata=None):
    """Return deterministic, JSON-safe metadata bound to all tensor contents."""
    import torch

    if (not isinstance(identity, dict) or not identity or not isinstance(tensors, dict) or not tensors
            or metadata is not None and not isinstance(metadata, dict)):
        raise ValueError("Numerical inputs require nonempty identity and named tensor mappings")
    if len(tensors) > 1024 or any(not isinstance(name, str) or not name or len(name) > 512 for name in tensors):
        raise ValueError("Numerical tensor names/count exceed the protocol limits")
    entries = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor is None:
            entries.append({"name": name, "shape": None, "dtype": None, "bytes": 0, "sha256": None})
            continue
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or tensor.requires_grad or str(tensor.dtype) not in _DTYPE_BYTES):
            raise ValueError(f"Numerical input {name} must be a detached dense CPU tensor")
        if tensor.ndim > 64:
            raise ValueError("Numerical tensor rank exceeds the protocol limit")
        entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "bytes": tensor.numel() * tensor.element_size(),
                        "sha256": _digest(_raw(tensor.contiguous()))})
    descriptor = {"schema_version": 1, "identity": identity, "metadata": metadata or {},
                  "tensors": entries, "total_bytes": sum(entry["bytes"] for entry in entries)}
    descriptor = json.loads(_json(descriptor))
    descriptor["descriptor_sha256"] = hashlib.sha256(_json(descriptor)).hexdigest()
    return descriptor


def _read(channel, size, *, stopped=None, deadline=None):
    result = bytearray(size)
    _read_into(channel, memoryview(result), stopped=stopped, deadline=deadline)
    return result


def _read_into(channel, data, *, stopped=None, deadline=None, digest=None):
    offset = 0
    while offset < len(data):
        if stopped is not None and stopped.is_set():
            raise RuntimeError("Numerical channel closed")
        if deadline is not None and monotonic() >= deadline:
            raise NumericalTransferError("Numerical channel transfer timed out")
        try:
            count = channel.recv_into(data[offset:offset + CHUNK_BYTES])
        except socket.timeout:
            continue
        if not count:
            raise NumericalTransferError("Numerical channel ended before the declared input was complete")
        if digest is not None:
            digest.update(data[offset:offset + count])
        offset += count


def _send_header(channel, value):
    header = _json(value)
    if len(header) > HEADER_BYTES:
        raise ValueError("Numerical descriptor exceeds the header limit")
    channel.sendall(struct.pack("!Q", len(header)))
    channel.sendall(header)


def _read_header(channel, **options):
    size = struct.unpack("!Q", _read(channel, 8, **options))[0]
    if not 0 < size <= HEADER_BYTES:
        raise ValueError("Invalid numerical descriptor length")
    return json.loads(_read(channel, size, **options))


def send_snapshot(identity, tensors, metadata=None, *, channel=None):
    """Send one snapshot; return its descriptor only after receiver validation.

    ``channel`` is the broker configuration dictionary, or omitted to use
    CHANNEL_ENV. The token is never included in the returned descriptor.
    """
    config = channel if channel is not None else json.loads(os.environ.get(CHANNEL_ENV, "null"))
    if not isinstance(config, dict) or config.get("host") != "127.0.0.1":
        raise ValueError("A private numerical channel is required")
    descriptor = snapshot_descriptor(identity, tensors, metadata)
    if descriptor["total_bytes"] > config.get("max_bytes", DEFAULT_MAX_BYTES):
        raise ValueError("Numerical snapshot exceeds the receiver's byte budget")
    with socket.create_connection((config["host"], config["port"]), timeout=config.get("timeout", 600)) as connection:
        _send_header(connection, {"token": config["token"], "descriptor": descriptor})
        for entry in descriptor["tensors"]:
            if entry["dtype"] is None:
                continue
            tensor = tensors[entry["name"]].contiguous()
            data = _raw(tensor)
            for offset in range(0, len(data), CHUNK_BYTES):
                connection.sendall(data[offset:offset + CHUNK_BYTES])
        connection.shutdown(socket.SHUT_WR)
        accepted = _read_header(connection, deadline=monotonic() + config.get("timeout", 600))
        if accepted != {"accepted": True, "descriptor_sha256": descriptor["descriptor_sha256"]}:
            raise ValueError("Numerical receiver rejected or changed the snapshot")
    return descriptor


class NumericalSnapshotBroker:
    """One private receiver with an explicit per-message memory ceiling.

    Call receive concurrently with the child process sending the snapshot.
    The caller owns cache limits across messages and must discard tensors after
    their direct comparisons. close interrupts pending receive calls.
    """

    def __init__(self, max_bytes=DEFAULT_MAX_BYTES, timeout=600, idle_timeout=None):
        if (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
                or idle_timeout is not None and (type(idle_timeout) not in (int, float)
                    or not math.isfinite(idle_timeout) or idle_timeout <= 0)):
            raise ValueError("Numerical channel limits must be positive")
        self.max_bytes, self.timeout = max_bytes, timeout
        self.idle_timeout = timeout if idle_timeout is None else idle_timeout
        self._stopped = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.listen(1)
        self._socket.settimeout(min(.2, self.idle_timeout))
        self.channel = {"host": "127.0.0.1", "port": self._socket.getsockname()[1],
                        "token": secrets.token_hex(32), "max_bytes": max_bytes, "timeout": timeout}

    @property
    def environment(self):
        return {CHANNEL_ENV: _json(self.channel).decode()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._stopped.set()
        self._socket.close()

    def receive(self, expected_identity=None, expected_shapes=None, *, on_snapshot=None):
        import torch
        from math import prod

        accept_deadline = monotonic() + self.idle_timeout
        while True:
            if self._stopped.is_set():
                raise RuntimeError("Numerical channel closed")
            if monotonic() >= accept_deadline:
                raise IdleTimeout("Numerical channel timed out waiting for a strategy")
            try:
                connection, _ = self._socket.accept()
                break
            except socket.timeout:
                continue
            except OSError:
                if self._stopped.is_set():
                    raise RuntimeError("Numerical channel closed") from None
                raise
        with connection:
            connection.settimeout(.2)
            deadline = monotonic() + self.timeout
            options = {"stopped": self._stopped, "deadline": deadline}
            request = _read_header(connection, **options)
            if (not isinstance(request, dict) or not isinstance(request.get("token"), str)
                    or not hmac.compare_digest(request["token"], self.channel["token"])):
                raise ValueError("Numerical channel authentication failed")
            descriptor = request.get("descriptor")
            if (not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1
                    or not isinstance(descriptor.get("identity"), dict) or not descriptor["identity"]
                    or expected_identity is not None and descriptor["identity"] != expected_identity
                    or not isinstance(descriptor.get("metadata"), dict)
                    or not isinstance(descriptor.get("tensors"), list) or not 0 < len(descriptor["tensors"]) <= 1024):
                raise ValueError("Numerical descriptor identity/schema differs from the planned strategy")
            signature = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
            if hashlib.sha256(_json(signature)).hexdigest() != descriptor.get("descriptor_sha256"):
                raise ValueError("Numerical descriptor fingerprint is invalid")
            names, total = set(), 0
            for entry in descriptor["tensors"]:
                if (not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
                        or not entry["name"] or len(entry["name"]) > 512 or entry["name"] in names):
                    raise ValueError("Numerical tensor name is missing, duplicated or unexpected")
                names.add(entry["name"])
                shape, dtype = entry.get("shape"), entry.get("dtype")
                if dtype is None:
                    if shape is not None or entry.get("bytes") != 0 or entry.get("sha256") is not None:
                        raise ValueError("Malformed absent numerical tensor")
                elif (dtype not in _DTYPE_BYTES or not isinstance(shape, list) or len(shape) > 64
                      or any(type(size) is not int or size < 0 for size in shape)
                      or type(entry.get("bytes")) is not int or entry["bytes"] != prod(shape) * _DTYPE_BYTES[dtype]
                      or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64):
                    raise ValueError("Malformed numerical tensor shape/dtype/size")
                if expected_shapes is not None:
                    if entry["name"] not in expected_shapes:
                        raise ValueError("Numerical tensor geometry differs from the fixture")
                    expected_shape = expected_shapes[entry["name"]]
                    if shape != (None if expected_shape is None else list(expected_shape)):
                        raise ValueError("Numerical tensor geometry differs from the fixture")
                total += entry["bytes"]
                if total > self.max_bytes:
                    raise ValueError("Numerical snapshot exceeds the receiver's byte budget")
            if (type(descriptor.get("total_bytes")) is not int or descriptor["total_bytes"] != total
                    or expected_shapes is not None and names != expected_shapes.keys()):
                raise ValueError("Numerical snapshot names/total bytes differ from its contract")
            tensors = {}
            for entry in descriptor["tensors"]:
                if entry["dtype"] is None:
                    tensors[entry["name"]] = None
                    continue
                tensor = torch.empty(entry["shape"], dtype=getattr(torch, entry["dtype"].removeprefix("torch.")))
                digest = hashlib.sha256()
                _read_into(connection, _raw(tensor), digest=digest, **options)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Numerical tensor content fingerprint differs from the descriptor")
                tensors[entry["name"]] = tensor
            # Require EOF before acknowledging: undeclared payload is invalid.
            while True:
                if self._stopped.is_set():
                    raise RuntimeError("Numerical channel closed")
                if monotonic() >= deadline:
                    raise NumericalTransferError("Numerical channel transfer timed out before payload EOF")
                try:
                    extra = connection.recv(1)
                    break
                except socket.timeout:
                    continue
            if extra:
                raise ValueError("Numerical channel contains trailing undeclared bytes")
            payload = {"identity": descriptor["identity"], "tensors": tensors,
                       "metadata": descriptor.get("metadata", {}), "descriptor": descriptor}
            # A live comparer may admit/consume this payload before the worker
            # receives its ACK. This avoids both a result/receipt race and CPU
            # comparison work overlapping the next benchmark's GPU timing.
            if on_snapshot is not None:
                on_snapshot(payload)
            _send_header(connection, {"accepted": True, "descriptor_sha256": descriptor["descriptor_sha256"]})
            return payload
