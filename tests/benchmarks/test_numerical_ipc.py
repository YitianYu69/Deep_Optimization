"""Numerical comparisons cross worker boundaries without tensor files."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import torch

from benchmarks.analysis import numerical_ipc as ipc


class NumericalIPCTests(unittest.TestCase):
    identity = {"suite_run_id": "current", "trial": 0, "strategy": "generated"}

    def test_import_and_help_helpers_need_no_torch(self):
        result = subprocess.run([sys.executable, "-S", "-c",
            "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules",
            str(Path(ipc.__file__))], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_roundtrip_preserves_shapes_dtypes_values_and_none_without_saving(self):
        tensors = {"strided": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
                   "bf16": torch.arange(15, dtype=torch.bfloat16).reshape(3, 5),
                   "scalar": torch.tensor(-3.5, dtype=torch.float64),
                   "empty": torch.empty(0, 2), "boolean": torch.tensor([True, False]),
                   "unused_gradient": None}
        shapes = {name: None if value is None else tuple(value.shape) for name, value in tensors.items()}
        metadata = {"original_shapes": {name: list(shape) for name, shape in shapes.items() if shape is not None}}
        with ipc.NumericalSnapshotBroker(timeout=3) as broker, ThreadPoolExecutor(1) as pool, \
                patch.object(torch, "save", side_effect=AssertionError("tensor file export")):
            pending = pool.submit(broker.receive, self.identity, shapes)
            descriptor = ipc.send_snapshot(self.identity, tensors, metadata, channel=broker.channel)
            received = pending.result(timeout=5)
            self.assertNotIn(broker.channel["token"], json.dumps(descriptor))
        self.assertEqual(received["descriptor"], descriptor)
        self.assertEqual(received["identity"], self.identity)
        self.assertEqual(received["metadata"], metadata)
        for name, expected in tensors.items():
            if expected is None:
                self.assertIsNone(received["tensors"][name])
            else:
                torch.testing.assert_close(received["tensors"][name], expected, rtol=0, atol=0)

    def test_raw_payload_spans_multiple_bounded_chunks(self):
        tensors = {"large": torch.arange(ipc.CHUNK_BYTES // 4 + 19, dtype=torch.float32)}
        with ipc.NumericalSnapshotBroker(timeout=3) as broker, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(broker.receive, self.identity)
            descriptor = ipc.send_snapshot(self.identity, tensors, channel=broker.channel)
            received = pending.result(timeout=5)
        self.assertGreater(descriptor["total_bytes"], ipc.CHUNK_BYTES)
        torch.testing.assert_close(received["tensors"]["large"], tensors["large"], rtol=0, atol=0)

    def test_live_comparison_receipt_is_published_before_worker_acknowledgement(self):
        accepted = []
        with ipc.NumericalSnapshotBroker(timeout=2) as broker, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(broker.receive, self.identity, on_snapshot=lambda payload: accepted.append(payload["descriptor"]))
            descriptor = ipc.send_snapshot(self.identity, {"output": torch.ones(2)}, channel=broker.channel)
            self.assertEqual(accepted, [descriptor])
            pending.result(timeout=3)

    def send_raw(self, broker, descriptor, payload=b"", *, token=None):
        with socket.create_connection((broker.channel["host"], broker.channel["port"]), timeout=2) as channel:
            request = {"token": broker.channel["token"] if token is None else token,
                       "descriptor": descriptor}
            try:
                ipc._send_header(channel, request)
                if payload:
                    channel.sendall(payload)
                channel.shutdown(socket.SHUT_WR)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def test_wrong_identity_authentication_and_geometry_fail_closed(self):
        tensors = {"value": torch.ones(2)}
        descriptor = ipc.snapshot_descriptor(self.identity, tensors)
        cases = (({"suite_run_id": "stale"}, None, None, "identity"),
                 (self.identity, {"value": (3,)}, None, "geometry"),
                 (self.identity, {"value": None}, None, "geometry"),
                 (self.identity, None, "wrong token", "authentication"))
        for expected, shapes, token, message in cases:
            with self.subTest(message=message), ipc.NumericalSnapshotBroker(timeout=2) as broker, ThreadPoolExecutor(1) as pool:
                pending = pool.submit(broker.receive, expected, shapes)
                self.send_raw(broker, descriptor, token=token)
                with self.assertRaisesRegex(ValueError, message):
                    pending.result(timeout=3)

    def test_budget_is_checked_before_allocating_received_tensors(self):
        descriptor = ipc.snapshot_descriptor(self.identity, {"value": torch.ones(2)})
        with ipc.NumericalSnapshotBroker(max_bytes=4, timeout=2) as broker, ThreadPoolExecutor(1) as pool, \
                patch.object(torch, "empty", side_effect=AssertionError("unadmitted allocation")) as allocate:
            pending = pool.submit(broker.receive, self.identity)
            self.send_raw(broker, descriptor)
            with self.assertRaisesRegex(ValueError, "byte budget"):
                pending.result(timeout=3)
            allocate.assert_not_called()

    def test_corruption_truncation_and_undeclared_bytes_cannot_be_accepted(self):
        tensors = {"value": torch.ones(2)}
        descriptor = ipc.snapshot_descriptor(self.identity, tensors)
        raw = bytes(ipc._raw(tensors["value"]))
        for payload, message in ((b"", "ended"), (bytes(len(raw)), "fingerprint"), (raw + b"x", "trailing")):
            with self.subTest(message=message), ipc.NumericalSnapshotBroker(timeout=2) as broker, ThreadPoolExecutor(1) as pool:
                pending = pool.submit(broker.receive, self.identity)
                self.send_raw(broker, descriptor, payload)
                with self.assertRaisesRegex(ValueError, message):
                    pending.result(timeout=3)

    def test_idle_polling_and_connected_transfer_timeouts_have_different_types(self):
        with ipc.NumericalSnapshotBroker(timeout=.1, idle_timeout=.03) as broker:
            with self.assertRaises(ipc.IdleTimeout):
                broker.receive()
        with ipc.NumericalSnapshotBroker(timeout=.05, idle_timeout=1) as broker, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(broker.receive)
            with socket.create_connection((broker.channel["host"], broker.channel["port"]), timeout=2) as channel:
                channel.sendall(b"\x00")  # Incomplete length prefix; keep the connection open.
                with self.assertRaisesRegex(ipc.NumericalTransferError, "timed out"):
                    pending.result(timeout=2)

    def test_oversized_header_is_rejected_before_json_or_tensor_allocation(self):
        with ipc.NumericalSnapshotBroker(timeout=2) as broker, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(broker.receive)
            with socket.create_connection((broker.channel["host"], broker.channel["port"]), timeout=2) as channel:
                channel.sendall(struct.pack("!Q", ipc.HEADER_BYTES + 1))
                with self.assertRaisesRegex(ValueError, "descriptor length"):
                    pending.result(timeout=3)

    def test_close_wakes_a_receiver_waiting_for_a_worker(self):
        ready = threading.Event()
        broker = ipc.NumericalSnapshotBroker(timeout=600)
        def receive():
            ready.set()
            return broker.receive()
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(receive)
            self.assertTrue(ready.wait(1))
            broker.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pending.result(timeout=2)

    def test_limits_and_tensor_inputs_are_validated_without_export(self):
        for arguments in ({"max_bytes": 0}, {"timeout": float("nan")}, {"timeout": True},
                          {"idle_timeout": 0}, {"idle_timeout": float("inf")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ipc.NumericalSnapshotBroker(**arguments)
        for tensor in (torch.ones(2, requires_grad=True), torch.ones(2, dtype=torch.complex64), "not a tensor"):
            with self.subTest(tensor=type(tensor).__name__), self.assertRaises(ValueError):
                ipc.snapshot_descriptor(self.identity, {"value": tensor})


if __name__ == "__main__":
    unittest.main()
