"""Optional, bounded suite diagnostics; never query during throughput timing."""

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
import subprocess
import threading
from time import monotonic


_TIMING_LOCK = threading.RLock()


@contextmanager
def quiet_timing_window():
    # A query already in flight completes before the caller starts its clock.
    # Samplers use a nonblocking acquire and skip the entire measured window.
    with _TIMING_LOCK:
        yield


def _device_selector(device):
    logical = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    return visible[logical].strip() if visible != [""] and logical < len(visible) else str(logical)


class UtilizationTelemetry:
    """Low-rate external telemetry, bounded in RAM and outside hot windows.

    Values describe the whole suite outside its synchronized throughput calls.
    They are not active-GPU utilization measurements for those hot calls.
    Missing nvidia-smi or unsupported counters remain explicitly unavailable.
    """

    def __init__(self, device, interval=0, max_samples=3600, *, query=None):
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError("Telemetry interval must be finite and nonnegative")
        if interval and interval < 1:
            raise ValueError("Telemetry is low-rate: choose an interval of at least one second")
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("Telemetry sample limit must be positive")
        self.device, self.interval = str(device), interval
        self.samples = deque(maxlen=max_samples)
        self.query = query or self._query
        self.enabled = bool(interval) and self.device.startswith("cuda")
        self.dropped = self.skipped = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def _query(self):
        selector = _device_selector(self.device)
        gpu = subprocess.run(["nvidia-smi", "--id", selector,
            "--query-gpu=uuid,utilization.gpu,power.draw,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True)
        rows = [line.split(",") for line in gpu.stdout.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError("Telemetry did not resolve exactly one physical GPU")
        uuid, utilization, power, used = (part.strip() for part in rows[0])
        processes = subprocess.run(["nvidia-smi", "--id", selector,
            "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True)
        process_rows = []
        for line in processes.stdout.splitlines():
            if not line.strip():
                continue
            pid, memory = (part.strip() for part in line.split(","))
            process_rows.append({"pid": int(pid), "used_memory_mib": float(memory)})
            if len(process_rows) > 256:
                raise ValueError("Telemetry process count exceeds its evidence limit")
        values = [float(utilization), float(power), float(used)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Telemetry contains invalid GPU counters")
        return {"gpu_uuid": uuid, "utilization_percent": values[0], "power_watts": values[1],
                "used_memory_mib": values[2], "processes": process_rows}

    def sample(self):
        if not self.enabled:
            return
        if not _TIMING_LOCK.acquire(blocking=False):
            self.skipped += 1
            return
        try:
            row = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": monotonic(), **self.query()}
            if len(self.samples) == self.samples.maxlen:
                self.dropped += 1
            self.samples.append(row)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            _TIMING_LOCK.release()

    def __enter__(self):
        if self.enabled:
            def collect():
                while not self._stop.is_set():
                    self.sample()
                    self._stop.wait(self.interval)
            self._thread = threading.Thread(target=collect, name="benchmark-suite-telemetry", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.error = "Telemetry collector did not stop within its bounded query deadline"

    def report(self):
        return {"status": "disabled" if not self.enabled else "observed" if self.samples else "unavailable",
                "device": self.device, "interval_seconds": self.interval,
                "samples": list(self.samples), "max_samples": self.samples.maxlen,
                "dropped_oldest_samples": self.dropped, "skipped_hot_windows": self.skipped,
                "error": self.error, "hot_window_sampling": False,
                "scope": "suite diagnostics outside synchronized throughput windows; not hot-window GPU utilization"}
