"""Capture native worker failures without changing distributed test behavior."""

from contextlib import contextmanager
import faulthandler
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def create_log_directory(temporary_directory, world):
    root = Path(os.environ.get("DEEPOPT_DDP_LOG_DIR") or temporary_directory)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"schedule-{world}rank-", dir=root))


@contextmanager
def rank_diagnostics(directory, rank, world, **metadata):
    """Capture Python and native stdout/stderr, plus fatal-signal tracebacks."""
    path = Path(directory, f"rank-{rank}.log")
    with path.open("w", buffering=1) as log:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        was_enabled = faulthandler.is_enabled()
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            faulthandler.enable(file=log, all_threads=True)

            def record(phase, **details):
                print(json.dumps({"time": time.monotonic(), "rank": rank,
                                  "world": world, "phase": phase, **details}),
                      file=log, flush=True)

            record("worker.start", pid=os.getpid(), **metadata)
            yield record
            record("worker.complete")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            faulthandler.disable()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            if was_enabled:
                faulthandler.enable(all_threads=True)


def print_rank_failures(processes, directory, *, tail_bytes=12000, stream=None):
    """Print bounded evidence while the rendezvous directory still exists."""
    stream = sys.stderr if stream is None else stream
    print(f"Schedule DDP worker failure; rank logs: {directory}", file=stream, flush=True)
    for rank, process in enumerate(processes):
        print(f"--- rank {rank}: pid={process.pid}, exitcode={process.exitcode} ---",
              file=stream, flush=True)
        try:
            with Path(directory, f"rank-{rank}.log").open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - tail_bytes))
                tail = log.read().decode("utf-8", errors="replace")
        except OSError as exc:
            tail = f"Rank log unavailable: {exc}"
        print(tail, file=stream, flush=True)
