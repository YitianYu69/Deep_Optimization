"""Bounded local torchrun execution without hostname-based rendezvous."""

import os
import signal
import socket
import subprocess
import sys


def run_torchrun(arguments, *, env=None, timeout=90, processes=2):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc-per-node={processes}",
               "--rdzv-backend=static", "--master-addr=127.0.0.1", f"--master-port={port}", *arguments]
    process = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name == "posix")
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        def stop(force=False):
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except ProcessLookupError:
                pass

        stop()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stop(force=True)
            stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError(f"torchrun exceeded {timeout}s; workers were stopped.\n{stdout[-2000:]}\n{stderr[-4000:]}") from None
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
