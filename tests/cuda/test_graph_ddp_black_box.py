"""Two real NCCL ranks compose model graphs with the existing DDP core."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import time
import unittest

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp

from deepopt import CUDAGraphCore, DDPCore, TrainingRuntime


def _worker(rank, directory):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=Path(directory, "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(71)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2)).to(device)
        reference = deepcopy(model)
        sample = torch.zeros(3, 4, device=device)
        capture = CUDAGraphCore(device).capture_model(model, (sample,))
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.02),
                                  device=device, execution_mode="max_speed")
        assert isinstance(runtime.core, DDPCore)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.02)
        for update in range(4):
            for micro in range(2):
                all_inputs = torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 4)
                all_inputs = (all_inputs + update + micro) / 20
                value = all_inputs[rank].clone()
                with runtime.core.accumulation_context(synchronize=micro == 1):
                    output = runtime.compute(value, model)
                    runtime.backward(output.square().mean())
                reference(all_inputs.reshape(6, 4)).square().mean().backward()
            runtime.step(gradient_divisor=2)
            for parameter in reference.parameters():
                parameter.grad.div_(2)
            reference_optimizer.step()
            reference_optimizer.zero_grad(set_to_none=True)
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        capture.close()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                     "Two CUDA GPUs and NCCL are required")
class CUDAGraphDDPBlackBoxTests(unittest.TestCase):
    def test_autograd_graphs_and_ddp_accumulation_match_global_eager_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_worker, args=(directory,), nprocs=2, join=False)
            deadline = time.monotonic() + 180
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("CUDA Graph/DDP workers exceeded 180 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
