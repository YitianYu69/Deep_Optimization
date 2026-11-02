"""Real Gloo ranks: equal values and no added per-window collectives."""

from datetime import timedelta
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp

from deepopt import Trainer
from deepopt.scheduling import ScheduleController, LinearWarmup


def _worker(rank, world, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(directory, "group").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=45))
    try:
        for accumulation in (1, 2, 4, 8):
            counts = []
            for enabled in (False, True):
                torch.manual_seed(11)
                model = nn.Linear(1, 1)
                curve = LinearWarmup(.1, 1., 10)
                schedules = ScheduleController({"weight": curve}) if enabled else None
                seen = []
                def step(model, batch):
                    weight = schedules["weight"] if enabled else curve.value_at(len(seen) // accumulation + 1)
                    seen.append(weight)
                    return model(batch).square().mean() * weight
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), step_fn=step,
                                  schedules=schedules, accumulation_steps=accumulation, execution_mode="max_speed")
                with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce, \
                        patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as gather:
                    trainer.train_epoch([torch.full((2, 1), (rank + 1) / 10)] * 9)
                    counts.append((reduce.call_count, gather.call_count))
                expected = [curve.value_at(i // accumulation + 1) for i in range(9)]
                assert seen == expected, (rank, seen, expected)
                if not enabled:
                    reference = [p.detach().clone() for p in model.parameters()]
                else:
                    for actual, target in zip(model.parameters(), reference):
                        torch.testing.assert_close(actual, target, rtol=0, atol=0)
                    assert schedules.completed_update == trainer.state.optimizer_steps
                    ranks = [None] * world
                    dist.all_gather_object(ranks, schedules.state_dict())
                    assert all(state == ranks[0] for state in ranks)
            assert counts[0] == counts[1], counts
        model = nn.Linear(1, 1)
        try:
            Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), loss_fn=nn.functional.mse_loss,
                    schedules=ScheduleController({"x": LinearWarmup(0, rank + 1, 10)}))
        except ValueError:
            pass
        else:
            raise AssertionError("Rank-inconsistent schedules were accepted")
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class ScheduleDistributedTests(unittest.TestCase):
    def run_ranks(self, world):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_worker, args=(world, directory), nprocs=world, join=False)
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Schedule DDP workers exceeded 120 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)

    def test_two_rank_consistency_and_collective_count(self):
        self.run_ranks(2)

    @unittest.skipUnless(os.environ.get("DEEPOPT_EXTENDED_DDP") == "1", "Extended four-rank matrix is opt-in")
    def test_four_rank_consistency_and_collective_count(self):
        self.run_ranks(4)


if __name__ == "__main__":
    unittest.main()
