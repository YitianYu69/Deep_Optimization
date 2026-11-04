"""Real Gloo ranks: equal values and no added per-window collectives."""

from datetime import timedelta
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp

from deepopt import Trainer
from deepopt.scheduling import ScheduleController, LinearWarmup
from tests.scheduling._ddp_diagnostics import (
    create_log_directory, print_rank_failures, rank_diagnostics,
)


def _worker(rank, world, directory, log_directory):
    with rank_diagnostics(log_directory, rank, world, torch_version=torch.__version__) as record:
        _run_worker(rank, world, directory, record)


def _run_worker(rank, world, directory, record):
    torch.set_num_threads(1)
    record("process_group.init.start")
    dist.init_process_group("gloo", init_method=Path(directory, "group").as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=45))
    record("process_group.init.complete")
    try:
        for accumulation in (1, 2, 4, 8):
            counts = []
            for enabled in (False, True):
                details = {"accumulation": accumulation, "enabled": enabled}
                record("trainer.init.start", **details)
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
                record("trainer.init.complete", **details)
                with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce, \
                        patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as gather:
                    record("train_epoch.start", **details)
                    trainer.train_epoch([torch.full((2, 1), (rank + 1) / 10)] * 9)
                    counts.append((reduce.call_count, gather.call_count))
                    record("train_epoch.complete", all_reduce=reduce.call_count,
                           all_gather_object=gather.call_count, **details)
                expected = [curve.value_at(i // accumulation + 1) for i in range(9)]
                assert seen == expected, (rank, seen, expected)
                if not enabled:
                    reference = [p.detach().clone() for p in model.parameters()]
                else:
                    for actual, target in zip(model.parameters(), reference):
                        torch.testing.assert_close(actual, target, rtol=0, atol=0)
                    assert schedules.completed_update == trainer.state.optimizer_steps
                    ranks = [None] * world
                    record("schedule_state.gather.start", **details)
                    dist.all_gather_object(ranks, schedules.state_dict())
                    record("schedule_state.gather.complete", **details)
                    assert all(state == ranks[0] for state in ranks)
            assert counts[0] == counts[1], counts
            record("accumulation.complete", accumulation=accumulation)
        model = nn.Linear(1, 1)
        record("inconsistent_schedule.init.start")
        try:
            Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), loss_fn=nn.functional.mse_loss,
                    schedules=ScheduleController({"x": LinearWarmup(0, rank + 1, 10)}))
        except ValueError as exc:
            record("inconsistent_schedule.rejected", error=str(exc))
            assert "matching trainer settings" in str(exc), str(exc)
        else:
            raise AssertionError("Rank-inconsistent schedules were accepted")
        record("inconsistent_schedule.complete")
    except BaseException as exc:
        record("worker.error", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        record("process_group.destroy.start")
        dist.destroy_process_group()
        record("process_group.destroy.complete")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class ScheduleDistributedTests(unittest.TestCase):
    def run_ranks(self, world):
        with tempfile.TemporaryDirectory() as directory:
            log_directory = create_log_directory(directory, world)
            context = mp.spawn(_worker, args=(world, directory, str(log_directory)),
                               nprocs=world, join=False)
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Schedule DDP workers exceeded 120 seconds")
            except BaseException:
                print_rank_failures(context.processes, log_directory)
                raise
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
