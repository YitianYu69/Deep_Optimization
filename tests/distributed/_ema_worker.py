"""Real distributed EMA acceptance checks, invoked through torchrun."""

import gc
import json
import math
import traceback

import torch
from torch import distributed as dist, nn

from deepopt import Trainer, initialize_distributed, shutdown_distributed
from deepopt.tasks.regression import RegressionTask


def gathered(value, world_size):
    tensors = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(tensors, value.detach())
    return tensors


def training_and_validation(context):
    torch.manual_seed(19 + context.global_rank)
    model = nn.Linear(1, 1, bias=False)
    original = gathered(model.weight, context.world_size)
    assert not torch.equal(original[0], original[1]), "Test needs different initial rank weights"
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.25), task=RegressionTask(),
                      device="cpu", config={"training": {"ema": {"enabled": True, "decay": 0.5}}})
    assert trainer.runtime.distributed
    # EMA must initialize after DDP broadcasts rank zero's model parameters.
    for value in gathered(trainer.ema.shadow["weight"], context.world_size):
        torch.testing.assert_close(value, original[0], rtol=0, atol=0)
    torch.testing.assert_close(model.weight, original[0], rtol=0, atol=0)
    expected_weight, expected_average = original[0].clone(), original[0].clone()
    observed_steps = []

    def verify_update(current, event):
        nonlocal expected_weight, expected_average
        # Rank targets are one and three, so DDP's global target mean is two.
        expected_weight = 0.5 * expected_weight + 1
        expected_average = 0.5 * expected_average + 0.5 * expected_weight
        torch.testing.assert_close(model.weight, expected_weight)
        torch.testing.assert_close(current.ema.shadow["weight"], expected_average)
        assert current.ema.updates == event["optimizer_step"]
        for value in gathered(current.ema.shadow["weight"], context.world_size):
            torch.testing.assert_close(value, current.ema.shadow["weight"], rtol=0, atol=0)
        observed_steps.append(event["optimizer_step"])

    inputs = torch.ones(2, 1)
    targets = torch.full_like(inputs, 1 + 2 * context.global_rank)
    trainer.train_epoch([(inputs, targets)] * 2, step_callbacks=[verify_update])
    assert observed_steps == [1, 2]
    live_weight = model.weight.detach().clone()
    parameter = model.weight
    assert not torch.equal(expected_average, live_weight)
    validation = trainer.evaluate([(inputs, torch.zeros_like(inputs))])
    assert math.isclose(validation["loss"], expected_average.square().item(), rel_tol=1e-6)
    torch.testing.assert_close(model.weight, live_weight, rtol=0, atol=0)
    assert model.weight is parameter and model.training
    assert trainer.ema.updates == trainer.state.optimizer_steps == 2


def reject_mismatched_decay(context):
    model = nn.Linear(1, 1, bias=False)
    try:
        Trainer(model, torch.optim.SGD(model.parameters(), lr=0.25), task=RegressionTask(),
                device="cpu", config={"training": {"ema": {
                    "enabled": True, "decay": 0.5 if context.global_rank == 0 else 0.8}}})
    except ValueError as exc:
        assert "matching" in str(exc), str(exc)
    else:
        raise AssertionError("Rank-inconsistent EMA decay was accepted")
    # A successful subsequent collective proves both ranks rejected together.
    rejected = [None] * context.world_size
    dist.all_gather_object(rejected, context.global_rank)
    assert rejected == [0, 1]
    return rejected


def main():
    torch.set_num_threads(1)
    context = initialize_distributed("cpu", timeout=45)
    try:
        training_and_validation(context)
        gc.collect()
        rejected = reject_mismatched_decay(context)
        if context.is_main_process:
            print(json.dumps({"world_size": context.world_size, "updates": 2,
                              "initial_synchronized": True, "ema_synchronized": True,
                              "validation_restored": True, "mismatch_rejected": rejected}), flush=True)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        shutdown_distributed()


if __name__ == "__main__":
    main()
