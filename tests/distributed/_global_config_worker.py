"""Global configuration acceptance on real Gloo ranks; invoked by torchrun."""

import gc
import json
import sys
import traceback
from unittest.mock import patch

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset

from deepopt import DDPCore, Trainer, TrainingRuntime, initialize_distributed, load_config, shutdown_distributed
from deepopt.config import get_config, reset_config
from deepopt.data import build_loaders


def configuration():
    return {"name": "distributed-defaults", "seed": 19,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.02, "accumulation_steps": 2},
            "runtime": {"device": "cpu", "fp8": {"enabled": True},
                        "ddp": {"broadcast_buffers": False, "bucket_cap_mb": 1}}}


def parameters_agree(model, initial, world):
    current = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert torch.isfinite(current).all()
    assert not torch.equal(current, initial)
    gathered = [torch.empty_like(current) for _ in range(world)]
    dist.all_gather(gathered, current)
    for value in gathered:
        torch.testing.assert_close(value, current, rtol=0, atol=0)


def global_trainer(context):
    load_config(configuration())
    torch.manual_seed(19)
    dataset = TensorDataset(torch.randn(32, 4), torch.randn(32, 3))
    loader, _, _ = build_loaders(dataset, None, batch_size=4, seed=19)
    model = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    runtime = TrainingRuntime(model, optimizer)
    trainer = Trainer(runtime, loss_fn=F.mse_loss)
    assert trainer.accumulation_steps == 2
    assert runtime.fp8.enabled
    assert runtime.core.options["broadcast_buffers"] is False
    assert runtime.core.options["bucket_cap_mb"] == 1

    changed = configuration()
    changed["training"]["accumulation_steps"] = 4
    changed["runtime"]["fp8"]["enabled"] = False
    changed["runtime"]["ddp"]["broadcast_buffers"] = True
    load_config(changed)
    assert trainer.accumulation_steps == 2
    assert runtime.fp8.enabled
    assert runtime.core.options["broadcast_buffers"] is False
    assert get_config().training.accumulation_steps == 4
    trainer.train_epoch(loader)
    assert trainer.state.optimizer_steps == 2
    parameters_agree(trainer.model, initial, context.world_size)
    return {"api": "trainer", "world_size": context.world_size, "updates": 2,
            "fp8": runtime.fp8.enabled, "accumulation": trainer.accumulation_steps,
            "broadcast_buffers": runtime.core.options["broadcast_buffers"]}


def per_call_override(context):
    load_config(configuration())
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(4, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    trainer = Trainer(
        model, optimizer, loss_fn=F.mse_loss, accumulation_steps=1,
        config={"training": {"accumulation_steps": 3},
                "runtime": {"fp8": {"enabled": False},
                            "ddp": {"broadcast_buffers": True, "bucket_cap_mb": 2}}},
        ddp_options={"bucket_cap_mb": 3},
    )
    assert trainer.accumulation_steps == 1
    assert not trainer.runtime.fp8.enabled
    assert type(trainer.model[0]) is nn.Linear
    assert trainer.runtime.core.options["broadcast_buffers"] is True
    assert trainer.runtime.core.options["bucket_cap_mb"] == 3
    assert trainer.optimizer is optimizer
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert get_config().runtime.ddp["broadcast_buffers"] is False
    assert get_config().runtime.fp8.enabled
    generator = torch.Generator().manual_seed(200 + context.global_rank)
    trainer.train_epoch([(torch.randn(4, 4, generator=generator), torch.randn(4, 3, generator=generator))])
    assert trainer.state.optimizer_steps == 1
    parameters_agree(trainer.model, initial, context.world_size)
    return {"api": "override", "world_size": context.world_size, "updates": 1,
            "fp8": trainer.runtime.fp8.enabled, "accumulation": trainer.accumulation_steps,
            "bucket_cap_mb": trainer.runtime.core.options["bucket_cap_mb"]}


def bare_core(context):
    load_config(configuration())
    torch.manual_seed(29)
    model = nn.Linear(4, 3)
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    initial = torch.cat([parameter.detach().flatten() for parameter in parameters]).clone()
    core = DDPCore(model, context.device, config={"runtime": {"ddp": {"bucket_cap_mb": 4}}})
    assert core.fp8.enabled
    assert core.options["broadcast_buffers"] is False
    assert core.options["bucket_cap_mb"] == 4
    assert all(a is b for a, b in zip(parameters, core.model.parameters()))
    generator = torch.Generator().manual_seed(300 + context.global_rank)
    inputs, targets = torch.randn(4, 4, generator=generator), torch.randn(4, 3, generator=generator)
    loss = core.compute((inputs, targets), lambda batch: F.mse_loss(core.model(batch[0]), batch[1]))
    core.backward(loss)
    optimizer.step()
    parameters_agree(core.model, initial, context.world_size)
    return {"api": "core", "world_size": context.world_size, "fp8": core.fp8.enabled,
            "bucket_cap_mb": core.options["bucket_cap_mb"]}


def mismatched_configuration(context):
    rejected = []
    for field in ("fp8", "ddp"):
        settings = configuration()
        if field == "fp8":
            settings["runtime"]["fp8"]["grad_dtype"] = "e4m3fn" if context.global_rank == 0 else "e5m2"
        else:
            settings["runtime"]["ddp"]["broadcast_buffers"] = bool(context.global_rank)
        load_config(settings)
        model = nn.Linear(4, 3)
        with patch.object(DistributedDataParallel, "__init__",
                          side_effect=AssertionError("DDP wrapped before configuration agreement")):
            try:
                TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.01))
            except ValueError as exc:
                assert "matching" in str(exc), str(exc)
                rejected.append(field)
            else:
                raise AssertionError("Rank-inconsistent configuration was accepted")

    settings = configuration()
    settings["training"]["accumulation_steps"] = context.global_rank + 1
    load_config(settings)
    model = nn.Linear(4, 3)
    try:
        Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), loss_fn=F.mse_loss)
    except ValueError as exc:
        assert "matching" in str(exc), str(exc)
        rejected.append("accumulation")
    else:
        raise AssertionError("Rank-inconsistent accumulation was accepted")
    return {"rejected": rejected, "world_size": context.world_size}


def invalid_configuration(context, api):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    local = {"runtime": {"checks": "invalid-rank-config" if context.global_rank == 0 else True}}
    with patch.object(DistributedDataParallel, "__init__",
                      side_effect=AssertionError("DDP wrapped before configuration validation")):
        try:
            if api == "runtime":
                TrainingRuntime(model, optimizer, config=local)
            elif api == "trainer":
                Trainer(model, optimizer, loss_fn=F.mse_loss, config=local)
            else:
                DDPCore(model, context.device, config=local)
        except (ValueError, RuntimeError) as exc:
            message = str(exc)
            assert "checks" in message, message
            assert "Connection closed" not in message, message
            assert "timed out" not in message.lower(), message
        else:
            raise AssertionError("Invalid rank-local configuration was accepted")
    # A successful exchange after rejection proves every rank left validation
    # together; a local raise followed by a peer's native timeout is not enough.
    statuses = [None] * context.world_size
    dist.all_gather_object(statuses, {"api": api, "rank": context.global_rank})
    assert all(status["api"] == api for status in statuses)
    return {"invalid_rejected": api, "ranks": [status["rank"] for status in statuses]}


def main():
    torch.set_num_threads(1)
    load_config(configuration())
    context = initialize_distributed("cpu", timeout=45)
    try:
        if sys.argv[1].startswith("invalid-"):
            rows = [invalid_configuration(context, sys.argv[1].removeprefix("invalid-"))]
        elif sys.argv[1] == "mismatch":
            rows = [mismatched_configuration(context)]
        else:
            rows = []
            for function in (global_trainer, per_call_override, bare_core):
                rows.append(function(context))
                gc.collect()
        if context.is_main_process:
            for row in rows:
                print(json.dumps(row), flush=True)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        shutdown_distributed()
        reset_config()


if __name__ == "__main__":
    main()
