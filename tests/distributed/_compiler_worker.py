"""Two-rank compiler integration, retaining the existing eager DDP reducer."""

from copy import deepcopy
import json

import torch
from torch import distributed as dist, nn

from deepopt import (
    TrainingRuntime,
    autocast,
    initialize_distributed,
    shutdown_distributed,
)
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


def main():
    context = initialize_distributed("cpu")
    try:
        for fp8 in (False, True):
            torch.compiler.reset()
            torch.manual_seed(91)
            model = prepare_fp8_model(
                nn.Sequential(nn.Linear(4, 8), nn.Sigmoid(), nn.Linear(8, 2)), fp8
            )[0]
            reference = deepcopy(model)
            identities = [id(parameter) for parameter in model.parameters()]
            compile_model(
                model, CompileConfig(enabled=True, backend="deepopt", dynamic=False)
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
            runtime = TrainingRuntime(
                model, optimizer, fp8=fp8, execution_mode="max_speed"
            )
            generator = torch.Generator().manual_seed(10 + context.global_rank)
            x, target = (
                torch.randn(3, 4, generator=generator),
                torch.randn(3, 2, generator=generator),
            )
            loss = runtime.compute(
                (x, target),
                lambda batch: nn.functional.mse_loss(model(batch[0]), batch[1]),
            )
            runtime.backward(loss)
            runtime.step()
            with autocast("cpu", enabled=False, fp8=fp8, check_finite=False):
                expected = nn.functional.mse_loss(reference(x), target)
            expected.backward()
            for parameter in reference.parameters():
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(context.world_size)
            reference_optimizer.step()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert identities == [id(parameter) for parameter in model.parameters()]
            assert compilation_report(model).captured_graphs > 0
            if fp8:
                assert compilation_report(model).deepopt_operations > 0
            if context.global_rank == 0:
                print(
                    json.dumps(
                        {
                            "compiled_ddp": True,
                            "fp8": fp8,
                            "world_size": context.world_size,
                        }
                    )
                )
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    main()
