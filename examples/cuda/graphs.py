"""Capture model training, callable inference, a decorator, and a raw block.

Run: python examples/cuda/graphs.py
     python examples/cuda/graphs.py --device cuda:1

Actual execution requires CUDA; --help works without GPU hardware.
"""

import argparse

import torch

from deepopt import CUDAGraphCore, TrainingRuntime
from deepopt.logger import log_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This example requires CUDA hardware and a CUDA-enabled PyTorch build")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(23)
    core = CUDAGraphCore(device)
    model = torch.nn.Linear(8, 2).to(device)
    sample = torch.randn(16, 8, device=device)
    captured = core.capture_model(model, (sample,))
    try:
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  device=device, execution_mode="max_speed")
        for _ in range(3):
            output = runtime.compute(torch.randn_like(sample), model)
            runtime.backward(output.square().mean())
            runtime.step()
    finally:
        captured.close()  # Restore eager forward after the final backward/update.

    # Plain callables keep trainable tensors explicit and replay through the handle.
    def project(x, weight):
        return x @ weight

    weight = torch.nn.Parameter(torch.randn(8, 2, device=device))
    callable_training = core.capture_callable(project, (sample, weight))
    try:
        callable_training(sample, weight).sum().backward()
        expected_gradient = sample.sum(dim=0)[:, None].expand_as(weight)
        torch.testing.assert_close(weight.grad, expected_gradient)
    finally:
        callable_training.close()

    def transform(x):
        return x.sin() * 2
    function_graph = core.capture_callable(transform, (sample,), autograd=False)
    try:
        for _ in range(2):
            inputs = torch.randn_like(sample)
            # Inference outputs share replay storage: clone values kept for later.
            saved = function_graph(inputs).clone()
            torch.testing.assert_close(saved, transform(inputs))
    finally:
        function_graph.close()

    @core.capture
    def decorated(x):
        return x.square() + 1
    try:
        torch.testing.assert_close(decorated(sample), sample.square() + 1)
        fresh = torch.randn_like(sample)
        torch.testing.assert_close(decorated(fresh), fresh.square() + 1)
    finally:
        decorated.close()

    core.warmup(lambda: sample.cos())
    with core.capture() as raw:
        result = sample.cos()
    try:
        sample.copy_(torch.randn_like(sample))  # Update values at the captured address.
        raw.replay()
        torch.testing.assert_close(result, sample.cos())
    finally:
        raw.close()
    log_output("Completed three model updates and verified callable gradients, inference, decorator, and raw replay.")


if __name__ == "__main__":
    main()
