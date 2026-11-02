"""Run: python examples/cuda/graphs.py (requires a CUDA GPU)."""

import torch

from deepopt import CUDAGraphCore, TrainingRuntime


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This example requires CUDA hardware and a CUDA-enabled PyTorch build")
    device = torch.device("cuda", torch.cuda.current_device())
    core = CUDAGraphCore(device)
    model = torch.nn.Linear(8, 2).to(device)
    sample = torch.randn(16, 8, device=device)
    captured = core.capture_model(model, (sample,))
    runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                              device=device, execution_mode="max_speed")
    for _ in range(3):
        output = runtime.compute(torch.randn_like(sample), model)
        runtime.backward(output.square().mean())
        runtime.step()
    captured.close()

    def transform(x):
        return x.sin() * 2
    function_graph = core.capture_callable(transform, (sample,), autograd=False)
    torch.testing.assert_close(function_graph(sample), transform(sample))
    function_graph.close()

    @core.capture
    def decorated(x):
        return x.square() + 1
    torch.testing.assert_close(decorated(sample), sample.square() + 1)
    decorated.close()

    core.warmup(lambda: sample.cos())
    with core.capture() as raw:
        result = sample.cos()
    raw.replay()
    torch.testing.assert_close(result, sample.cos())
    raw.close()
    print("Model training, callable inference, decorator and raw CUDA capture completed.")


if __name__ == "__main__":
    main()
