"""Benchmark installed DeepOpt fused ReLU precision conversion and saved masks."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import median
from time import perf_counter


def shape(text):
    try:
        dimensions = tuple(int(value) for value in text.lower().split("x"))
    except ValueError:
        dimensions = ()
    if not dimensions or any(value <= 0 for value in dimensions):
        raise argparse.ArgumentTypeError("use positive dimensions, for example 4096x768")
    return dimensions


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--shape", type=shape, default=(4096, 768))
    result.add_argument("--inputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    result.add_argument("--outputs", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"])
    result.add_argument("--fp8-block", type=shape)
    result.add_argument("--output-scaling", choices=("current", "provided"), default="current")
    result.add_argument("--layout", choices=("last", "channels"), default="last")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--output", type=Path)
    return result


def measure(function, args, torch, device):
    for _ in range(args.warmup):
        function()
    wall, events = [], []
    for _ in range(args.repeats):
        if device.type == "cuda":
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
        start = perf_counter()
        if device.type == "cuda":
            first.record()
        for _ in range(args.iterations):
            function()
        if device.type == "cuda":
            last.record()
            last.synchronize()
        wall.append((perf_counter() - start) * 1000 / args.iterations)
        if device.type == "cuda":
            events.append(first.elapsed_time(last) / args.iterations)
    return {"wall_ms": wall, "wall_ms_median": median(wall), "cuda_stream_ms": events,
            "cuda_stream_ms_median": median(events) if events else None}


def main():
    arguments = parser()
    args = arguments.parse_args()
    if min(args.iterations, args.repeats) <= 0 or args.warmup < 0:
        arguments.error("iterations/repeats must be positive; warmup may be zero")
    if args.fp8_block is not None and len(args.fp8_block) != 2:
        arguments.error("--fp8-block requires two dimensions")
    if args.layout == "channels" and len(args.shape) < 2:
        arguments.error("channels layout requires at least two dimensions")

    import torch
    import deepopt
    from deepopt.opt.nn.functional import do_relu
    from deepopt.opt.policy import relu_plan
    from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation

    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        arguments.error("--device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise SystemExit("An NVIDIA CUDA device is required")
        torch.cuda.set_device(device)
    report = {"deepopt": {"version": version("deep-optimization"), "path": str(Path(deepopt.__file__).resolve())},
              "torch": torch.__version__, "device": str(device),
              "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "notes": ["Eager full forward/backward includes output casting, mask writing/decoding and gradient clearing.",
                        "Input FP8 preparation, warmup and numerical diagnostics are outside timing.",
                        "FP8 inputs are leaf wrappers, so conversion into FP8 is not part of their backward timing.",
                        "CUDA intervals include host submission gaps. Finite checks are disabled during timing.",
                        "Launch counts describe ReLU/scaling kernels, excluding autograd accumulation and gradient clearing.",
                        "The native baseline is BF16 ReLU on the represented input rounded to BF16.",
                        "CPU executes the reference implementation; no Triton speed or fusion claim follows from CPU timing."],
              "results": []}
    for input_type in args.inputs:
        for output_type in args.outputs:
            for compressed in (False, True):
                torch.manual_seed(14)
                dense = torch.randn(args.shape, dtype=torch.bfloat16, device=device)
                source = (to_fp8_activation(dense, layout=args.layout, block_shape=args.fp8_block)
                          if input_type == "fp8" else dense)
                source.requires_grad_()
                represented = source.dequantize(torch.float32).detach() if isinstance(source, FP8Activation) else dense.detach().float()
                policy = relu_plan(fp8={"enabled": True, "backend": "triton" if device.type == "cuda" else "reference",
                                        "block_shapes": {"activation": args.fp8_block}}, activation_compression=compressed)
                gradient = torch.randn_like(dense)
                dtype = torch.float8_e4m3fn if output_type == "fp8" else torch.bfloat16
                info = {}
                supplied = None
                if args.output_scaling == "provided" and output_type == "fp8":
                    rows, cols = ((dense.numel() // dense.shape[1], dense.shape[1]) if args.layout == "channels" else
                                  (dense.numel() // dense.shape[-1], dense.shape[-1]))
                    scale_shape = ((1, 1) if args.fp8_block is None else
                                   tuple((size + block - 1) // block for size, block in zip((rows, cols), args.fp8_block)))
                    supplied = torch.ones(scale_shape, device=device, dtype=torch.float32)

                def step():
                    source.grad = None
                    output = do_relu(source, plan=policy, output_dtype=dtype, output_layout=args.layout,
                                     output_scale=supplied, _execution_info=info)
                    output.backward(gradient)
                    return output

                native_source = represented.bfloat16().detach().requires_grad_()

                def native_step():
                    native_source.grad = None
                    torch.relu(native_source).backward(gradient)

                baseline = measure(native_step, args, torch, device)
                with deepopt.autocast(device.type, enabled=False, check_finite=False,
                                      activation_compression=compressed):
                    timings = measure(step, args, torch, device)
                    saved = []
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel() * t.element_size()) or t,
                                                                lambda t: t):
                        actual = step()
                actual_dense = actual.dequantize(torch.float32) if isinstance(actual, FP8Activation) else actual.float()
                expected_dx = torch.where(represented <= 0, 0, gradient).to(source.dtype)
                torch.testing.assert_close(dequantize_activation(source.grad), expected_dx, rtol=0, atol=0)
                row = {"input": input_type, "output": output_type, "compression": compressed,
                       "timings": timings, "native_bf16_timings": baseline, "saved_mask_bytes": sum(saved),
                       "gradient_exact": True, "forward_max_abs_error": (actual_dense - represented.relu()).abs().max().item(),
                       "execution": dict(info)}
                report["results"].append(row)
                print(f"{input_type}->{output_type} compression={compressed} "
                      f"wall={timings['wall_ms_median']:.4f} ms saved={sum(saved):,} bytes", flush=True)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
