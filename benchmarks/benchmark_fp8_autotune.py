"""Compare FP8 classifier implementations on a compatible CUDA GPU.

python benchmarks/benchmark_fp8_autotune.py
python benchmarks/benchmark_fp8_autotune.py --source src --source /path/to/baseline/src

Each source/scope runs in a fresh process. By default its Triton disk cache is
also isolated, so first-use latency includes compilation. Warmed timings use
synchronized wall-clock measurements. No datasets or model downloads are needed.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter


_RESULT_PREFIX = "DEEPOPT_BENCHMARK_RESULT="


def _autotune_diagnostics():
    """Read cached winners after timing; older static snapshots need no tuner."""
    backend = sys.modules.get("deepopt.opt.backends.triton_fp8")
    cache = getattr(getattr(backend, "_gemm_nt_kernel", None), "cache", None)
    if not isinstance(cache, dict):
        return None
    policy = sys.modules.get("deepopt.opt.backends._triton_autotune")
    key_names = getattr(policy, "AUTOTUNE_KEY", ())

    def serializable(value):
        return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)

    winners = []
    for key, config in cache.items():
        tile = getattr(config, "kwargs", None)
        if not isinstance(tile, dict):
            continue
        key_values = key if isinstance(key, (tuple, list)) else (key,)
        metadata = ({name: serializable(value) for name, value in zip(key_names, key_values)}
                    if key_names else {"cache_key": [serializable(value) for value in key_values]})
        winners.append({
            "key": metadata,
            "tile": {name: serializable(value) for name, value in tile.items()},
            "num_warps": serializable(getattr(config, "num_warps", None)),
            "num_stages": serializable(getattr(config, "num_stages", None)),
        })
    return winners


def _measure(operation, torch, device, args):
    def elapsed(iterations):
        torch.cuda.synchronize(device)
        started = perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
        return perf_counter() - started

    first_use = elapsed(1)
    for _ in range(args.warmups):
        operation()
    samples = [elapsed(args.iterations) * 1000 / args.iterations
               for _ in range(args.repeats)]
    return {
        "first_use_seconds": first_use,
        "warm_median_ms": median(samples),
        "warm_samples_ms": samples,
    }


def _worker(args):
    # Keep --help and orchestration usable without importing torch or Triton.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA-enabled PyTorch and a compatible NVIDIA GPU.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select a CUDA device.")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise SystemExit("The Triton TMA FP8 backend requires compute capability 9.0 or newer.")
    sys.path.insert(0, str(args.source[0]))
    import deepopt
    from deepopt.opt.backends import gemm_nt
    from deepopt.opt.do_nn.functional import do_linear
    from deepopt.opt.quantization.quantizer import quantize_fp8

    torch.manual_seed(42)
    # Initialize the CUDA context before timing any projection.
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    plan = {
        "fp8_training": {
            "backend": "triton",
            "fp8_dtypes": {
                "activation": torch.float8_e4m3fn,
                "weight": torch.float8_e4m3fn,
                "grad": torch.float8_e5m2,
            },
            "num_stages": 3,
            "out_dtype": torch.bfloat16,
        },
        "block_shapes": {"activation": (1, 32), "weight": (32, 32), "grad": (32, 32)},
    }
    rows = []
    for batch in args.batches:
        x = torch.randn(batch, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
        # Keep FP32 master parameters, as with ordinary BF16 autocast training.
        weight = torch.randn(10, 512, device=device, requires_grad=True)
        bias = torch.randn(10, device=device, requires_grad=True)
        grad = torch.randn(batch, 10, device=device, dtype=torch.bfloat16) / batch
        if args.scope == "linear":
            def projection():
                x.grad = weight.grad = bias.grad = None
                do_linear(x, weight, bias, plan=plan).backward(grad)

            rows.append({"batch": batch, "operation": "forward_backward",
                         **_measure(projection, torch, device, args)})
        else:
            qx = quantize_fp8(x, fp8_dtype=torch.float8_e4m3fn, block_shape=(1, 32))
            qw = quantize_fp8(weight, fp8_dtype=torch.float8_e4m3fn, block_shape=(32, 32))
            qg = quantize_fp8(grad, fp8_dtype=torch.float8_e5m2, block_shape=(32, 32))
            # Match each autograd GEMM's orientation and result dtype, including
            # FP32 weight gradients for FP32 master parameters.
            products = (
                ("forward", qx, qw, bias, torch.bfloat16),
                ("input_gradient", qg, qw.transpose(), None, torch.bfloat16),
                ("weight_gradient", qg.transpose(), qx.transpose(), None, torch.float32),
            )
            for name, left, right, product_bias, output_dtype in products:
                def product():
                    return gemm_nt(left, right, backend="triton", bias=product_bias,
                                   out_dtype=output_dtype, num_stages=3)

                rows.append({"batch": batch, "operation": name,
                             **_measure(product, torch, device, args)})
    return {
        "source": str(args.source[0]),
        "scope": args.scope,
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "deepopt_source": str(Path(deepopt.__file__).resolve()),
                        "gpu": torch.cuda.get_device_name(device),
                        "capability": torch.cuda.get_device_capability(device)},
        "timings": rows,
        "autotune_selections": _autotune_diagnostics(),
    }


def _run(args):
    results = []
    scopes = ("linear", "gemm") if args.scope == "both" else (args.scope,)
    for source in args.source:
        for scope in scopes:
            print(f"Measuring {source} ({scope}) in a fresh process...", file=sys.stderr, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
                       "--source", str(source), "--scope", scope, "--device", args.device,
                       "--warmups", str(args.warmups), "--iterations", str(args.iterations),
                       "--repeats", str(args.repeats), "--batches", *map(str, args.batches)]
            cache = nullcontext(None) if args.reuse_cache else tempfile.TemporaryDirectory(prefix="deepopt-fp8-")
            with cache as cache_dir:
                environment = os.environ.copy()
                if cache_dir is not None:
                    environment["TRITON_CACHE_DIR"] = cache_dir
                completed = subprocess.run(command, env=environment, capture_output=True, text=True)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            if completed.returncode:
                print(completed.stdout, file=sys.stderr, end="")
                raise SystemExit(completed.returncode)
            result = None
            for line in completed.stdout.splitlines():
                if line.startswith(_RESULT_PREFIX):
                    result = json.loads(line[len(_RESULT_PREFIX):])
                else:
                    print(line, file=sys.stderr)
            if result is None:
                raise RuntimeError("Benchmark worker returned no timing result")
            results.append(result)
    return {
        "settings": {**{key: value for key, value in vars(args).items() if key not in ("_worker", "source")},
                     "source": [str(source) for source in args.source]},
        "disk_cache": "existing" if args.reuse_cache else "isolated per process",
        "notes": [
            "First-use measurements include compilation and, when enabled, candidate benchmarking.",
            "With --reuse-cache, earlier processes may warm the Triton disk cache for later processes.",
            "Later batch shapes may reuse already compiled kernels from earlier shapes in the same process.",
            "Warm timings are synchronized wall-clock milliseconds per invocation.",
            "Linear scope includes quantization, forward/backward, and gradient reset; no backbone, optimizer, or data loading.",
            "GEMM scope uses prequantized inputs but includes packing, padding, and host dispatch; these are not kernel-only timings.",
            "Forward output and input gradients are BF16; master parameters and weight gradients are FP32.",
            "Autotune selections are inspected after timing; null means the source exposes no compatible tuner cache.",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, action="append",
                        help="Directory containing deepopt; repeat to compare snapshots. Defaults to this checkout's src.")
    parser.add_argument("--scope", choices=("both", "linear", "gemm"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", nargs="+", type=int, default=[2048, 848, 1808],
                        help="Batch sizes; defaults include CIFAR-10 training and validation tails.")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse the normal Triton disk cache.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source = [source.resolve() for source in (args.source or [Path(__file__).resolve().parents[1] / "src"])]
    if any(not (source / "deepopt").is_dir() for source in args.source):
        parser.error("Each --source must be a directory containing the deepopt package.")
    if any(batch < 1 for batch in args.batches) or args.iterations < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error("Batches, iterations, and repeats must be positive; warmups must be nonnegative.")
    if args._worker:
        if len(args.source) != 1 or args.scope == "both":
            parser.error("A worker needs one source and one scope.")
        print(_RESULT_PREFIX + json.dumps(_worker(args)))
    else:
        print(json.dumps(_run(args), indent=2))


if __name__ == "__main__":
    main()
