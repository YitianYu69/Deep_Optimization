"""Compare FP8 projection implementations on a compatible CUDA GPU.

python benchmarks/benchmark_fp8_autotune.py
python benchmarks/benchmark_fp8_autotune.py --source src --source /path/to/baseline/src --num-stages 3

Each source/scope runs in a fresh process. By default its Triton disk cache is
also isolated, so first-use latency includes compilation. Warmed timings use
synchronized wall-clock measurements. No datasets or model downloads are needed.
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


_RESULT_PREFIX = "DEEPOPT_BENCHMARK_RESULT="


def _config_record(config):
    return {"tile": dict(config.kwargs), "num_warps": config.num_warps,
            "num_stages": config.num_stages}


@contextmanager
def _record_autotune_trials():
    """Record cold-search timings for every key, outside the warmed path."""
    backend = sys.modules.get("deepopt.opt.backends.triton_fp8")
    kernel = getattr(backend, "_gemm_nt_kernel", None)
    trials = {}
    if kernel is None or not hasattr(kernel, "_bench"):
        yield trials
        return
    original = kernel._bench

    def record(*args, config, **kwargs):
        all_args = {**kernel.nargs, **kwargs}
        named = {name: value for name, value in all_args.items() if name in kernel.arg_names}
        key = (tuple(named[name] for name in kernel.keys if name in named)
               + tuple(str(value.dtype) for value in named.values() if hasattr(value, "dtype")))
        started = perf_counter()
        timings = original(*args, config=config, **kwargs)
        samples = timings if isinstance(timings, (list, tuple)) else [timings]
        trials.setdefault(key, []).append({
            **_config_record(config),
            "quantiles_ms": [float(value) if isfinite(value) else None for value in samples],
            "status": "ok" if all(isfinite(value) for value in samples) else "unavailable",
            "compile_and_benchmark_seconds": perf_counter() - started,
        })
        return timings

    with patch.object(kernel, "_bench", new=record):
        yield trials


def _autotune_diagnostics(trials=None):
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
            "implicit_dtype_key": [serializable(value) for value in key_values[len(key_names):]],
            "candidates": (trials or {}).get(key, []),
        })
    return winners


def _prepare_gemm_products(x, weight, bias, grad, plan):
    """Capture each snapshot's real autograd operands without warming GEMMs.

    The explicit upstream gradient determines backward inputs independently of
    the placeholder forward result. Quantization and saved orientations still
    execute the source snapshot's own implementation.
    """
    import torch
    from deepopt.opt.do_nn.functional import do_linear
    from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func

    captured = []

    def capture(a, b, **kwargs):
        captured.append((a, b, kwargs.get("bias"), kwargs["out_dtype"]))
        return torch.zeros((a.data.shape[0], b.data.shape[0]),
                           dtype=kwargs["out_dtype"], device=a.data.device)

    try:
        with patch.object(linear_autograd_func, "gemm_nt", new=capture):
            do_linear(x, weight, bias, plan=plan).backward(grad)
    finally:
        x.grad = weight.grad = bias.grad = None
    if len(captured) != 3:
        raise RuntimeError("Expected forward, input-gradient, and weight-gradient GEMMs")
    return [(name, *product) for name, product in
            zip(("forward", "input_gradient", "weight_gradient"), captured)]


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


def _benchmark_plan(args, torch):
    formats = {"e4m3fn": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
    training = {"backend": "triton", "num_stages": args.num_stages, "out_dtype": torch.bfloat16,
                "fp8_dtypes": {role: formats[getattr(args, role + "_format", default)]
                               for role, default in (("activation", "e4m3fn"), ("weight", "e4m3fn"), ("grad", "e5m2"))}}
    if args.max_num_imprecise_acc != 32:
        training["max_num_imprecise_acc"] = args.max_num_imprecise_acc
    blocks = (dict.fromkeys(("activation", "weight", "grad")) if args.scaling == "tensorwise" else
              {"activation": (1, args.scale_k), "weight": (args.scale_k, args.scale_k),
               "grad": (args.scale_k, args.scale_k)})
    return {"fp8_training": training, "block_shapes": blocks}


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
    # Load the tuner before measuring so the recorder sees the very first key.
    if args.implementation == "fp8":
        from deepopt.opt.backends import triton_fp8  # noqa: F401

    torch.manual_seed(42)
    # Initialize the CUDA context before timing any projection.
    torch.empty(1, device=device)
    torch.cuda.synchronize(device)
    plan = _benchmark_plan(args, torch)
    # Keep strict comparisons compatible with snapshots predating check_finite.
    # A persistent disabled AMP context around BF16 measurements would keep
    # native weight-cast caches alive across forwards, unlike a training step.
    check_scope = (nullcontext() if args.check_finite or args.implementation == "bf16" else
                   deepopt.autocast("cuda", enabled=False, fp8=None, check_finite=False))
    with check_scope, _record_autotune_trials() as trials:
        rows = _measure_batches(args, torch, device, plan, do_linear, gemm_nt)
    return {
        "source": str(args.source[0]),
        "scope": args.scope,
        "implementation": args.implementation,
        "scaling": args.scaling,
        "fp8_policy": {"block_shapes": plan["block_shapes"], "max_num_imprecise_acc": args.max_num_imprecise_acc,
                       "formats": {role: str(dtype) for role, dtype in plan["fp8_training"]["fp8_dtypes"].items()}},
        "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "deepopt_source": str(Path(deepopt.__file__).resolve()),
                        "gpu": torch.cuda.get_device_name(device),
                        "capability": torch.cuda.get_device_capability(device)},
        "timings": rows,
        "autotune_selections": _autotune_diagnostics(trials),
        "tuning_result_cache_enabled": bool(getattr(
            getattr(sys.modules.get("deepopt.opt.backends.triton_fp8"), "_gemm_nt_kernel", None),
            "cache_results", False)),
    }


def _projection_cases(args):
    if getattr(args, "preset", "classifier") == "vit":
        return [(name, batch if name == "head" else batch * args.tokens_per_sample, inputs, outputs)
                for batch in args.batches
                for name, inputs, outputs in (("qkv", 768, 2304), ("attention", 768, 768),
                                              ("fc1", 768, 3072), ("fc2", 3072, 768),
                                              ("head", 768, args.output_features))]
    return [("classifier", batch, args.input_features, args.output_features) for batch in args.batches]


def _dense_products(x, weight, bias, grad):
    import torch

    return [("forward", x, weight.to(torch.bfloat16), bias.to(torch.bfloat16), torch.bfloat16),
            ("input_gradient", grad, weight.T.to(torch.bfloat16), None, torch.bfloat16),
            ("weight_gradient", grad.T, x.T, None, torch.float32)]


def _error_metrics(actual, expected, torch):
    actual, expected = actual.detach().float(), expected.detach().float()
    error = actual - expected
    values = {"relative_l2": (error.norm() / expected.norm().clamp_min(1e-30)).item(),
              "max_abs": error.abs().max().item(), "rmse": error.square().mean().sqrt().item()}
    return {**{name: value if isfinite(value) else None for name, value in values.items()},
            "finite": bool(torch.isfinite(actual).all().item())}


def _reference_product(name, x, weight, bias, grad, torch):
    # Accuracy is computed after timing; IEEE FP32 references must not silently
    # use a process-wide TF32 setting intended for performance measurements.
    cuda = x.device.type == "cuda"
    previous = torch.backends.cuda.matmul.allow_tf32 if cuda else None
    try:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        if name == "forward":
            return x.detach().float() @ weight.detach().float().T + bias.detach().float()
        if name == "input_gradient":
            return grad.float() @ weight.detach().float()
        return grad.float().T @ x.detach().float()
    finally:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous


def _measure_batches(args, torch, device, plan, do_linear, gemm_nt):
    rows = []
    dense = getattr(args, "implementation", "fp8") == "bf16"
    for projection_name, batch, inputs, outputs in _projection_cases(args):
        x = torch.randn(batch, inputs, device=device, dtype=torch.bfloat16, requires_grad=True)
        # Keep FP32 master parameters, as with ordinary BF16 autocast training.
        weight = torch.randn(outputs, inputs, device=device, requires_grad=True)
        bias = torch.randn(outputs, device=device, requires_grad=True)
        grad = torch.randn(batch, outputs, device=device, dtype=torch.bfloat16) / batch
        if args.scope == "linear":
            def projection():
                x.grad = weight.grad = bias.grad = None
                if dense:
                    with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                        output = torch.nn.functional.linear(x, weight, bias)
                else:
                    output = do_linear(x, weight, bias, plan=plan)
                output.backward(grad)
                return output.detach()

            row = {"projection": projection_name, "batch": batch, "operation": "forward_backward",
                   **_measure(projection, torch, device, args)}
            if getattr(args, "accuracy", False):
                actual = projection()
                row["errors_vs_fp32"] = {
                    name: _error_metrics(value, _reference_product(name, x, weight, bias, grad, torch), torch)
                    for name, value in (("forward", actual), ("input_gradient", x.grad),
                                        ("weight_gradient", weight.grad))}
            rows.append(row)
        elif args.scope in ("quantize", "scale", "dequantize"):
            from deepopt.opt.quantization.quantizer import quantize_fp8
            from deepopt.opt.scaling_strategy.amax import AmaxScaling

            operands = [("activation", x, "activation"), ("weight", weight, "weight"),
                        ("gradient", grad, "grad")]
            for name, value, role in tuple(operands):
                blocks = plan["block_shapes"][role]
                if role in ("activation", "grad") and blocks is not None and blocks[0] == 1 and blocks[1] > 1:
                    operands.append((name + "_transposed", value.T, role))
            for name, value, role in operands:
                def quantize():
                    return quantize_fp8(value, fp8_dtype=plan["fp8_training"]["fp8_dtypes"][role],
                                        block_shape=plan["block_shapes"][role])

                if args.scope == "scale":
                    def operation():
                        return AmaxScaling(plan["block_shapes"][role]).compute(
                            value, fp8_dtype=plan["fp8_training"]["fp8_dtypes"][role])
                elif args.scope == "dequantize":
                    # Prepare FP8 operands outside reconstruction timing.
                    operation = quantize().dequantize
                else:
                    operation = quantize
                rows.append({"projection": projection_name, "batch": batch, "operation": name, "shape": list(value.shape),
                             "stride": list(value.stride()), **_measure(operation, torch, device, args)})
        else:
            products = _dense_products(x, weight, bias, grad) if dense else _prepare_gemm_products(x, weight, bias, grad, plan)
            for name, left, right, product_bias, output_dtype in products:
                def product():
                    if dense:
                        return torch.nn.functional.linear(left, right, product_bias).to(output_dtype)
                    options = ({"max_num_imprecise_acc": args.max_num_imprecise_acc}
                               if getattr(args, "max_num_imprecise_acc", 32) != 32 else {})
                    return gemm_nt(left, right, backend="triton", bias=product_bias,
                                   out_dtype=output_dtype, num_stages=args.num_stages, **options)

                row = {"projection": projection_name, "batch": batch, "operation": name,
                       "M_N_K": [left.shape[0] if dense else left.data.shape[0],
                                 right.shape[0] if dense else right.data.shape[0],
                                 left.shape[1] if dense else left.data.shape[1]],
                       **_measure(product, torch, device, args)}
                if getattr(args, "accuracy", False):
                    row["errors_vs_fp32"] = _error_metrics(
                        product(), _reference_product(name, x, weight, bias, grad, torch), torch)
                rows.append(row)
    return rows


def _run(args):
    results = []
    scopes = ("linear", "gemm") if args.scope == "both" else (args.scope,)
    for source in args.source:
        for implementation, scope in ((impl, scope) for impl in
                                       (("fp8", "bf16") if args.implementation == "both" else (args.implementation,))
                                       for scope in scopes):
            print(f"Measuring {source} ({scope}) in a fresh process...", file=sys.stderr, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--_worker",
                       "--source", str(source), "--scope", scope, "--device", args.device,
                       "--warmups", str(args.warmups), "--iterations", str(args.iterations),
                       "--repeats", str(args.repeats), "--batches", *map(str, args.batches)]
            command.extend(["--num-stages", str(args.num_stages)])
            command.extend(["--implementation", implementation, "--scaling", args.scaling,
                            "--preset", args.preset, "--tokens-per-sample", str(args.tokens_per_sample),
                            "--max-num-imprecise-acc", str(args.max_num_imprecise_acc)])
            for role in ("activation", "weight", "grad"):
                command.extend(["--" + role + "-format", getattr(args, role + "_format")])
            if not args.accuracy:
                command.append("--no-accuracy")
            command.extend(["--input-features", str(args.input_features),
                            "--output-features", str(args.output_features), "--scale-k", str(args.scale_k)])
            if not args.check_finite:
                command.append("--no-check-finite")
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
            "Quantize/scale/dequantize scopes measure one operation and allocation per operand and independently scaled transpose.",
            "Dequantize scope prepares FP8 operands before timing and does not perform finite validation during reconstruction.",
            "Finite checks are strict by default; --no-check-finite models unchecked runtime updates in supporting snapshots.",
            "GEMM scope uses prequantized inputs but includes packing, padding, and host dispatch; these are not kernel-only timings.",
            "Forward output and input gradients are BF16; master parameters and weight gradients are FP32.",
            "Autotune selections are inspected after timing; null means the source exposes no compatible tuner cache.",
            "Candidate quantiles are search-time median/p20/p80 device milliseconds, including split-K reduction when used.",
            "GEMM operands are captured from each source's actual autograd path before timing.",
            "Accuracy compares forward/dX/dW with IEEE FP32 after timing; changing accumulation requires training-quality validation.",
            "ViT preset uses separate qkv/attention/fc1/fc2/head projections; it does not measure a whole ViT model.",
            "Candidate wall times include compilation and benchmarking; native tuning-result persistence is reported separately.",
        ],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, action="append",
                        help="Directory containing deepopt; repeat to compare snapshots. Defaults to this checkout's src.")
    parser.add_argument("--scope", choices=("both", "linear", "gemm", "quantize", "scale", "dequantize"), default="both")
    parser.add_argument("--check-finite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accuracy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--implementation", choices=("fp8", "bf16", "both"), default="fp8")
    parser.add_argument("--scaling", choices=("blockwise", "tensorwise"), default="blockwise")
    parser.add_argument("--preset", choices=("classifier", "vit"), default="classifier")
    parser.add_argument("--tokens-per-sample", type=int)
    parser.add_argument("--max-num-imprecise-acc", type=int, default=32)
    for role, default in (("activation", "e4m3fn"), ("weight", "e4m3fn"), ("grad", "e5m2")):
        parser.add_argument("--" + role + "-format", choices=("e4m3fn", "e5m2"), default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-stages", choices=("auto", "1", "2", "3", "4", "5"), default="auto",
                        help="Tune pipeline stages by default; use an integer to compare older source snapshots.")
    parser.add_argument("--batches", nargs="+", type=int,
                        help="Sample batches; defaults are 2048/848/1808 for classifier or 512 for ViT.")
    parser.add_argument("--input-features", type=int, default=512)
    parser.add_argument("--output-features", type=int, default=10)
    parser.add_argument("--scale-k", type=int, choices=(32, 64, 128, 256, 512, 1024), default=32,
                        help="Scale width; 128+ permits K=128 FP8 tiles, and larger grids exercise chunked fusion.")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse the normal Triton disk cache.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = args.batches or ([512] if args.preset == "vit" else [2048, 848, 1808])
    args.tokens_per_sample = args.tokens_per_sample if args.tokens_per_sample is not None else (197 if args.preset == "vit" else 1)
    if args.implementation != "fp8" and args.scope not in ("both", "linear", "gemm"):
        parser.error("BF16 comparison supports linear and gemm scopes")
    if not 32 <= args.max_num_imprecise_acc <= 2147483616 or args.max_num_imprecise_acc % 32:
        parser.error("--max-num-imprecise-acc must be a multiple of 32 up to 2147483616")
    if args.tokens_per_sample < 1:
        parser.error("--tokens-per-sample must be positive")
    args.num_stages = args.num_stages if args.num_stages == "auto" else int(args.num_stages)
    args.source = [source.resolve() for source in (args.source or [Path(__file__).resolve().parents[1] / "src"])]
    if any(not (source / "deepopt").is_dir() for source in args.source):
        parser.error("Each --source must be a directory containing the deepopt package.")
    if (any(batch < 1 for batch in args.batches) or args.input_features < 1 or args.output_features < 1
            or args.iterations < 1 or args.repeats < 1 or args.warmups < 0):
        parser.error("Dimensions, iterations, and repeats must be positive; warmups must be nonnegative.")
    if args._worker:
        if len(args.source) != 1 or args.scope == "both" or args.implementation == "both":
            parser.error("A worker needs one source and one scope.")
        print(_RESULT_PREFIX + json.dumps(_worker(args)))
    else:
        print(json.dumps(_run(args), indent=2))


if __name__ == "__main__":
    main()
