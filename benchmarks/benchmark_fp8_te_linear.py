"""Standalone Linear comparison using installed DeepOpt and optional TE 2.9.0.

Copy outside the checkout and use the environment's Python. TE is optional;
missing or incompatible versions are reported explicitly, never emulated.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib import import_module
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter


TE_VERSION = "2.9.0"
IMPLEMENTATIONS = ("bf16", "deepopt-current", "te-current", "te-delayed", "deepopt-delayed")
PREFIX = "DEEPOPT_TE_LINEAR="


def cases(args):
    return [(batch, name, batch if name == "head" else batch * args.tokens, inputs, outputs,
             "fp32" if name in ("qkv", "fc1") else "bf16", 1 if name == "head" else 12)
            for batch in args.batches for name, inputs, outputs in
            (("qkv", 768, 2304), ("attention", 768, 768), ("fc1", 768, 3072),
             ("fc2", 3072, 768), ("head", 768, 10))]


def te_recipe(args, recipes):
    fp8_format = recipes.Format.E4M3 if args.format == "e4m3fn" else recipes.Format.HYBRID
    if args.implementation == "te-current":
        result = recipes.Float8CurrentScaling(fp8_format=fp8_format, use_power_2_scales=False)
    else:
        result = recipes.DelayedScaling(fp8_format=fp8_format, amax_history_len=args.history_length,
                                        amax_compute_algo=args.amax_algorithm, margin=args.margin, reduce_amax=True)
    if getattr(args, "math_policy", "native") == "matched-fp32":
        # TE 2.9 Linear consults these recipe overrides for all three GEMMs.
        for role in ("fprop", "dgrad", "wgrad"):
            setattr(result, f"fp8_gemm_{role}", recipes.MMParams(use_split_accumulator=False))
    return result


def matched_projection(projection, bias, torch):
    class MatchedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = projection
            self.bias = bias

        def forward(self, x):
            output = self.projection(x.float())
            if self.bias is not None:
                output = output + self.bias
            return output.to(torch.bfloat16)

    return MatchedProjection()


def numerical_error(value, expected, torch):
    difference = value.detach().float() - expected
    relative = (difference.norm() / expected.norm().clamp_min(1e-30)).item()
    maximum = difference.abs().max().item()
    return {"relative_l2": relative if relative < float("inf") else None,
            "max_abs": maximum if maximum < float("inf") else None,
            "finite": bool(torch.isfinite(value).all())}


def worker(args):
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if not torch.cuda.is_available():
        return {"implementation": args.implementation, "status": "unavailable", "reason": "CUDA is unavailable"}
    torch.cuda.set_device(args.device)
    if args.implementation != "bf16" and torch.cuda.get_device_capability() < (9, 0):
        return {"implementation": args.implementation, "status": "unavailable", "reason": "This comparison requires SM90 or newer"}
    torch.use_deterministic_algorithms(args.deterministic)
    torch.utils.deterministic.fill_uninitialized_memory = args.fill_uninitialized
    # References below disable TF32 explicitly, outside all timing passes.
    from deepopt import FP8Config, DelayedScalingConfig, autocast, fp8_scaling_scope
    from deepopt.opt.policy import prepare_fp8_model
    from deepopt.opt._recipe import fp8_recipe

    te, recipe, te_version, te_defaults = None, None, None, {}
    if args.implementation.startswith("te-"):
        try:
            te_version = version("transformer-engine")
        except PackageNotFoundError:
            return {"implementation": args.implementation, "status": "unavailable", "reason": "TE is not installed"}
        if te_version != args.te_version:
            return {"implementation": args.implementation, "status": "unavailable",
                    "reason": f"TE version {te_version} does not match requested pin {args.te_version}"}
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe as recipes
        except (ImportError, OSError) as error:
            return {"implementation": args.implementation, "status": "unavailable", "reason": str(error)}

        recipe = te_recipe(args, recipes)
        base = import_module("transformer_engine.pytorch.module.base")
        te_defaults = {role: getattr(base, f"_2X_ACC_{short}", None)
                       for role, short in (("fprop", "FPROP"), ("dgrad", "DGRAD"), ("wgrad", "WGRAD"))}
    matched_math = args.math_policy == "matched-fp32"
    options = FP8Config(enabled=True, backend="triton", intercept_linear=False, out_dtype="fp32" if matched_math else "bf16",
                        scaling_recipe="delayed" if args.implementation.endswith("delayed") else "current",
                        grad_dtype="e5m2" if args.format == "hybrid" else "e4m3fn",
                        delayed_scaling=DelayedScalingConfig(args.history_length, args.amax_algorithm, args.margin))
    rows = []
    for index, (batch, name, m, k, n, input_dtype, weight) in enumerate(cases(args)):
        torch.manual_seed(args.seed + index)
        source = torch.nn.Linear(k, n, bias=args.bias, device=args.device, dtype=torch.float32)
        source.weight.requires_grad_(args.gradients in ("all", "weight"))
        if source.bias is not None:
            source.bias.requires_grad_(args.gradients in ("all", "weight", "bias"))
        x = torch.randn(m, k, device=args.device, dtype=torch.float32 if input_dtype == "fp32" else torch.bfloat16,
                        requires_grad=args.gradients in ("all", "input"))
        grad = torch.randn(m, n, device=args.device, dtype=torch.bfloat16) / m
        fingerprint = hashlib.sha256(source.weight.detach().cpu().numpy().tobytes()).hexdigest()
        input_fingerprint = hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        grad_fingerprint = hashlib.sha256(grad.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        parameters = tuple(source.parameters())
        dense = args.implementation == "bf16" or name == "head"
        matched = matched_math and not dense
        model = source
        if not dense and te is not None:
            model = te.Linear(k, n, bias=args.bias and not matched, params_dtype=torch.float32, device=args.device,
                              fuse_wgrad_accumulation=False)
            model.weight = source.weight
            if args.bias and not matched:
                model.bias = source.bias
        elif not dense:
            projection = source
            if matched:
                projection = torch.nn.Linear(k, n, bias=False, device="meta")
                projection.weight = source.weight
            model = prepare_fp8_model(projection, options)[0]
            del projection
        if matched:
            model = matched_projection(model, source.bias, torch)
        if {id(parameter) for parameter in model.parameters()} != {id(parameter) for parameter in parameters}:
            raise RuntimeError("Projection changed FP32 optimizer parameter ownership")
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate) if args.optimizer_updates else None

        def operation(*, update=True, model=model, x=x, grad=grad, optimizer=optimizer, matched=matched):
            model.zero_grad(set_to_none=True)
            x.grad = None
            quantization = (te.fp8_autocast(enabled=True, fp8_recipe=recipe) if te is not None and not dense else
                            autocast("cuda", dtype=torch.bfloat16, enabled=not matched, fp8=options, check_finite=args.check_finite)
                            if not dense else nullcontext())
            boundary = (fp8_scaling_scope() if args.implementation == "deepopt-delayed" and not dense else nullcontext())
            with boundary:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not matched), quantization:
                    output = model(x)
                output.backward(grad)
            if update and optimizer is not None:
                optimizer.step()
            return output.detach()

        torch.cuda.synchronize()
        started = perf_counter()
        operation()
        torch.cuda.synchronize()
        cold = perf_counter() - started
        for _ in range(args.warmups):
            operation()
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = perf_counter()
            for _ in range(args.iterations):
                operation()
            torch.cuda.synchronize()
            samples.append((perf_counter() - started) * 1000 / args.iterations)
        model.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        observed = operation(update=False)
        torch.cuda.synchronize()
        memory = {"baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        errors = {}
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.no_grad():
                expected = x.float() @ source.weight.T
                if source.bias is not None:
                    expected += source.bias
                errors["forward"] = numerical_error(observed, expected, torch)
                if x.grad is not None:
                    errors["input_gradient"] = numerical_error(x.grad, grad.float() @ source.weight, torch)
                if source.weight.grad is not None:
                    errors["weight_gradient"] = numerical_error(source.weight.grad, grad.float().T @ x.float(), torch)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
        del expected
        rows.append({"sample_batch": batch, "projection": name, "M_N_K": [m, n, k], "vit_weight": weight,
                     "input_dtype": input_dtype, "head_policy": "bf16" if name == "head" else None,
                     "initial_weight_sha256": fingerprint, "cold_seconds": cold, "warm_median_ms": median(samples),
                     "input_sha256": input_fingerprint, "upstream_gradient_sha256": grad_fingerprint,
                     "warm_samples_ms": samples, "memory": memory, "errors_vs_fp32": errors,
                     "gradient_dtypes": {"input": str(x.grad.dtype) if x.grad is not None else None,
                                         "weight": str(source.weight.grad.dtype) if source.weight.grad is not None else None},
                     "output_dtype": str(observed.dtype)})
        del observed, optimizer, model, source, parameters, x, grad, operation
    import deepopt
    from deepopt.opt.backends import triton_fp8

    digest = hashlib.sha256()
    root = Path(deepopt.__file__).resolve().parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")

    def policy(role):
        return getattr(getattr(recipe, f"fp8_gemm_{role}", None), "use_split_accumulator", te_defaults.get(role))

    return {"status": "ok", "implementation": args.implementation, "timings": rows,
            "contract": {"deepopt_configuration": asdict(options) if args.implementation.startswith("deepopt") else None,
                         "math_policy": args.math_policy,
                         "te_recipe": str(recipe), "gradient_mode": args.gradients,
                         "bias": args.bias, "optimizer_updates": args.optimizer_updates,
                         "deterministic": args.deterministic, "fill_uninitialized_memory": args.fill_uninitialized,
                         "finite_validation_matched": not args.check_finite,
                         "deepopt_internal_outputs": "FP32 fprop/dX/dW" if matched_math else "dX follows input dtype; dW FP32",
                         "te_internal_outputs": "FP32 fprop/dX/dW" if matched_math else "activation dtype; external parameter gradients may cast to FP32",
                         "bias_policy": ("Common FP32 bias addition outside both linears before BF16 output cast" if matched_math else
                                         "TE casts bias to AMP dtype; DeepOpt adds bias in FP32 before output cast"),
                         "te_split_accumulation": {role: policy(role)
                                                   for role in ("fprop", "dgrad", "wgrad")}},
            "environment": {"torch": str(torch.__version__), "triton": sys.modules["triton"].__version__,
                            "transformer_engine": te_version, "deepopt_source": str(Path(deepopt.__file__).resolve()),
                            "deepopt_source_sha256": digest.hexdigest(),
                            "numerical_recipe": (fp8_recipe("triton", options) if args.implementation.startswith("deepopt") else
                                                 {"transformer_engine": te_version, "recipe": str(recipe)} if te else
                                                 {"scaling": "none", "matmul": "bf16"}),
                            "gpu": torch.cuda.get_device_name()},
            "selected_configs": [{"key": list(key), "kwargs": config.kwargs,
                                  "warps": config.num_warps, "stages": config.num_stages}
                                 for key, config in triton_fp8._gemm_nt_kernel.cache.items()]}


def summary(results):
    totals = {}
    for result in results:
        if result["status"] != "ok":
            continue
        by_batch = {}
        for row in result["timings"]:
            key = str(row["sample_batch"])
            by_batch[key] = by_batch.get(key, 0.) + row["vit_weight"] * row["warm_median_ms"]
        totals[result["implementation"]] = by_batch
    ratios = {batch: value / totals["te-delayed"][batch]
              for batch, value in totals.get("deepopt-delayed", {}).items()
              if batch in totals.get("te-delayed", {})}
    compared = {row["implementation"]: row for row in results if row["status"] == "ok"}
    controls = [compared.get(name, {}).get("contract", {}) for name in ("deepopt-delayed", "te-delayed")]
    matched = (all(row.get("math_policy") == "matched-fp32" and row.get("finite_validation_matched") for row in controls)
               and all(controls[1].get("te_split_accumulation", {}).get(role) is False for role in ("fprop", "dgrad", "wgrad")))
    return {"weighted_projection_ms": totals, "deepopt_delayed_over_te_delayed": ratios,
            "target_ratio": 1.05, "internal_policies_matched": matched,
            "acceptance": "Requires matched internal GEMM policies and training-quality validation; ratios alone do not establish parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=(*IMPLEMENTATIONS, "all"), default="all")
    parser.add_argument("--te-version", default=TE_VERSION)
    parser.add_argument("--math-policy", choices=("native", "matched-fp32"), default="native",
                        help="matched-fp32 uses FP32 internal GEMM outputs, common external bias, and native fast accumulation in both FP8 paths.")
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 336, 272])
    parser.add_argument("--tokens", type=int, default=197)
    parser.add_argument("--format", choices=("e4m3fn", "hybrid"), default="e4m3fn")
    parser.add_argument("--history-length", type=int, default=1024)
    parser.add_argument("--amax-algorithm", choices=("max", "most_recent"), default="max")
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--gradients", choices=("all", "weight", "input", "bias"), default="all")
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-finite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer-updates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fill-uninitialized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (any(batch < 1 for batch in args.batches) or args.tokens < 1 or args.iterations < 1
            or args.repeats < 1 or args.warmups < 0 or not 1 <= args.history_length <= 4096
            or not -32 <= args.margin <= 32 or (args.gradients == "bias" and not args.bias)):
        parser.error("Invalid shapes, measurement counts, recipe, or requested gradients")
    if args._worker:
        print(PREFIX + json.dumps(worker(args), allow_nan=False))
        return
    results = []
    for implementation in IMPLEMENTATIONS if args.implementation == "all" else (args.implementation,):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--implementation", implementation, "--_worker"]
        with tempfile.TemporaryDirectory(prefix="deepopt-te-linear-") as cache:
            completed = subprocess.run(command, env={**os.environ, "TRITON_CACHE_DIR": cache}, capture_output=True, text=True)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        result = next((json.loads(line[len(PREFIX):]) for line in completed.stdout.splitlines() if line.startswith(PREFIX)), None)
        if result is None:
            raise RuntimeError("Benchmark worker returned no result")
        results.append(result)
    report = {"settings": {key: value for key, value in vars(args).items() if key not in ("output", "_worker")},
              "results": results, "summary": summary(results)}
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
