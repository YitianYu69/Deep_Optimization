#!/usr/bin/env python3
"""Standalone QuEST-inspired saved-activation compression experiments.

Examples:
  python toy_quest_saved_linear.py --device cuda --bits 2 4 --steps 200
  python toy_quest_saved_linear.py --device cpu --in-features 128 --hidden 128 \
      --group-size 128 --rows 64 --out-features 64 --steps 20
  python toy_quest_saved_linear.py --device cuda --steps 0 --benchmark
  python toy_quest_saved_linear.py --device cuda --distributions laplace student-t sparse-spikes \
      --noise-std 0.25 --train-distribution student-t --steps 200
  python toy_quest_saved_linear.py --device cuda --steps 0 --modules all \
      --models vit-b-16 efficientnet-b0 resnet18 --model-distributions gaussian outliers

The original Linear experiment adapts QuEST's quantizer to saved inputs.
Its forward GEMMs and dx remain ordinary BF16/FP32 computations; only dw
uses reconstructed inputs.
There is NO trust estimator, STE, quantized-weight training, or QuEST gradient mask.
Other MLP operators remain native. This is not a reproduction of QuEST QAT.

Additional module/model probes use saved-tensor hooks across the entire
forward graph, including functional operations. These approximate backward
activations throughout the model, so dx and nonlinear gradients can change.
Mask-only saves from ReLU, ReLU6/Hardtanh, LeakyReLU and dropout use exact
one-bit predicates, independent of the floating-value codec and bit width.
Functional and inplace calls are included. A later consumer that needs the
activation's magnitudes still uses the floating-value codec for its own save.
Smooth activations save local derivative coefficients. Softmax/log-softmax
save probabilities and restore row sums; norms save normalized xhat and
lossless reciprocal std, with per-domain mean/variance correction after decoding.
Norm Jacobians include their coupled reductions; no dense Jacobian is stored.
--saved-state values retains the generic floating-save approach for comparisons,
with exact mask-only saves in both modes. Operator-state probes are first-order.
Parameters (including AMP casts/views), buffers, integer indices, masks and
detached statistics remain exact. Eligible activations are flattened, padded
at the final group, and reconstructed to their original shape and dtype.
Coverage and exact fallbacks are reported; an executed module need not save
compressible tensors. Reconstructed one-sided domains are clamped to their
observed range to keep sqrt/log/power inputs valid. These probes are untimed
and run one matched backward without optimizer updates. Full torchvision architectures are opt-in, use
weights=None and synthetic images, and require torchvision. They do not
measure accuracy on natural images or convergence. ViT's zero-initialized
classifier is randomized so backbone gradient comparisons are meaningful.

Gradient cosines compare with native gradients at the same weights, batch,
and compute dtype. Global cosine concatenates all weight and bias gradients;
per-parameter cosines are also saved. Training measures these before the
optimizer update at logged steps. Extra native passes are outside benchmark
timings. Two zero vectors have cosine 1; one zero or nonfinite vectors use null.

Probes default to all input distributions, including heavy-tailed, skewed,
multimodal, sparse-spike, correlated, and heteroscedastic inputs. --noise-std
adds independent zero-mean Gaussian noise after sampling, before teacher
labels are generated. Training inputs and noise stay fixed across steps and
methods; all sampling uses the seeded generator. See --help for definitions.

Methods: minmax = group affine codec; gaussian = RMS Gaussian grid without HT;
quest = block Hadamard + Gaussian grid; quest-centered additionally stores a
group mean and recenters the decoded residual, an extension for nonzero-mean
saved activations. All scales are
per group, unlike the reference's full-row RMS. The normalized Hadamard uses
FP32 butterflies here, rather than the reference's BF16 block matrix multiply.

Quantizer equations and Gaussian clipping constants checked against:
https://arxiv.org/html/2502.05003v2 (section 3.1)
https://github.com/IST-DASLab/QuEST/blob/c295afda142c03a89d2af04653881beccf459aa1/src/models/quantization/base_linear.py

Timing, when requested, measures this UNFUSED PyTorch prototype. Packed saved
input bytes exclude parameters, caller-owned inputs, other saved activations,
and temporary workspaces. Neither metric establishes production speed or ViT
convergence. No datasets, pretrained weights or external kernels are fetched.
"""

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch.overrides import TorchFunctionMode


REVISION = 6
METHODS = ("minmax", "gaussian", "quest", "quest-centered")
DISTRIBUTIONS = {
    "gaussian": "Standard normal N(0, 1).",
    "outliers": "Standard normal with persistent outlier channels amplified by 12.",
    "shifted": "Standard normal shifted by +3.",
    "gelu": "GELU applied to N(0, 4).",
    "uniform": "Uniform on [-sqrt(3), sqrt(3)], with unit variance.",
    "laplace": "Zero-mean Laplace with unit variance.",
    "student-t": "Student-t with 3 degrees of freedom, divided by sqrt(3) for unit variance.",
    "lognormal": "exp(N(0, 1)), centered and scaled to zero mean and unit variance.",
    "gaussian-mixture": "Equal mixture of N(-2, 0.25) and N(2, 0.25), divided by sqrt(4.25).",
    "sparse-spikes": "N(0, 1) plus independent 1% N(0, 400) spikes, divided by sqrt(5).",
    "correlated": "Standard normal marginals with pairwise feature correlation 0.9.",
    "heteroscedastic": "Gaussian rows with independent log-uniform standard deviations in [0.1, 10].",
}
GAUSSIAN_CLIP = {1: 0.7978845587140913, 2: 1.4935346200015913,
                 4: 2.513930578568423, 8: 3.884938678807525}
MAX_CODEC_VALUES = 1 << 20
REFERENCE = "https://github.com/IST-DASLab/QuEST/blob/c295afda142c03a89d2af04653881beccf459aa1/src/models/quantization/base_linear.py"


def hadamard(x, block_size):
    """Normalized, self-inverse Sylvester HT along feature blocks only."""
    if block_size < 1 or block_size & (block_size - 1) or x.shape[-1] % block_size:
        raise ValueError("Hadamard size must be a power of two dividing the last dimension")
    shape = x.shape
    values = x.reshape(-1, block_size)
    stride = 1
    while stride < block_size:
        pair = values.reshape(-1, block_size // (2 * stride), 2, stride)
        left, right = pair[:, :, 0], pair[:, :, 1]
        values = torch.stack((left + right, left - right), dim=2).reshape(-1, block_size)
        stride *= 2
    return (values * (block_size ** -.5)).reshape(shape)


def pack_codes(codes, bits):
    """Real 32-bit packing, including signed words and partially filled tails."""
    vpw = 32 // bits
    words = (codes.shape[-1] + vpw - 1) // vpw
    codes = F.pad(codes.long(), (0, words * vpw - codes.shape[-1]))
    shifts = torch.arange(vpw, device=codes.device, dtype=torch.int64) * bits
    return (codes.reshape(codes.shape[0], words, vpw) << shifts).sum(-1).int()


def unpack_codes(data, bits, count):
    shifts = torch.arange(32 // bits, device=data.device, dtype=torch.int64) * bits
    return ((data.long()[..., None] >> shifts) & ((1 << bits) - 1)).reshape(data.shape[0], -1)[:, :count]


@dataclass(frozen=True)
class Spec:
    method: str
    bits: int
    shape: tuple
    group_size: int
    hadamard_size: int


@dataclass
class Packed:
    data: torch.Tensor
    scale: torch.Tensor
    offset: torch.Tensor | None
    spec: Spec

    @property
    def saved_bytes(self):
        return sum(v.numel() * v.element_size() for v in (self.data, self.scale, self.offset) if v is not None)


@torch.no_grad()
def encode(x, method="quest", bits=2, group_size=256, hadamard_size=128):
    if method not in METHODS or bits not in GAUSSIAN_CLIP:
        raise ValueError("Choose a supported method and bits in 1, 2, 4, 8")
    if x.ndim != 2 or group_size < 1 or x.shape[1] % group_size:
        raise ValueError("Inputs must be a matrix whose feature width is divisible by group size")
    if method.startswith("quest") and (hadamard_size < 1 or hadamard_size & (hadamard_size - 1)
                                       or group_size % hadamard_size):
        raise ValueError("Hadamard size must be a power of two dividing group size")
    source = x.reshape(-1, group_size)
    count = source.shape[0]
    words = (group_size + 32 // bits - 1) // (32 // bits)
    data = torch.empty((count, words), device=x.device, dtype=torch.int32)
    scale = torch.empty((count, 1), device=x.device, dtype=torch.float32)
    offset = torch.empty_like(scale) if method in ("minmax", "quest-centered") else None
    # Device-resident division avoids the CUDA Python-scalar reciprocal path.
    divisor = torch.tensor((1 << bits) - 1, device=x.device, dtype=torch.float32)
    chunk = max(1, MAX_CODEC_VALUES // group_size)
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        value = source[start:end].float()
        if method == "minmax":
            low, high = value.amin(-1, keepdim=True), value.amax(-1, keepdim=True)
            step = (high - low) / divisor
            codes = ((value - low) / torch.where(step > 0, step, 1.)).round()
            offset[start:end] = low
        else:
            if method == "quest-centered":
                mean = value.mean(-1, keepdim=True)
                offset[start:end] = mean
                value = value - mean
            if method.startswith("quest"):
                value = hadamard(value, hadamard_size)
            rms = value.square().mean(-1, keepdim=True).sqrt()
            step = (2 * GAUSSIAN_CLIP[bits] * rms) / divisor
            safe_step = torch.where(step > 0, step, 1.)
            limit = step * (float((1 << bits) - 1) * .5)
            clipped = torch.maximum(torch.minimum(value, limit), -limit)
            # Preserve QuEST's half-step grid/tie phase, including at zero.
            # Zero-RMS blocks use step=0, reconstructing exact zeros.
            codes = (clipped / safe_step + .5).round() + ((1 << bits) - 2) // 2
        codes = codes.clamp(0, (1 << bits) - 1)
        data[start:end] = pack_codes(codes, bits)
        scale[start:end] = step
    return Packed(data, scale, offset, Spec(method, bits, tuple(x.shape), group_size, hadamard_size))


@torch.no_grad()
def decode(packed, dtype=torch.bfloat16):
    spec = packed.spec
    output = torch.empty(spec.shape, device=packed.data.device, dtype=dtype)
    target = output.reshape(-1, spec.group_size)
    chunk = max(1, MAX_CODEC_VALUES // spec.group_size)
    for start in range(0, packed.data.shape[0], chunk):
        end = min(start + chunk, packed.data.shape[0])
        codes = unpack_codes(packed.data[start:end], spec.bits, spec.group_size).float()
        if spec.method == "minmax":
            value = codes * packed.scale[start:end] + packed.offset[start:end]
        else:
            value = (codes - (((1 << spec.bits) - 2) // 2 + .5)) * packed.scale[start:end]
            if spec.method.startswith("quest"):
                value = hadamard(value, spec.hadamard_size)
            if spec.method == "quest-centered":
                value = value - value.mean(-1, keepdim=True) + packed.offset[start:end]
        target[start:end] = value.to(dtype)
    return output


class _SavedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, method, bits, group_size, hadamard_size):
        y = F.linear(x, weight, bias)
        need_dx, need_dw = ctx.needs_input_grad[:2]
        p = encode(x.reshape(-1, weight.shape[1]).to(y.dtype), method, bits, group_size, hadamard_size) if need_dw else None
        ctx.save_for_backward(p.data if p else None, p.scale if p else None, p.offset if p else None,
                              weight.to(y.dtype) if need_dx else None)
        ctx.spec = p.spec if p else None
        ctx.input_shape, ctx.input_dtype = x.shape, x.dtype
        ctx.weight_dtype, ctx.bias_dtype = weight.dtype, bias.dtype if bias is not None else None
        ctx.compute_dtype, ctx.out_features = y.dtype, weight.shape[0]
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        data, scale, offset, weight = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        with torch.autocast(grad_output.device.type, enabled=False):
            dy = grad_output.reshape(-1, ctx.out_features).to(ctx.compute_dtype)
            dx = (dy @ weight).reshape(ctx.input_shape).to(ctx.input_dtype) if need_dx else None
            x_hat = decode(Packed(data, scale, offset, ctx.spec), ctx.compute_dtype) if need_dw else None
            dw = (dy.T @ x_hat).to(ctx.weight_dtype) if need_dw else None
            db = dy.sum(0).to(ctx.bias_dtype) if need_db else None
        return dx, dw, db, None, None, None, None


class ToyLinear(nn.Linear):
    def __init__(self, in_features, out_features, method="native", bits=2, group_size=256, hadamard_size=128, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.codec = method, bits, group_size, hadamard_size

    def forward(self, x):
        if self.codec[0] == "native" or not torch.is_grad_enabled():
            return F.linear(x, self.weight, self.bias)
        return _SavedLinear.apply(x, self.weight, self.bias, *self.codec)


def amp(device, dtype):
    return torch.autocast(device.type, dtype=dtype, cache_enabled=False) if dtype == torch.bfloat16 else nullcontext()


def error(actual, reference):
    a, b = actual.detach().cpu().double().flatten(), reference.detach().cpu().double().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "relative_l2": None, "cosine": None, "max_abs": None,
                "mean_error": None, "mean_error_over_reference_rms": None}
    an, bn, dn = float(a.norm()), float(b.norm()), float((a - b).norm())
    cosine = float(a @ b) / (an * bn) if an and bn else 1. if an == bn else None
    return {"finite": True, "relative_l2": dn / bn if bn else 0. if not dn else None,
            "cosine": max(-1., min(1., cosine)) if cosine is not None else None,
            "max_abs": float((a - b).abs().max()), "actual_l2": an, "reference_l2": bn,
            "mean_error": float((a - b).mean()),
            "mean_error_over_reference_rms": float((a - b).mean()) * math.sqrt(b.numel()) / bn if bn else None}


def gradient_cosine_similarity(actual, reference):
    """Compare named gradients using bounded FP64 chunks, without a model-sized concatenation."""
    def cosine(aa, bb, ab, finite):
        if not finite:
            return None
        if aa == 0 or bb == 0:
            return 1. if aa == bb else None
        return max(-1., min(1., ab / (math.sqrt(aa) * math.sqrt(bb))))

    totals = [0., 0., 0.]
    all_finite, per_parameter = True, {}
    for name, value in actual.items():
        a, b = value.detach().reshape(-1), reference[name].detach().reshape(-1)
        if value.shape != reference[name].shape:
            raise ValueError(f"Gradient shape mismatch for {name}")
        sums, finite = [0., 0., 0.], True
        for start in range(0, a.numel(), MAX_CODEC_VALUES):
            left = a[start:start + MAX_CODEC_VALUES].cpu().double()
            right = b[start:start + MAX_CODEC_VALUES].cpu().double()
            finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            if finite:
                for index, number in enumerate((float(left @ left), float(right @ right), float(left @ right))):
                    sums[index] += number
        per_parameter[name] = cosine(*sums, finite)
        all_finite = all_finite and finite
        for index, number in enumerate(sums):
            totals[index] += number
    return {"global": cosine(*totals, all_finite) if actual else None, "per_parameter": per_parameter}


def format_metric(value, decimals=5):
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def samples(rows, features, distribution, generator, noise_std=0.):
    """Sample CPU inputs and optional additive noise using only this generator."""
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown input distribution: {distribution}")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise std must be finite and nonnegative")
    x = torch.randn(rows, features, generator=generator)
    if distribution == "outliers":
        # Persistent outlier channels, not independently resampled per entry.
        x[:, ::max(1, features // max(1, features // 50))] *= 12.
    elif distribution == "shifted":
        x += 3.
    elif distribution == "gelu":
        x = F.gelu(x * 2.)
    elif distribution == "uniform":
        x = (torch.rand(rows, features, generator=generator) * 2 - 1) * math.sqrt(3.)
    elif distribution == "laplace":
        eps = torch.finfo(x.dtype).eps
        u = torch.rand(rows, features, generator=generator).clamp(eps, 1 - eps) - .5
        x = -u.sign() * torch.log1p(-2 * u.abs()) / math.sqrt(2.)
    elif distribution == "student-t":
        # A normal divided by sqrt(chi-square(3)) has variance one.
        denominator = torch.randn(rows, features, 3, generator=generator).square().sum(-1)
        x = x / denominator.clamp_min(torch.finfo(x.dtype).tiny).sqrt()
    elif distribution == "lognormal":
        x = (x.exp() - math.exp(.5)) / math.sqrt(math.e * (math.e - 1))
    elif distribution == "gaussian-mixture":
        centers = torch.where(torch.rand(rows, features, generator=generator) < .5, -2., 2.)
        x = (x * .5 + centers) / math.sqrt(4.25)
    elif distribution == "sparse-spikes":
        mask = torch.rand(rows, features, generator=generator) < .01
        spikes = torch.randn(rows, features, generator=generator) * 20.
        x = (x + mask * spikes) / math.sqrt(5.)
    elif distribution == "correlated":
        shared = torch.randn(rows, 1, generator=generator)
        x = x * math.sqrt(.1) + shared * math.sqrt(.9)
    elif distribution == "heteroscedastic":
        scales = (torch.rand(rows, 1, generator=generator) * math.log(100.) + math.log(.1)).exp()
        x = x * scales
    if noise_std > 0:
        x = x + torch.randn(rows, features, generator=generator) * noise_std
    return x


def cases(args):
    return [("native", 16 if args.dtype == "bf16" else 32)] + [(method, bits) for bits in args.bits for method in args.methods]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(step, device, args):
    for _ in range(args.warmup):
        step()
    sync(device)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(args.timing_iters):
            step()
        sync(device)
        times.append((time.perf_counter() - start) * 1000 / args.timing_iters)
    return {"median_wall_ms": statistics.median(times), "repeats_ms": times,
            "scope": "unfused eager Linear forward/backward including Python/codec overhead; no optimizer"}


def probe(args, device, dtype):
    generator = torch.Generator().manual_seed(args.seed)
    w = torch.randn(args.out_features, args.in_features, generator=generator) / math.sqrt(args.in_features)
    b = torch.randn(args.out_features, generator=generator) * .01
    dy = torch.randn(args.rows, args.out_features, generator=generator).to(device=device, dtype=dtype)
    rows = []
    for distribution in args.distributions:
        source = samples(args.rows, args.in_features, distribution, generator, args.noise_std).to(device)
        baseline = None
        for method, bits in cases(args):
            model = ToyLinear(args.in_features, args.out_features, method, bits, args.group_size, args.hadamard_size).to(device)
            with torch.no_grad():
                model.weight.copy_(w); model.bias.copy_(b)
            x = source.detach().clone().requires_grad_(True)
            def step():
                model.zero_grad(set_to_none=True); x.grad = None
                with amp(device, dtype):
                    y = model(x)
                y.backward(dy)
                return y
            y = step()
            values = {"output": y.detach().cpu(), "dx": x.grad.detach().cpu(),
                      "dw": model.weight.grad.detach().cpu(), "db": model.bias.grad.detach().cpu()}
            if method == "native":
                baseline = values
                saved_bytes = source.numel() * torch.empty((), dtype=dtype).element_size()
                reconstruction = {"relative_l2": 0., "cosine": 1., "max_abs": 0.}
            else:
                p = encode(source.to(dtype), method, bits, args.group_size, args.hadamard_size)
                saved_bytes = p.saved_bytes
                reconstruction = error(decode(p, dtype), source.to(dtype))
            row = {"distribution": distribution, "method": method, "bits": bits,
                   "errors_vs_native": {key: error(value, baseline[key]) for key, value in values.items()},
                   "gradient_cosine_similarity": gradient_cosine_similarity(
                       {"weight": values["dw"], "bias": values["db"]},
                       {"weight": baseline["dw"], "bias": baseline["db"]}),
                   "reconstruction": reconstruction, "saved_input_bytes": saved_bytes,
                   "saved_input_ratio_vs_native": source.numel() * torch.empty((), dtype=dtype).element_size() / saved_bytes}
            if args.benchmark:
                row["timing"] = benchmark(step, device, args)
            rows.append(row)
            print(f"probe {distribution:16} {method:14} {bits:2}bit  "
                  f"x_err={format_metric(reconstruction['relative_l2'], 4)}  "
                  f"dw_err={format_metric(row['errors_vs_native']['dw']['relative_l2'], 4)}  "
                  f"dx_cos={format_metric(row['errors_vs_native']['dx']['cosine'])}  "
                  f"dw_cos={format_metric(row['errors_vs_native']['dw']['cosine'])}  "
                  f"db_cos={format_metric(row['errors_vs_native']['db']['cosine'])}  "
                  f"grad_cos={format_metric(row['gradient_cosine_similarity']['global'])}  saved={saved_bytes} B", flush=True)
    return rows


def train(args, device, dtype):
    if not args.steps:
        return []
    generator = torch.Generator().manual_seed(args.seed + 1000)
    count = args.train_samples + args.valid_samples
    x = samples(count, args.in_features, args.train_distribution, generator, args.noise_std)
    teacher = torch.randn(args.classes, args.in_features, generator=generator) / math.sqrt(args.in_features)
    labels = (x @ teacher.T).argmax(-1)
    x, labels = x.to(device), labels.to(device)
    batches = torch.randint(args.train_samples, (args.steps, args.batch_size), generator=generator).to(device)
    torch.manual_seed(args.seed)
    def model_for(method, bits):
        return nn.Sequential(ToyLinear(args.in_features, args.hidden, method, bits, args.group_size, args.hadamard_size),
                             nn.GELU(), ToyLinear(args.hidden, args.classes, method, bits, args.group_size, args.hadamard_size))
    native_model = model_for("native", 16)
    initial = deepcopy(native_model.state_dict())
    native_model = native_model.to(device)
    results = []
    for method, bits in cases(args):
        model = model_for(method, bits).to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9)
        def evaluate():
            with torch.inference_mode(), amp(device, dtype):
                logits = model(x[args.train_samples:])
                loss = float(F.cross_entropy(logits.float(), labels[args.train_samples:]))
                finite = math.isfinite(loss) and bool(torch.isfinite(logits).all())
                accuracy = float((logits.argmax(-1) == labels[args.train_samples:]).float().mean()) if finite else None
            return {"loss": loss if finite else None, "accuracy": accuracy, "finite": finite}
        row = {"method": method, "bits": bits, "initial_validation": evaluate(), "curve": [], "status": "ok"}
        for index, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            with amp(device, dtype):
                logits = model(x[batch])
                loss = F.cross_entropy(logits.float(), labels[batch])
            if not bool(torch.isfinite(loss)):
                row.update(status="nonfinite_loss", stopped_before_update=index + 1)
                break
            loss.backward()
            if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                row.update(status="nonfinite_gradient", stopped_before_update=index + 1)
                break
            record_metrics = index == 0 or (index + 1) % max(1, args.steps // 5) == 0 or index + 1 == args.steps
            if record_metrics:
                actual_gradients = {name: p.grad for name, p in model.named_parameters()}
                native_gradients = actual_gradients
                if method != "native":
                    # Match this run's current parameters, before its optimizer update.
                    native_model.load_state_dict(model.state_dict())
                    native_model.zero_grad(set_to_none=True)
                    with amp(device, dtype):
                        native_loss = F.cross_entropy(native_model(x[batch]).float(), labels[batch])
                    native_loss.backward()
                    native_gradients = {name: p.grad for name, p in native_model.named_parameters()}
                gradient_similarity = gradient_cosine_similarity(actual_gradients, native_gradients)
            optimizer.step()
            if record_metrics:
                row["curve"].append({"step": index + 1, "batch_loss": float(loss.detach()), "validation": evaluate(),
                                     "gradient_cosine_similarity": gradient_similarity})
        row["final_validation"] = evaluate()
        if not row["final_validation"]["finite"] and row["status"] == "ok":
            row["status"] = "nonfinite_update"
        results.append(row)
        last_cosine = row["curve"][-1]["gradient_cosine_similarity"]["global"] if row["curve"] else None
        print(f"train {method:14} {bits:2}bit  validation={row['final_validation']}  "
              f"grad_cos={format_metric(last_cosine)}  status={row['status']}", flush=True)
    return results


@dataclass
class SavedActivation:
    packed: Packed
    shape: tuple
    dtype: torch.dtype
    count: int
    bounds: torch.Tensor | None = None
    state_kind: str = "values"
    row_width: int | None = None
    correction: torch.Tensor | None = None


@dataclass
class SavedMask:
    data: torch.Tensor
    shape: tuple
    dtype: torch.dtype
    true_value: float | torch.Tensor = 1.
    false_value: float = 0.

    @property
    def saved_bytes(self):
        return self.data.numel() * self.data.element_size() + (
            self.true_value.numel() * self.true_value.element_size() if isinstance(self.true_value, torch.Tensor) else 0)


def option(args, kwargs, name, index, default=None):
    return kwargs.get(name, args[index] if len(args) > index else default)


def local_derivative(value, kind, args, kwargs):
    """Precompute first-order coefficients, before any inplace forward update."""
    x = value.detach().double() if value.dtype == torch.float64 else value.detach().float()
    with torch.autocast(x.device.type, enabled=False):
        if kind == "sigmoid":
            y = x.sigmoid()
            return y * (1 - y)
        if kind == "tanh":
            return 1 - x.tanh().square()
        if kind == "gelu":
            return torch.ops.aten.gelu_backward(torch.ones_like(x), x,
                                                approximate=option(args, kwargs, "approximate", 0, "none"))
        if kind == "silu":
            return torch.ops.aten.silu_backward(torch.ones_like(x), x)
        if kind == "elu":
            return torch.ops.aten.elu_backward(torch.ones_like(x), option(args, kwargs, "alpha", 0, 1.),
                                               1., 1., False, x)
        if kind == "softplus":
            return torch.ops.aten.softplus_backward(torch.ones_like(x), x,
                                                    option(args, kwargs, "beta", 0, 1.),
                                                    option(args, kwargs, "threshold", 1, 20.))
    raise ValueError(f"Unsupported derivative: {kind}")


class _SavedDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, function, args, kwargs, codec, kind):
        derivative = local_derivative(x, kind, args, kwargs)
        y = function(x, *args, **kwargs)
        inplace = function in (torch.Tensor.sigmoid_, torch.Tensor.tanh_)
        if kind in ("silu", "elu"):
            inplace = option(args, kwargs, "inplace", 0 if kind == "silu" else 1, False)
        if inplace:
            ctx.mark_dirty(x)
        ctx.input_dtype = x.dtype
        codec.register_state(derivative, "derivative")
        ctx.save_for_backward(derivative)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        derivative, = ctx.saved_tensors
        with torch.autocast(dy.device.type, enabled=False):
            dx = (dy.to(derivative.dtype) * derivative).to(ctx.input_dtype)
        return dx, None, None, None, None, None


class _SavedProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, function, args, kwargs, codec, kind):
        y = function(x, *args, **kwargs)
        dim = option(args, kwargs, "dim", 0)
        if dim is None:
            dim = 0 if x.ndim in (0, 1, 3) else 1
        ctx.dim = dim % x.ndim if x.ndim else 0
        ctx.input_shape, ctx.input_dtype, ctx.kind = tuple(x.shape), x.dtype, kind
        with torch.autocast(x.device.type, enabled=False):
            probability = y.double() if y.dtype == torch.float64 else y.float()
            if kind == "log_softmax":
                probability = probability.exp()
            width = x.shape[ctx.dim] if x.ndim else 1
            rows = math.prod(x.shape[:ctx.dim] + x.shape[ctx.dim + 1:]) if x.ndim else 1
            matrix = probability.movedim(ctx.dim, -1).reshape(rows, width)
        codec.register_state(matrix, "probability")
        ctx.save_for_backward(matrix)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        matrix, = ctx.saved_tensors
        shape = list(ctx.input_shape)
        width = shape.pop(ctx.dim) if shape else 1
        with torch.autocast(dy.device.type, enabled=False):
            p = matrix.reshape(*shape, width).movedim(-1, ctx.dim).reshape(ctx.input_shape)
            grad = dy.to(p.dtype)
            if ctx.kind == "softmax":
                dx = p * (grad - (grad * p).sum(ctx.dim, keepdim=True))
            else:
                dx = grad - p * grad.sum(ctx.dim, keepdim=True)
        return dx.to(ctx.input_dtype), None, None, None, None, None


@dataclass(frozen=True)
class NormGeometry:
    kind: str
    shape: tuple
    groups: int
    normalized_shape: tuple
    batch_stats: bool

    def matrix(self, value):
        if self.kind == "batch_norm":
            return value.movedim(1, 0).reshape(self.shape[1], -1)
        rows = (self.shape[0] * self.groups if self.kind in ("group_norm", "instance_norm") else
                math.prod(self.shape[:-len(self.normalized_shape)]))
        width = math.prod(self.normalized_shape) if self.kind == "layer_norm" else math.prod(self.shape[1:]) // self.groups
        return value.reshape(rows, width)

    def restore(self, value):
        if self.kind == "batch_norm":
            return value.reshape(self.shape[1], self.shape[0], *self.shape[2:]).movedim(0, 1)
        return value.reshape(self.shape)

    @property
    def affine_shape(self):
        if self.kind == "layer_norm":
            return (1,) * (len(self.shape) - len(self.normalized_shape)) + self.normalized_shape
        return (1, self.shape[1]) + (1,) * (len(self.shape) - 2)

    @property
    def affine_reductions(self):
        if self.kind == "layer_norm":
            return tuple(range(len(self.shape) - len(self.normalized_shape)))
        return (0, *range(2, len(self.shape)))


class _SavedNorm(torch.autograd.Function):
    """Save xhat and reciprocal std for the coupled normalization Jacobian."""
    @staticmethod
    def forward(ctx, x, weight, bias, config, codec):
        kind, eps = config["kind"], config["eps"]
        normalized_shape = tuple(config.get("normalized_shape", ()))
        groups = config.get("groups", x.shape[1] if x.ndim > 1 else 1)
        batch_stats = config.get("batch_stats", True)
        if kind == "layer_norm":
            y = F.layer_norm(x, normalized_shape, weight, bias, eps)
        elif kind == "group_norm":
            y = F.group_norm(x, groups, weight, bias, eps)
        elif kind == "batch_norm":
            y = F.batch_norm(x, config["running_mean"], config["running_var"], weight, bias,
                             batch_stats, config["momentum"], eps)
        else:
            y = F.instance_norm(x, config["running_mean"], config["running_var"], weight, bias,
                                batch_stats, config["momentum"], eps)
        geometry = NormGeometry(kind, tuple(x.shape), groups, normalized_shape, batch_stats)
        need_dx, need_dw = ctx.needs_input_grad[:2]
        h, inverse = None, None
        if need_dx or need_dw:
            # Keep the native forward (and its running-buffer updates) intact.
            # Recompute FP32 statistics without touching those buffers. This
            # eager prototype trades extra forward work for explicit state.
            with torch.autocast(x.device.type, enabled=False):
                value = x.double() if x.dtype == torch.float64 else x.float()
                matrix = geometry.matrix(value)
                if kind == "layer_norm":
                    _, mean, inverse = torch.ops.aten.native_layer_norm(value, normalized_shape, None, None, eps)
                elif kind in ("group_norm", "instance_norm") and batch_stats:
                    _, mean, inverse = torch.ops.aten.native_group_norm(
                        value, None, None, x.shape[0], x.shape[1], math.prod(x.shape[2:]), groups, eps)
                elif batch_stats:
                    _, mean, inverse = torch.ops.aten.native_batch_norm(value, None, None, None, None, True, 0., eps)
                else:
                    mean = config["running_mean"].to(value.dtype)
                    inverse = (config["running_var"].to(value.dtype) + eps).rsqrt()
                    if kind == "instance_norm":
                        mean, inverse = mean.repeat(x.shape[0]), inverse.repeat(x.shape[0])
                mean, inverse = mean.reshape(-1, 1), inverse.reshape(-1, 1)
                if need_dw or need_dx and batch_stats:
                    h = (matrix - mean) * inverse
                    codec.register_state(h, "normalized_state", restore_moments=batch_stats)
                if need_dx:
                    codec.register_state(inverse, "normalization_statistics")
                else:
                    inverse = None
        ctx.geometry, ctx.input_dtype = geometry, x.dtype
        ctx.weight_dtype = weight.dtype if weight is not None else None
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(h, inverse, weight if need_dx else None)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        h, inverse, weight = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        geometry = ctx.geometry
        with torch.autocast(dy.device.type, enabled=False):
            grad = dy.double() if ctx.input_dtype == torch.float64 else dy.float()
            # Empty reductions mean an affine shape equal to the whole input.
            def reduce(value):
                return value.sum(geometry.affine_reductions) if geometry.affine_reductions else value
            dw = reduce(grad * geometry.restore(h)).to(ctx.weight_dtype) if need_dw else None
            db = reduce(grad).to(ctx.bias_dtype) if need_db else None
            dx = None
            if need_dx:
                if weight is not None:
                    grad = grad * weight.to(grad.dtype).reshape(geometry.affine_shape)
                u = geometry.matrix(grad)
                if geometry.batch_stats:
                    u = u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)
                dx = geometry.restore(u * inverse).to(ctx.input_dtype)
        return dx, dw, db, None, None


class OperatorSaves(TorchFunctionMode):
    """Select backward state at the saving operation, including functional calls."""
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        x = args[0] if args else kwargs.get("input")
        if (self.codec.method != "native" and self.codec.saved_state == "operator" and
                torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad):
            tail, options = args[1:], {key: value for key, value in kwargs.items() if key != "input"}
            derivatives = {F.gelu: "gelu", F.silu: "silu", F.sigmoid: "sigmoid", torch.sigmoid: "sigmoid",
                           torch.Tensor.sigmoid: "sigmoid", torch.Tensor.sigmoid_: "sigmoid",
                           F.tanh: "tanh", torch.tanh: "tanh", torch.Tensor.tanh: "tanh",
                           torch.Tensor.tanh_: "tanh", F.elu: "elu", F.softplus: "softplus"}
            if func in derivatives:
                kind = derivatives[func]
                inplace = func in (torch.Tensor.sigmoid_, torch.Tensor.tanh_)
                if kind in ("silu", "elu"):
                    inplace = option(tail, options, "inplace", 0 if kind == "silu" else 1, False)
                base = x._base
                if inplace and (x.is_leaf or base is not None and base.is_leaf and base.requires_grad):
                    # Let native autograd reject a leaf update before mutation.
                    return func(*args, **kwargs)
                return _SavedDerivative.apply(x, func, tail, options, self.codec, derivatives[func])
            if func in (F.softmax, torch.softmax, torch.Tensor.softmax,
                        F.log_softmax, torch.log_softmax, torch.Tensor.log_softmax):
                kind = "log_softmax" if "log_softmax" in func.__name__ else "softmax"
                return _SavedProbability.apply(x, func, tail, options, self.codec, kind)
        if (self.codec.method != "native" and self.codec.saved_state == "operator" and
                torch.is_grad_enabled() and func in (F.layer_norm, F.group_norm, F.batch_norm, F.instance_norm)):
            tail = args[1:]
            config = {"kind": func.__name__}
            if func in (F.layer_norm, F.group_norm):
                key = "normalized_shape" if func is F.layer_norm else "groups"
                config[key] = option(tail, kwargs, "normalized_shape" if func is F.layer_norm else "num_groups", 0)
                weight, bias = option(tail, kwargs, "weight", 1), option(tail, kwargs, "bias", 2)
                config["eps"] = option(tail, kwargs, "eps", 3, 1e-5)
            else:
                weight, bias = option(tail, kwargs, "weight", 2), option(tail, kwargs, "bias", 3)
                config.update(running_mean=option(tail, kwargs, "running_mean", 0),
                              running_var=option(tail, kwargs, "running_var", 1),
                              batch_stats=option(tail, kwargs, "training" if func is F.batch_norm else "use_input_stats",
                                                 4, func is F.instance_norm),
                              momentum=option(tail, kwargs, "momentum", 5, .1),
                              eps=option(tail, kwargs, "eps", 6, 1e-5))
            if any(value is not None and value.requires_grad for value in (x, weight, bias)):
                return _SavedNorm.apply(x, weight, bias, config, self.codec)
        role = None
        if func in (F.relu, torch.relu, torch.relu_, torch.Tensor.relu, torch.Tensor.relu_):
            role = ("relu", 0., 0.)
        elif func is F.relu6:
            role = ("hardtanh", 0., 6.)
        elif func is F.hardtanh:
            role = ("hardtanh", kwargs.get("min_val", args[1] if len(args) > 1 else -1.),
                    kwargs.get("max_val", args[2] if len(args) > 2 else 1.))
        elif func is F.leaky_relu:
            role = ("leaky_relu", 0., 0.)
        elif func in (F.dropout, F.dropout1d, F.dropout2d, F.dropout3d, torch.dropout):
            role = ("dropout", 0., 0.)
        if role is None:
            return func(*args, **kwargs)
        previous = self.codec.mask_role
        self.codec.mask_role = role
        try:
            return func(*args, **kwargs)
        finally:
            self.codec.mask_role = previous


class SavedActivations:
    """Experimental graph-wide codec with explicit operator backward state."""
    def __init__(self, model, method, bits, group_size, hadamard_size, saved_state="operator"):
        self.model, self.method, self.bits = model, method, bits
        self.group_size, self.hadamard_size = group_size, hadamard_size
        if saved_state not in ("operator", "values"):
            raise ValueError("saved_state must be operator or values")
        self.saved_state, self.pending_states = saved_state, {}
        self.parameter_ids = {id(value) for value in model.parameters()}
        self.protected_storage = {self.storage(value) for value in (*model.parameters(), *model.buffers())
                                  if value.layout == torch.strided and value.numel()}
        self.stack, self.records, self.calls = [], [], {}
        self.mask_role = None

    def register_state(self, value, kind, **options):
        # The Function's save hooks run after forward returns. Register only an
        # identity and metadata: never retain the dense coefficient as a side copy.
        self.pending_states[id(value)] = {"kind": kind, **options}

    @staticmethod
    def storage(value):
        return str(value.device), value.untyped_storage().data_ptr()

    def parameter_or_buffer(self, value):
        if self.storage(value) in self.protected_storage:
            return True
        # AMP creates new storage. Follow only casts/views back to their leaf,
        # never computation nodes whose activations also depend on parameters.
        node = value.grad_fn
        views = ("ToCopy", "TBackward", "Transpose", "Permute", "View", "ReshapeAlias", "Clone", "Slice",
                 "Select", "Split", "Unbind", "Expand", "Repeat", "Squeeze", "Unsqueeze", "AsStrided", "Alias")
        for _ in range(32):
            if node is None:
                break
            variable = getattr(node, "variable", None)
            if variable is not None:
                return id(variable) in self.parameter_ids
            if not type(node).__name__.startswith(views):
                break
            parents = [parent for parent, _ in node.next_functions if parent is not None]
            node = parents[0] if len(parents) == 1 else None
        return False

    def pack(self, value):
        owner = self.stack[-1] if self.stack else ("<functional>", "functional")
        original = value.numel() * value.element_size()
        record = {"module": owner[0], "type": owner[1], "shape": list(value.shape), "dtype": str(value.dtype),
                  "original_bytes": original, "saved_bytes": original}
        state = self.pending_states.pop(id(value), None)
        if state is not None:
            record["saved_state"] = state["kind"]
            if state["kind"] == "normalization_statistics":
                record.update(kind="exact", reason="normalization_statistics")
                self.records.append(record)
                return value.detach()
            return self.pack_preprocessed(value, record, state)
        if value.layout == torch.strided and value.numel() and (self.mask_role is not None or value.dtype == torch.bool):
            if self.method == "native":
                record.update(kind="native_activation", encoding="native_mask")
                self.records.append(record)
                return value.detach()
            return self.pack_mask(value, record)
        reason = None
        if value.layout != torch.strided or not value.numel():
            reason = "layout_or_empty"
        elif self.parameter_or_buffer(value):
            reason = "parameter_or_buffer"
        elif not value.is_floating_point():
            reason = "indices_or_integer_mask"
        elif not value.requires_grad:
            reason = "detached_mask_or_statistics"
        if reason is not None:
            record.update(kind="exact", reason=reason)
            self.records.append(record)
            return value.detach()
        record["kind"] = "native_activation"
        result = value.detach()
        if self.method != "native":
            flat = value.detach().reshape(1, -1)
            padding = (-value.numel()) % self.group_size
            if padding:
                flat = F.pad(flat, (0, padding))
            packed = encode(flat, self.method, self.bits, self.group_size, self.hadamard_size)
            # Retain valid one-sided domains for sqrt/log/power and bounded
            # nonlinearities. Mixed-sign activations use the unmodified codec.
            low, high = value.detach().aminmax()
            bounds = torch.stack((low, high)).float() if bool(low >= 0 or high <= 0) else None
            saved = packed.saved_bytes + (bounds.numel() * bounds.element_size() if bounds is not None else 0)
            if saved < original:
                record.update(kind="compressed", encoding="lossy_values", saved_bytes=saved, padded_values=padding)
                result = SavedActivation(packed, tuple(value.shape), value.dtype, value.numel(), bounds)
            else:
                record["reason"] = "metadata_would_expand_tensor"
        self.records.append(record)
        return result

    def pack_preprocessed(self, value, record, state):
        kind = state["kind"]
        record.update(kind="native_activation", encoding="native_" + kind)
        if not value.numel():
            self.records.append(record)
            return value.detach()
        matrix = value.detach() if kind != "derivative" else value.detach().reshape(1, -1)
        width = matrix.shape[1]
        group, ht = self.group_size, self.hadamard_size
        if kind != "derivative":
            group = min(group, width)
            if self.method.startswith("quest"):
                ht = min(ht, 1 << (group.bit_length() - 1))
                group = max(ht, group // ht * ht)
        padding = (-width) % group
        padded = F.pad(matrix, (0, padding)) if padding else matrix
        packed = encode(padded, self.method, self.bits, group, ht)
        record.update(codec_group_size=group, codec_hadamard_size=ht if self.method.startswith("quest") else None)
        bounds, correction = None, None
        if kind == "derivative":
            bounds = torch.stack(value.detach().aminmax()).float()
        if kind == "normalized_state" and state["restore_moments"]:
            # Calibrate within the true normalization domains, never across
            # unrelated channels/rows or the codec's padded tail. Preserve the
            # actual forward mean/variance, including epsilon (not forced to 1).
            decoded = decode(packed, torch.float32)[:, :width]
            mean = matrix.float().mean(1, keepdim=True)
            target = (matrix.float() - mean).square().mean(1, keepdim=True)
            center = decoded.mean(1, keepdim=True)
            variance = (decoded - center).square().mean(1, keepdim=True)
            gain = torch.where(variance > 0, (target / variance.clamp_min(1e-30)).sqrt(), 0.)
            correction = torch.cat((mean - center * gain, gain), dim=1)
            record["degenerate_decoded_domains"] = int(((variance == 0) & (target > 0)).sum())
        saved = packed.saved_bytes + sum(t.numel() * t.element_size() for t in (bounds, correction) if t is not None)
        if saved < record["original_bytes"]:
            record.update(kind="compressed", encoding="lossy_" + kind, saved_bytes=saved,
                          padded_values=padding * matrix.shape[0],
                          correction_bytes=correction.numel() * correction.element_size() if correction is not None else 0)
            result = SavedActivation(packed, tuple(value.shape), value.dtype, value.numel(), bounds,
                                     kind, width, correction)
        else:
            record["reason"] = "metadata_would_expand_tensor"
            result = value.detach()
        self.records.append(record)
        return result

    def pack_mask(self, value, record):
        kind, low, high = self.mask_role or ("boolean", 0., 0.)
        value = value.detach()
        true, false = 1., 0.
        if value.dtype == torch.bool:
            predicate = value
        elif kind == "relu":
            # Native threshold backward passes a NaN saved value; > 0 does not.
            predicate = ~(value <= 0)
        elif kind == "hardtanh":
            true, false = low / 2 + high / 2, low
            proxy = torch.tensor(true, device=value.device, dtype=value.dtype)
            if not math.isfinite(low) or not math.isfinite(high) or not bool((proxy > low) & (proxy < high)):
                record.update(kind="native_activation", reason="mask_proxy_not_representable")
                self.records.append(record)
                return value
            # Use native branch semantics, including dtype/backend-specific
            # NaN handling, instead of assuming two comparisons are identical.
            predicate = torch.ops.aten.hardtanh_backward(torch.ones_like(value), value, low, high) != 0
        elif kind == "leaky_relu":
            predicate, false = value > 0, -1.
        elif kind == "dropout":
            predicate = value != 0
            # CPU dropout saves an already-scaled floating mask. Preserve its
            # actual coefficient, including BF16 rounding, rather than recalculate it.
            true = value.amax().clone()
        else:
            predicate = value.bool()
        count = value.numel()
        data = torch.empty((1, (count + 31) // 32), dtype=torch.int32, device=value.device)
        block = max(32, MAX_CODEC_VALUES // 32 * 32)
        flat = predicate.reshape(-1)
        for start in range(0, count, block):
            end = min(start + block, count)
            data[:, start // 32:(end + 31) // 32] = pack_codes(flat[start:end].reshape(1, -1), 1)
        packed = SavedMask(data, tuple(value.shape), value.dtype, true, false)
        if kind == "dropout" and packed.saved_bytes >= record["original_bytes"]:
            record.update(kind="native_activation", reason="mask_metadata_would_expand_tensor")
            result = value
        else:
            record.update(kind="compressed", encoding="exact_1bit_mask", mask_kind=kind,
                          saved_bits=1, saved_bytes=packed.saved_bytes)
            result = packed
        self.records.append(record)
        return result

    @staticmethod
    def unpack(value):
        if isinstance(value, SavedMask):
            count = math.prod(value.shape)
            output = torch.empty(count, dtype=value.dtype, device=value.data.device)
            true = (value.true_value.to(device=value.data.device, dtype=value.dtype)
                    if isinstance(value.true_value, torch.Tensor) else
                    torch.tensor(value.true_value, device=value.data.device, dtype=value.dtype))
            false = torch.tensor(value.false_value, device=value.data.device, dtype=value.dtype)
            block = max(32, MAX_CODEC_VALUES // 32 * 32)
            for start in range(0, count, block):
                end = min(start + block, count)
                predicate = unpack_codes(value.data[:, start // 32:(end + 31) // 32], 1, end - start).bool().reshape(-1)
                output[start:end] = (predicate if value.dtype == torch.bool else
                                     torch.where(predicate, true, false))
            return output.reshape(value.shape)
        if not isinstance(value, SavedActivation):
            return value
        decoded = decode(value.packed, value.dtype)
        if value.row_width is None:
            decoded = decoded.reshape(-1)[:value.count].reshape(value.shape)
        else:
            decoded = decoded[:, :value.row_width]
            if value.correction is not None:
                decoded = decoded * value.correction[:, 1:] + value.correction[:, :1]
            if value.state_kind == "probability":
                decoded = decoded.clamp(min=0., max=1.)
                mass = decoded.sum(1, keepdim=True)
                decoded = torch.where(mass > 0, decoded / mass.clamp_min(1e-30), 1. / value.row_width)
            decoded = decoded.reshape(value.shape)
        if value.bounds is not None:
            decoded = decoded.clamp(min=value.bounds[0].to(value.dtype), max=value.bounds[1].to(value.dtype))
        return decoded

    @contextmanager
    def capture(self):
        handles = []
        def enter(module, inputs, path):
            self.stack.append((path, type(module).__name__))
            self.calls[type(module).__name__] = self.calls.get(type(module).__name__, 0) + 1
        def leave(module, inputs, output):
            self.stack.pop()
        try:
            for path, module in self.model.named_modules():
                def before(module, inputs, path=path or "<root>"):
                    enter(module, inputs, path)
                handles.append(module.register_forward_pre_hook(before))
                handles.append(module.register_forward_hook(leave, always_call=True))
            with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack), OperatorSaves(self):
                yield self
        finally:
            for handle in handles:
                handle.remove()
            self.stack.clear()
            self.pending_states.clear()

    def summary(self):
        activations = [record for record in self.records if record["kind"] != "exact"]
        before = sum(record["original_bytes"] for record in activations)
        after = sum(record["saved_bytes"] for record in activations)
        by_module, exact, fallbacks, states = {}, {}, {}, {}
        encodings = ("exact_1bit_mask", "lossy_values", "lossy_derivative", "lossy_normalized_state", "lossy_probability")
        prefixes = ("exact_mask", "lossy_value", "derivative", "normalized_state", "probability")
        def encoding_counts(records):
            result = {}
            for encoding, prefix in zip(encodings, prefixes):
                selected = [record for record in records if record.get("encoding") == encoding]
                result[prefix + "_tensors"] = len(selected)
                result[prefix + "_bytes"] = sum(record["saved_bytes"] for record in selected)
            return result
        for record in self.records:
            if "saved_state" in record:
                states[record["saved_state"]] = states.get(record["saved_state"], 0) + 1
            if record["kind"] == "native_activation" and "reason" in record:
                fallbacks[record["reason"]] = fallbacks.get(record["reason"], 0) + 1
            if record["kind"] == "exact":
                exact[record["reason"]] = exact.get(record["reason"], 0) + 1
            else:
                row = by_module.setdefault(record["type"], {"saved_activations": 0, "compressed_tensors": 0,
                                                           **encoding_counts([]),
                                                           "original_bytes": 0, "saved_bytes": 0})
                row["saved_activations"] += 1
                row["compressed_tensors"] += record["kind"] == "compressed"
                for key, count in encoding_counts([record]).items():
                    row[key] += count
                row["original_bytes"] += record["original_bytes"]
                row["saved_bytes"] += record["saved_bytes"]
        return {"compressed_tensors": sum(record["kind"] == "compressed" for record in activations),
                **encoding_counts(activations), "saved_state": self.saved_state, "saved_state_tensors": states,
                "eligible_saved_tensors": len(activations), "activation_bytes_before": before,
                "activation_bytes_after": after, "activation_ratio": before / after if after else None,
                "exact_saved_bytes": sum(record["saved_bytes"] for record in self.records if record["kind"] == "exact"),
                "exact_auxiliary_bytes": sum(record["saved_bytes"] for record in self.records
                                              if record["kind"] == "exact" and record["reason"] != "parameter_or_buffer"),
                "normalization_correction_bytes": sum(record.get("correction_bytes", 0) for record in activations),
                "degenerate_normalization_domains": sum(record.get("degenerate_decoded_domains", 0)
                                                        for record in activations if record["kind"] == "compressed"),
                "exact_reasons": exact, "native_fallback_reasons": fallbacks,
                "module_calls": dict(self.calls), "by_module": by_module,
                "scope": "logical saved-tensor bytes including codec metadata and domain bounds; duplicates count "
                         "separately rather than unique storage; excludes allocator peaks and temporary workspaces"}


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(32, 4, dropout=0., batch_first=True)

    def forward(self, x):
        return self.attention(x, x, x, need_weights=False)[0]


class PoolUnpool(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.pool = getattr(nn, f"MaxPool{rank}d")(2, return_indices=True)
        self.unpool = getattr(nn, f"MaxUnpool{rank}d")(2)

    def forward(self, x):
        values, indices = self.pool(x)
        return self.unpool(values, indices, output_size=x.shape)


def module_cases():
    """Factories and non-batch shapes covering the standard vision operation families."""
    result = {"linear": (lambda: nn.Linear(32, 16), (8, 32)),
              "layernorm": (lambda: nn.LayerNorm(32), (8, 32)),
              "groupnorm": (lambda: nn.GroupNorm(4, 8), (8, 8, 8)),
              "multihead-attention": (SelfAttention, (8, 32)),
              "transformer-encoder": (lambda: nn.TransformerEncoderLayer(32, 4, 64, dropout=0., batch_first=True), (8, 32)),
              "flatten": (nn.Flatten, (8, 8, 8)), "identity": (nn.Identity, (8, 32))}
    for name, factory in (("relu", nn.ReLU), ("relu-inplace", lambda: nn.ReLU(inplace=True)),
                          ("relu6", nn.ReLU6), ("leaky-relu", nn.LeakyReLU), ("gelu", nn.GELU),
                          ("silu", nn.SiLU), ("sigmoid", nn.Sigmoid), ("tanh", nn.Tanh),
                          ("elu", nn.ELU), ("softplus", nn.Softplus),
                          ("softmax", lambda: nn.Softmax(dim=-1)), ("log-softmax", lambda: nn.LogSoftmax(dim=-1)),
                          ("dropout", lambda: nn.Dropout(.25))):
        result[name] = factory, (8, 32)
    for rank in (1, 2, 3):
        shape = (8,) + (8,) * rank
        conv = getattr(nn, f"Conv{rank}d")
        transpose = getattr(nn, f"ConvTranspose{rank}d")
        result[f"conv{rank}d"] = (lambda cls=conv: cls(8, 8, 3, padding=1)), shape
        result[f"conv-transpose{rank}d"] = (lambda cls=transpose: cls(8, 8, 3, stride=2, padding=1, output_padding=1)), shape
        batchnorm = getattr(nn, f"BatchNorm{rank}d")
        instance = getattr(nn, f"InstanceNorm{rank}d")
        result[f"batchnorm{rank}d"] = (lambda cls=batchnorm: cls(8)), shape
        result[f"instancenorm{rank}d"] = (lambda cls=instance: cls(8, affine=True, track_running_stats=True)), shape
        result[f"max-unpool{rank}d"] = (lambda rank=rank: PoolUnpool(rank)), shape
        for prefix, class_name in (("maxpool", "MaxPool"), ("avgpool", "AvgPool"),
                                   ("adaptive-maxpool", "AdaptiveMaxPool"), ("adaptive-avgpool", "AdaptiveAvgPool")):
            cls = getattr(nn, f"{class_name}{rank}d")
            result[f"{prefix}{rank}d"] = (lambda cls=cls: cls(2)), shape
        if hasattr(nn, f"LPPool{rank}d"):
            cls = getattr(nn, f"LPPool{rank}d")
            result[f"lp-pool{rank}d"] = (lambda cls=cls: cls(2, 2)), shape
        if rank > 1:
            cls = getattr(nn, f"FractionalMaxPool{rank}d")
            result[f"fractional-maxpool{rank}d"] = (lambda cls=cls, rank=rank: cls(2, output_size=(3,) * rank)), shape
            cls = getattr(nn, f"Dropout{rank}d")
            result[f"dropout{rank}d"] = (lambda cls=cls: cls(.25)), shape
    result["depthwise-conv2d"] = (lambda: nn.Conv2d(8, 8, 3, padding=1, groups=8)), (8, 8, 8)
    result["grouped-conv2d"] = (lambda: nn.Conv2d(8, 8, 3, padding=1, groups=2, dilation=2)), (8, 8, 8)
    return result


VISION_MODELS = ("vit-b-16", "efficientnet-b0", "resnet18")


def make_vision_model(name, classes=10, image_size=224):
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("Real-model probes require optional torchvision; install the project's model-tests extra") from exc
    factories = {"vit-b-16": models.vit_b_16, "efficientnet-b0": models.efficientnet_b0, "resnet18": models.resnet18}
    kwargs = {"weights": None, "num_classes": classes}
    if name == "vit-b-16":
        kwargs["image_size"] = image_size
    model = factories[name](**kwargs)
    if name == "vit-b-16":
        nn.init.normal_(model.heads.head.weight, std=.02)
    return model


def graph_probe(model, source, args, device, dtype, *, classification=False):
    """One matched forward/backward per codec; buffers and dropout RNG reset each time."""
    model = model.to(device).train()
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    labels = torch.arange(source.shape[0], device=device) % args.classes if classification else None
    baseline, rows = None, []
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for method, bits in cases(args):
        gradients, comparison = {}, {}
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        torch.manual_seed(args.seed + 2000)
        x = source.detach().clone().to(device).requires_grad_(True)
        codec = SavedActivations(model, method, bits, args.group_size, args.hadamard_size,
                                 getattr(args, "saved_state", "operator"))
        try:
            # The math backend exposes attention intermediates to saved-tensor
            # hooks, avoiding opaque fused-kernel workspaces in this experiment.
            with sdpa_kernel(SDPBackend.MATH), codec.capture(), amp(device, dtype):
                output = model(x + 0.)
            if classification:
                loss = F.cross_entropy(output.float(), labels)
            else:
                generator = torch.Generator().manual_seed(args.seed + 3000)
                upstream = torch.randn(output.shape, generator=generator).to(device)
                loss = (output.float() * upstream).sum() / math.sqrt(output.numel())
            loss.backward()
            gradients = {name: p.grad.detach() for name, p in model.named_parameters() if p.grad is not None}
            values = {"output": output.detach().cpu(), "dx": x.grad.detach().cpu(), "loss": loss.detach().cpu()}
            if method == "native":
                baseline = {"values": values, "gradients": {name: grad.cpu() for name, grad in gradients.items()}}
            if baseline is None:
                raise RuntimeError("The native baseline did not complete")
            missing = sorted(baseline["gradients"].keys() - gradients.keys())
            unexpected = sorted(gradients.keys() - baseline["gradients"].keys())
            comparison = {name: grad for name, grad in gradients.items() if name in baseline["gradients"]}
            finite = all(bool(torch.isfinite(value).all()) for value in (*values.values(), *gradients.values()))
            row = {"method": method, "bits": bits, "status": "ok" if finite and not missing and not unexpected else "nonfinite_or_missing_gradient",
                   "errors_vs_native": {name: error(value, baseline["values"][name]) for name, value in values.items()},
                   "gradient_cosine_similarity": gradient_cosine_similarity(comparison, baseline["gradients"]),
                   "parameter_gradients": len(gradients), "missing_parameter_gradients": missing,
                   "unexpected_parameter_gradients": unexpected, "saved_activations": codec.summary()}
            saved = row["saved_activations"]
            total = saved["activation_bytes_after"] + saved["exact_auxiliary_bytes"]
            if method == "native":
                baseline["saved_bytes"] = total
            saved["activation_and_auxiliary_bytes"] = total
            saved["ratio_vs_native"] = baseline["saved_bytes"] / total if total else None
            row["forward_matches_native"] = row["errors_vs_native"]["output"]["max_abs"] == 0.
            if not row["forward_matches_native"] and row["status"] == "ok":
                row["status"] = "forward_mismatch"
        except (RuntimeError, NotImplementedError) as exc:
            row = {"method": method, "bits": bits, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "saved_activations": codec.summary()}
        rows.append(row)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    return rows


def module_probes(args, device, dtype):
    factories, results = module_cases(), []
    for index, name in enumerate(args.modules):
        for distribution in args.distributions:
            torch.manual_seed(args.seed + index)
            factory, dimensions = factories[name]
            model = factory()
            generator = torch.Generator().manual_seed(args.seed + 4000 + index)
            source = samples(args.module_batch_size, math.prod(dimensions), distribution, generator, args.noise_std)
            source = source.reshape(args.module_batch_size, *dimensions)
            for row in graph_probe(model, source, args, device, dtype):
                row.update(module=name, distribution=distribution, input_shape=list(source.shape))
                results.append(row)
                cosine = row.get("gradient_cosine_similarity", {}).get("global")
                dx = row.get("errors_vs_native", {}).get("dx", {}).get("cosine")
                print(f"module {name:23} {distribution:16} {row['method']:14} {row['bits']:2}bit  "
                      f"grad_cos={format_metric(cosine)} dx_cos={format_metric(dx)} "
                      f"packed={row['saved_activations']['compressed_tensors']} status={row['status']}", flush=True)
    return results


def model_probes(args, device, dtype):
    results = []
    for name in args.models:
        torch.manual_seed(args.seed)
        model = make_vision_model(name, args.classes, args.model_image_size)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        for distribution in args.model_distributions:
            generator = torch.Generator().manual_seed(args.seed + 5000)
            source = samples(args.model_batch_size, 3 * args.model_image_size ** 2, distribution, generator, args.noise_std)
            source = source.reshape(args.model_batch_size, 3, args.model_image_size, args.model_image_size)
            for row in graph_probe(model, source, args, device, dtype, classification=True):
                row.update(model=name, distribution=distribution, parameters=parameters, input_shape=list(source.shape),
                           pretrained=False, optimizer_steps=0)
                results.append(row)
                cosine = row.get("gradient_cosine_similarity", {}).get("global")
                print(f"model {name:16} {distribution:16} {row['method']:14} {row['bits']:2}bit  "
                      f"grad_cos={format_metric(cosine)} packed={row['saved_activations']['compressed_tensors']} "
                      f"status={row['status']}", flush=True)
        del model
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Input distributions (N(mean, variance)):\n" + "\n".join(
                                         f"  {name}: {description}" for name, description in DISTRIBUTIONS.items()))
    parser.add_argument("--version", action="version", version=f"QuEST saved-activation toy revision {REVISION}")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda[:index]")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bits", nargs="+", type=int, choices=tuple(GAUSSIAN_CLIP), default=[2, 4])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["minmax", "quest", "quest-centered"])
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--hadamard-size", type=int, default=128)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--in-features", type=int, default=768)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument("--distributions", nargs="+", choices=tuple(DISTRIBUTIONS), default=list(DISTRIBUTIONS),
                        help="input distributions for gradient probes (default: all)")
    parser.add_argument("--noise-std", type=float, default=0.,
                        help="extra additive Gaussian noise std for probe/training inputs (default: 0)")
    parser.add_argument("--modules", nargs="+", choices=("all", "none", *module_cases()), default=["all"],
                        help="additional graph-wide module probes (default: all); none keeps only the original Linear experiment")
    parser.add_argument("--saved-state", choices=("operator", "values"), default="operator",
                        help="module/model backward state: derivatives and normalized states (default), or generic saved values; masks stay exact")
    parser.add_argument("--module-batch-size", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=("all", *VISION_MODELS), default=[],
                        help="opt-in full torchvision architecture probes; requires torchvision, never downloads weights")
    parser.add_argument("--model-batch-size", type=int, default=2)
    parser.add_argument("--model-image-size", type=int, default=224)
    parser.add_argument("--model-distributions", nargs="+", choices=("all", *DISTRIBUTIONS), default=["gaussian"],
                        help="full-model input distributions (default: gaussian); all uses every distribution")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--valid-samples", type=int, default=512)
    parser.add_argument("--train-distribution", choices=tuple(DISTRIBUTIONS), default="outliers")
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-iters", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("quest-toy-results.json"))
    args = parser.parse_args(argv)
    positive = (args.group_size, args.hadamard_size, args.rows, args.in_features, args.out_features,
                args.hidden, args.classes, args.batch_size, args.train_samples, args.valid_samples, args.threads,
                args.timing_iters, args.module_batch_size, args.model_batch_size, args.model_image_size)
    if min(positive) < 1 or args.steps < 0 or args.warmup < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("sizes/learning rate must be positive and steps/warmup nonnegative")
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("noise std must be finite and nonnegative")
    if "none" in args.modules and len(args.modules) != 1:
        parser.error("--modules none cannot be combined with other module selections")
    args.modules = [] if args.modules == ["none"] else list(module_cases()) if "all" in args.modules else list(dict.fromkeys(args.modules))
    args.models = list(VISION_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    args.model_distributions = list(DISTRIBUTIONS) if "all" in args.model_distributions else list(dict.fromkeys(args.model_distributions))
    if args.models and (args.model_image_size < 32 or "vit-b-16" in args.models and args.model_image_size % 16):
        parser.error("model image size must be at least 32, and divisible by 16 for ViT-B/16")
    if any(name != "vit-b-16" for name in args.models) and args.model_image_size == 32 and args.model_batch_size < 2:
        parser.error("32-pixel model probes require batch size >= 2 for training BatchNorm")
    if args.models:
        try:
            import torchvision
        except ImportError:
            parser.error("--models requires optional torchvision; install the project's model-tests extra")
    if args.in_features % args.group_size or (args.steps and args.hidden % args.group_size):
        parser.error("input width and training hidden width must be divisible by group size")
    if any(m.startswith("quest") for m in args.methods) and (args.hadamard_size & (args.hadamard_size - 1)
                                                           or args.group_size % args.hadamard_size):
        parser.error("Hadamard size must be a power of two dividing group size")
    args.methods, args.bits = list(dict.fromkeys(args.methods)), list(dict.fromkeys(args.bits))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        parser.error("this toy requires an available CPU or CUDA device")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"revision": REVISION, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference": REFERENCE, "configuration": configuration,
              "distribution_definitions": DISTRIBUTIONS,
              "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(device),
                              "torchvision": str(torchvision.__version__) if args.models else None},
              "scope": "original Linear experiment plus graph-wide saved-activation module/model probes; no trust estimator",
              "memory_scope": "Linear probe counts packed inputs; graph probes count logical saved tensors, not peak GPU memory",
              "gradient_cosine_scope": "native reference at the same weights, batch and compute dtype; global concatenates "
                                       "all parameter gradients; training metrics precede the optimizer update at logged steps",
              "graph_probe_scope": "one matched forward/backward with dropout RNG and buffers reset; math SDPA; no optimizer "
                                   "updates or throughput measurements; exact 1-bit mask-only saves; operator mode quantizes derivatives, "
                                   "probabilities and normalized states, with native statistics and domain corrections; "
                                   "all full models use random weights and synthetic inputs",
              "probe": probe(args, device, dtype), "training": train(args, device, dtype),
              "module_probes": module_probes(args, device, dtype), "model_probes": model_probes(args, device, dtype)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Report: " + str(args.output), flush=True)
    return int(any(row["status"] != "ok" for group in ("training", "module_probes", "model_probes") for row in report[group]))


if __name__ == "__main__":
    raise SystemExit(main())
