"""Compile VAE V2 at (16, 3, 224, 224) and save code beside this script.

Run: python test_inductor_vae_v2.py
Requires CUDA-enabled PyTorch with a working TorchInductor/Triton toolchain.
Only mbconv_vae_v2_torch.py must accompany this file; DeepOpt is not required.
The default is BF16 autocast with FP32 parameters and FP32 L1 + KL loss.
One forward/loss/backward pass triggers both lazy compiler phases. No optimizer,
dataset, pretrained weights or tensor checkpoints are needed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


INPUT_SHAPE = (16, 3, 224, 224)  # PyTorch Conv2d takes NCHW, not NHWC.
MODEL_OPTIONS = dict(base_channels=32, max_channels=512, blocks_per_stage=2, latent_channels=4)
EXPECTED_PARAMETERS = 32_317_999
SEED = 42
BETA = 1e-6


def compiler_environment(directory):
    """Configure a fresh worker before importing Torch; never serialize tensors."""
    directory = Path(directory).resolve()
    environment = dict(os.environ)
    for key in (
        "TORCHDYNAMO_REPRO_AFTER", "TORCH_TRACE", "TORCH_LOGS", "TRITON_OVERRIDE_DIR",
    ):
        environment.pop(key, None)
    environment.update(
        TORCH_COMPILE_DEBUG="1",
        TORCH_COMPILE_DEBUG_DIR=str(directory / "debug"),
        TORCH_COMPILE_DEBUG_SAVE_REAL="0",
        TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor_cache"),
        TORCHINDUCTOR_FX_GRAPH_CACHE="0",
        TORCHINDUCTOR_AUTOGRAD_CACHE="0",
        TORCHINDUCTOR_SAVE_COMPILE_ARGS="0",
        TORCHDYNAMO_REPRO_LEVEL="0",
        TRITON_CACHE_DIR=str(directory / "triton_cache"),
        TRITON_DUMP_DIR=str(directory / "triton_dumps"),
        TRITON_KERNEL_DUMP="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    return environment


def configure_diagnostics(directory):
    """Enable readable FX, scheduling IR, generated wrappers and kernel sources."""
    dynamo = importlib.import_module("torch._dynamo.config")
    dynamo.repro_after = None  # Empty strings and the string "None" are invalid.
    dynamo.repro_level = 0
    dynamo.suppress_errors = False
    inductor = importlib.import_module("torch._inductor.config")
    inductor.fx_graph_cache = False
    inductor.trace.enabled = True
    inductor.trace.save_real_tensors = False
    inductor.trace.debug_dir = str(Path(directory) / "debug")
    for option in ("fx_graph", "fx_graph_transformed", "ir_pre_fusion", "ir_post_fusion", "output_code"):
        setattr(inductor.trace, option, True)
    # Keep the VAE's stochastic sampling on the ordinary Torch RNG stream.
    inductor.fallback_random = True
    functorch = importlib.import_module("torch._functorch.config")
    backward_autocast = hasattr(functorch, "backward_pass_autocast")
    if backward_autocast:
        functorch.backward_pass_autocast = "off"
    import torch

    flags = {name: True for name in ("aot_graphs", "output_code", "graph_breaks", "recompiles")
             if name in inspect.signature(torch._logging.set_logs).parameters}
    torch._logging.set_logs(**flags)
    return dict(real_tensor_dumping=False, dynamo_repro_after=dynamo.repro_after,
                dynamo_repro_level=dynamo.repro_level, fallback_random=True,
                backward_pass_autocast="off" if backward_autocast else "unavailable",
                logging_flags=list(flags))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_worker(args):
    directory = args.worker
    report = dict(status="started", input_shape=list(INPUT_SHAPE), layout="NCHW contiguous",
                  model="MBConvVAE_V2", model_options=MODEL_OPTIONS, precision=args.precision,
                  parameter_dtype="float32", input_dtype="float32", device=args.device,
                  compile=dict(backend="inductor", mode="default", fullgraph=True, dynamic=False),
                  scope="forward + FP32 L1/KL loss + PyTorch AOTAutograd backward",
                  seed=SEED, beta=BETA, input_range=[-1, 1], python_version=sys.version,
                  optimizer=None, gpu_capture_observed=False, numerical_parity_checked=False)
    exit_code = 1
    try:
        import torch
        from torch.nn import functional as F

        if __package__:
            from .mbconv_vae_v2_torch import MBConvVAE_V2
        else:
            from mbconv_vae_v2_torch import MBConvVAE_V2

        report.update(torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda,
                      torch_git=getattr(torch.version, "git_version", None))
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This fixed full-model capture requires CUDA-enabled PyTorch and a CUDA GPU.")
        torch.cuda.set_device(device)
        if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; select --precision fp32 explicitly.")
        properties = torch.cuda.get_device_properties(device)
        report.update(device_name=torch.cuda.get_device_name(device),
                      device_capability=list(torch.cuda.get_device_capability(device)),
                      device_total_memory=properties.total_memory,
                      device_uuid=str(properties.uuid) if hasattr(properties, "uuid") else None,
                      algorithm_controls=dict(
                          deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                          cudnn_deterministic=torch.backends.cudnn.deterministic,
                          cudnn_benchmark=torch.backends.cudnn.benchmark,
                          tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                          tf32_cudnn=torch.backends.cudnn.allow_tf32))
        try:
            report["triton_version"] = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError:
            report["triton_version"] = None
        report["diagnostics"] = configure_diagnostics(directory)
        torch.manual_seed(SEED)
        # Construct on the GPU: the original orthogonal initialization can be
        # expensive on CPU. Parameters keep their original FP32 dtype.
        with torch.device(device):
            model = MBConvVAE_V2(**MODEL_OPTIONS).train()
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        if report["parameter_count"] != EXPECTED_PARAMETERS:
            raise RuntimeError("VAE architecture does not match the expected full model.")
        images = torch.rand(INPUT_SHAPE, device=device, dtype=torch.float32) * 2 - 1

        def forward_loss(x):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                reconstruction, mu, logvar = model(x)
            l1 = F.l1_loss(reconstruction.float(), x.float())
            mu32, logvar32 = mu.float(), logvar.float()
            kl = 0.5 * (mu32.square() + logvar32.exp() - 1 - logvar32).sum((1, 2, 3)).mean()
            return l1 + BETA * kl, reconstruction, mu, logvar

        compiled = torch.compile(forward_loss, backend="inductor", fullgraph=True, dynamic=False)
        print("Compiling and running forward/loss ...", flush=True)
        loss, reconstruction, mu, logvar = compiled(images)
        print("Compiling and running backward ...", flush=True)
        loss.backward()  # Backward code is generated lazily on the first call.
        torch.cuda.synchronize(device)
        report["gpu_capture_observed"] = True
        if tuple(reconstruction.shape) != INPUT_SHAPE:
            raise RuntimeError(f"Unexpected reconstruction shape: {tuple(reconstruction.shape)}")
        if any(p.grad is None for p in model.parameters() if p.requires_grad):
            raise RuntimeError("A trainable parameter did not receive a gradient.")
        finite = [value.detach().isfinite().all() for value in (loss, reconstruction, mu, logvar)]
        finite.extend(p.grad.detach().isfinite().all() for p in model.parameters() if p.grad is not None)
        if not torch.stack(finite).all().item():
            raise RuntimeError("Nonfinite model output, loss or gradient; inspect compiler.log.")
        report.update(loss=loss.item(), reconstruction_shape=list(reconstruction.shape),
                      latent_shape=list(mu.shape), finite_outputs_and_gradients=True)
        phases = {"forward": [], "backward": []}
        for path in sorted((directory / "debug").rglob("output_code.py")):
            header = path.read_text().splitlines()[0]
            for phase in phases:
                if phase in header:
                    phases[phase].append(str(path.relative_to(directory)))
        report["generated_code"] = phases
        if not all(phases.values()):
            raise RuntimeError("Forward/backward generated source is missing; inspect compiler.log and debug/.")
        report["status"] = "complete"
        exit_code = 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        # Retain the exact scripts and hashed code/IR identity without saving
        # input, parameter, optimizer or activation tensor payloads.
        for name in (Path(__file__).name, "mbconv_vae_v2_torch.py"):
            source = Path(__file__).resolve().parent / name
            (directory / name).write_bytes(source.read_bytes())
        report["artifacts"] = [dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size,
                                    sha256=sha256(p)) for p in sorted(directory.rglob("*"))
                               if p.is_file() and p.name != "compiler.log"]
        write_json(directory / "manifest.json", report)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker is not None:
        return run_worker(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(tempfile.mkdtemp(prefix=f"inductor_vae_v2_b16_224_{stamp}_",
                                     dir=Path(__file__).resolve().parent))
    print(f"Input: {INPUT_SHAPE}; precision: {args.precision}", flush=True)
    print(f"Dump directory: {directory}", flush=True)
    print("Compiling forward and backward; this can take several minutes. See compiler.log.", flush=True)
    command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker", str(directory),
               "--device", args.device, "--precision", args.precision]
    with (directory / "compiler.log").open("w") as log:
        result = subprocess.run(command, env=compiler_environment(directory), stdout=log,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        print(f"Capture failed (exit {result.returncode}). Details: {directory / 'compiler.log'}", file=sys.stderr)
    else:
        report = json.loads((directory / "manifest.json").read_text())
        print("Saved generated code:")
        for paths in report["generated_code"].values():
            for path in paths:
                print(f"  {directory / path}")
        print(f"Graphs, IR, Triton compiler files, logs and manifest: {directory}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
