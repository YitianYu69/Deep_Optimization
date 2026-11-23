# Qualify a DeepOpt build

Use [qualify_backend.py](../../benchmarks/compiler/qualify_backend.py) with the
non-editable installed DeepOpt wheel under test. The runner pins the package
identity across workers and writes one log and result per stage under your
output directory.

Run the CPU contracts first:

```sh
python benchmarks/compiler/qualify_backend.py --profile cpu --output work/qualify/cpu
```

Check the CUDA prerequisites and expand the test plan with `--dry-run`:

```sh
python benchmarks/compiler/qualify_backend.py \
  --profile gpu-full --device cuda:0 --vae-audit --dry-run --output work/qualify/plan
```

Run the CUDA profiles in an environment with compatible PyTorch and Triton:

```sh
python benchmarks/compiler/qualify_backend.py \
  --profile gpu-smoke --device cuda:0 --output work/qualify/gpu-smoke
python benchmarks/compiler/qualify_backend.py \
  --profile gpu-full --device cuda:0 --vae-audit --output work/qualify/gpu-full
```

The profiles exercise compiler ownership, saved-state contracts, isolated
fusion, and model numerics. Compression comparisons use `group_size=1024`.
The VAE metadata audit uses fake tensors to inventory saved state. Use
`--vae-target` with an explicit `--vae-optimizer-factory` and
`--vae-capacity-batches` for real VAE training and capacity checks. Numerical
workers compare named tensors in bounded chunks. Their reports separate
numerical acceptance, retained storage, and allocator peaks.

For distributed transport, see the [packed-offload DDP guide](packed_activation_offload_ddp.md).
For matched TorchInductor comparisons, use the
[local Inductor guide](../../compiler_references/inductor/README.md) and match
the recipe and target hardware. The CPU contract runs and CUDA profiles qualify
their recorded test cases. Throughput and capacity need separate target runs.
