# FP8 validation and test environments

[FP8 guide](README.md) · [Test groups](../../tests/README.md) · [CUDA kernels](cuda.md)

Reference execution, real-model CPU tests, and CUDA tests verify different
parts of the FP8 implementation. Their requirements and limits are below.

## Black-box and white-box validation

Install optional real-model dependencies without downloading pretrained weights:

```bash
python -m pip install -e '.[model-tests]'
DEEPOPT_REQUIRE_MODEL_TESTS=1 python -m unittest discover -s tests/opt -v
python -m unittest discover -s tests/distributed -p 'test_fp8_*.py' -v
```

For a fresh Linux/Windows CPU environment, install PyTorch and torchvision
together from the CPU wheel index **before** installing the model-test extra:

```bash
python -m pip install 'torch>=2.4,<3' 'torchvision>=0.19,<1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model-tests]'
python -m pip check
```

Mixing CPU PyTorch with a torchvision binary from a different build channel can
cause `RuntimeError: operator torchvision::nms does not exist`. That import error
also affects timm, Transformers and TorchMetrics. Recreate the environment with
a matching pair instead of skipping the model tests or registering a fake NMS
operator. For macOS or CUDA, use the corresponding matching pair from the
[official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).
The CPU CI job installs both wheels from the CPU index and executes real NMS
before running the test groups.

CPU CI also installs Triton and checks language attributes referenced inside
the kernel bodies, which Python imports and compileall do not resolve. This
check needs no GPU and catches missing APIs before a CUDA launch. It does not
replace compilation and numerical validation on real hardware.

The tiny randomly initialized real architectures are torchvision ViT, timm ViT,
HF GPT-2, and HF Llama. Black-box tests compare outputs/gradients against dense or
independently quantized references, perform optimizer updates, and verify exact
checkpoint continuation. White-box tests inspect replacement identity, tied
embeddings, graph targets, saved FP8 tensors/scales, gradient dispatch, and DDP
reducer ownership. Native attention tests with quantization disabled isolate
adapter semantics from quantization error.

On a compatible GPU:

```bash
DEEPOPT_REQUIRE_MODEL_TESTS=1 DEEPOPT_REQUIRE_CUDA_FP8=1 \
  python -m unittest discover -s tests/cuda -p 'test_fp8_*.py' -v
```

CUDA tests execute actual Triton kernels against dequantized FP32 or reference
oracles, covering edge tiles, scale boundaries, strided layouts, backward,
real-model training, and conditional two-GPU NCCL resume. One GPU runs the kernel
and full-model cases; two compatible GPUs and NCCL are needed for DDP cases.
The required-environment switches turn missing prerequisites into failures.
Ordinary local discovery skips unavailable hardware/packages explicitly.
The existing opt-in GPU CI job requires two FP8-capable GPUs and model packages
so an all-skipped FP8 job cannot be mistaken for validation.

CPU/reference validation was executed in the development environment with
PyTorch 2.14.1, torchvision 0.29.1, Transformers 4.57.6 and timm 1.0.31. No CUDA
device is available in that macOS environment; actual Triton/NCCL FP8 results
remain pending a compatible GPU run. CPU passes and hardware skips are not GPU
validation or performance results.
