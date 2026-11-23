# VAE V2 Inductor code capture

Keep [test_inductor_vae_v2.py](./test_inductor_vae_v2.py) and
[mbconv_vae_v2_torch.py](./mbconv_vae_v2_torch.py) together. Run in a Python
environment with CUDA-enabled PyTorch and its matching Triton/toolchain:

```bash
python test_inductor_vae_v2.py
```

From the repository root, use:

```bash
python examples/vae/test_inductor_vae_v2.py
```

The runner uses only this fixed input: **16 RGB images, 224 × 224**, represented
as a contiguous PyTorch NCHW tensor of shape **`(16, 3, 224, 224)`**. Inputs are
synthetic FP32 images in `[-1, 1]`. The full MBConvVAE_V2 has base/max channels
32/512, two blocks per stage, four latent channels and 32,317,999 parameters.
The plain Torch model preserves the original architecture, initialization,
parameter names, BlurPool recipe and stochastic latent sampling.

The default uses BF16 autocast, FP32 parameters, and the original FP32
`L1 + 1e-6 * KL` loss. To explicitly capture FP32 or select another GPU:

```bash
python test_inductor_vae_v2.py --precision fp32 --device cuda:0
```

`torch.compile(backend="inductor", fullgraph=True, dynamic=False)` compiles
forward and loss; the actual `loss.backward()` triggers PyTorch AOTAutograd and
Inductor backward compilation. A graph break or compiler error fails the run.
There is no optimizer step. No DeepOpt installation, dataset or weights download
is required. The runner never changes the batch size after an OOM.

Every invocation creates a new permanent dump folder **beside the test script**,
regardless of the shell's current directory. Earlier captures are preserved:

```text
test_inductor_vae_v2.py
mbconv_vae_v2_torch.py
inductor_vae_v2_b16_224_<UTC timestamp>_<unique suffix>/
    manifest.json
    compiler.log
    test_inductor_vae_v2.py
    mbconv_vae_v2_torch.py
    debug/
        torchinductor/
            model__...forward.../
                output_code.py
                fx_graph_readable.py
                fx_graph_transformed.py
                ir_pre_fusion.txt
                ir_post_fusion.txt
            model__...backward.../
                output_code.py
                ...
    inductor_cache/
    triton_cache/
    triton_dumps/
```

Exact generated directory names depend on PyTorch. The runner prints both
`output_code.py` paths after success. These files contain the generated Python
wrappers and embedded Triton kernel source. Available lower-level compiler files
remain in the adjacent caches/dumps. Calls into cuDNN/cuBLAS/ATen may remain
external library calls; Inductor cannot dump their internal implementation.
`compiler.log` contains AOT forward/backward graph logging and diagnostics.
`manifest.json` records software/GPU identity, fixed geometry, finite checks,
generated-code paths and artifact SHA256 hashes. No tensor checkpoints are saved.

Use the retained forward and backward code to inspect the captured execution.
Numerical and runtime comparisons require a separate matched benchmark.
