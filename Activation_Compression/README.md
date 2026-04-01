# Activation Compression

`Activation_Compression/` is the low-precision systems core of **Deep_Optimization**.

It provides an end-to-end activation quantization pipeline designed for memory/throughput optimization during training, while preserving model quality with frequency-aware and layer-adaptive heuristics.

---

## What this module does

At a high level, this subsystem:

1. **Analyzes a model graph** with `torch.fx`.
2. **Selects layers/metadata** for quantized activation handling.
3. **Rewrites layers** into custom DO-* modules (`DOLinear`, `DOConv*`, etc.).
4. **Quantizes activations in forward** (bit-packing with Triton/CUDA).
5. **Dequantizes activations in backward** to compute gradients.
6. Optionally applies **frequency-domain heuristics** and **layer-adaptive rules**.

The result is a research-oriented framework for activation compression under demanding training conditions (mixed precision, DDP, CUDA Graph, etc.).

---

## Folder structure

```text
Activation_Compression/
├─ controller.py                  # FX tracing + module replacement orchestration
├─ quantizer.py                   # layer scoring / bit decisions / analysis helpers
├─ modules/
│  ├─ layers.py                   # DO* replacement layer classes
│  ├─ ops.py                      # custom autograd Functions for quantized ops
│  ├─ module_utils.py             # quantize/dequantize orchestration helpers
│  ├─ tensor_act_reshape_utils.py # group/padding/reshape logic for activation packing
│  ├─ activations/
│  └─ normalization/
├─ triton_kernel/                 # Triton kernels and registration helpers
├─ act_triton_kernel.py           # main Triton bit-pack/dequant kernels
├─ cpp_extension/
│  ├─ bind.cpp                    # PyBind bindings
│  ├─ kernels/                    # CUDA kernels
│  ├─ src/                        # C++/CUDA dispatch & utilities
│  └─ setup.py                    # extension build config
├─ fusion/                        # fused ops/layers utilities
├─ freq_utils.py                  # frequency-domain spectrum helpers
└─ cuda_graph_utils.py            # CUDA graph integration utilities
```

---

## Core components

### 1) Controller (`controller.py`)

`Controller` is the entry point for activation compression graph transformation.

- Symbolically traces the model (`torch.fx`).
- Propagates shape metadata with fake tensors.
- Builds per-layer quantization metadata (group size, padding, bit config, optional division settings).
- Replaces eligible modules with quantized DO-* implementations.

Typical flow:

```python
controller = Controller(model, act_config, train_loader, criterion, test=False)
controller.iterate(criterion=criterion)
controller.warp_model(graph_mode=True, quantizer=True)
compressed_model = controller.traced_model
```

---

### 2) Quantizer (`quantizer.py`)

`Quantizer` manages quantization policy and statistics.

Capabilities include:

- Filtering non-target tensors.
- Layer-level signal analysis (e.g., low-frequency energy ratio, trust-like gradient/weight ratio, SNR, activation variance).
- Optional low-rank activation analysis via autocorrelation eigendecomposition.

This file is intentionally research-friendly: it is built to iterate on precision heuristics quickly.

---

### 3) Quantized ops and layers (`modules/`)

- `modules/layers.py` defines DO-* layer wrappers used after graph rewrite.
- `modules/ops.py` contains custom autograd logic:
  - Forward: activation quantization + packed storage.
  - Backward: dequantization + gradient reconstruction.
- `modules/module_utils.py` provides unified quantize/dequantize wrappers and reshaping logic.

These parts are where most of the training-time compression behavior is implemented.

---

### 4) Triton kernels (`act_triton_kernel.py`, `triton_kernel/`)

This module implements custom Triton kernels for:

- Quantize + bit-pack.
- Dequantize + unpack.
- Pack-only/unpack-only modes.

Registered custom ops are exposed under `torch.ops.act_lib.*` for use in Python-side quantization paths.

---

### 5) C++/CUDA extension (`cpp_extension/`)

Includes lower-level CUDA kernels and pybind bindings for specialized pack/unpack and related routines.

Build notes:

- The provided setup compiles multiple GPU targets (e.g., SM80/89/90/90a).
- Use this path when benchmarking low-level packing performance or integrating custom kernels beyond Triton coverage.

---

## Integration with the Trainer

In `Train/Trainer.py`, ACT is integrated by creating a `Controller`, running analysis, and replacing model layers before wrapping under DDP (or raw mode).

This means activation compression can be plugged directly into your regular training loop configuration.

---

## Minimal usage pattern

```python
from Activation_Compression.controller import Controller

act_config = {
    "batch_size": 64,
    "default_bits": 2,
    "group_size": 256,
    "fp8": False,
    "depth_point_conv": False,
    # optional keys:
    # "AVG_ALAM": True,
    # "AVG_ALAM_BTS": 4,
    # "DIVISION": {"pool_kernel_size": 4},
}

controller = Controller(model, act_config, train_loader, criterion, test=False)
controller.iterate(criterion=criterion)
controller.warp_model(graph_mode=True, quantizer=True)
model = controller.traced_model
```

---

## Research notes

This folder is a **research-first implementation**. You should expect active iteration on:

- Layer selection heuristics
- Bit allocation strategies
- Quantization scaling/clamping behavior
- Kernel-level performance tuning

For publications or long-running benchmarks, pin commit hashes and exact environment versions.

---

## Recommended future additions

- Add unit tests for `quantize/dequantize` invariants.
- Add benchmark scripts comparing memory + throughput vs baseline.
- Add config schema validation for ACT config safety.
- Add worked examples for CNN and ViT use cases.
