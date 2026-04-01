# Deep_Optimization

Deep_Optimization is a modular computer-vision training framework focused on **efficient, stable, and research-oriented deep learning**.

It combines distributed training, mixed precision, advanced optimization utilities, activation compression, and frequency-domain regularization under one project.

---

## Highlights

- **Modular architecture** for model, training, and optimization components.
- **Distributed training support** (PyTorch DDP / DeepSpeed paths).
- **AMP support** (fp16 / bf16) with gradient-scaling flow.
- **CUDA Graph-ready training path** for reduced runtime overhead.
- **Activation Compression (ACT)** with custom Triton/CUDA kernels.
- **Frequency-domain tooling** for spectral diagnostics/regularization.
- **Built-in model family support** for CNN and ViT variants.

---

## Repository layout

```text
Deep_Optimization/
├─ Model/
│  ├─ CNN/                        # ResNet, ConvNeXt, MobileNet, EfficientNet, PyramidNet
│  └─ ViT/                        # ViT/DeiT variants
├─ Train/
│  ├─ Trainer.py                  # main training engine
│  ├─ data.py                     # dataloader and transform helpers
│  ├─ experiment.py               # EMA and related experiment utilities
│  ├─ utils_train.py
│  ├─ utils_ddp.py
│  └─ log.py
├─ Optimizer/
│  └─ SGD_geometry.py             # custom geometry/noise-aware optimizer experiments
├─ Activation_Compression/        # low-precision activation compression subsystem
└─ README.md
```

---

## Core design

### 1) Training Engine

`Train/Trainer.py` orchestrates model execution and wraps features such as:

- DDP / DeepSpeed wrapping
- optional teacher-model distillation flow
- AMP and scaler behavior
- gradient accumulation
- optional CUDA Graph capture path

The goal is a single training surface that can toggle advanced strategies by config.

### 2) Optimization & Stability

`Train/experiment.py` includes EMA utilities with options such as:

- dynamic decay schedule support
- optional FP32 master shadow params
- buffer handling
- optional Kahan-style compensation path

`Optimizer/SGD_geometry.py` contains experimental optimization work that blends geometric preconditioning and noise shaping.

### 3) Activation Compression (ACT)

`Activation_Compression/` is the main research subsystem for low-bit activation handling:

- FX-based graph rewrite
- custom autograd operators
- Triton kernels for packing/unpacking
- optional C++/CUDA extension path
- frequency-aware and layer-adaptive heuristics

See `Activation_Compression/README.md` for full details.

---

## Quick start

## 1) Clone

```bash
git clone https://github.com/YitianYu69/Deep_Optimization.git
cd Deep_Optimization
```

## 2) Environment

Use a Python environment with PyTorch + CUDA, then install project dependencies and editable package setup as appropriate for your workflow.

(If you rely on ACT kernels, ensure your CUDA/Triton/PyTorch versions are mutually compatible.)

## 3) Train

Use your training entry script/config to initialize:

- model from `Model/`
- dataloaders from `Train/data.py`
- optimizer/scheduler/loss
- `Trainer` from `Train/Trainer.py`

Then run epoch loops with `trainer.train(...)` and validation hooks.

---

## Who this project is for

This project is especially suitable for:

- Researchers exploring low-precision training stability
- Engineers benchmarking memory/performance tradeoffs
- Practitioners who want a flexible CV experimentation framework

---

## Roadmap ideas

- Add standardized config system and launch scripts
- Add reproducible benchmark suite (speed/memory/accuracy)
- Expand docs for common training recipes
- Add CI tests for core training and ACT paths

---

## License

MIT
