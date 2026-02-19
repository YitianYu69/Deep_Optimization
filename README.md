# Deep_Optimization

**Deep_Optimization** is a self-built computer vision & deep learning toolkit that unifies  
state-of-the-art training strategies into a clean, modular, and extensible engine.

It is designed for **fast prototyping, stable training, efficient large-scale experiments,  
and low-precision / compressed training research**.

---

## ✨ Features

### 🔧 Engine & Training
- 🔧 **Modular design**: `Model/`, `Train/`, `Utils/`
- ⚡ **Distributed training** with PyTorch DDP or DeepSpeed
- 🧮 **Mixed precision** (fp16 / bf16) with `torch.amp`
- ➕ **Gradient Accumulation** for large effective batch sizes
- 🔒 **CUDA Graph capture** for faster & more deterministic execution

---

### ⚡ Optimization & Stability
- 🔄 **Exponential Moving Average (EMA)**
- 🔁 **Stochastic Weight Averaging (SWA)**
- 🛠️ Gradient checkpointing hooks
- 📉 Training stabilization utilities

---

### 🧠 ACT Trainer (Activation Compression Trainer)
- 🧮 Extreme activation quantization (2-bit / 1-bit research modes)
- ⚡ Triton-based custom kernels
- 💾 Memory-efficient forward/backward passes
- 🚀 Designed for:
  - Activation compression research
  - Low-precision training stability
  - GPU memory reduction
  - Throughput optimization

---

### 🌊 Frequency-Domain Learning
- 📈 Radial spectrum regularization
- 🎛️ Magnitude / phase-aware losses
- 📏 Wasserstein / Log-Huber spectral distances
- ✅ Observed benefits:
  - Stabilized quantized training
  - Improved convergence in low-precision regimes
  - Reduced frequency shortcut learning
  - More stable validation dynamics

---

### 🛠️ Model Registry
Supports extensible registration of:

- ResNet  
- PyramidNet  
- ConvNeXt  
- Vision Transformer (ViT / FreqViT variants)  

---

### 📊 Metrics & Diagnostics
- 📊 Top-1 / Top-5 Accuracy
- 📉 RMSE / regression metrics
- 🌊 Spectral statistics & frequency diagnostics

---

### 📦 Deployment
- 📦 **ONNX export**
- ⚡ TensorRT-ready inference pipeline

---

## 🚀 Quickstart

### 1️⃣ Clone & Install

```bash
git clone https://github.com/YitianYu69/Deep_Optimization.git
cd Deep_Optimization
pip install -e .
