# Neural Router Model & Hardware Requirements

This document outlines the hardware requirements, recommended lightweight models, quantization strategies, and fallback configurations for deploying and fine-tuning neural router models in the Smart AI Router framework.

---

## 🎯 Target Hardware Specifications

The neural router is designed to run efficiently on low-resource environments, including developer laptops and entry-level GPU instances.

| Spec | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | **2 GB VRAM** | 4 GB – 8 GB VRAM |
| **System RAM** | 8 GB RAM | 16 GB RAM |
| **CPU** | 4-core modern x86/ARM | 8-core CPU |
| **Disk Space** | 5 GB available | 20 GB NVMe SSD |

---

## 🤖 Recommended Models

Lightweight open-weight models (< 1B parameters) optimal for document context routing:

| Model | Parameters | Context Window | Recommended Quantization | Notes |
|---|---|---|---|---|
| **Qwen2.5-0.5B** | 490M | 32,768 tokens | 4-bit NF4 / 8-bit | High accuracy, multi-lingual |
| **Qwen3-0.6B** | 600M | 32,768 tokens | 4-bit NF4 | Advanced reasoning router |
| **SmolLM2-135M** | 135M | 8,192 tokens | FP16 / INT8 | Ultra-fast low latency (< 5ms) |
| **SmolLM2-360M** | 360M | 8,192 tokens | 4-bit NF4 | Excellent speed/relevance balance |

---

## ⚡ Quantization & QLoRA Configuration

To fit within 2GB VRAM constraints, models should be loaded using **4-bit NF4 quantization** via `bitsandbytes` and fine-tuned with **PEFT LoRA**:

```yaml
# training/configs/neural_router.yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj"]
```

### Estimated Memory Footprint (Qwen2.5-0.5B)

- **FP16 baseline**: ~1.0 GB VRAM
- **4-bit NF4 QLoRA**: ~0.4 GB VRAM
- **Forward batch size = 16**: ~0.8 GB peak VRAM

---

## 💻 CPU Fallback Strategy

When a GPU is not present:
1. The framework automatically falls back to `device="cpu"`.
2. PyTorch uses multi-threaded CPU inference.
3. For ultra-low CPU latency, smaller models like `SmolLM2-135M` are recommended.

---

## ⚡ Training-Gated Routing Principles & Training Triggers

routing principles are categorized into **Heuristic-based** (zero training required) and **Training-gated** (require offline or online fitting before benchmark inclusion):

### 1. Heuristic-Based Principles (Pre-registered)
- `random`: Reproducible random seed baseline.
- `bm25`: Classic text-overlap keyword similarity scorer.
- `tfidf`: Local term frequency-inverse document frequency scorer.
- `embedding`: Vector space cosine similarity.

### 2. Training-Gated Principles
- `logistic_regression`: A learned linear model fitted on extracted sample feature vectors.
- `neural_router`: A fine-tuned sequence classifier representing LLM attention or gradient importance.

These scorers check their internal training condition flags. If untrained, they fall back to baseline-random values (scoring 0.5) to isolate benchmarks from throwing runtime errors.

### Training Trigger Example
Before executing benchmarks, check that models are trained:
```bash
# Train the logistic regression classifier on CUAD + LexGLUE compatibility datasets
PYTHONPATH=. python -m training.train
```
Model checkpoints and serialized `.pkl` or `.pt` weights are automatically saved to `models/saved/` (which is skipped by git tracking).

