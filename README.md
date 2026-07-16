# FDRect: Flow-Distilled Image Rectangling for Consumer Devices via Dynamic Asymmetric Knowledge Transfer

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/FuShu24/FDRect-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **FDRect**, a hardware-efficient and high-fidelity image rectangling framework designed for resource-constrained consumer electronics (CE) edge platforms.

<p align="center">
  <strong>KaiJun Wu<sup>1</sup>, XiaoRong Xu<sup>2 *</sup>, DanDan Feng<sup>3</sup>, Yuan Mei<sup>4</sup>, ChongKai Zhu<sup>5</sup></strong>
</p>
<p align="center">
  <sup>1</sup> Lanzhou Jiaotong University, <sup>2</sup> Hong Kong Polytechnic University
</p>

---

## 🚀 News & Demo
* **[2026/07]** 🔥 We release **FDRect-Lite**, an efficiency-scalable variant that completely decouples motion estimation from geometric warping, reducing peak memory by **77.3%** and achieving ultra-high throughput on consumer edge platforms!
* **[Interactive Live Demo]** Click the badge above to try our online web-based demonstration. Upload your input stitched images with irregular boundaries and custom masks to verify real-time performance on standard hardware.

---

## 🛠️ Architecture Overview

FDRect addresses the mobile "Memory Wall" and thermal throttling issues in multi-frame camera pipelines by reframing complex cascading distortion modeling into an efficient, single-stage flow knowledge transfer task.

<div align="center">
  <img src="https://raw.githubusercontent.com/XiaoRong24/FDRect/main/Network.png" width="90%" alt="FDRect Architecture"/>
</div>

### Core Highlights:
* **Dynamic Asymmetric Distillation (DAD):** Achieves an **86% reduction in parameters** by adaptively modulating the knowledge transfer intensity based on training trajectories.
* **Hierarchical Hybrid Block (HHB):** Strategically blends the inductive biases of CNNs, Mamba, and Transformers to secure a Pareto-optimal trade-off between computational overhead and content structural integrity.
* **FDRect-Lite Optimization:** Specifically tailored for edge SoCs by disabling heavy cross-attention on ARM architectures, bypassing mobile memory access bottlenecks while preserving high-fidelity output.

---

## 📊 System-Level CE Benchmarks

Our model delivers competitive reconstruction quality while operating within the strict latency and hardware limitations of mainstream consumer devices:

| Method | #Params (M) | GFLOPs $\downarrow$ | Throughput (FPS) | Peak VRAM | PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DeepRectangling | 50.91 | 35.09 | 22.99 | 148.2 MB | 21.27 |
| **FDRect (Full)** | **4.56** | **2.51** | **31.34** | **33.91 MB** | **22.70** |
| **FDRect-Lite** | **1.69** | **1.14** | **55.57** | **7.68 MB** | **22.36** |

---

## ⚡ Quick Start

### 1. Environment Setup
Clone the repository and install the required dependencies (compatible with Python >= 3.8 and PyTorch >= 1.12):
```bash
git clone [https://github.com/XiaoRong24/FDRect.git](https://github.com/XiaoRong24/FDRect.git)
cd FDRect
pip install -r requirements.txt
 ```

### 2. Run Inference and Profiling
We provide standard inference scripts alongside a system-level hardware profiling suite to verify deployment metrics
```bash
# Run the standard high-fidelity model demo
python demo.py --variant full --input assets/input.png --mask assets/mask.png

# Run the hardware-optimized ultra-lightweight variant (FDRect-Lite)
python demo.py --variant lite --input assets/input.png --mask assets/mask.png

# Perform a systematic on-device profiling (latency, throughput, and memory tracking)
python benchmark.py --device cpu --batch_size 1
```

### 📝 Citation
If you find our work or the FDRect-Lite deployment scripts helpful in your research, please consider citing our paper:
