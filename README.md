# FDRect : Flow-Distilled Image Rectangling for Consumer Devices via Dynamic Asymmetric Knowledge Transfer



<p align="center">KaiJun Wu<sup>1</sup>, XiaoRong Xu<sup>2 *</sup>, DanDan Feng<sup>3</sup>, Yuan Mei<sup>4</sup>, ChongKai Zhu<sup>5</sup></p>

<p align="center"><sup>1</sup>Lanzhou Jiaotong University, <sup>2</sup>Hong Kong Polytechnic University</p>



<div align=center>

<img src="https://github.com/XiaoRong24/FDRect/blob/main/Network.png"/>

</div>

---

## 🚀 News & Demo
* **[2026/06]** 🔥 We release **FDRect-Lite**, an efficiency-scalable variant that completely decouples motion estimation from geometric warping, reducing peak memory by **77.3%** and achieving ultra-high throughput on consumer edge platforms!
* **[Interactive Live Demo]** Click the badge above to try our online web-based demonstration. Upload your input stitched images with irregular boundaries and custom masks to verify real-time performance on standard hardware.

---


## 📊 System-Level CE Benchmarks

Our model delivers competitive reconstruction quality while operating within the strict latency and hardware limitations of mainstream consumer devices:

| Method | #Params (M) | GFLOPs $\downarrow$ | Throughput (FPS) | Peak VRAM | SSIM $\uparrow$ | PSNR (dB) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepRectangling | 50.91 | 35.09 | 22.99 | 148.2 MB | 0.7141 | 21.27 |
| **FDRect (Full)** | **4.56** | **2.51** | **31.34** | **33.91 MB** | **0.7960** | **22.70** |
| **FDRect-Lite** | **1.69** | **1.14** | **55.57** | **7.68 MB** | **0.7835** | **22.36** |

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
