# FDRect : Flow-Distilled Image Rectangling for Consumer Devices via Dynamic Asymmetric Knowledge Transfer

<p align="center">KaiJun Wu<sup>1</sup>, XiaoRong Xu<sup>2 *</sup>, DanDan Feng<sup>3</sup>, Yuan Mei<sup>4</sup>, ChongKai Zhu<sup>5</sup></p>
<p align="center"><sup>1</sup>Lanzhou Jiaotong University, <sup>2</sup>Hong Kong Polytechnic University</p>

<div align=center>
<img src="https://github.com/XiaoRong24/FDRect/blob/main/Network.png"/>
</div>

## 🚀 News & Demo
* **[2026/07]** We release **FDRect-Lite**, an efficiency-scalable variant that completely decouples motion estimation from warping, reducing peak GPU memory by 77.3% and achieving ultra-high throughput on consumer edge devices!
* [![Hugging Face Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/FuShu24/FDRect-Demo) Upload input and mask images, then click Start Rectification.

### Quick Start
```bash
pip install -r requirements.txt
# Run standard model demo
python demo.py --variant full
# Run the newly proposed lightweight variant (FDRect-Lite)
python demo.py --variant lite
