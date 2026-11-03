---
license: mit
tags:
  - shot-boundary-detection
  - scene-detection
  - video
  - transformer
arxiv: 2604.24762
---

## OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer

**Authors:** Boyang Wang, Guangyi Xu, Jiahui Zhang, Zhipeng Tang, Zezhou Cheng  
**Affiliation:** UVA Computer Vision Lab

[![Project Page](https://img.shields.io/badge/Project%20Page-333399.svg?logo=homepage)](https://uva-computer-vision-lab.github.io/OmniShotCut_website/)
[![Online Demo](https://img.shields.io/badge/%F0%9F%A4%97%20OnlineDemo-276cb4.svg)](https://huggingface.co/spaces/uva-cv-lab/OmniShotCut)
[![Code](https://img.shields.io/badge/Code-black.svg?logo=github)](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)
[![arXiv](https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv)](https://arxiv.org/abs/2604.24762)

OmniShotCut is a sensitive and informative shot boundary detector. It detects shot changes across diverse sources (anime, vlog, game, shorts, sports, screen recording, etc.) and recognizes Sudden Jumps and Transitions (dissolve, fade, wipe, ...) with a Shot-Query-based Video Transformer.

## Model Zoo

| Model | Description | Link |
|-------|-------------|------|
| **OmniShotCut v1.5** (recommended) | Hidden dim 576, 52.8M params. Best real-world quality: fewer false cuts on continuous camera motion / lighting changes, and better dense fast-cut recall. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5) |
| OmniShotCut (paper) | 3 encoders, 34.5M params. Reproduces the paper (Table 1) numbers. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut_paper) |
| OmniShotCut (preview) | 6 encoders, 41M params. Early preview release. | [Download](https://huggingface.co/uva-cv-lab/OmniShotCut) |

## Usage

Install:
```bash
pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git
```

Run shot boundary detection:
```python
import omnishotcut

cut_model = omnishotcut.load("uva-cv-lab/OmniShotCut_v1.5", filename="OmniShotCut_ckpt.pth")
ranges = cut_model.inference("video.mp4", mode="clean_shot")
```

## Citation

```bibtex
@article{wang2026omnishotcut,
  title={OmniShotCut: Holistic Relational Shot Boundary Detection with Shot-Query Transformer},
  author={Wang, Boyang and Xu, Guangyi and Zhang, Jiahui and Tang, Zhipeng and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2604.24762},
  year={2026}
}
```
