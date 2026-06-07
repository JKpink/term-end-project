# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **not a software codebase** — it is an academic course materials repository for **智能科学综合课程设计五** (Intelligent Science Comprehensive Course Design 5) at **河北师范大学 (Hebei Normal University)**. It contains reference papers, the course paper manuscript, and course administrative documents. There is no source code, build system, or test infrastructure.

## Course Requirements

- **Week 8**: Submit project topic
- **Week 10**: Submit paper draft
- **Week 17**: Present PPT (5 minutes)
- **Week 18 (deadline)**: Complete a small AI application or computer vision project using large models, on a topic of personal interest within the field
- **Submission**: Printed paper via class committee + electronic copy to `huolina@hebtu.edu.cn`
- **File naming**: `name+studentID+major+topic`

### Paper Structure

Must include: abstract, methods/algorithm description, experiments/dataset description, runtime environment, results analysis and comparison, references. No plagiarism; must follow journal formatting.

## Key Files

| File | Purpose |
|------|---------|
| `智能科学设计五考核方式+封面.docx` | Course assessment criteria, schedule, and cover page template |
| `论文软件-ww.docx` | Research paper manuscript: **DilatedGait** — gait recognition using dilated reparameterization and atrous convolution architecture. Corresponding author: Wang Wei (wangwei2021@hebtu.edu.cn) |
| `投稿-终版.doc` | Final submission version of the paper |
| `2024-软件学院--基于扩张重参数化和空洞卷积架构的步态识别方法_霍丽娜 (1).pdf` | Published reference: DilatedGait paper (Huo Lina et al.) |
| `2020--河北工业科技--本科---基于CenterNet的小学生英文手写体区域检测_张朝晖.pdf` | Reference paper: CenterNet-based handwriting detection |

## Research Topic

The paper proposes **DilatedGait**, a gait recognition method addressing the mismatch between effective receptive field and human silhouette region in outdoor scenes. Key technical components:
- **Atrous convolution** to expand neuron receptive field without downsampling resolution loss
- **Dilated Reparameterization Module (DRM)** to fuse multi-scale convolution kernels and optimize effective receptive field focus
- Based on a modified ResNet backbone with residual atrous convolution blocks
- Evaluated on Gait3D and GREW outdoor gait datasets
- Environment: Python 3.8, CUDA 11.3, PyTorch 1.11.0, NVIDIA RTX 3090
