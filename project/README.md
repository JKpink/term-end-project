# Qwen3-VL-2B + LoRA 细粒度食物图像分类

## 项目简介

本项目研究小型视觉语言模型(VLM)的专项能力增强。通过 LoRA 微调 Qwen3-VL-2B（600M参数），使其在细粒度食物分类任务上超越未训练的 4B/8B 大模型。

**核心叙事**：精准的专才 > 平庸的通才 — 参数少 4 倍，专项能力更强。

## 环境要求

- Python 3.10+
- CUDA 12.1+
- RTX 5060 8GB（或以上）
- Windows / Linux

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型和数据集
python prepare.py --target all

# 3. 验证安装
python prepare.py --target verify

# 4. 运行完整实验
python run.py all

# 5. 启动 Gradio Demo
python run.py demo
```

## 项目结构

```
project/
├── README.md
├── requirements.txt
├── prepare.py              # 下载模型+数据集
├── run.py                  # 实验编排
├── src/
│   ├── data.py             # Food101 数据加载
│   ├── train.py            # QLoRA 训练
│   ├── inference.py        # 多模型对比推理
│   ├── evaluate.py         # 评估指标
│   └── utils.py            # 工具函数
└── app/
    └── gradio_app.py       # Web Demo
```

## Baseline 对比

| 模型 | 参数 | 方式 |
|------|:---:|------|
| Qwen3-VL-2B (zero-shot) | 2B | 零样本推理 |
| Qwen3-VL-4B (zero-shot) | 4B | 零样本推理 |
| Qwen3-VL-8B (zero-shot) | 8B | 零样本推理 |
| ResNet-50 (fine-tuned) | 25M | 传统 CNN 微调 |
| CLIP ViT-B/32 (zero-shot) | 150M | 零样本分类 |
| **Qwen3-VL-2B + LoRA** | 2B | **我们的方法** |

## 单独运行

```bash
# 仅训练
python src/train.py --lora_r 16 --num_classes 20 --max_train_samples 2000

# 仅推理
python src/inference.py --lora_path ./outputs/lora_food/final

# 仅评估
python src/evaluate.py --predictions_dir ./outputs/inference
```

## 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-VL-2B-Instruct (4-bit nf4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| 学习率 | 2e-4 |
| Epochs | 2 |
| Batch size | 1 × grad_accum=8 |
| 显存峰值 | ~6 GB |
