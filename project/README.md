# 基于 GRPO 的 Qwen3.0 小模型理科推理能力增强研究

智能科学综合课程设计五 · 课程项目

---

## 项目简介

使用 **GRPO（Group Relative Policy Optimization，DeepSeek-R1 同款算法）** 对 **Qwen3.0** 小模型进行推理能力训练，打造一个**理科题目求解助手**——输入物理、化学或数学题，AI 逐步展示推理过程并给出答案。

```
输入："一个 2kg 的物体从 10m 高处落下，求落地速度（g=10m/s²）"

输出：<think>
      1. 已知 m=2kg, h=10m, g=10m/s²
      2. 机械能守恒：mgh = ½mv²
      3. 代入：2×10×10 = ½×2×v²
      4. v = √200 ≈ 14.14 m/s
      </think>
      答案：14.14 m/s
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型和数据（国内用 ModelScope，国外用 HuggingFace）
python prepare.py --target all --source modelscope

# 3. 一键运行实验
python run_5060.py all          # RTX 5060 (8GB)
python run_local.py all         # RTX 2080Ti ×4 (44GB)
# Kaggle: 上传 kaggle_notebook.ipynb

# 4. 启动演示
python run_5060.py demo
```

---

## 环境要求

| 环境 | GPU | 脚本 |
|------|-----|------|
| 笔记本 | RTX 5060 8GB ×1 | `run_5060.py` |
| 服务器 | RTX 2080Ti 11GB ×4 | `run_local.py` |
| 云端 | Kaggle T4 16GB ×2 | `kaggle_notebook.ipynb` |

**软件：** Python 3.10+ / PyTorch 2.4+ / CUDA 12.1+

---

## 命令说明

### 数据准备

```bash
python prepare.py --target all --source modelscope   # 国内推荐
python prepare.py --target all --source huggingface   # 国外/镜像
python prepare.py --target verify                      # 验证环境
```

### 分步运行

```bash
python run_5060.py 1        # Step 1: 下载模型+数据
python run_5060.py 2        # Step 2: SFT 预热训练
python run_5060.py 3        # Step 3: GRPO 主实验
python run_5060.py 4        # Step 4: 消融实验
python run_5060.py 5        # Step 5: 评估
python run_5060.py demo     # Step 6: Gradio 演示
```

### 单独使用

```bash
# SFT 训练
python src/train_sft.py --model_name Qwen/Qwen3.0-0.6B --output_dir ./outputs/sft

# GRPO 训练
python src/train_grpo.py --model_name Qwen/Qwen3.0-0.6B --sft_adapter_path ./outputs/sft/final

# 评估
python src/evaluate.py --model_name qwen3.0-0.6b --dataset gsm8k --max_samples 200
```

---

## 实验设计

### Baseline（5 组）

| 方法 | 说明 |
|------|------|
| 基座模型直接回答 | Qwen3.0-0.6B，不开 thinking |
| 基座模型 + thinking | 原生 thinking 模式 |
| QLoRA SFT | 监督微调 |
| **SFT + GRPO（本文）** | 核心方法 |
| 1.7B + SFT + GRPO | 规模对比 |

### 消融实验（5 组）

| 维度 | 对比 |
|------|------|
| SFT 预热 | 有 vs 无 |
| 奖励函数 | 纯正确性 vs 格式+正确性 vs 完整 |
| Group Size | G=4 vs G=8 vs G=16 |
| KL 系数 | β=0.01 vs 0.04 vs 0.1 |

---

## 项目结构

```
project/
├── README.md                 # 本文件
├── REPORT.md                 # 详细项目报告
├── requirements.txt          # Python 依赖
├── prepare.py                # 数据与模型准备
├── run_5060.py               # RTX 5060 一键运行
├── run_local.py              # RTX 2080Ti 并行运行
├── kaggle_notebook.ipynb     # Kaggle 运行
├── src/
│   ├── reward.py             # 奖励函数
│   ├── train_sft.py          # SFT 训练
│   ├── train_grpo.py         # GRPO 训练
│   └── evaluate.py           # 评估脚本
├── app/
│   └── gradio_app.py         # Web 演示
├── models/                   # 本地模型
├── outputs/                  # 训练输出
├── paper/                    # 论文
└── ppt/                      # PPT
```

---

## 数据集与模型

| 类型 | 名称 | 来源 |
|------|------|------|
| 数据集 | GSM8K (7473题) | `openai/gsm8k` (HF) / `swift/gsm8k` (ModelScope) |
| 数据集 | SciBench (~695题) | `xw27/scibench` (HF) |
| 模型 | Qwen3.0-0.6B-Instruct | `Qwen/Qwen3.0-0.6B` |
| 模型 | Qwen3.0-1.7B-Instruct | `Qwen/Qwen3.0-1.7B` |

---

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥2.4.0 | 深度学习框架 |
| transformers | ≥4.47.0 | 模型加载 |
| TRL | ≥0.12.0 | GRPO 训练 |
| PEFT | ≥0.13.0 | LoRA/QLoRA |
| bitsandbytes | ≥0.44.0 | 4-bit 量化 |
| Gradio | ≥5.0.0 | Web 演示 |

---

## 参考文献

1. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (2025)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning (2024)
3. Qwen3.0 Technical Report (2026)
4. LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
5. QLoRA: Efficient Finetuning of Quantized LLMs (NeurIPS 2023)
6. SciBench: Evaluating College-Level Scientific Problem-Solving (ICML 2024)

---

> **课程：** 智能科学综合课程设计五 · 河北师范大学  
> **指导教师：** 霍丽娜
