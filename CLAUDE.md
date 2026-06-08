# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Repository Overview

Course project for **智能科学综合课程设计五** at **河北师范大学** (Hebei Normal University).

## Project: Multi-Agent LLM Collaborative Writing

Train two small LLM agents with role specialization for collaborative writing.
Adapted from CoMLRL (github.com/OpenMLRL/CoMLRL).

### Core idea

**2 × Qwen3-0.6B + LoRA (collaborative) vs single Qwen3-1.7B**

- Agent A: concise summarizer (extract key points)
- Agent B: detailed writer (expand into full summary)

### Baselines

| # | Method | Description |
|---|--------|-------------|
| B1 | Single Model | One model does everything |
| B2 | Parallel | Two models, no communication |
| B3 | Sequential | A→B, no training |
| B4 | Discussion | One-round discussion |
| Ours | Collaborative | LoRA-trained role specialization |

### Key source files

| File | Role |
|------|------|
| `project/src/train.py` | QLoRA + LoRA training, reward functions, formatters |
| `project/src/evaluate.py` | 5-baseline comparison, metrics |
| `project/app/gradio_app.py` | Web demo: 5-column comparison |
| `project/kaggle_notebook.ipynb` | Kaggle T4×2 training notebook |

### Common commands

```bash
cd project
pip install -r requirements.txt

# Local training (small scale)
python src/train.py --model_name Qwen/Qwen3-0.6B --task tldr --dataset_size 100

# Local evaluation
python src/evaluate.py --model_name Qwen/Qwen3-0.6B --num_samples 20

# Kaggle: upload kaggle_notebook.ipynb for T4×2 training

# Gradio demo
python app/gradio_app.py
```

### Key design decisions

- **QLoRA (4-bit)**: Enables training on 8GB consumer GPU
- **Simple REINFORCE**: Not full MAGRPO — uses single reward signal per step
- **Length ratio reward**: Agent B should produce 2-3x longer output than Agent A
- **320 training samples**: Matches CoMLRL's TLDR example
- **Two separate LoRA adapters**: Agent A and Agent B have different LoRA weights

### Hardware

- Training: Kaggle T4×2 (16GB each) or RTX 5060 8GB
- Inference: 2×Qwen3-0.6B 4-bit ≈ 2GB

## Instructor

霍丽娜 (Huo Lina) — huolina@hebtu.edu.cn
