# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **dual-purpose** repository for **智能科学综合课程设计五** (Intelligent Science Comprehensive Course Design 5) at **河北师范大学 (Hebei Normal University)**, containing:

1. **A functioning PyTorch project** (`project/`) — GRPO-based reasoning enhancement for Qwen3.5 small models
2. **Academic materials** — course admin documents, the DilatedGait reference paper, and submission templates

## Project: GRPO-based Qwen3.5 Reasoning Enhancement

The project trains **Qwen3.5-0.8B-Instruct** (a GatedDeltaNet + Attention hybrid architecture) to solve science problems (physics, chemistry, math) with step-by-step `<think>` reasoning, using **GRPO (Group Relative Policy Optimization)** — the same RL algorithm behind DeepSeek-R1.

### Architecture (two-stage training pipeline)

```
Phase 1: SFT warmup           Phase 2: GRPO RL
═══════════════════           ═══════════════
Qwen3.5-0.8B                  SFT adapter (init)
  + QLoRA (4-bit)       →       + fresh LoRA
  + GSM8K formatted               + GRPO reward-driven
    (<think>...</think>)            optimization
  → SFT adapter                    → GRPO model
```

- **QLoRA** (4-bit nf4 quantization, r=8–16) throughout — both phases fit in 8GB VRAM
- **Reward function** (`src/reward.py`): 0.7 correctness + 0.2 format + 0.1 step count (3 variants for ablation)
- **GRPO vs PPO**: no critic/value network needed — advantage estimated via group-normalized comparison, saving ~75% VRAM
- **Baselines** (5): raw model, native thinking, SFT-only, SFT+GRPO (core), 2B-scale comparison
- **Ablations** (5): no SFT warmup, reward type, group size (4/8/16), KL coefficient (0.01/0.04/0.1)

### Key source files

| File | Role |
|------|------|
| `project/src/reward.py` | 3 reward strategies (`correctness_only`, `correctness_and_format`, `full`), answer extraction, format checking |
| `project/src/train_sft.py` | SFT phase: QLoRA + GSM8K formatted with `<think>` tags, uses `HfArgumentParser` for config |
| `project/src/train_grpo.py` | GRPO phase: loads SFT adapter, configures `GRPOTrainer` with reward function, QLoRA |
| `project/src/evaluate.py` | Evaluation on GSM8K/SciBench, measures accuracy, thinking-rate, avg reasoning steps |
| `project/app/gradio_app.py` | Gradio web demo — `solve()` loads model and streams `<think>` reasoning output |
| `project/prepare.py` | Downloads models/datasets from ModelScope (China access) or HuggingFace (with hf-mirror.com) |

### Common commands

```bash
# Environment
cd project
pip install -r requirements.txt

# Download models + datasets (China: modelscope; elsewhere: huggingface)
python prepare.py --target all --source modelscope
python prepare.py --target verify                      # Check if everything is accessible

# Run experiments (pick one environment)
python run_5060.py all         # RTX 5060 8GB, single-GPU, serial
python run_local.py all        # RTX 2080Ti x4, multi-GPU parallel via threads
# Or upload project/kaggle_notebook.ipynb for Kaggle T4 x2

# Run individual steps
python run_5060.py 1           # Step 1: download models + data
python run_5060.py 2           # Step 2: SFT warmup
python run_5060.py 3           # Step 3: GRPO main experiment
python run_5060.py 4           # Step 4: ablation experiments (serial)
python run_5060.py 5           # Step 5: evaluate all baselines
python run_5060.py 6           # Step 6: launch Gradio demo

# Run individual scripts directly
python src/train_sft.py --model_name Qwen/Qwen3.5-0.8B-Instruct --output_dir ./outputs/sft
python src/train_grpo.py --model_name Qwen/Qwen3.5-0.8B-Instruct --sft_adapter_path ./outputs/sft/final
python src/evaluate.py --model_name qwen3.5-0.8b --dataset gsm8k --max_samples 200

# Gradio demo
python app/gradio_app.py
```

### Run scripts: when to use which

- `run_5060.py` — **single GPU ≤8GB**, serial execution, conservative batch/generation configs. Use for development/debugging.
- `run_local.py` — **4-GPU parallel** via `threading.Thread`, spawns concurrent SFT + GRPO experiments across GPUs. Use for full experiment grid.
- `kaggle_notebook.ipynb` — **cloud**, Kaggle T4 x2. Mirrors the same training logic in notebook form.

### Environment requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.11+ |
| PyTorch | 2.4.0 | 2.5+ |
| CUDA | 12.1 | 12.4+ |
| GPU VRAM | 8GB (RTX 5060) | 11GB x4 (RTX 2080Ti) |

Key dependencies: `transformers>=4.47.0`, `trl>=0.12.0`, `peft>=0.13.0`, `bitsandbytes>=0.44.0`, `gradio>=5.0.0`

### Training output structure

```
project/outputs/
├── sft_*/              # LoRA adapter from Phase 1
│   └── final/          #  adapter_model.bin, adapter_config.json
├── grpo_*/             # GRPO model from Phase 2
│   └── final_grpo/     #  merged or adapter weights
└── eval_*/             # CSV results + JSON metrics per baseline
```

### Reward function design

Three strategies in `src/reward.py`, selectable via `--reward_type`:
1. `correctness_only` — binary 0/1 for answer match
2. `correctness_and_format` — 0.8 correct + 0.2 has `<think>` tags
3. `full` (default) — 0.7 correct + 0.2 format + 0.1 steps ≥ 3

Answer matching supports: exact string, floating-point tolerance (1e-6), substring containment, and `\boxed{}` extraction.

## Academic Course Materials

### Course Deadlines

- **Week 8**: Submit project topic
- **Week 10**: Submit paper draft
- **Week 17**: Present PPT (5 minutes)
- **Week 18** (deadline): Submit final project — printed paper + electronic copy to `huolina@hebtu.edu.cn`
- **File naming**: `name+studentID+major+topic`

### Paper Structure

Must include: abstract, methods/algorithm description, experiments/dataset description, runtime environment, results analysis and comparison, references. No plagiarism; must follow journal formatting.

### Key Reference Documents

| File | Content |
|------|---------|
| `智能科学设计五考核方式+封面.docx` | Course assessment criteria, schedule, and cover page template |
| `论文软件-ww.docx` | DilatedGait paper manuscript (Wang Wei, wangwei2021@hebtu.edu.cn) |
| `投稿-终版.doc` | Final submission version of the DilatedGait paper |
| `2024-软件学院--基于扩张重参数化和空洞卷积架构的步态识别方法_霍丽娜 (1).pdf` | Published DilatedGait reference (Huo Lina et al.) |
| `2020--河北工业科技--本科---基于CenterNet的小学生英文手写体区域检测_张朝晖.pdf` | CenterNet handwriting detection reference |

### DilatedGait Reference Paper Summary

- Task: gait recognition addressing ERF (effective receptive field) mismatch with human silhouette regions in outdoor scenes
- Key techniques: atrous convolution (expand receptive field without resolution loss), Dilated Reparameterization Module (multi-scale kernel fusion)
- Backbone: modified ResNet with residual atrous convolution blocks
- Datasets: Gait3D, GREW
- Environment: Python 3.8, CUDA 11.3, PyTorch 1.11.0, NVIDIA RTX 3090

## Instructor

霍丽娜 (Huo Lina) — huolina@hebtu.edu.cn
