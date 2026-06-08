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

| File | Role | Notes |
|------|------|-------|
| `project/src/reward.py` | 3 reward strategies, answer extraction, format checking | Exports `REWARD_FUNCTIONS` dict + helpers: `extract_final_answer()`, `is_correct()`, `check_thinking_format()`, `count_reasoning_steps()` |
| `project/src/train_sft.py` | SFT phase: QLoRA + GSM8K formatted with `<think>` tags | Uses `HfArgumentParser(SFTConfig)`. Formats data via `format_gsm8k()` / `format_scibench()` into `{"text": ...}` |
| `project/src/train_grpo.py` | GRPO phase: loads SFT adapter, configures `GRPOTrainer` | Imports reward helpers from `reward.py`. Reads `GRPO_REWARD_TYPE` env var at runtime in `grpo_reward_func()` |
| `project/src/evaluate.py` | Evaluation on GSM8K/SciBench | Model shorthands resolved via `MODEL_TO_HF` dict (only maps `qwen3.5-0.8b` and `qwen3.5-1.7b` — pass full HF path for other models) |
| `project/app/gradio_app.py` | Gradio web demo | **Requires manual setup:** set `ADAPTER_PATH` (line 17) to your trained adapter path, otherwise it runs the raw base model |
| `project/prepare.py` | Downloads models/datasets from ModelScope or HuggingFace | Supports `--target all/models/datasets/verify`, `--source modelscope/huggingface`, `--model` filter |

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

### Script Argument Reference

**`train_sft.py`** (uses `HfArgumentParser(SFTConfig)`):
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen3.5-0.8B` | HF model ID |
| `--dataset_name` | `openai/gsm8k` | Training dataset |
| `--dataset_config` | `main` | Dataset config/subset |
| `--output_dir` | `/kaggle/working/sft_output` | Output directory |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--num_epochs` | 2 | Training epochs |
| `--per_device_batch_size` | 4 | Per-GPU batch size |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation |
| `--learning_rate` | 2e-4 | Learning rate |
| `--max_seq_length` | 1024 | Max sequence length |
| `--max_train_samples` | None | Cap training samples (None=all) |
| `--use_flash_attention` | True | Use Flash Attention 2 |

**`train_grpo.py`** (uses `HfArgumentParser(GRPOTrainingConfig)`):
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen3.5-0.8B` | HF model ID |
| `--sft_adapter_path` | None | SFT LoRA adapter path (None = skip warmup) |
| `--output_dir` | `/kaggle/working/grpo_output` | Output directory |
| `--reward_type` | `full` | One of: `correctness_only`, `correctness_and_format`, `full` |
| `--use_qlora` | True | Use 4-bit QLoRA |
| `--lora_r` | 16 | LoRA rank |
| `--num_generations` | 8 | Group size G (completions per prompt) |
| `--max_completion_length` | 512 | Max tokens per generated completion |
| `--temperature` | 1.0 | Sampling temperature |
| `--kl_coef` | 0.04 | KL penalty coefficient (mapped to `GRPOConfig(beta=...)`) |
| `--num_train_epochs` | 2 | Training epochs |
| `--per_device_batch_size` | 2 | Per-GPU batch size |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation |
| `--learning_rate` | 5e-5 | Learning rate (GRPO: use lower than SFT) |
| `--max_prompt_length` | 512 | Max prompt tokens |
| `--max_train_samples` | None | Cap training samples |
| `--use_vllm` | False | vLLM acceleration (deprecated in newer TRL) |

**`evaluate.py`** (uses `argparse`):
| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `qwen3.5-0.8b` | Shorthand (resolved via `MODEL_TO_HF`) or full HF path |
| `--adapter_path` | None | LoRA/GRPO adapter path |
| `--dataset` | `gsm8k` | `gsm8k`, `scibench`, or `all` |
| `--enable_thinking` | True | Enable `<think>` reasoning mode |
| `--no_thinking` | False | Disable thinking (sets enable_thinking=False) |
| `--max_samples` | 200 | Number of evaluation samples |
| `--max_new_tokens` | 1024 | Max generation tokens |
| `--output_dir` | `./eval_output` | Output for CSVs + JSON metrics |
| `--verbose` | False | Print first 3 Q&A pairs |

**`prepare.py`** (uses `argparse`):
| Flag | Default | Description |
|------|---------|-------------|
| `--target` | `all` | `all`, `models`, `datasets`, or `verify` |
| `--source` | `modelscope` | `modelscope` (国内) or `huggingface` (uses hf-mirror.com) |
| `--model` | None | Specific model to download (e.g. `qwen3.5-0.8b`) |

### Architecture Patterns

**Padding side matters.** SFT training sets `padding_side="right"` (standard for causal LM training — labels align with right-padded tokens). GRPO training sets `padding_side="left"` (required for batch generation — prompts must left-align so generated completions share the same starting position).

**Env var as reward config channel.** `run_local.py` sets `GRPO_REWARD_TYPE` in the subprocess environment before launching `train_grpo.py`. Inside `train_grpo.py`, `grpo_reward_func()` reads `os.environ.get("GRPO_REWARD_TYPE", "full")` to select the reward strategy. The `--reward_type` CLI flag also sets this env var, so direct invocation works too. When modifying reward behavior, check both paths.

**KL coefficient mapping.** The CLI flag is `--kl_coef` (dataclass field), but `GRPOConfig` expects `beta`. The mapping is explicit: `GRPOConfig(beta=config.kl_coef, ...)`.

**SFTTrainer API — post-`c866efd`.** Uses the **newest TRL API**: `processing_class=tokenizer` + `formatting_func=lambda x: x["text"]`. The old parameters (`tokenizer=`, `packing=`, `dataset_text_field=`, `max_seq_length=`) are **deprecated and intentionally removed** — do not add them back. The data is pre-formatted into a `"text"` column before reaching the trainer.

**No gradient checkpointing in SFT.** Removed in commit `ccb1612` — batch sizes are small (1–4) so it provides no memory benefit, only slows training. Do not re-add.

**Parallelism approach.** `run_local.py` uses `threading.Thread` (not multiprocessing) with `CUDA_VISIBLE_DEVICES` set per-thread to assign different GPUs. GPU-bound threads may contend for the GIL on CPU work, but GPU ops release the GIL so this is acceptable.

**SciBench path duality.** Training loads SciBench from `xw27/scibench`. Evaluation tries `lupantech/SciBench` first, falling back to GSM8K — these are different HF repos. Prepare uses `xw27/scibench`.

### Gotchas

- **`gradio_app.py`** has `ADAPTER_PATH = None` at line 17 — you **must** change this to your trained adapter path before the demo will use a trained model. Without it, the raw Qwen3.5-0.8B model runs instead.
- **`wandb`** is in `requirements.txt` but training scripts only report to `tensorboard` by default. If you want wandb logging, run `wandb login` first and change `report_to=["tensorboard"]` to include `"wandb"`.
- **`evaluate.py --model_name`**: accepts shorthand (`qwen3.5-0.8b`, `qwen3.5-1.7b`) resolved via `MODEL_TO_HF`, or any full HuggingFace path. Qwen3.5-2B is **not** in the shorthand map — use `Qwen/Qwen3.5-2B-Instruct`.
- **`--use_vllm`** in `train_grpo.py`: deprecated in TRL ≥0.15, defaults to `False`. Wrapped in try/except so it won't crash.
- **bitsandbytes** 4-bit quantization needs a CUDA-compiled build. On Windows, use prebuilt wheels from https://github.com/jllllll/bitsandbytes-windows-webui if needed.
- **`.gitignore`** only covers `.DS_Store`. Directories like `outputs/`, `models/`, `__pycache__/`, `*.pyc` are not ignored — be careful not to commit large model files or training artifacts.

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
│   └── final_grpo/     #  saved via trainer.save_model()
└── eval_*/             # CSV results + JSON metrics per baseline
```

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
