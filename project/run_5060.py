"""
RTX 5060 8GB 本地一键运行脚本
用法: python run_5060.py [step]
  step 1: 准备数据+模型 (prepare)
  step 2: SFT 预热训练
  step 3: GRPO 训练
  step 4: 消融实验
  step 5: 评估
  step 6: 启动 Gradio 演示
  all   : 按顺序执行全部
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

# RTX 5060 8GB 专用配置 (单卡，显存友好)
CONFIG = {
    "model_08b": "Qwen/Qwen3-0.6B",
    "model_2b":  "Qwen/Qwen3-1.7B",
    "sft": {
        "num_epochs": 2,
        "batch_size": 2,           # 5060 8GB: 保守设 2
        "grad_accum": 8,            # 有效 batch = 2*8 = 16
        "lr": 2e-4,
        "max_seq_length": 768,      # 缩短序列省显存
        "lora_r": 8,                # 小 rank 省显存
        "max_samples": 2000,
    },
    "grpo": {
        "num_epochs": 2,
        "batch_size": 1,            # GRPO 每个 prompt 生成多个回答，batch=1 稳
        "grad_accum": 8,
        "lr": 5e-5,
        "num_generations": 4,       # Group Size (5060 设小点)
        "max_prompt_length": 384,
        "max_completion_length": 512,
        "kl_coef": 0.04,
        "lora_r": 8,
        "max_samples": 1000,
    },
}


def step1_prepare():
    """Step 1: 下载模型和数据集"""
    print("\n" + "="*60)
    print("STEP 1: 准备数据与模型")
    print("="*60)
    subprocess.run([
        sys.executable, str(PROJECT_ROOT / "prepare.py"),
        "--target", "all",
        "--source", "modelscope",
        "--model", "qwen3-0.6b",  # 先只下 0.6B
    ], check=True)


def step2_sft():
    """Step 2: SFT 预热 (基线③)"""
    print("\n" + "="*60)
    print("STEP 2: SFT 预热训练")
    print(f"  GPU: RTX 5060 8GB")
    print(f"  Model: Qwen3-0.6B QLoRA")
    print("="*60)

    cfg = CONFIG["sft"]
    cmd = [
        sys.executable, str(SRC_DIR / "train_sft.py"),
        "--model_name", CONFIG["model_08b"],
        "--output_dir", str(OUTPUT_DIR / "sft_5060"),
        "--num_epochs", str(cfg["num_epochs"]),
        "--per_device_batch_size", str(cfg["batch_size"]),
        "--gradient_accumulation_steps", str(cfg["grad_accum"]),
        "--learning_rate", str(cfg["lr"]),
        "--lora_r", str(cfg["lora_r"]),
        "--max_seq_length", str(cfg["max_seq_length"]),
        "--max_train_samples", str(cfg["max_samples"]),
        "--logging_steps", "5",
        "--save_steps", "200",
    ]
    subprocess.run(cmd, check=True)
    print("✅ SFT done")
    return str(OUTPUT_DIR / "sft_5060" / "final")


def step3_grpo(sft_adapter=None):
    """Step 3: GRPO 主实验 (基线④)"""
    print("\n" + "="*60)
    print("STEP 3: GRPO 训练 (主实验)")
    print(f"  GPU: RTX 5060 8GB")
    print(f"  Group Size: {CONFIG['grpo']['num_generations']}")
    print("="*60)

    cfg = CONFIG["grpo"]
    cmd = [
        sys.executable, str(SRC_DIR / "train_grpo.py"),
        "--model_name", CONFIG["model_08b"],
        "--output_dir", str(OUTPUT_DIR / "grpo_5060_main"),
        "--reward_type", "full",
        "--num_generations", str(cfg["num_generations"]),
        "--kl_coef", str(cfg["kl_coef"]),
        "--learning_rate", str(cfg["lr"]),
        "--lora_r", str(cfg["lora_r"]),
        "--num_train_epochs", str(cfg["num_epochs"]),
        "--per_device_batch_size", str(cfg["batch_size"]),
        "--gradient_accumulation_steps", str(cfg["grad_accum"]),
        "--max_prompt_length", str(cfg["max_prompt_length"]),
        "--max_completion_length", str(cfg["max_completion_length"]),
        "--max_train_samples", str(cfg["max_samples"]),
        "--logging_steps", "5",
        "--save_steps", "100",
    ]
    if sft_adapter and os.path.exists(sft_adapter):
        cmd.extend(["--sft_adapter_path", sft_adapter])

    subprocess.run(cmd, check=True)
    print("✅ GRPO main done")
    return str(OUTPUT_DIR / "grpo_5060_main")


def step4_ablation(sft_adapter=None):
    """Step 4: 消融实验 (逐个跑，保存显存)"""
    print("\n" + "="*60)
    print("STEP 4: 消融实验 (串行)")
    print("="*60)

    cfg = CONFIG["grpo"]
    ablations = [
        # (name, reward_type, sft, group_size, kl)
        ("abl_no_sft",     "full",             None,    4, 0.04),
        ("abl_correctness","correctness_only",  sft_adapter, 4, 0.04),
        ("abl_g8",         "full",             sft_adapter, 8, 0.04),
        ("abl_kl001",      "full",             sft_adapter, 4, 0.01),
        ("abl_kl01",       "full",             sft_adapter, 4, 0.10),
    ]

    for name, reward, sft, gsize, kl in ablations:
        print(f"\n  --- Ablation: {name} ---")
        cmd = [
            sys.executable, str(SRC_DIR / "train_grpo.py"),
            "--model_name", CONFIG["model_08b"],
            "--output_dir", str(OUTPUT_DIR / f"grpo_5060_{name}"),
            "--reward_type", reward,
            "--num_generations", str(gsize),
            "--kl_coef", str(kl),
            "--learning_rate", str(cfg["lr"]),
            "--lora_r", str(cfg["lora_r"]),
            "--num_train_epochs", "1",         # 消融只跑1轮
            "--per_device_batch_size", "1",
            "--gradient_accumulation_steps", "8",
            "--max_prompt_length", str(cfg["max_prompt_length"]),
            "--max_completion_length", str(cfg["max_completion_length"]),
            "--max_train_samples", "500",       # 消融用500样本
            "--logging_steps", "5",
            "--save_steps", "100",
        ]
        if sft and os.path.exists(sft):
            cmd.extend(["--sft_adapter_path", sft])
        subprocess.run(cmd, check=True)

    print("✅ All ablations done")


def step5_evaluate():
    """Step 5: 评估所有模型"""
    print("\n" + "="*60)
    print("STEP 5: 评估")
    print("="*60)

    evals = [
        # (label, model, adapter, thinking)
        ("base_no_think", "qwen3-0.6b", None, False),
        ("base_think",    "qwen3-0.6b", None, True),
        ("sft",           "qwen3-0.6b", str(OUTPUT_DIR / "sft_5060" / "final"), True),
        ("grpo_main",     "qwen3-0.6b", str(OUTPUT_DIR / "grpo_5060_main" / "final_grpo" / "final_grpo"), True),
    ]

    results = {}
    for label, model, adapter, thinking in evals:
        if adapter and not os.path.exists(adapter):
            print(f"  ⚠️  Skip {label}: adapter not found at {adapter}")
            continue

        cmd = [
            sys.executable, str(SRC_DIR / "evaluate.py"),
            "--model_name", model,
            "--dataset", "gsm8k",
            "--max_samples", "100",
            "--output_dir", str(OUTPUT_DIR / f"eval_{label}"),
        ]
        if adapter:
            cmd.extend(["--adapter_path", adapter])
        if not thinking:
            cmd.append("--no_thinking")

        print(f"\n  Evaluating: {label}")
        subprocess.run(cmd, check=True)

    print("✅ Evaluation done")


def step6_demo():
    """Step 6: 启动 Gradio"""
    print("\n" + "="*60)
    print("STEP 6: 启动 Gradio 演示")
    print("="*60)
    subprocess.run([sys.executable, str(PROJECT_ROOT / "app" / "gradio_app.py")])


def main():
    parser = argparse.ArgumentParser(description="RTX 5060 8GB 一键运行")
    parser.add_argument("step", type=str, default="all", nargs="?",
                       choices=["1", "2", "3", "4", "5", "6", "all", "prepare", "sft", "grpo", "ablation", "eval", "demo"])
    args = parser.parse_args()

    step_map = {
        "1": step1_prepare, "prepare": step1_prepare,
        "2": step2_sft,     "sft": step2_sft,
        "3": step3_grpo,    "grpo": step3_grpo,
        "4": step4_ablation,"ablation": step4_ablation,
        "5": step5_evaluate, "eval": step5_evaluate,
        "6": step6_demo,    "demo": step6_demo,
    }

    if args.step == "all":
        step1_prepare()

        print("\n⚠️  内存提示: 每个训练步骤可能需要 2-6GB 显存")
        print("   确保关闭其他 GPU 程序\n")

        sft_path = step2_sft()
        step3_grpo(sft_path)
        step4_ablation(sft_path)
        step5_evaluate()

        print("\n" + "="*60)
        print("🎉 全部完成!")
        print(f"  模型输出: {OUTPUT_DIR}")
        print(f"  启动演示: python run_5060.py demo")
        print("="*60)
    else:
        step_map[args.step]()


if __name__ == "__main__":
    main()
