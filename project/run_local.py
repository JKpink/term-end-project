"""
本地 2080Ti × 4 实验启动脚本 (Windows)
用法: python run_local.py [--experiment all|sft|grpo|eval|demo]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_gpu_list():
    """获取可用 GPU 列表"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        return result.stdout.strip().split("\n")
    except Exception:
        return []


def run_sft(gpu_id: int = 0, model_size: str = "0.8B"):
    """运行 SFT 预热训练"""
    print(f"\n{'='*60}")
    print(f"Step 1: SFT 预热 — GPU {gpu_id}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, str(SRC_DIR / "train_sft.py"),
        "--model_name", "Qwen/Qwen3.5-0.8B-Instruct" if "0.8" in model_size else "Qwen/Qwen3.5-2B-Instruct",
        "--output_dir", str(OUTPUT_DIR / f"sft_{model_size.lower()}"),
        "--num_epochs", "2",
        "--per_device_batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-4",
        "--lora_r", "16",
        "--max_seq_length", "1024",
        "--max_train_samples", "4000",
        "--logging_steps", "5",
        "--save_steps", "200",
    ]

    subprocess.run(cmd, env=env, check=True)
    return str(OUTPUT_DIR / f"sft_{model_size.lower()}" / "final")


def run_grpo_experiment(
    gpu_id: int,
    name: str,
    model_size: str = "0.8B",
    sft_adapter: str = None,
    reward_type: str = "full",
    num_generations: int = 8,
    kl_coef: float = 0.04,
    learning_rate: float = 5e-5,
    max_train_samples: int = 2000,
):
    """运行单个 GRPO 实验"""
    print(f"\n{'='*60}")
    print(f"GRPO Experiment: {name} — GPU {gpu_id}")
    print(f"  Model: {model_size}, Reward: {reward_type}")
    print(f"  Group Size: {num_generations}, KL: {kl_coef}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GRPO_REWARD_TYPE"] = reward_type

    model_name = "Qwen/Qwen3.5-0.8B-Instruct" if "0.8" in model_size else "Qwen/Qwen3.5-2B-Instruct"
    output_dir = str(OUTPUT_DIR / f"grpo_{name}")

    cmd = [
        sys.executable, str(SRC_DIR / "train_grpo.py"),
        "--model_name", model_name,
        "--output_dir", output_dir,
        "--reward_type", reward_type,
        "--num_generations", str(num_generations),
        "--kl_coef", str(kl_coef),
        "--learning_rate", str(learning_rate),
        "--lora_r", "16",
        "--num_train_epochs", "2",
        "--per_device_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--max_prompt_length", "512",
        "--max_completion_length", "1024",
        "--max_train_samples", str(max_train_samples),
        "--logging_steps", "5",
        "--save_steps", "100",
    ]

    if sft_adapter and os.path.exists(sft_adapter):
        cmd.extend(["--sft_adapter_path", sft_adapter])

    subprocess.run(cmd, env=env, check=True)
    return output_dir


def run_all_experiments():
    """运行全部实验 — 利用 4 卡并行"""
    gpus = get_gpu_list()
    print(f"Available GPUs:\n" + "\n".join(gpus))

    # ===== Step 1: SFT 预热 (GPU 0 & 1 各跑一个模型) =====
    print("\n" + "="*70)
    print("PHASE 1: SFT Warmup")
    print("="*70)

    # 0.8B SFT on GPU 0
    sft_08b_path = run_sft(gpu_id=0, model_size="0.8B")

    # ===== Step 2: GRPO 主实验 + 消融 (4 GPUs 并行) =====
    print("\n" + "="*70)
    print("PHASE 2: GRPO Experiments (并行)")
    print("="*70)

    import threading

    threads = []

    # GPU 0: 主实验 — 完整 GRPO
    t0 = threading.Thread(target=run_grpo_experiment, args=(0, "main", "0.8B", sft_08b_path, "full", 8, 0.04))
    threads.append(t0)

    # GPU 1: 消融 A — 无 SFT 预热
    t1 = threading.Thread(target=run_grpo_experiment, args=(1, "abl_no_sft", "0.8B", None, "full", 8, 0.04))
    threads.append(t1)

    # GPU 2: 消融 B — 仅正确性奖励
    t2 = threading.Thread(target=run_grpo_experiment, args=(2, "abl_correctness_only", "0.8B", sft_08b_path, "correctness_only", 8, 0.04))
    threads.append(t2)

    # GPU 3: 消融 C — Group Size 变化 (G=4)
    t3 = threading.Thread(target=run_grpo_experiment, args=(3, "abl_g4", "0.8B", sft_08b_path, "full", 4, 0.04))
    threads.append(t3)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # ===== Step 3: 额外消融 =====
    print("\n" + "="*70)
    print("PHASE 3: Additional Ablation Experiments")
    print("="*70)

    threads = []

    # GPU 0: 消融 C — G=16
    t4 = threading.Thread(target=run_grpo_experiment, args=(0, "abl_g16", "0.8B", sft_08b_path, "full", 16, 0.04))
    threads.append(t4)

    # GPU 1: 消融 D — KL=0.01
    t5 = threading.Thread(target=run_grpo_experiment, args=(1, "abl_kl001", "0.8B", sft_08b_path, "full", 8, 0.01))
    threads.append(t5)

    # GPU 2: 消融 D — KL=0.1
    t6 = threading.Thread(target=run_grpo_experiment, args=(2, "abl_kl01", "0.8B", sft_08b_path, "full", 8, 0.1))
    threads.append(t6)

    # GPU 3: Baseline ⑤ — 2B + GRPO
    sft_2b_path = run_sft(gpu_id=3, model_size="2B")
    t7 = threading.Thread(target=run_grpo_experiment, args=(3, "main_2b", "2B", sft_2b_path, "full", 8, 0.04))
    threads.append(t7)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("="*70)


def run_eval_only():
    """仅运行评估"""
    print("Running evaluation on all baselines...")
    # GPU 0: base model no thinking
    subprocess.run([
        sys.executable, str(SRC_DIR / "evaluate.py"),
        "--model_name", "qwen3.5-0.8b",
        "--no_thinking",
        "--dataset", "gsm8k",
        "--max_samples", "200",
        "--output_dir", str(OUTPUT_DIR / "eval_base_no_thinking"),
        "--verbose",
    ])

    # GPU 0: base model with thinking
    subprocess.run([
        sys.executable, str(SRC_DIR / "evaluate.py"),
        "--model_name", "qwen3.5-0.8b",
        "--enable_thinking",
        "--dataset", "gsm8k",
        "--max_samples", "200",
        "--output_dir", str(OUTPUT_DIR / "eval_base_thinking"),
        "--verbose",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local 2080Ti experiment launcher")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "sft", "grpo", "eval", "demo"])
    args = parser.parse_args()

    if args.experiment == "all":
        run_all_experiments()
    elif args.experiment == "sft":
        run_sft(gpu_id=0)
    elif args.experiment == "grpo":
        sft_path = str(OUTPUT_DIR / "sft_0.8b" / "final")
        run_grpo_experiment(gpu_id=0, name="main", sft_adapter=sft_path)
    elif args.experiment == "eval":
        run_eval_only()
    elif args.experiment == "demo":
        subprocess.run([sys.executable, str(PROJECT_ROOT / "app" / "gradio_app.py")])
