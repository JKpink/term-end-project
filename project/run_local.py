"""
本地 2080Ti × 4 实验启动脚本
用法:
  python run_local.py --experiment all         # 一键运行全部
  python run_local.py --experiment sft         # 仅 SFT
  python run_local.py --experiment grpo        # 仅 GRPO
  python run_local.py --experiment eval        # 仅评估
  python run_local.py --experiment demo        # Gradio 演示
  python run_local.py -c my_config.yaml        # 使用自定义配置

超参数集中管理在 config.yaml，修改它即可调整所有参数，无需改源码。
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

# ============================================================
# 项目路径 & 配置加载
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_config(path: Path = None) -> dict:
    """加载 YAML 配置，若文件不存在则返回空"""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    if not path.exists():
        print(f"⚠️  配置文件不存在: {path}，使用内置默认值")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"📋 已加载配置: {path}")
    return cfg


# ============================================================
# 工具函数
# ============================================================
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


MODELS_DIR = PROJECT_ROOT / "models"


def _resolve_model(config: dict, model_size: str) -> str:
    """优先使用本地模型目录，不存在则用 HF ID"""
    hf_id = config.get("models", {}).get(model_size, f"Qwen/Qwen3-{model_size}")
    local_dir = MODELS_DIR / f"qwen3-{model_size.lower()}-Instruct"
    if local_dir.exists():
        return str(local_dir)
    return hf_id


def _sft_cmd(config: dict, model_size: str, output_name: str = None) -> list:
    """根据 SFT 配置节构建命令行参数"""
    sft = config.get("sft", {})
    model_id = _resolve_model(config, model_size)
    if output_name is None:
        output_name = f"sft_{model_size.lower()}"
    return [
        sys.executable, str(SRC_DIR / "train_sft.py"),
        "--model_name", model_id,
        "--output_dir", str(OUTPUT_DIR / output_name),
        "--num_epochs", str(sft.get("num_epochs", 2)),
        "--per_device_batch_size", str(sft.get("per_device_batch_size", 4)),
        "--gradient_accumulation_steps", str(sft.get("gradient_accumulation_steps", 4)),
        "--learning_rate", str(sft.get("learning_rate", 2e-4)),
        "--lora_r", str(sft.get("lora_r", 16)),
        "--max_seq_length", str(sft.get("max_seq_length", 1024)),
        "--max_train_samples", str(sft.get("max_train_samples", 4000)),
        "--logging_steps", str(sft.get("logging_steps", 5)),
        "--save_steps", str(sft.get("save_steps", 200)),
    ]


def _grpo_cmd(config: dict, model_size: str, name: str,
              reward_type: str = "full", num_generations: int = 8,
              kl_coef: float = 0.04, learning_rate: float = None,
              num_epochs: int = None, max_train_samples: int = None,
              sft_adapter: str = None) -> list:
    """根据 GRPO 配置节构建命令行参数，可单独覆盖关键字段"""
    grpo = config.get("grpo", {})
    model_id = _resolve_model(config, model_size)
    cmd = [
        sys.executable, str(SRC_DIR / "train_grpo.py"),
        "--model_name", model_id,
        "--output_dir", str(OUTPUT_DIR / f"grpo_{name}"),
        "--reward_type", reward_type,
        "--num_generations", str(num_generations),
        "--kl_coef", str(kl_coef),
        "--learning_rate", str(learning_rate if learning_rate is not None else grpo.get("learning_rate", 5e-5)),
        "--lora_r", str(grpo.get("lora_r", 16)),
        "--num_train_epochs", str(num_epochs if num_epochs is not None else grpo.get("num_epochs", 2)),
        "--per_device_batch_size", str(grpo.get("per_device_batch_size", 2)),
        "--gradient_accumulation_steps", str(grpo.get("gradient_accumulation_steps", 8)),
        "--max_prompt_length", str(grpo.get("max_prompt_length", 512)),
        "--max_completion_length", str(grpo.get("max_completion_length", 1024)),
        "--max_train_samples", str(max_train_samples if max_train_samples is not None else grpo.get("max_train_samples", 2000)),
        "--logging_steps", str(grpo.get("logging_steps", 5)),
        "--save_steps", str(grpo.get("save_steps", 100)),
    ]
    if sft_adapter and os.path.exists(sft_adapter):
        cmd.extend(["--sft_adapter_path", sft_adapter])
    return cmd


# ============================================================
# 单步执行
# ============================================================
def run_sft(config: dict, gpu_id: int = 0, model_size: str = "0.6B"):
    """运行 SFT 预热训练"""
    print(f"\n{'='*60}")
    print(f"SFT 预热 — GPU {gpu_id}, Model {model_size}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["HF_HUB_OFFLINE"] = "1"       # 强制离线，只用本地缓存

    cmd = _sft_cmd(config, model_size)
    subprocess.run(cmd, env=env, check=True)
    return str(OUTPUT_DIR / f"sft_{model_size.lower()}" / "final")


def run_grpo(config: dict, gpu_id: int, name: str, model_size: str = "0.6B",
             sft_adapter: str = None, reward_type: str = "full",
             num_generations: int = 8, kl_coef: float = 0.04,
             learning_rate: float = None, num_epochs: int = None,
             max_train_samples: int = None):
    """运行单个 GRPO 实验"""
    print(f"\n{'='*60}")
    print(f"GRPO: {name} — GPU {gpu_id}")
    print(f"  Model: {model_size}, Reward: {reward_type}, G={num_generations}, KL={kl_coef}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GRPO_REWARD_TYPE"] = reward_type
    env["HF_HUB_OFFLINE"] = "1"       # 强制离线

    cmd = _grpo_cmd(config, model_size, name, reward_type=reward_type,
                    num_generations=num_generations, kl_coef=kl_coef,
                    learning_rate=learning_rate, num_epochs=num_epochs,
                    max_train_samples=max_train_samples, sft_adapter=sft_adapter)
    subprocess.run(cmd, env=env, check=True)
    return str(OUTPUT_DIR / f"grpo_{name}")


# ============================================================
# 一键运行
# ============================================================
def run_all_experiments(config: dict):
    """运行全部实验 — 逐张卡串行执行"""
    gpus = get_gpu_list()
    print(f"Available GPUs:\n" + "\n".join(gpus))

    grpo_cfg = config.get("grpo", {})

    # ===== Phase 1: SFT 预热 (GPU 0) =====
    print("\n" + "=" * 70)
    print("PHASE 1: SFT Warmup (GPU 0)")
    print("=" * 70)
    sft_06b_path = run_sft(config, gpu_id=0, model_size="0.6B")

    # ===== Phase 2: 主实验 (GPU 0) =====
    print("\n" + "=" * 70)
    print("PHASE 2: Main Experiment (GPU 0)")
    print("=" * 70)
    run_grpo(config, gpu_id=0, name="main", model_size="0.6B",
             sft_adapter=sft_06b_path, reward_type="full",
             num_generations=grpo_cfg.get("num_generations", 8),
             kl_coef=grpo_cfg.get("kl_coef", 0.04))

    # ===== Phase 3: 消融实验 (GPU 0, 逐一串行) =====
    print("\n" + "=" * 70)
    print("PHASE 3: Ablation Experiments (GPU 0, serial)")
    print("=" * 70)

    all_ablations = (config.get("ablations", {}).get("phase2", []) +
                     config.get("ablations", {}).get("phase3", []))

    for i, abl in enumerate(all_ablations):
        model_size = abl.get("model", "0.6B")
        adapter = sft_06b_path if abl.get("sft", True) else None

        # 1.7B 需要先跑 SFT
        if model_size == "1.7B" and abl.get("sft", True):
            print(f"\n  --- 1.7B SFT warmup (GPU 0) ---")
            sft_17b_path = run_sft(config, gpu_id=0, model_size="1.7B")
            adapter = sft_17b_path

        print(f"\n  --- Ablation {i+1}/{len(all_ablations)}: {abl['name']} ---")
        run_grpo(config, gpu_id=0, name=abl["name"], model_size=model_size,
                 sft_adapter=adapter,
                 reward_type=abl.get("reward", "full"),
                 num_generations=abl.get("generations", 8),
                 kl_coef=abl.get("kl", 0.04),
                 num_epochs=abl.get("epoch", grpo_cfg.get("num_epochs", 2)),
                 max_train_samples=abl.get("samples", grpo_cfg.get("max_train_samples", 2000)))

    print("\n" + "=" * 70)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


def run_eval_only(config: dict):
    """仅运行评估"""
    eval_cfg = config.get("evaluate", {})
    shorthand = _resolve_model(config, "0.6B")  # 优先本地路径
    dataset = eval_cfg.get("dataset", "gsm8k")
    max_samples = str(eval_cfg.get("max_samples", 200))
    out_base = str(OUTPUT_DIR)
    verbose = ["--verbose"] if eval_cfg.get("verbose", False) else []

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"       # 强制离线

    print("Running evaluation on all baselines...")

    # Base model — no thinking
    subprocess.run([
        sys.executable, str(SRC_DIR / "evaluate.py"),
        "--model_name", shorthand, "--no_thinking",
        "--dataset", dataset, "--max_samples", max_samples,
        "--output_dir", f"{out_base}/eval_base_no_thinking",
    ] + verbose, env=env)

    # Base model — with thinking
    subprocess.run([
        sys.executable, str(SRC_DIR / "evaluate.py"),
        "--model_name", shorthand, "--enable_thinking",
        "--dataset", dataset, "--max_samples", max_samples,
        "--output_dir", f"{out_base}/eval_base_thinking",
    ] + verbose, env=env)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 2080Ti ×4 实验启动器")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "sft", "grpo", "eval", "demo"],
                        help="要运行的实验阶段 (默认 all)")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="YAML 配置文件路径 (默认 project/config.yaml)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="单步模式 (sft/grpo/eval/demo) 使用的 GPU 编号 (默认 0)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    if args.experiment == "all":
        run_all_experiments(cfg)
    elif args.experiment == "sft":
        run_sft(cfg, gpu_id=args.gpu)
    elif args.experiment == "grpo":
        sft_path = str(OUTPUT_DIR / "sft_0.6b" / "final")
        run_grpo(cfg, gpu_id=args.gpu, name="main", sft_adapter=sft_path)
    elif args.experiment == "eval":
        run_eval_only(cfg)
    elif args.experiment == "demo":
        subprocess.run([sys.executable, str(PROJECT_ROOT / "app" / "gradio_app.py")])
