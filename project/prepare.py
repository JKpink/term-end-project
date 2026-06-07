"""
数据与模型准备脚本 (支持 ModelScope 国内加速)
用法:
  python prepare.py --target all --source modelscope    # 国内推荐
  python prepare.py --target all --source huggingface   # HF 直连
  python prepare.py --target verify                     # 验证现有数据
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# ModelScope 资源 ID（国内访问快）
# ============================================================
MODELSCOPE_MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B-Instruct",
    "qwen3.5-2b":  "Qwen/Qwen3.5-2B-Instruct",
}

MODELSCOPE_DATASETS = {
    "gsm8k": "swift/gsm8k",           # modelscope 镜像
    "scibench": "xw27/scibench",      # 仅 HF 有，需设 HF_ENDPOINT 镜像
}

HF_DATASETS = {
    "gsm8k": ("openai/gsm8k", "main"),
    "scibench": ("xw27/scibench", None),
}


def install_modelscope():
    """安装 ModelScope SDK"""
    print("Installing modelscope...")
    subprocess.run([sys.executable, "-m", "pip", "install", "modelscope", "-q"], check=True)


def download_model_modelscope(model_id: str, local_dir: Path):
    """从 ModelScope 下载模型"""
    install_modelscope()
    from modelscope import snapshot_download
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[ModelScope] Downloading: {model_id}")
    print(f"To: {local_dir}")
    print(f"{'='*60}")
    snapshot_download(model_id=model_id, local_dir=str(local_dir))
    print(f"✅ Downloaded to {local_dir}")


def download_model_hf(model_id: str, local_dir: Path):
    """从 HuggingFace 下载模型 (使用 hf-mirror.com 镜像)"""
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[HuggingFace] Downloading: {model_id}")
    print(f"To: {local_dir}")
    print(f"{'='*60}")

    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"], check=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅ Downloaded to {local_dir}")


def download_dataset_modelscope(ds_id: str):
    """从 ModelScope 加载并缓存数据集"""
    install_modelscope()
    from modelscope import MsDataset
    print(f"\n[ModelScope] Loading dataset: {ds_id}")
    ds = MsDataset.load(ds_id)
    print(f"✅ Dataset loaded: {len(ds)} samples")
    return ds


def download_dataset_hf(ds_name: str, ds_config: str = None):
    """从 HuggingFace 加载并缓存数据集"""
    from datasets import load_dataset
    print(f"\n[HF] Loading dataset: {ds_name} (config={ds_config})")
    if ds_config:
        ds = load_dataset(ds_name, ds_config)
    else:
        ds = load_dataset(ds_name)
    print(f"✅ Dataset loaded")
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data)} samples")
    return ds


def verify_local():
    """验证本地模型和数据集可访问性"""
    print("\n" + "="*60)
    print("Verifying...")
    print("="*60)

    # 检查模型
    for name in ["qwen3.5-0.8b", "qwen3.5-2b"]:
        local_path = MODELS_DIR / f"{name}-Instruct"
        if local_path.exists():
            size_gb = sum(f.stat().st_size for f in local_path.rglob("*")) / 1024**3
            print(f"✅ {name}: {size_gb:.1f} GB at {local_path}")
        else:
            print(f"⚠️  {name}: not downloaded (transformers will auto-download from HF/ModelScope)")

    # 测试数据集加载 (通过 HF mirror)
    print("\nTesting HF dataset access (via hf-mirror.com)...")
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        print(f"✅ GSM8K: {len(ds)} train samples (via mirror)")
    except Exception as e:
        print(f"⚠️  GSM8K HF: {str(e)[:80]}")

    try:
        from datasets import load_dataset
        ds = load_dataset("xw27/scibench")
        first_split = list(ds.values())[0]
        print(f"✅ SciBench: {len(first_split)} samples (via mirror)")
    except Exception as e:
        print(f"⚠️  SciBench HF: {str(e)[:80]}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data and models")
    parser.add_argument("--target", type=str, default="all",
                       choices=["all", "models", "datasets", "verify"])
    parser.add_argument("--source", type=str, default="modelscope",
                       choices=["modelscope", "huggingface"],
                       help="下载源 (modelscope 国内快, huggingface 需镜像)")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    # 国内 HF 访问设置镜像
    if args.source == "huggingface":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF mirror: hf-mirror.com")

    if args.target in ["all", "models"]:
        models_to_dl = MODELSCOPE_MODELS
        if args.model:
            models_to_dl = {args.model: MODELSCOPE_MODELS[args.model]}

        for name, model_id in models_to_dl.items():
            local_dir = MODELS_DIR / f"{name}-Instruct"
            if args.source == "modelscope":
                download_model_modelscope(model_id, local_dir)
            else:
                download_model_hf(model_id, local_dir)

    if args.target in ["all", "datasets"]:
        if args.source == "modelscope":
            for name, ds_id in MODELSCOPE_DATASETS.items():
                try:
                    download_dataset_modelscope(ds_id)
                    print(f"  -> {name}: cached via ModelScope")
                except Exception as e:
                    print(f"  ⚠️ {name} ModelScope failed ({e}), trying HF mirror...")
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    info = HF_DATASETS[name]
                    download_dataset_hf(info[0], info[1])
        else:
            for name, (ds_name, ds_config) in HF_DATASETS.items():
                download_dataset_hf(ds_name, ds_config)

    if args.target == "verify":
        verify_local()

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
