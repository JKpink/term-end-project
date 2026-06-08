"""
Download models and datasets for the project.

Supports:
- Models from HuggingFace / ModelScope (国内加速)
- Food101 dataset from torchvision
"""

import os
import argparse
import sys


def download_from_huggingface(model_id: str, save_dir: str) -> bool:
    """Download a model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_id} from HuggingFace...")
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(save_dir, model_id.replace("/", "_")),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  ✓ {model_id} downloaded")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {model_id}: {e}")
        return False


def download_from_modelscope(model_id: str, save_dir: str) -> bool:
    """Download a model from ModelScope (国内镜像)."""
    try:
        from modelscope import snapshot_download
        print(f"Downloading {model_id} from ModelScope...")
        snapshot_download(
            model_id=model_id,
            cache_dir=os.path.join(save_dir, model_id.replace("/", "_")),
        )
        print(f"  ✓ {model_id} downloaded")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {model_id}: {e}")
        return False


def download_food101(save_dir: str) -> bool:
    """Download Food101 dataset from torchvision."""
    try:
        from torchvision.datasets import Food101
        print("Downloading Food101 dataset...")
        Food101(root=save_dir, split="train", download=True)
        Food101(root=save_dir, split="test", download=True)
        print(f"  ✓ Food101 downloaded to {save_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download Food101: {e}")
        return False


def verify_models(models_dir: str) -> dict:
    """Check if required models exist locally."""
    required = [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
    ]
    status = {}
    for model_id in required:
        local_path = os.path.join(models_dir, model_id.replace("/", "_"))
        exists = os.path.isdir(local_path) and os.listdir(local_path)
        status[model_id] = exists
    return status


def verify_data(data_dir: str) -> bool:
    """Check if Food101 data exists."""
    train_dir = os.path.join(data_dir, "food101", "food-101", "images")
    return os.path.isdir(train_dir) and len(os.listdir(train_dir)) > 0


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets")
    parser.add_argument(
        "--target",
        default="all",
        choices=["all", "models", "datasets", "verify"],
        help="What to download",
    )
    parser.add_argument(
        "--source",
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="Download source (modelscope for China)",
    )
    parser.add_argument(
        "--models_dir",
        default="./models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory to save datasets",
    )

    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.target in ("all", "models"):
        print("\n" + "=" * 50)
        print("Downloading Models")
        print("=" * 50)

        models = [
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
        ]
        download_fn = download_from_modelscope if args.source == "modelscope" else download_from_huggingface

        success = 0
        for model_id in models:
            if download_fn(model_id, args.models_dir):
                success += 1

        print(f"\nModels downloaded: {success}/{len(models)}")

    if args.target in ("all", "datasets"):
        print("\n" + "=" * 50)
        print("Downloading Datasets")
        print("=" * 50)

        if download_food101(args.data_dir):
            print("Food101 ready!")
        else:
            print("Food101 download failed. Try manually from torchvision.")

    if args.target == "verify":
        print("\n" + "=" * 50)
        print("Verifying Installation")
        print("=" * 50)

        print("\nModels:")
        model_status = verify_models(args.models_dir)
        for model_id, exists in model_status.items():
            status_icon = "✓" if exists else "✗ (will download on first use)"
            print(f"  {status_icon} {model_id}")

        print("\nDatasets:")
        data_ok = verify_data(args.data_dir)
        if data_ok:
            print("  ✓ Food101 dataset found")
        else:
            print("  ✗ Food101 not found - run 'python prepare.py --target datasets'")

    print("\nDone!")


if __name__ == "__main__":
    main()
