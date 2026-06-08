"""Utility functions: seed setting, device info, result formatting."""

import os
import random
import json
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_info() -> dict[str, Any]:
    """Get GPU device information."""
    info = {}
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    else:
        info["cuda_available"] = False
    return info


def format_results_table(
    results: list[dict[str, Any]],
    model_names: list[str],
) -> str:
    """Format evaluation results as a markdown table.

    Args:
        results: List of result dicts with keys: model_name, top1_acc, top5_acc, avg_desc_score
        model_names: Ordered list of model names for display

    Returns:
        Markdown formatted table string.
    """
    header = "| 模型 | Top-1 Acc | Top-5 Acc | 描述质量 | 推理速度 (img/s) | 显存 (GB) |"
    sep = "|------|:--------:|:--------:|:--------:|:---------------:|:---------:|"

    rows = []
    for r in results:
        name = r.get("model_name", "Unknown")
        top1 = r.get("top1_acc", 0)
        top5 = r.get("top5_acc", 0)
        desc = r.get("avg_desc_score", 0)
        speed = r.get("inference_speed", 0)
        vram = r.get("vram_gb", 0)
        rows.append(f"| {name} | {top1:.1%} | {top5:.1%} | {desc:.2f} | {speed:.1f} | {vram:.1f} |")

    return "\n".join([header, sep] + rows)


def save_results_json(results: list[dict], path: str) -> None:
    """Save results to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")
