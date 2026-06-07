"""
评估脚本 — 在 SciBench/GSM8K 测试集上评估模型推理能力
支持评估: 基座模型 zero-shot / thinking on / SFT 微调 / GRPO 训练
"""

import os
import sys
import json
import torch
import re
import argparse
from typing import Optional, List, Dict
from tqdm import tqdm
import pandas as pd

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import extract_final_answer, check_thinking_format, count_reasoning_steps, is_correct


MODEL_TO_HF = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-1.7b": "Qwen/Qwen3.5-1.7B",
}


def load_model(model_name: str, adapter_path: Optional[str] = None):
    """加载模型和 tokenizer"""
    hf_name = MODEL_TO_HF.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {hf_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_prompt(question: str, enable_thinking: bool = True, system_prompt: str = "") -> str:
    """构建推理 prompt"""
    if not system_prompt:
        system_prompt = "你是一个理科题目求解助手。请一步步思考，将推理过程写在 <think> 和 </think> 之间，最后给出答案。"

    if enable_thinking:
        sys_part = f"<|im_start|>system\n{system_prompt}<|im_end|>"
    else:
        sys_part = f"<|im_start|>system\n你是一个理科题目求解助手。请直接给出答案，不需要展示思考过程。<|im_end|>"

    return f"{sys_part}\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_gsm8k(model, tokenizer, args) -> Dict:
    """在 GSM8K 测试集上评估"""
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return run_evaluation(model, tokenizer, dataset, args, "gsm8k")


def evaluate_scibench(model, tokenizer, args) -> Dict:
    """在 SciBench 上评估"""
    try:
        dataset = load_dataset("lupantech/SciBench", split="test")
    except Exception:
        print("SciBench not available, falling back to GSM8K test")
        return evaluate_gsm8k(model, tokenizer, args)
    return run_evaluation(model, tokenizer, dataset, args, "scibench")


def run_evaluation(model, tokenizer, dataset, args, dataset_name: str) -> Dict:
    """运行评估"""
    results = []
    correct = 0
    has_thinking = 0
    total_steps = 0

    max_samples = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)

    for i in tqdm(range(max_samples), desc=f"Evaluating {dataset_name}"):
        example = dataset[i]

        # 提取问题和答案
        if "question" in example:
            question = example["question"]
        elif "problem_text" in example:
            question = example["problem_text"]
        else:
            continue

        if "answer" in example:
            gt = example["answer"]
            if "####" in gt:
                gt = gt.split("####")[1].strip()
        elif "solution" in example:
            gt = example["solution"]
        elif "correct" in example:
            gt = str(example["correct"])
        else:
            gt = ""

        # 生成
        prompt = build_prompt(question, enable_thinking=args.enable_thinking)
        response = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

        # 提取答案并判断
        pred = extract_final_answer(response)
        correct_flag = is_correct(pred, gt)
        thinking_flag = check_thinking_format(response)

        if correct_flag:
            correct += 1
        if thinking_flag:
            has_thinking += 1
            total_steps += count_reasoning_steps(response)

        results.append({
            "index": i,
            "question": question[:200],
            "ground_truth": gt,
            "predicted_answer": pred,
            "response": response[:500],
            "correct": correct_flag,
            "has_thinking": thinking_flag,
        })

        if args.verbose and i < 3:
            print(f"\n{'='*60}")
            print(f"Q: {question[:200]}")
            print(f"GT: {gt}")
            print(f"Pred: {pred}")
            print(f"Response: {response[:300]}")
            print(f"Correct: {correct_flag}, Thinking: {thinking_flag}")

    # 统计
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0
    thinking_rate = has_thinking / total * 100 if total > 0 else 0
    avg_steps = total_steps / has_thinking if has_thinking > 0 else 0

    metrics = {
        "dataset": dataset_name,
        "model": args.model_name,
        "adapter": args.adapter_path,
        "enable_thinking": args.enable_thinking,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "thinking_rate": round(thinking_rate, 2),
        "avg_reasoning_steps": round(avg_steps, 2),
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, f"{dataset_name}_results.csv"), index=False)
    with open(os.path.join(args.output_dir, f"{dataset_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model reasoning")
    parser.add_argument("--model_name", type=str, default="qwen3.5-0.8b")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "scibench", "all"])
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False

    model, tokenizer = load_model(args.model_name, args.adapter_path)

    if args.dataset == "all":
        metrics_gsm8k = evaluate_gsm8k(model, tokenizer, args)
        metrics_sci = evaluate_scibench(model, tokenizer, args)
        all_metrics = {"gsm8k": metrics_gsm8k, "scibench": metrics_sci}
        with open(os.path.join(args.output_dir, "all_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    elif args.dataset == "gsm8k":
        evaluate_gsm8k(model, tokenizer, args)
    else:
        evaluate_scibench(model, tokenizer, args)


if __name__ == "__main__":
    main()
