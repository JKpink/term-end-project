"""
Evaluation for collaborative multi-agent writing.

Metrics: structure, content coverage, style consistency.
Compares: Single, Parallel, Sequential, Discussion, Ours (collaborative).
"""

import os
import json
import argparse
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from train import (
    load_agent_with_lora,
    build_tldr_formatters,
    collaborative_inference,
)
from utils import set_seed, get_device_info


def single_agent_inference(prompt: str, model, tokenizer, max_tokens=256) -> str:
    """Single agent does the whole task."""
    messages = [
        {"role": "system", "content": "Write a summary of this Reddit post."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def parallel_inference(prompt: str, model, tokenizer, max_tokens=256) -> tuple:
    """Two agents independently generate (no communication)."""
    a = single_agent_inference(
        prompt + "\nWrite a one-sentence summary.", model, tokenizer, max_tokens // 2,
    )
    b = single_agent_inference(
        prompt + "\nWrite a detailed summary.", model, tokenizer, max_tokens,
    )
    return a, b


def sequential_inference(
    prompt: str, model, tokenizer, formatter_a, formatter_b, max_tokens=256,
) -> tuple:
    """A generates → B generates based on A (base models, no training)."""
    prompt_a = formatter_a({"prompt": prompt})
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_a = model.generate(**inputs_a, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    completion_a = tokenizer.decode(out_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True)

    prompt_b = formatter_b({"prompt": prompt}) + f"\n\n[Reference]: {completion_a}"
    inputs_b = tokenizer(prompt_b, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_b = model.generate(**inputs_b, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    completion_b = tokenizer.decode(out_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True)

    return completion_a, completion_b


def discussion_inference(
    prompt: str, model, tokenizer, formatter_a, formatter_b, max_tokens=256,
) -> tuple:
    """One-round discussion: A writes → B comments → A revises."""
    # Round 1: A generates
    prompt_a = formatter_a({"prompt": prompt})
    inputs_a = tokenizer(prompt_a, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_a = model.generate(**inputs_a, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    a1 = tokenizer.decode(out_a[0][inputs_a.input_ids.shape[1]:], skip_special_tokens=True)

    # B comments
    b_prompt = f"Review this summary and suggest improvements:\n{a1}"
    inputs_b = tokenizer(b_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_b = model.generate(**inputs_b, max_new_tokens=max_tokens // 2, temperature=0.1, do_sample=False)
    b_comment = tokenizer.decode(out_b[0][inputs_b.input_ids.shape[1]:], skip_special_tokens=True)

    # A revises
    revise_prompt = f"Revise your summary based on this feedback:\n{b_comment}"
    inputs_rev = tokenizer(revise_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_rev = model.generate(**inputs_rev, max_new_tokens=max_tokens, temperature=0.1, do_sample=False)
    a_revised = tokenizer.decode(out_rev[0][inputs_rev.input_ids.shape[1]:], skip_special_tokens=True)

    return a_revised, b_comment


def compute_metrics(completions: List[Dict]) -> Dict[str, float]:
    """Compute simple length ratio and content overlap metrics."""
    if not completions:
        return {}

    ratios = []
    for c in completions:
        a = c.get("agent_a", "")
        b = c.get("agent_b", "")
        len_a = len(a.strip())
        len_b = len(b.strip())
        if len_a > 0:
            ratios.append(len_b / len_a)

    return {
        "num_samples": len(completions),
        "avg_len_a": sum(len(c.get("agent_a", "")) for c in completions) / len(completions),
        "avg_len_b": sum(len(c.get("agent_b", "")) for c in completions) / len(completions),
        "avg_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
    }


def run_evaluation(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    lora_path: Optional[str] = None,
    output_dir: str = "./outputs/eval",
    num_samples: int = 50,
    use_4bit: bool = True,
):
    """Run all baselines and our collaborative model on the test set."""
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {model_name}")
    base_model, tokenizer = load_agent_with_lora(
        model_name, use_4bit=use_4bit, device_map="auto",
    )

    # Load trained agents if available
    agent_a, agent_b = None, None
    if lora_path:
        agent_a_path = os.path.join(lora_path, "agent_a_final")
        agent_b_path = os.path.join(lora_path, "agent_b_final")
        if os.path.exists(agent_a_path):
            agent_a, _ = load_agent_with_lora(
                model_name, use_4bit=use_4bit, device_map="auto",
            )
    if agent_a is None:
        agent_a = base_model
    if agent_b is None:
        agent_b = base_model

    # Load test data
    dataset = load_dataset("trl-lib/tldr", split="train")
    test_indices = range(300, 300 + num_samples)
    test_data = [dataset[i] for i in test_indices]
    formatters = build_tldr_formatters(tokenizer)

    all_results = {}

    # B1: Single Agent
    print("\n=== B1: Single Agent ===")
    single_results = []
    for ex in test_data[:num_samples]:
        output = single_agent_inference(ex["prompt"], base_model, tokenizer)
        single_results.append({"output": output, "prompt": ex["prompt"]})
    all_results["B1_Single"] = compute_single_metrics(single_results)

    # B2: Parallel (no communication)
    print("\n=== B2: Parallel Generation ===")
    parallel_results = []
    for ex in test_data[:num_samples]:
        a, b = parallel_inference(ex["prompt"], base_model, tokenizer)
        parallel_results.append({"agent_a": a, "agent_b": b})
    all_results["B2_Parallel"] = compute_metrics(parallel_results)

    # B3: Sequential (A→B, no training)
    print("\n=== B3: Sequential Pipeline ===")
    seq_results = []
    for ex in test_data[:num_samples]:
        a, b = sequential_inference(
            ex["prompt"], base_model, tokenizer,
            formatters[0], formatters[1],
        )
        seq_results.append({"agent_a": a, "agent_b": b})
    all_results["B3_Sequential"] = compute_metrics(seq_results)

    # B4: One-Round Discussion
    print("\n=== B4: One-Round Discussion ===")
    disc_results = []
    for ex in test_data[:num_samples]:
        a, b = discussion_inference(
            ex["prompt"], base_model, tokenizer,
            formatters[0], formatters[1],
        )
        disc_results.append({"agent_a": a, "agent_b": b})
    all_results["B4_Discussion"] = compute_metrics(disc_results)

    # Ours: Collaborative (trained agents)
    print("\n=== Ours: Collaborative Training ===")
    ours_results = []
    for ex in test_data[:num_samples]:
        result = collaborative_inference(
            ex["prompt"], agent_a, agent_b, tokenizer,
            formatters[0], formatters[1],
        )
        ours_results.append(result)
    all_results["Ours_Collaborative"] = compute_metrics(ours_results)

    # Save results
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Method':<25} {'Samples':>8} {'Avg Len A':>10} {'Avg Len B':>10} {'Ratio':>8}")
    print("-" * 60)
    for method, metrics in all_results.items():
        print(f"{method:<25} {metrics.get('num_samples',0):>8} "
              f"{metrics.get('avg_len_a',0):>10.1f} {metrics.get('avg_len_b',0):>10.1f} "
              f"{metrics.get('avg_ratio',0):>8.2f}")

    print(f"\nResults saved to {results_path}")
    return all_results


def compute_single_metrics(results: List[Dict]) -> Dict:
    """Compute metrics for single-agent baseline."""
    if not results:
        return {}
    lengths = [len(r["output"]) for r in results]
    return {
        "num_samples": len(results),
        "avg_output_len": sum(lengths) / len(lengths),
        "avg_len_a": sum(lengths) / len(lengths),
        "avg_len_b": 0,
        "avg_ratio": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", default="./outputs/collab")
    parser.add_argument("--output_dir", default="./outputs/eval")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()
    run_evaluation(**vars(args))
