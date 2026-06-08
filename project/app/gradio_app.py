"""
Gradio Demo: Multi-Agent Collaborative Writing

Side-by-side comparison of 5 methods on the same input.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import gradio as gr
import torch
from transformers import AutoTokenizer

from train import (
    load_agent_with_lora,
    build_tldr_formatters,
    collaborative_inference,
    single_agent_inference as _single,
    parallel_inference as _parallel,
    sequential_inference as _sequential,
    discussion_inference as _discussion,
)

# Cache
_cache: dict = {}

def load_all(model_name="Qwen/Qwen3-0.6B", lora_path=None):
    global _cache
    if _cache:
        return _cache

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model, _ = load_agent_with_lora(model_name, use_4bit=True, device_map="auto")

    agent_a = agent_b = base_model
    if lora_path and os.path.exists(os.path.join(lora_path, "agent_a_final")):
        from peft import PeftModel
        agent_a = PeftModel.from_pretrained(base_model, os.path.join(lora_path, "agent_a_final"))
        agent_b = PeftModel.from_pretrained(base_model, os.path.join(lora_path, "agent_b_final"))
        agent_a = agent_a.merge_and_unload()
        agent_b = agent_b.merge_and_unload()

    formatters = build_tldr_formatters(tokenizer)
    _cache = {
        "base_model": base_model, "agent_a": agent_a, "agent_b": agent_b,
        "tokenizer": tokenizer, "formatters": formatters,
    }
    print("Models loaded!")
    return _cache


def compare_all(text):
    """Run all 5 methods and return results."""
    m = load_all()
    tokenizer = m["tokenizer"]
    fmt = m["formatters"]

    # B1: Single
    b1 = _single(text, m["base_model"], tokenizer)

    # B2: Parallel
    b2_a, b2_b = _parallel(text, m["base_model"], tokenizer)

    # B3: Sequential
    b3_a, b3_b = _sequential(text, m["base_model"], tokenizer, fmt[0], fmt[1])

    # B4: Discussion
    b4_a, b4_b = _discussion(text, m["base_model"], tokenizer, fmt[0], fmt[1])

    # Ours: Collaborative
    ours = collaborative_inference(text, m["agent_a"], m["agent_b"], tokenizer, fmt[0], fmt[1])

    return (
        b1,
        f"**Agent A:**\n{b2_a}\n\n**Agent B:**\n{b2_b}",
        f"**Agent A:**\n{b3_a}\n\n**Agent B:**\n{b3_b}",
        f"**讨论后输出:**\n{b4_a}\n\n**反馈:**\n{b4_b}",
        f"**Agent A (精炼):**\n{ours['agent_a']}\n\n**Agent B (展开):**\n{ours['agent_b']}",
    )


def create_demo(lora_path=None):
    try:
        load_all(lora_path=lora_path)
    except Exception as e:
        print(f"Models will load on first request: {e}")

    with gr.Blocks(title="🤖 Multi-Agent Collaborative Writing") as demo:
        gr.Markdown("""
        # 🤖 多智能体协作写作 — 小模型大能力

        **研究问题**：两个小模型通过角色分工协作，能否超越单个大模型？

        上传一段文本，对比 **5 种方法** 的输出效果。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="📝 输入文本",
                    placeholder="Paste a Reddit post or any long text...",
                    lines=8,
                )
                btn = gr.Button("🚀 对比所有方法", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("B1: Single Model"):
                        out_b1 = gr.Textbox(label="单模型输出", lines=6)
                    with gr.TabItem("B2: Parallel"):
                        out_b2 = gr.Markdown("Waiting...")
                    with gr.TabItem("B3: Sequential"):
                        out_b3 = gr.Markdown("Waiting...")
                    with gr.TabItem("B4: Discussion"):
                        out_b4 = gr.Markdown("Waiting...")
                    with gr.TabItem("🔥 Ours: Collaborative"):
                        out_ours = gr.Markdown("Waiting...")

        btn.click(
            fn=compare_all,
            inputs=[input_text],
            outputs=[out_b1, out_b2, out_b3, out_b4, out_ours],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="./outputs/collab")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.lora_path)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
