"""
Gradio 演示界面 — 理科题目求解助手
"""

import os
import sys
import torch
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


MODEL_NAME = "Qwen/Qwen3.5-0.8B-Instruct"
ADAPTER_PATH = None  # 设置训练好的 LoRA/GRPO adapter 路径

model = None
tokenizer = None


def load_model_once():
    global model, tokenizer
    if model is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    print("Model loaded!")


SYSTEM_PROMPT = "你是一个理科题目求解助手。请一步步思考，将推理过程写在 <think> 和 </think> 之间，最后给出答案。"

EXAMPLES = [
    ["一个质量为 2kg 的物体从 10m 高处自由落下，求落地时的速度（g=10m/s²）"],
    ["计算 25°C 时 0.1mol/L 醋酸溶液的 pH 值（Ka=1.8×10⁻⁵）"],
    ["小明有 15 颗糖，给了小红 1/3，又给了小刚 2/5，还剩几颗？"],
    ["一辆汽车以 72km/h 的速度行驶，突然刹车，刹车后滑行 40m 停下，求加速度。"]
]


def solve(question: str, enable_thinking: bool = True, temperature: float = 0.7) -> str:
    load_model_once()

    if enable_thinking:
        sys_part = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"
    else:
        sys_part = f"<|im_start|>system\n请直接给出答案。<|im_end|>"

    prompt = f"{sys_part}\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def create_ui():
    with gr.Blocks(title="理科题目求解助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔬 理科题目求解助手
        ### 基于 Qwen3.5 + GRPO 的小模型推理增强

        输入物理、化学或数学题目，AI 将展示逐步推理过程并给出答案。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="输入题目",
                    placeholder="请输入物理、化学或数学题目...",
                    lines=3,
                )
                with gr.Row():
                    thinking_toggle = gr.Checkbox(label="启用推理过程 (Think Mode)", value=True)
                    temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                submit_btn = gr.Button("求解", variant="primary", size="lg")

            with gr.Column(scale=4):
                output = gr.Textbox(
                    label="求解结果",
                    lines=20,
                    max_lines=30,
                )

        submit_btn.click(
            fn=solve,
            inputs=[question, thinking_toggle, temperature],
            outputs=output,
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[question],
            label="示例题目",
        )

        gr.Markdown("""
        ---
        ### 关于本项目
        本项目使用 **GRPO (Group Relative Policy Optimization)** 算法对 **Qwen3.5** 小模型进行推理能力训练。

        - **模型**: Qwen3.5-0.8B (Apache 2.0)
        - **方法**: GRPO (DeepSeek-R1 同款算法)
        - **数据集**: GSM8K + SciBench
        """)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)
