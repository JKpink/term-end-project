"""
SFT 预热训练脚本 (Kaggle 环境)
使用 QLoRA 对 Qwen3-0.6B.8B 进行监督微调，让模型学会推理格式
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer


@dataclass
class SFTConfig:
    """SFT 训练配置"""
    model_name: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "基座模型名称"}
    )
    dataset_name: str = field(
        default="openai/gsm8k",
        metadata={"help": "训练数据集"}
    )
    dataset_config: str = field(
        default="main",
        metadata={"help": "数据集配置名"}
    )
    output_dir: str = field(
        default="/kaggle/working/sft_output",
        metadata={"help": "输出目录"}
    )
    # QLoRA 参数
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    # 训练参数
    num_epochs: int = field(default=2, metadata={"help": "训练轮数"})
    per_device_batch_size: int = field(default=4, metadata={"help": "每卡 batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "梯度累积步数"})
    learning_rate: float = field(default=2e-4, metadata={"help": "学习率"})
    max_seq_length: int = field(default=1024, metadata={"help": "最大序列长度"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "预热比例"})
    logging_steps: int = field(default=10, metadata={"help": "日志间隔"})
    save_steps: int = field(default=200, metadata={"help": "保存间隔"})
    # 数据参数
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "最大训练样本数（None=全部）"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "是否使用 Flash Attention 2"}
    )


SYSTEM_PROMPT = """你是一个理科题目求解助手。请一步步思考，将推理过程写在 <think> 和 </think> 之间，最后给出答案。

示例格式：
<think>
1. 分析题目条件：...
2. 列出已知公式：...
3. 代入计算：...
</think>
答案：..."""


def format_gsm8k(example: dict) -> dict:
    """将 GSM8K 数据格式化为带推理格式的训练数据"""
    question = example["question"]
    # 从 answer 中提取推理步骤
    answer = example["answer"]
    # GSM8K 的 answer 格式: "#### 数字"，前面是推理步骤
    if "####" in answer:
        reasoning = answer.split("####")[0].strip()
        final_answer = answer.split("####")[1].strip()
    else:
        reasoning = answer
        final_answer = answer

    text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
{reasoning}
</think>
答案：{final_answer}<|im_end|>"""

    return {"text": text}


def format_scibench(example: dict) -> dict:
    """将 SciBench 数据格式化为训练数据"""
    problem = example.get("problem_text", example.get("question", ""))
    solution = example.get("solution", "")
    answer = example.get("answer_number", example.get("answer_latex", ""))
    unit = example.get("unit", "")

    answer_str = f"{answer} {unit}".strip() if unit else str(answer)

    text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
<think>
{solution}
</think>
答案：{answer_str}<|im_end|>"""

    return {"text": text}


def load_and_prepare_data(config: SFTConfig) -> Dataset:
    """加载并预处理数据集"""
    print(f"Loading dataset: {config.dataset_name}")

    if "gsm8k" in config.dataset_name.lower():
        dataset = load_dataset(config.dataset_name, config.dataset_config, split="train")
        dataset = dataset.map(format_gsm8k, remove_columns=dataset.column_names)
    elif "scibench" in config.dataset_name.lower() or "xw27" in config.dataset_name:
        # SciBench: xw27/scibench — 可能没有 train/test 分割
        try:
            dataset = load_dataset(config.dataset_name, split="train")
        except Exception:
            dataset = load_dataset(config.dataset_name)
            if isinstance(dataset, dict):
                dataset = list(dataset.values())[0]
        dataset = dataset.map(format_scibench, remove_columns=dataset.column_names)
    else:
        # 通用格式：假设有 question 和 answer 列
        dataset = load_dataset(config.dataset_name, config.dataset_config, split="train")

    if config.max_train_samples:
        dataset = dataset.select(range(min(config.max_train_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample:\n{dataset[0]['text'][:500]}...")
    return dataset


def train_sft(config: SFTConfig):
    """执行 SFT 训练"""
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 加载模型
    print(f"Loading model: {config.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if config.use_flash_attention else "sdpa",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 数据集
    dataset = load_and_prepare_data(config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    # SFT Trainer — TRL 最新 API: processing_class + formatting_func
    # 废弃: tokenizer, packing, dataset_text_field, max_seq_length
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=lambda x: x["text"],
    )

    print("Starting SFT training...")
    trainer.train()

    # 保存模型
    final_path = os.path.join(config.output_dir, "final")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return final_path


if __name__ == "__main__":
    parser = HfArgumentParser(SFTConfig)
    config = parser.parse_args_into_dataclasses()[0]
    train_sft(config)
