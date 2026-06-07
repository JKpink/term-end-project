"""
GRPO 训练脚本 (Kaggle 环境)
基于 TRL GRPOTrainer，使用 Qwen3.5 + QLoRA 进行 GRPO 强化学习训练

参考:
- nanoRL: https://github.com/ethanhe42/nanoRL
- microR1: https://github.com/tyler-romero/micror1
"""

import os
import sys
import json
import torch
import re
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import GRPOConfig, GRPOTrainer

# 添加 src 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward import (
    extract_final_answer,
    check_thinking_format,
    count_reasoning_steps,
    is_correct,
    REWARD_FUNCTIONS,
)


@dataclass
class GRPOTrainingConfig:
    """GRPO 训练配置"""
    model_name: str = field(
        default="Qwen/Qwen3.5-0.8B",
        metadata={"help": "基座模型名称"}
    )
    sft_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "SFT 预热的 LoRA 适配器路径（None 则直接 GRPO）"}
    )
    output_dir: str = field(
        default="/kaggle/working/grpo_output",
        metadata={"help": "输出目录"}
    )
    dataset_name: str = field(
        default="openai/gsm8k",
        metadata={"help": "训练数据集"}
    )
    dataset_config: str = field(
        default="main",
        metadata={"help": "数据集配置名"}
    )
    # 奖励函数
    reward_type: str = field(
        default="full",
        metadata={"help": "奖励函数类型: correctness_only, correctness_and_format, full"}
    )
    # QLoRA
    use_qlora: bool = field(default=True, metadata={"help": "是否使用 QLoRA"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    # GRPO 参数
    num_generations: int = field(default=8, metadata={"help": "每组生成数量 (Group Size)"})
    max_completion_length: int = field(default=512, metadata={"help": "最大生成长度"})
    temperature: float = field(default=1.0, metadata={"help": "采样温度"})
    # KL 惩罚
    kl_coef: float = field(default=0.04, metadata={"help": "KL 散度惩罚系数 (β)"})
    # 训练参数
    num_train_epochs: int = field(default=2, metadata={"help": "训练轮数"})
    per_device_batch_size: int = field(default=2, metadata={"help": "每卡 batch size"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "梯度累积步数"})
    learning_rate: float = field(default=5e-5, metadata={"help": "学习率 (GRPO 建议比 SFT 小)"})
    max_prompt_length: int = field(default=512, metadata={"help": "最大 prompt 长度"})
    logging_steps: int = field(default=5, metadata={"help": "日志间隔"})
    save_steps: int = field(default=100, metadata={"help": "保存间隔"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "最大训练样本数"})
    use_vllm: bool = field(default=False, metadata={"help": "是否使用 vLLM 加速推理"})


SYSTEM_PROMPT = """你是一个理科题目求解助手。请一步步思考，将推理过程写在 <think> 和 </think> 之间，最后给出答案。

格式要求：
<think>
1. 分析题目
2. 列出公式
3. 代入计算
4. 验证结果
</think>
答案：[你的答案]"""


def format_scibench_grpo(example: dict) -> dict:
    """将 SciBench 数据格式化为 GRPO 训练数据"""
    problem = example.get("problem_text", example.get("question", ""))
    answer = example.get("answer_number", example.get("answer_latex", ""))
    unit = example.get("unit", "")
    gt = f"{answer} {unit}".strip() if unit else str(answer)
    return {
        "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                  f"<|im_start|>user\n{problem}<|im_end|>\n"
                  f"<|im_start|>assistant\n",
        "ground_truth": gt,
    }


def prepare_grpo_dataset(config: GRPOTrainingConfig) -> Dataset:
    """准备 GRPO 训练数据集，支持 GSM8K 和 SciBench"""
    print(f"Loading dataset: {config.dataset_name}")

    if "gsm8k" in config.dataset_name.lower():
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        train_data = dataset["train"]

        def format_gsm8k_grpo(example):
            answer = example["answer"]
            if "####" in answer:
                gt = answer.split("####")[1].strip()
            else:
                gt = answer.strip()
            return {
                "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                          f"<|im_start|>user\n{example['question']}<|im_end|>\n"
                          f"<|im_start|>assistant\n",
                "ground_truth": gt,
            }

        train_data = train_data.map(format_gsm8k_grpo, remove_columns=train_data.column_names)

    elif "scibench" in config.dataset_name.lower() or "xw27" in config.dataset_name:
        try:
            dataset = load_dataset(config.dataset_name, config.dataset_config, split="train")
        except Exception:
            dataset = load_dataset(config.dataset_name, config.dataset_config)
            if isinstance(dataset, dict):
                dataset = list(dataset.values())[0]
        train_data = dataset.map(format_scibench_grpo, remove_columns=dataset.column_names)

    else:
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        train_data = dataset["train"]

    if config.max_train_samples:
        train_data = train_data.select(range(min(config.max_train_samples, len(train_data))))

    print(f"Training samples: {len(train_data)}")
    return train_data


def grpo_reward_func(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """GRPO 奖励函数（供 TRL GRPOTrainer 调用）"""
    reward_fn = REWARD_FUNCTIONS.get(
        os.environ.get("GRPO_REWARD_TYPE", "full"), REWARD_FUNCTIONS["full"]
    )

    # kwargs 中可能包含 prompt 等信息
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        r = reward_fn(completion, gt)
        rewards.append(r)
    return rewards


def train_grpo(config: GRPOTrainingConfig):
    """执行 GRPO 训练"""
    # 加载 tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    if config.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # T4 用 sdpa，flash_attention_2 不一定有
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="sdpa",
            )

        # 加载 SFT adapter（如果提供）
        if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
            print(f"Loading SFT adapter from {config.sft_adapter_path}")
            model = PeftModel.from_pretrained(model, config.sft_adapter_path)

        # LoRA 配置
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if config.sft_adapter_path and os.path.exists(config.sft_adapter_path):
            model = PeftModel.from_pretrained(model, config.sft_adapter_path)

    model.print_trainable_parameters() if hasattr(model, "print_trainable_parameters") else None

    # 准备数据集
    dataset = prepare_grpo_dataset(config)

    # GRPO 训练参数
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if config.use_qlora else "adamw_torch",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        # GRPO 特有参数
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        max_prompt_length=config.max_prompt_length,
        beta=config.kl_coef,  # KL 惩罚系数
        use_vllm=config.use_vllm,
    )

    # 确保 CLI --reward_type 生效（传递到奖励函数内）
    os.environ["GRPO_REWARD_TYPE"] = config.reward_type

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=grpo_reward_func,
    )

    print("Starting GRPO training...")
    trainer.train()

    # 保存模型
    final_path = os.path.join(config.output_dir, "final_grpo")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"GRPO model saved to {final_path}")

    return final_path


if __name__ == "__main__":
    parser = HfArgumentParser(GRPOTrainingConfig)
    config = parser.parse_args_into_dataclasses()[0]
    train_grpo(config)
