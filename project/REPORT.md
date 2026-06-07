# 基于 GRPO 的 Qwen3.5 小模型理科推理能力增强研究

> 智能科学综合课程设计五 · 项目报告  
> 河北师范大学 · 计算机与网络安全学院  
> 日期：2026 年 6 月

---

## 一、项目概述

### 1.1 题目

**基于 GRPO 的 Qwen3.5 小模型理科推理能力增强研究**

### 1.2 案例

**理科题目求解助手** —— 用户输入物理、化学或数学题目，AI 模型展示逐步推理过程（`<think>...</think>`），最后给出答案。

### 1.3 核心思路

```
基座模型 (Qwen3.5-0.8B)
    ↓ SFT 预热 (监督微调，学会推理格式)
    ↓ GRPO 强化学习 (组内对比优化，激发推理能力)
    → 理科推理助手 (小模型也能"思考")
```

### 1.4 四要素

| 要素 | 内容 |
|------|------|
| **案例** | 理科题目求解助手 |
| **方向** | GRPO 强化学习训练（DeepSeek-R1 同款算法） |
| **创新点** | Qwen3.5 + GRPO，研究 GatedDeltaNet 混合架构小模型的推理涌现条件 |
| **Baseline** | 5 组基线 + 5 组消融实验 |

---

## 二、方法

### 2.1 GRPO 算法

GRPO (Group Relative Policy Optimization) 由 DeepSeek 在 DeepSeekMath 和 DeepSeek-R1 中提出。

**与 PPO 的核心区别：**

| | PPO | GRPO |
|---|-----|------|
| 所需模型数 | 4 (Policy + Reference + Critic + Reward) | 1 (仅 Policy) |
| 优势估计 | 需要 Value 网络 | 组内标准化比较 |
| 显存占用 | 高 | **节省 75%** |

**GRPO 工作原理：**
1. 对每个 prompt，生成 G 个回答（Group Size = G）
2. 用奖励函数对 G 个回答打分
3. 组内标准化：(reward - mean(rewards)) / std(rewards) → 组内优势
4. 用 PPO 风格的 clipped loss 更新策略，加上 KL 惩罚防止偏离太远

### 2.2 奖励函数设计

```python
reward = 0.0
if 答案正确:      reward += 0.7
if 包含 <think>:  reward += 0.2
if 推理步骤 ≥ 3:  reward += 0.1
```

三种策略用于消融：纯正确性 / 正确性+格式 / 完整奖励。

### 2.3 训练流程

```
Phase 1: SFT 预热
  Qwen3.5-0.8B + QLoRA (4-bit)
  在 GSM8K 上监督微调，学会 <think> 推理格式
  训练 2 epoch，LoRA r=8~16

Phase 2: GRPO 训练
  SFT 适配器作为初始化
  GRPO 强化学习，奖励驱动推理能力提升
  Group Size=4~8, KL β=0.04
```

---

## 三、实验设计

### 3.1 数据集

| 数据集 | 来源 | 规模 | 用途 |
|--------|------|------|------|
| GSM8K | `openai/gsm8k` (HF) / `swift/gsm8k` (ModelScope) | 7473 训练 / 1319 测试 | SFT 预热 + GRPO 训练 |
| SciBench | `xw27/scibench` (HF) | ~695 题 | 主评估集 |

### 3.2 模型

| 模型 | 参数 | 架构 | 协议 |
|------|------|------|------|
| Qwen3.5-0.8B-Instruct | 0.8B | GatedDeltaNet + Attention | Apache 2.0 |
| Qwen3.5-2B-Instruct | 2B | GatedDeltaNet + Attention | Apache 2.0 |

### 3.3 运行环境

| 环境 | GPU | 用途 |
|------|-----|------|
| 本地 RTX 5060 | 1× 8GB | 单卡开发调试 |
| 本地 RTX 2080Ti | 4× 11GB | 并行实验 |
| Kaggle T4 | 2× 16GB | 云端训练 |

**软件环境：** Python 3.10+ / PyTorch 2.4+ / transformers 4.47+ / TRL 0.12+ / PEFT 0.13+ / bitsandbytes 0.44+

### 3.4 Baseline（5 组）

| # | 方法 | 说明 |
|---|------|------|
| ① | Qwen3.5-0.8B 直接回答 | 不开 thinking，不训练 |
| ② | Qwen3.5-0.8B + thinking on | 原生 thinking 模式 |
| ③ | Qwen3.5-0.8B + QLoRA SFT | 监督微调 |
| ④ | **Qwen3.5-0.8B + SFT + GRPO（本文）** | 核心方法 |
| ⑤ | Qwen3.5-2B + SFT + GRPO | 规模对比 |

### 3.5 消融实验（5 组）

| # | 消融维度 | 变化 | 回答的问题 |
|---|---------|------|-----------|
| A | SFT 预热 | GRPO 有/无 SFT 预热 | SFT 预热是否必要？ |
| B | 奖励函数 | 纯正确性 / +格式 / 完整 | 格式奖励是否有用？ |
| C | Group Size | G=4 / G=8 / G=16 | 组大小对效果的影响？ |
| D | KL 系数 | β=0.01 / 0.04 / 0.1 | KL 约束多强合适？ |

---

## 四、预期结果

```
                              GSM8K 准确率
                              20%   40%   60%   80%
                               
① 直接回答 (no thinking)       ████
② thinking on                  ██████
③ QLoRA SFT                    ████████
④ GRPO (本文)                  ██████████████  ← 核心贡献
⑤ 2B + GRPO                    ██████████████████

消融 A: 无SFT预热 → GRPO       < ④ → SFT 预热必要
消融 B: 仅正确性奖励            < ④ → 格式奖励有效
消融 C: G 变化                  → 找最优 Group Size
消融 D: β 变化                  → 找最优 KL 约束
```

### 预期结论

1. GRPO 能在极小模型（0.8B）上显著激发推理能力
2. SFT 预热对 GRPO 收敛至关重要
3. 格式奖励 + 正确性奖励的组合优于纯正确性奖励
4. Qwen3.5 的 GatedDeltaNet 架构在推理任务上表现出色

---

## 五、参考文献

1. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024
2. Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", 2025
3. Qwen Team, "Qwen3.5 Technical Report", 2026
4. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
5. Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
6. Wang et al., "SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models", ICML 2024
7. Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K), 2021
8. von Werra et al., "TRL: Transformer Reinforcement Learning", Hugging Face, 2024
9. ethanhe42, "nanoRL: Minimal GRPO Implementation", GitHub, 2025
10. tyler-romero, "microR1: Minimal DeepSeek-R1 Reproduction", GitHub, 2025

---

## 六、项目结构

```
project/
├── REPORT.md                  # 本报告
├── requirements.txt           # Python 依赖
├── prepare.py                 # 数据与模型准备 (ModelScope/HF)
├── run_5060.py                # RTX 5060 单卡一键运行
├── run_local.py               # RTX 2080Ti ×4 并行运行
├── kaggle_notebook.ipynb      # Kaggle T4 ×2 云端运行
├── src/
│   ├── reward.py              # 奖励函数 (3种策略)
│   ├── train_sft.py           # SFT QLoRA 预热训练
│   ├── train_grpo.py          # GRPO 强化学习训练
│   └── evaluate.py            # 模型评估 (GSM8K + SciBench)
├── app/
│   └── gradio_app.py          # Gradio Web 演示界面
├── models/                    # 本地模型缓存
│   ├── qwen3.5-0.8b-Instruct/
│   └── qwen3.5-2b-Instruct/
├── outputs/                   # 训练输出
│   ├── sft_*/                 # SFT 适配器
│   ├── grpo_*/                # GRPO 模型
│   └── eval_*/                # 评估结果
├── paper/                     # 论文
└── ppt/                       # PPT 演示
```

---

## 七、实施计划

| 阶段 | 时间 | 内容 | 产出 |
|------|------|------|------|
| 环境搭建 | 1 天 | 下载模型/数据，跑通 demo | 可训练的环境 |
| SFT 预热 | 1 天 | QLoRA SFT 训练 | SFT 适配器 |
| GRPO 训练 | 3-4 天 | 主实验 + 5 组消融 | GRPO 模型 |
| 结果分析 | 2 天 | 统计数据、画图、案例分析 | 实验结果图表 |
| 论文撰写 | 5-7 天 | 按模板撰写论文 | 完整论文 |
| PPT + 演示 | 1-2 天 | Gradio 界面 + PPT | 可演示系统 |

---

## 八、快速开始

```bash
# 1. 准备环境
cd project
pip install -r requirements.txt

# 2. 下载模型和数据集
python prepare.py --target all --source modelscope

# 3. 运行实验 (选择你的环境)
python run_5060.py all       # RTX 5060
python run_local.py all      # RTX 2080Ti ×4
# Kaggle: 上传 kaggle_notebook.ipynb

# 4. 启动演示
python run_5060.py demo
```

---

> **指导教师：** 霍丽娜 (huolina@hebtu.edu.cn)  
> **课程：** 智能科学综合课程设计五  
> **学期：** 2025-2026 学年第二学期
