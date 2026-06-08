# 基于 LoRA 的小型视觉语言模型专项能力增强研究

## —— Qwen3-VL-2B 在细粒度食物图像分类上的实证分析

> 河北师范大学 智能科学综合课程设计五 课程论文

---

## 摘要

大型视觉语言模型(VLM)在通用图像理解任务中表现出色，但部署成本高昂。本文研究小
型 VLM 通过参数高效微调(PEFT)在特定领域超越大模型的可能性。我们使用 QLoRA 技术
对 Qwen3-VL-2B（6亿参数）在 Food101 数据集上进行专项训练，并与 Qwen3-VL-4B/8B
零样本、ResNet-50、CLIP 等基线模型对比。实验结果表明，经过 LoRA 专项训练的小
VLM 在细粒度食物分类任务上可超越参数规模 4 倍的大模型零样本表现，验证了"精准专
才优于平庸通才"的假设。

**关键词**：视觉语言模型；LoRA；参数高效微调；细粒度图像分类；模型压缩

---

## 1. 引言

（描述研究背景、VLM 发展现状、小模型研究意义、本文贡献）

## 2. 相关工作

（VLM微调方法、LoRA相关研究、小模型专项能力研究）

## 3. 方法

### 3.1 模型架构

Qwen3-VL-2B + QLoRA (4-bit nf4) + LoRA (r=16)

### 3.2 训练设置

（训练参数、数据集、硬件环境）

### 3.3 Baseline 设计

（5个baseline的选取理由和实验设置）

### 3.4 评估指标

（Top-1/Top-5准确率、描述质量、推理速度、显存占用）

## 4. 实验

### 4.1 数据集

Food101 (101类, 10万张图片)

### 4.2 主实验结果

（模型 vs 5个baseline的对比表）

### 4.3 消融实验

（LoRA rank消融、数据规模消融、QLoRA vs LoRA对比）

### 4.4 案例分析

（典型成功/失败案例展示）

## 5. 结果分析与讨论

（为什么小VLM+LoRA能超越大VLM？训练策略的启示？局限性和未来工作）

## 6. 结论

（总结发现，重申"精准专才 > 平庸通才"）

## 参考文献

[1] Qwen Team. Qwen3-VL: Multimodal Large Language Model. 2025.
[2] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
[3] Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
[4] Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
[5] Bossard et al. Food-101 – Mining Discriminative Components with Random Forests. ECCV 2014.

## 附录

- 实验详细配置
- 完整结果表格
- 模型推理示例
