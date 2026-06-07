"""
GRPO 奖励函数模块
支持多种奖励策略用于消融实验
"""

import re
from typing import Optional


def extract_final_answer(text: str) -> Optional[str]:
    """从模型输出中提取最终答案"""
    # 匹配 </think> 后的内容，或整个回答
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 尝试多种答案格式
    patterns = [
        r"(?:答案|最终答案|所以|因此)[：:]\s*(.+)",
        r"(?:answer|Answer)\s*[:：]\s*(.+)",
        r"\\boxed\{([^}]+)\}",
        r"(\d+\.?\d*)\s*(?:$|\n)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).strip()
    return None


def check_thinking_format(text: str) -> bool:
    """检查是否包含 <think>...</think> 推理格式"""
    return "<think>" in text and "</think>" in text


def count_reasoning_steps(text: str) -> int:
    """统计推理步骤数量"""
    if "<think>" not in text:
        return 0
    think_content = text.split("<think>")[-1].split("</think>")[0]
    # 统计以数字编号或步骤标记开头的行
    steps = re.findall(
        r"(?:^\d+[\.\)、]|步骤\s*\d|第[一二三四五六七八九十\d]+步)",
        think_content,
        re.MULTILINE,
    )
    # 也统计换行符作为粗略替代
    line_count = len([l for l in think_content.split("\n") if l.strip()])
    return max(len(steps), line_count)


def normalize_answer(answer: str) -> str:
    """标准化答案字符串"""
    answer = answer.strip().lower()
    # 移除常见后缀
    for suffix in ["。", ".", "，", ",", "!", "！"]:
        if answer.endswith(suffix):
            answer = answer[:-1]
    # 移除空格
    answer = answer.replace(" ", "")
    return answer


def is_correct(predicted: Optional[str], ground_truth: str) -> bool:
    """判断预测答案是否正确"""
    if predicted is None:
        return False
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    # 精确匹配
    if pred == gt:
        return True
    # 数值近似匹配
    try:
        pred_num = float(pred)
        gt_num = float(gt)
        return abs(pred_num - gt_num) < 1e-6
    except (ValueError, TypeError):
        pass
    # 包含匹配
    if gt in pred or pred in gt:
        return True
    return False


def reward_correctness_only(response: str, ground_truth: str) -> float:
    """仅根据答案正确性给奖励"""
    answer = extract_final_answer(response)
    return 1.0 if is_correct(answer, ground_truth) else 0.0


def reward_correctness_and_format(response: str, ground_truth: str) -> float:
    """答案正确性 + 格式完整性"""
    score = 0.0
    answer = extract_final_answer(response)
    if is_correct(answer, ground_truth):
        score += 0.8
    if check_thinking_format(response):
        score += 0.2
    return score


def reward_full(response: str, ground_truth: str) -> float:
    """完整奖励：正确性 + 格式 + 步骤数量"""
    score = 0.0
    answer = extract_final_answer(response)
    if is_correct(answer, ground_truth):
        score += 0.7
    if check_thinking_format(response):
        score += 0.2
        # 步骤越多越好（至少3步）
        steps = count_reasoning_steps(response)
        if steps >= 3:
            score += 0.1
    return score


REWARD_FUNCTIONS = {
    "correctness_only": reward_correctness_only,
    "correctness_and_format": reward_correctness_and_format,
    "full": reward_full,
}


def compute_grpo_reward(
    responses: list[str],
    ground_truths: list[str],
    reward_type: str = "full",
) -> list[float]:
    """批量计算 GRPO 奖励"""
    reward_fn = REWARD_FUNCTIONS.get(reward_type, reward_full)
    return [reward_fn(r, gt) for r, gt in zip(responses, ground_truths)]
