from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


class DeepSeekError(Exception):
    pass


MAX_INPUT_CHARS = 120_000

VALID_PRACTICE_TYPES = frozenset({"mcq", "short_answer", "calculation", "mixed"})

VALID_IMPORTANCE = frozenset({"必考", "一般", "了解"})

IMPORTANCE_ALIASES = {
    "重点": "必考",
    "高": "必考",
    "中": "一般",
    "普通": "一般",
    "低": "了解",
    "拓展": "了解",
    "次要": "了解",
}

# 写入用户消息，指导练习题格式（system 中保留 JSON 结构约定）
PRACTICE_TYPE_INSTRUCTIONS: dict[str, str] = {
    "mcq": """【练习题型：选择题 MCQ】
请生成单项选择题。每道题的 question 字段中必须包含：题干段落，以及换行后四个选项，格式严格为：
A. …
B. …
C. …
D. …
reference_answer 写明正确选项字母（如「B」），并可用一两句话说明为何正确；solution_approach 写解题思路（可用排除法或推导正确项）。""",
    "short_answer": """【练习题型：简答题】
每道题的 question 为简答设问，表述清晰；reference_answer 为条理清楚的要点式或简短段落式参考答案（覆盖得分点）；solution_approach 说明如何依据讲义组织答案、常见踩分点。""",
    "calculation": """【练习题型：计算题】
每道题的 question 须给出已知条件与待求量（必要时说明单位或近似）；reference_answer 给出主要公式、代入过程与数值结果（含单位）；solution_approach 说明选用公式理由与计算步骤要点。""",
    "mixed": """【练习题型：混合】
请在 practice_questions 中合理搭配至少两种题型（可含选择题 MCQ、简答题、计算题等）。每道小题的实际形式须在 question 中体现；reference_answer 与 solution_approach 须与该小题题型一致。""",
}

SYSTEM_PROMPT = """你是面向留学生的课程复习助教。用户会提供课程讲义摘录（语言任意）以及练习题型要求（在用户消息开头）。
你必须全程使用简体中文作答，并只输出一个 JSON 对象，不要 Markdown 围栏、不要前后说明文字。
JSON 的键必须严格为：
- "core_points": 数组。每一项必须是对象，且恰好包含两个键（字符串）：
  - "point": 考点描述，简练清晰；
  - "importance": 只能是以下三个字面量之一（勿加括号或其它符号）："必考"、"一般"、"了解"。
  含义：必考=极可能在考试中出现或分值高；一般=常见考点；了解=拓展、背景或次要内容。
  请根据讲义合理分配比例，使「必考」条目精炼、有区分度。
- "difficult_analysis": 单个字符串，对重难点做解析，可含多段，使用 \\n 换行；
- "practice_questions": 数组，每一项必须是对象，且恰好包含三个键（字符串）：
  - "question": 题干（格式须符合用户消息中的题型要求）；
  - "reference_answer": 参考答案；
  - "solution_approach": 解题思路（关键步骤、知识点、理由；可含多段，用 \\n 换行）。

练习题数量适中（例如 3～8 题），每题必须同时给出 question、reference_answer、solution_approach，且内容具体、可核对。

确保 JSON 合法，字符串内的换行和引号需正确转义。"""


def _read_deepseek_api_key() -> str:
    """从环境变量读取 Key，兼容 UTF-8 BOM、首尾空白与引号包裹。"""
    raw = os.environ.get("DEEPSEEK_API_KEY", "") or ""
    key = raw.strip().lstrip("\ufeff")
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    return key


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


def _normalize_core_points(items: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            t = item.strip()
            if t:
                out.append({"point": t, "importance": "一般"})
            continue
        if not isinstance(item, dict):
            raise DeepSeekError(f"core_points[{i}] 应为对象或字符串")

        point = str(
            item.get("point")
            or item.get("考点")
            or item.get("content")
            or item.get("text")
            or ""
        ).strip()
        if not point:
            raise DeepSeekError(f"core_points[{i}] 缺少考点内容 point")

        imp_raw = str(
            item.get("importance")
            or item.get("重要程度")
            or item.get("level")
            or "一般"
        ).strip()
        imp_clean = imp_raw.replace("【", "").replace("】", "").strip()
        if imp_clean not in VALID_IMPORTANCE:
            imp_clean = IMPORTANCE_ALIASES.get(imp_clean, "一般")
        if imp_clean not in VALID_IMPORTANCE:
            imp_clean = "一般"

        out.append({"point": point, "importance": imp_clean})
    return out


def parse_model_json(content: str) -> dict[str, Any]:
    raw = _strip_code_fence(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise DeepSeekError("模型返回不是合法 JSON，请稍后重试。")
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise DeepSeekError("模型返回不是合法 JSON，请稍后重试。") from exc

    for key in ("core_points", "difficult_analysis", "practice_questions"):
        if key not in data:
            raise DeepSeekError(f"模型返回缺少字段：{key}")

    if not isinstance(data["core_points"], list):
        raise DeepSeekError("core_points 应为数组")
    if not isinstance(data["difficult_analysis"], str):
        raise DeepSeekError("difficult_analysis 应为字符串")
    if not isinstance(data["practice_questions"], list):
        raise DeepSeekError("practice_questions 应为数组")

    data["core_points"] = _normalize_core_points(data["core_points"])
    data["difficult_analysis"] = str(data["difficult_analysis"]).strip()
    data["practice_questions"] = _normalize_practice_questions(data["practice_questions"])
    return data


def _normalize_practice_questions(items: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            q = item.strip()
            if q:
                out.append(
                    {
                        "question": q,
                        "reference_answer": "",
                        "solution_approach": "",
                    }
                )
            continue
        if not isinstance(item, dict):
            raise DeepSeekError(f"practice_questions[{i}] 应为对象或字符串")

        q = str(
            item.get("question")
            or item.get("题目")
            or item.get("stem")
            or ""
        ).strip()
        if not q:
            raise DeepSeekError(f"practice_questions[{i}] 缺少题干 question")

        ans = str(
            item.get("reference_answer")
            or item.get("answer")
            or item.get("参考答案")
            or item.get("reference")
            or ""
        ).strip()
        sol = str(
            item.get("solution_approach")
            or item.get("solution")
            or item.get("解题思路")
            or item.get("explanation")
            or ""
        ).strip()

        out.append(
            {
                "question": q,
                "reference_answer": ans,
                "solution_approach": sol,
            }
        )
    return out


def normalize_practice_type(value: Any) -> str:
    if not isinstance(value, str):
        return "mixed"
    v = value.strip().lower()
    # 兼容前端可能传的大写
    aliases = {
        "mcq": "mcq",
        "short_answer": "short_answer",
        "shortanswer": "short_answer",
        "calculation": "calculation",
        "calc": "calculation",
        "mixed": "mixed",
        "mix": "mixed",
    }
    v = aliases.get(v, v)
    if v in VALID_PRACTICE_TYPES:
        return v
    return "mixed"


def analyze_course_text(
    document_text: str,
    practice_type: str = "mixed",
) -> dict[str, Any]:
    api_key = _read_deepseek_api_key()
    if not api_key:
        raise DeepSeekError("未配置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek。")

    ptype = normalize_practice_type(practice_type)
    type_block = PRACTICE_TYPE_INSTRUCTIONS.get(ptype, PRACTICE_TYPE_INSTRUCTIONS["mixed"])

    trimmed = document_text
    if len(trimmed) > MAX_INPUT_CHARS:
        trimmed = trimmed[:MAX_INPUT_CHARS]

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=300.0,
    )

    user_content = (
        f"{type_block}\n\n"
        f"以下是从讲义中提取的文本（可能已截断至约 {MAX_INPUT_CHARS} 字）：\n\n{trimmed}"
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        raise DeepSeekError(f"调用 DeepSeek 失败：{exc}") from exc

    choice = completion.choices[0].message.content
    if not choice:
        raise DeepSeekError("模型未返回内容。")

    return parse_model_json(choice)
