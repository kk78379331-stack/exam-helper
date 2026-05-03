from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


class DeepSeekError(Exception):
    pass


MAX_INPUT_CHARS = 120_000


def _read_deepseek_api_key() -> str:
    """从环境变量读取 Key，兼容 UTF-8 BOM、首尾空白与引号包裹。"""
    raw = os.environ.get("DEEPSEEK_API_KEY", "") or ""
    key = raw.strip().lstrip("\ufeff")
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    return key

SYSTEM_PROMPT = """你是面向留学生的课程复习助教。用户会提供课程讲义摘录（语言任意）。
你必须全程使用简体中文作答，并只输出一个 JSON 对象，不要 Markdown 围栏、不要前后说明文字。
JSON 的键必须严格为：
- "core_points": 字符串数组，列出核心考点，条目清晰简练；
- "difficult_analysis": 单个字符串，对重难点做解析，可含多段，使用 \\n 换行；
- "practice_questions": 数组，每一项必须是对象，且恰好包含三个键（字符串）：
  - "question": 题干，与讲义风格一致；
  - "reference_answer": 该题的参考答案（结论、要点或完整作答，视题型而定）；
  - "solution_approach": 解题思路（关键步骤、所用知识点、为何如此做；可含多段，用 \\n 换行）。

练习题数量适中（例如 3～8 题），每题必须同时给出 question、reference_answer、solution_approach，且内容具体、可核对。

确保 JSON 合法，字符串内的换行和引号需正确转义。"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if m:
        return m.group(1).strip()
    return text


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

    data["core_points"] = [str(x).strip() for x in data["core_points"] if str(x).strip()]
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


def analyze_course_text(document_text: str) -> dict[str, Any]:
    api_key = _read_deepseek_api_key()
    if not api_key:
        raise DeepSeekError("未配置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek。")

    trimmed = document_text
    if len(trimmed) > MAX_INPUT_CHARS:
        trimmed = trimmed[:MAX_INPUT_CHARS]

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=300.0,
    )

    user_content = f"以下是从讲义中提取的文本（可能已截断至约 {MAX_INPUT_CHARS} 字）：\n\n{trimmed}"

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
