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
请生成单项选择题。每道题的 question 字段中必须包含：题干段落，以及换行后四个选项，格式严格为（每行一项）：
A. …
B. …
C. …
D. …
除下列公共字段外，每题必须设置 question_format 为 "mcq"，并必须设置 correct_option 为单个字母 A/B/C/D（大写）表示唯一正确选项。
reference_answer 在选项字母之外，可用一两句话解释为何该选项正确；solution_approach 写完整解题思路（排除法或推导）。""",
    "short_answer": """【练习题型：简答题】
每道题 question_format 必须为 "written"，correct_option 必须为 ""。
每道题的 question 为简答设问；reference_answer 为要点式或简短段落式参考答案；solution_approach 说明如何依据讲义组织答案、踩分点。""",
    "calculation": """【练习题型：计算题】
每道题 question_format 必须为 "written"，correct_option 必须为 ""。
每道题的 question 须给出已知与待求；reference_answer 含公式、过程与数值结果（含单位）；solution_approach 说明公式选用与步骤要点。""",
    "mixed": """【练习题型：混合】
请在 practice_questions 中合理搭配至少两种题型。客观单选题对应 question_format 为 "mcq"，须含 A.–D. 四行选项及 correct_option（A–D）；简答、计算等主观题对应 question_format 为 "written"，correct_option 为 ""。
每题均须含 question、reference_answer、solution_approach。""",
}

SYSTEM_PROMPT = """你是面向留学生的课程复习助教。用户会提供课程讲义摘录（语言任意）以及练习题型要求（在用户消息开头）。
你必须全程使用简体中文作答，并只输出一个 JSON 对象，不要 Markdown 围栏、不要前后说明文字。
JSON 的键必须严格为：
- "core_points": 数组。每一项必须是对象，且恰好包含两个键（字符串）：
  - "point": 考点表述，格式严格为「中文考点（English keyword）」：先写简练的中文，紧跟一对中文全角括号，括号内为该考点在学术/课程中常用的英文关键词或短语（首字母大小写符合英语习惯，勿再套一层括号）。示例："知识产权的定义（Definition of Intellectual Property）"、"邻接权（Related Rights）"。不得只写中文不写英文，也不得只有英文没有中文。
  - "importance": 只能是以下三个字面量之一（字符串本身勿加【】等符号）："必考"、"一般"、"了解"。
  【必考】数量与门槛（必须严格遵守）：importance 为 "必考" 的条目总数硬性上限为 5 条，不得超过。在讲义信息量正常时，请将必考控制在 3～5 条：仅保留本门课最核心、期末与平时测验中最高频考查、分值或命题概率明显最高的知识点；若讲义很短、真正够格的核心点不足 3 条，可按实际条数标注必考，禁止为凑数虚标。严禁把大半考点标为 "必考"；其余考点必须降为 "一般" 或 "了解"。
  "一般"=常见考点、值得掌握；"了解"=拓展、背景、次要或低频内容。
- "difficult_analysis": 单个字符串，对重难点做解析，可含多段，使用 \\n 换行；
- "practice_questions": 数组，每一项必须是对象，且必须包含以下字符串键：
  - "question": 题干（格式须符合用户消息中的题型要求；MCQ 须在题干后换行给出 A. B. C. D. 四行选项）；
  - "reference_answer": 参考答案（MCQ 除选项字母外可附简短文字说明）；
  - "solution_approach": 解题思路（关键步骤、知识点、理由；可含多段，用 \\n 换行）；
  - "question_format": 取值为 "mcq" 或 "written"（"mcq"=单选题；"written"=简答、计算等非点击选项题）；
  - "correct_option": 当 question_format 为 "mcq" 时，必须是 "A"、"B"、"C"、"D" 之一；为 "written" 时必须为空字符串 ""。

练习题数量适中（例如 3～8 题），每题必须同时给出上述五键，且内容具体、可核对。

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


def parse_model_json(content: str, practice_type_hint: str = "mixed") -> dict[str, Any]:
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
    hint = normalize_practice_type(practice_type_hint)
    data["practice_questions"] = _normalize_practice_questions(
        data["practice_questions"], hint
    )
    return data


def _normalize_practice_questions(
    items: list[Any], practice_type_hint: str = "mixed"
) -> list[dict[str, str]]:
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
                        "question_format": "written",
                        "correct_option": "",
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

        fmt_raw = str(item.get("question_format") or item.get("format") or "").strip().lower()
        if fmt_raw in ("multiple_choice", "choice"):
            fmt_raw = "mcq"
        if fmt_raw not in ("mcq", "written"):
            if practice_type_hint == "mcq":
                fmt_raw = "mcq"
            else:
                fmt_raw = "written"

        correct = str(item.get("correct_option") or item.get("correct") or "").strip().upper()
        if fmt_raw == "mcq":
            if correct not in ("A", "B", "C", "D"):
                m = re.search(r"\b([ABCD])\b", ans, re.I)
                correct = m.group(1).upper() if m else ""
            if correct not in ("A", "B", "C", "D"):
                fmt_raw = "written"
                correct = ""
        else:
            correct = ""

        out.append(
            {
                "question": q,
                "reference_answer": ans,
                "solution_approach": sol,
                "question_format": fmt_raw,
                "correct_option": correct,
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

    return parse_model_json(choice, practice_type_hint=ptype)
