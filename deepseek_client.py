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
请生成单项选择题。题干、四个选项的正文、reference_answer、solution_approach 均须以**简体中文**为主（见系统对练习题语言的强制规则）；英文仅允许以「中文（English term）」括注专业术语。
每道题的 question 字段中必须包含：题干段落，以及换行后四个选项，格式严格为（每行一项）：
A. …
B. …
C. …
D. …
除下列公共字段外，每题必须设置 question_format 为 "mcq"，并必须设置 correct_option 为单个字母 A/B/C/D（大写）表示唯一正确选项。
reference_answer 在选项字母之外，可用一两句话解释为何该选项正确；solution_approach 写完整解题思路（排除法或推导）。""",
    "short_answer": """【练习题型：简答题】
每道题 question_format 必须为 "written"，correct_option 必须为 ""。
题干、参考答案、解题思路须以简体中文书写（见系统强制语言规则）。
每道题的 question 为简答设问；reference_answer 为要点式或简短段落式参考答案；solution_approach 说明如何依据讲义组织答案、踩分点。""",
    "calculation": """【练习题型：计算题】
每道题 question_format 必须为 "written"，correct_option 必须为 ""。
题干、文字说明、参考答案、解题思路须以简体中文书写（见系统强制语言规则）；公式与变量符号可用常规数学/物理记号。
每道题的 question 须给出已知与待求；reference_answer 含公式、过程与数值结果（含单位）；solution_approach 说明公式选用与步骤要点。""",
    "mixed": """【练习题型：混合】（硬性结构，必须严格遵守）
本批 practice_questions 中**必须同时包含**以下三类，**缺一不可**，且**禁止**整批题目几乎全是同一种题型：
1）**至少一道** question_format 为 "mcq" 的单项选择题（题干 + A.–D. 四行选项 + correct_option）；
2）**至少一道** question_format 为 "written" 的**简答题**（以文字阐述、列举、比较为主，不要求数值推导为主）；
3）**至少一道** question_format 为 "written" 的**计算题**（须含明确已知量与待求量、公式代入或数值运算为主，与纯简答在形式上可区分）。
若总题数为 N，则三类题目的数量宜**相对均衡**（例如 N=6 时 MCQ、简答、计算各约 2 道；允许小幅偏差，但**禁止**出现某一类仅 1 道而其余绝大部分集中在另一类的情况）。
每题均须含 question、reference_answer、solution_approach；全部文字说明须遵守系统对练习题中文主写的强制规则。""",
}

# 换一批练习题时追加到用户消息末尾，约束本批题型分布（与所选题型一致、不畸形集中）
REROLL_DISTRIBUTION_APPEND: dict[str, str] = {
    "mcq": "【本批换题】仍为纯选择题：各题应覆盖讲义中不同知识点或设问角度，避免本批多题高度同质或仅围绕同一结论反复提问；难度与干扰项设计宜有梯度。",
    "short_answer": "【本批换题】仍为纯简答题：各题设问切入点、作答篇幅与得分点结构应有差异，避免多题几乎雷同；勿把本应属于计算推导的主干放在简答里凑数。",
    "calculation": "【本批换题】仍为纯计算题：各题涉及的公式链、未知量类型与数值情境宜多样化，避免本批几乎全部同一套路或仅改数字的重复题。",
    "mixed": "【本批换题】用户选择为**混合题型**：本批仍须**至少含 1 道 MCQ、1 道简答 written、1 道计算 written**（与首次生成规则相同），且三类数量宜**均衡**，禁止本批又倒向「几乎全是选择题」或「几乎全是简答而无计算」等畸形分布；设问须与上一批明显不同。",
}

SYSTEM_PROMPT = """你是面向留学生的课程复习助教。用户会提供课程讲义摘录（语言任意）以及练习题型要求（在用户消息开头）。
你必须全程使用简体中文作答，并只输出一个 JSON 对象，不要 Markdown 围栏、不要前后说明文字。
JSON 的键必须严格为：
- "core_points": 数组。每一项必须是对象，且恰好包含两个键（字符串）：
  - "point": 考点表述，格式严格为「中文考点（English keyword）」：先写简练的中文，紧跟一对中文全角括号，括号内为该考点在学术/课程中常用的英文关键词或短语（首字母大小写符合英语习惯，勿再套一层括号）。示例："知识产权的定义（Definition of Intellectual Property）"、"邻接权（Related Rights）"。不得只写中文不写英文，也不得只有英文没有中文。
  - "importance": 只能是以下三个字面量之一（字符串本身勿加【】等符号）："必考"、"一般"、"了解"。
  【必考】数量与门槛（必须严格遵守）：importance 为 "必考" 的条目总数硬性上限为 5 条，不得超过。在讲义信息量正常时，请将必考控制在 3～5 条：仅保留本门课最核心、期末与平时测验中最高频考查、分值或命题概率明显最高的知识点；若讲义很短、真正够格的核心点不足 3 条，可按实际条数标注必考，禁止为凑数虚标。严禁把大半考点标为 "必考"；其余考点必须降为 "一般" 或 "了解"。
  "一般"=常见考点、值得掌握；"了解"=拓展、背景、次要或低频内容。
- "concept_explanations": 数组，长度必须与 "core_points" 完全一致且顺序一一对应：第 i 条详解对应第 i 个考点。每一项必须是对象，且恰好包含三个字符串键（均用简体中文书写）：
  - "what_it_is": 用最直白、最简单的语言说明「这个概念到底是什么、在讲什么」，假设读者完全没听过课也能读懂；避免堆砌术语，必要时用类比。
  - "formulas_notes": 若该考点涉及公式或符号，请写出公式（可用 LaTeX 风格或纯文本如 E=mc²），并逐项说明每个符号/变量代表什么、常用单位是什么、在题目或应用中如何代入使用；若本考点基本无公式，则写一两句话说明「本概念以定性理解为主」或「无常用公式」即可，勿留空键。
  - "life_example": 举一个贴近日常生活的短例子或小故事，帮助读者建立直觉（一两段即可）。
- "difficult_analysis": 单个字符串，对重难点做解析，可含多段，使用 \\n 换行；
- "practice_questions": 数组，每一项必须是对象，且必须包含以下字符串键：
  - "question": 题干（格式须符合用户消息中的题型要求；MCQ 须在题干后换行给出 A. B. C. D. 四行选项）；
  - "reference_answer": 参考答案（MCQ 除选项字母外可附简短文字说明）；
  - "solution_approach": 解题思路（关键步骤、知识点、理由；可含多段，用 \\n 换行）；
  - "question_format": 取值为 "mcq" 或 "written"（"mcq"=单选题；"written"=简答、计算等非点击选项题）；
  - "correct_option": 当 question_format 为 "mcq" 时，必须是 "A"、"B"、"C"、"D" 之一；为 "written" 时必须为空字符串 ""。

【练习题语言（强制）】practice_questions 中每一题的 question、reference_answer、solution_approach 必须使用**简体中文**进行出题与解析（国际通用计量单位符号如 m、s、N 等可保留）。**禁止**用整句、整段英文来出题或写解析；**仅允许**在确有必要时用「中文术语（English term）」这一对中文全角括号的形式夹注专业英文名词，且括号外主体仍为中文。选择题 A.–D. 各选项的正文同样以中文为主，英文仅限括注术语。化学式、公认数学/物理符号、公式本身不受「必须中文」限制，但其前后文字说明仍须为中文。

练习题数量适中（例如 3～8 题），每题必须同时给出上述五键，且内容具体、可核对。
若用户消息中的题型为**混合（mixed）**，则 practice_questions 须满足该混合说明中的**硬性三类齐全与均衡**要求，不得用「几乎全是同一类」冒充混合。

确保 JSON 合法，字符串内的换行和引号需正确转义。"""

REGENERATE_PRACTICE_PROMPT = """你是面向留学生的课程复习助教。
用户会再次提供**同一份**讲义摘录，且刚刚已完成过一批练习题；你的任务是根据该讲义**重新生成一整批全新的练习题**（设问角度、情境、选项或数值须与常见套路有明显变化，**禁止**仅对上一批题目做同义改写或微调）。
只输出**一个** JSON 对象，不要 Markdown 代码围栏、不要任何前后说明文字。该对象**只包含一个键**：
- "practice_questions": 数组。每一项的结构与主分析接口中的 practice_questions **完全一致**：须含 question、reference_answer、solution_approach、question_format（"mcq" 或 "written"）、correct_option（mcq 时为 A/B/C/D 之一，written 时为 ""）；题型与选项格式须符合用户消息开头的题型说明。

【练习题语言（强制）】须与主分析接口相同：question、reference_answer、solution_approach 一律以**简体中文**出题与解析；英文仅允许「中文（English term）」括注专业术语；禁止整段英文叙述。

【换一批时的题型分布】本批题目须**严格服从**用户消息开头的题型说明；**不得**为求新而偏离用户所选题型。若用户选择**混合题型**：本批仍须含**至少一道 MCQ、至少一道简答、至少一道计算**，且三类数量宜**相对均衡**，禁止整批几乎只剩一种子题型。若用户选择**单一题型**（仅 MCQ / 仅简答 / 仅计算）：本批须**全部为**该题型，但**内部**仍须丰富设问角度与考查点，避免本批多题高度同质、或（在单一题型允许范围内）明显倒向某一固定套路。

练习题数量适中（例如 3～8 题），内容具体可作答。

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


def _empty_concept_row() -> dict[str, str]:
    return {
        "what_it_is": "",
        "formulas_notes": "",
        "life_example": "",
    }


def _normalize_concept_explanations(
    core_len: int, items: Any
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        items = []

    def one(x: Any) -> dict[str, str]:
        if not isinstance(x, dict):
            return _empty_concept_row()
        return {
            "what_it_is": str(
                x.get("what_it_is")
                or x.get("plain_explanation")
                or x.get("是什么")
                or ""
            ).strip(),
            "formulas_notes": str(
                x.get("formulas_notes")
                or x.get("formulas_and_variables")
                or x.get("公式")
                or ""
            ).strip(),
            "life_example": str(
                x.get("life_example")
                or x.get("everyday_example")
                or x.get("例子")
                or ""
            ).strip(),
        }

    out: list[dict[str, str]] = [one(items[i]) for i in range(min(len(items), core_len))]
    while len(out) < core_len:
        out.append(_empty_concept_row())
    return out[:core_len]


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

    for key in (
        "core_points",
        "concept_explanations",
        "difficult_analysis",
        "practice_questions",
    ):
        if key not in data:
            raise DeepSeekError(f"模型返回缺少字段：{key}")

    if not isinstance(data["core_points"], list):
        raise DeepSeekError("core_points 应为数组")
    if not isinstance(data["concept_explanations"], list):
        raise DeepSeekError("concept_explanations 应为数组")
    if not isinstance(data["difficult_analysis"], str):
        raise DeepSeekError("difficult_analysis 应为字符串")
    if not isinstance(data["practice_questions"], list):
        raise DeepSeekError("practice_questions 应为数组")

    data["core_points"] = _normalize_core_points(data["core_points"])
    data["concept_explanations"] = _normalize_concept_explanations(
        len(data["core_points"]),
        data["concept_explanations"],
    )
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


def parse_practice_questions_only(content: str, practice_type_hint: str) -> list[dict[str, str]]:
    """解析仅含 practice_questions 的模型 JSON（或根为数组）。"""
    raw = _strip_code_fence(content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end <= start:
                raise DeepSeekError("模型返回不是合法 JSON，请稍后重试。")
            try:
                arr = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise DeepSeekError("模型返回不是合法 JSON，请稍后重试。") from exc
            if not isinstance(arr, list):
                raise DeepSeekError("根数组格式错误。")
            data = {"practice_questions": arr}
        else:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError as exc:
                raise DeepSeekError("模型返回不是合法 JSON，请稍后重试。") from exc

    if isinstance(data, list):
        data = {"practice_questions": data}

    if "practice_questions" not in data:
        raise DeepSeekError("模型返回缺少字段：practice_questions")
    if not isinstance(data["practice_questions"], list):
        raise DeepSeekError("practice_questions 应为数组")

    hint = normalize_practice_type(practice_type_hint)
    return _normalize_practice_questions(data["practice_questions"], hint)


def regenerate_practice_questions(document_text: str, practice_type: str = "mixed") -> list[dict[str, str]]:
    """同一讲义 + 同一题型下，重新生成一批练习题。"""
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

    reroll_append = REROLL_DISTRIBUTION_APPEND.get(ptype, REROLL_DISTRIBUTION_APPEND["mixed"])

    user_content = (
        f"{type_block}\n\n"
        "【任务】上一批练习题已完成。请仅根据下列同一讲义内容，输出**全新一批**练习题（JSON 仅含 practice_questions 键）。\n\n"
        f"{reroll_append}\n\n"
        f"讲义文本（可能已截断至约 {MAX_INPUT_CHARS} 字）：\n\n{trimmed}"
    )

    try:
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": REGENERATE_PRACTICE_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.55,
        )
    except Exception as exc:
        raise DeepSeekError(f"调用 DeepSeek 失败：{exc}") from exc

    choice = completion.choices[0].message.content
    if not choice:
        raise DeepSeekError("模型未返回内容。")

    return parse_practice_questions_only(choice, practice_type_hint=ptype)


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
