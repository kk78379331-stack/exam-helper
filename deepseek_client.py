from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI


class DeepSeekError(Exception):
    pass


MAX_INPUT_CHARS = 120_000

# 讲义 + 可选复习提纲合并为单段用户材料时的固定片段（长度计入 MAX_INPUT_CHARS）
_MERGE_OUTLINE_PRIORITY = (
    "【分析指令】请优先根据下方「复习提纲」段落判断各 core_points 的 star_rating："
    "提纲中出现、着重强调、列为重点或反复展开的内容应优先标注为高星级（仍须遵守本系统对五星考点总数上限等全部 JSON 硬性规则）；"
    "「课程讲义」段落用于补充与核对知识点。若提纲与讲义在「考频判断」上侧重不同，以提纲为准确定星级。\n\n"
)
_OUTLINE_BLOCK_HEAD = "【复习提纲（教师提供）】\n\n"
_MAIN_BLOCK_HEAD = "\n\n【课程讲义】\n\n"


def merge_material_with_outline(main: str, outline: str | None) -> tuple[str, bool]:
    """将讲义与可选提纲合并为一条材料字符串，总长度不超过 MAX_INPUT_CHARS。

    返回 (merged_text, truncated) ；truncated 表示任一段因长度被截断。
    """
    main_st = (main or "").strip()
    o_st = (str(outline).strip() if outline else "") or ""
    if not o_st:
        if len(main_st) > MAX_INPUT_CHARS:
            return main_st[:MAX_INPUT_CHARS], True
        return main_st, False

    fixed = len(_MERGE_OUTLINE_PRIORITY) + len(_OUTLINE_BLOCK_HEAD) + len(_MAIN_BLOCK_HEAD)
    avail = MAX_INPUT_CHARS - fixed
    if avail < 2000:
        avail = 2000

    max_o = min(len(o_st), avail * 45 // 100)
    max_m = avail - max_o
    if max_m < 8000:
        max_m = min(avail, 8000)
        max_o = max(0, avail - max_m)
        max_o = min(len(o_st), max_o)
    o_trim = o_st[:max_o]
    m_trim = main_st[:max_m]
    truncated = len(o_st) > len(o_trim) or len(main_st) > len(m_trim)
    merged = _MERGE_OUTLINE_PRIORITY + _OUTLINE_BLOCK_HEAD + o_trim + _MAIN_BLOCK_HEAD + m_trim
    if len(merged) > MAX_INPUT_CHARS:
        merged = merged[:MAX_INPUT_CHARS]
        truncated = True
    return merged, truncated

VALID_PRACTICE_TYPES = frozenset(
    {"mcq", "true_false", "short_answer", "calculation", "mixed"}
)

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
请生成单项选择题，**本批题目总数不得超过 10 道**。
题干、四个选项的正文、reference_answer、solution_approach 均须以**简体中文**为主（见系统对练习题语言的强制规则）；英文仅允许以「中文（English term）」括注专业术语。
每道题的 question 字段中必须包含：题干段落，以及换行后四个选项，格式严格为（每行一项）：
A. …
B. …
C. …
D. …
除下列公共字段外，每题必须设置 question_format 为 "mcq"，并必须设置 correct_option 为单个字母 A/B/C/D（大写）表示唯一正确选项。
reference_answer 在选项字母之外，可用一两句话解释为何该选项正确；solution_approach 写完整解题思路（排除法或推导）。""",
    "true_false": """【练习题型：判断题 True/False】
请生成判断题，**本批题目总数不得超过 10 道**。
每道题 question_format 必须为 "tf"；correct_option 必须为字符串 "TRUE" 或 "FALSE"（大写），表示唯一正确答案。
question 为判断陈述（可含多句，用 \\n 换行）；**不要**在 question 里再写 True/False 选项行（前端固定展示 True / False 两个按钮）。
reference_answer 用一两句话说明为何为真或为假；solution_approach 写完整辨析思路。
题干、参考答案、解题思路须以简体中文书写（见系统强制语言规则）。""",
    "short_answer": """【练习题型：简答题】
请生成简答题，**本批题目总数不得超过 10 道**。
每道题 question_format 必须为 "short_answer"，correct_option 必须为 ""。
题干、参考答案、解题思路须以简体中文书写（见系统强制语言规则）。
每道题的 question 为简答设问；reference_answer 为要点式或简短段落式参考答案；solution_approach 说明如何依据讲义组织答案、踩分点。""",
    "calculation": """【练习题型：计算题】
请生成计算题，**本批题目总数不得超过 10 道**。
每道题 question_format 必须为 "calculation"，correct_option 必须为 ""。
题干、文字说明、参考答案、解题思路须以简体中文书写（见系统强制语言规则）；公式与变量符号可用常规数学/物理记号。
每道题的 question 须给出已知与待求；reference_answer 含公式、过程与数值结果（含单位）；solution_approach 说明公式选用与步骤要点。""",
    "mixed": """【练习题型：混合】（硬性结构，必须严格遵守）
本批 practice_questions **总题数 N 须满足 4 ≤ N ≤ 10**，**不得超过 10 道**。
须**同时包含以下四类子题型，每一类至少 1 道**，缺一不可；禁止整批几乎只剩某一类：
1）question_format 为 "mcq" 的单项选择题（题干 + 换行后 A.–D. 四行选项 + correct_option 为 A/B/C/D）；
2）question_format 为 "tf" 的判断题（题干为陈述；correct_option 为 "TRUE" 或 "FALSE"；不要在题干中再写 True/False 选项行）；
3）question_format 为 "short_answer" 的简答题（以阐述、列举、比较为主，不以数值推导为主）；
4）question_format 为 "calculation" 的计算题（须含明确已知量与待求量、公式或数值运算为主，与简答题可区分）。
在 N 允许范围内四类数量宜**相对均衡**（例如 N=8 时每类约 2 道）；**禁止**某一类仅 1 道而其余绝大部分挤在另一类。
每题均须含 question、reference_answer、solution_approach；全部文字说明须遵守系统对练习题中文主写的强制规则。""",
}

# 换一批练习题时追加到用户消息末尾，约束本批题型分布（与所选题型一致、不畸形集中）
REROLL_DISTRIBUTION_APPEND: dict[str, str] = {
    "mcq": "【本批换题】仍为纯选择题：题目数不超过 10；各题应覆盖讲义中不同知识点或设问角度，避免本批多题高度同质或仅围绕同一结论反复提问；难度与干扰项设计宜有梯度。",
    "true_false": "【本批换题】仍为纯判断题：题目数不超过 10；各题陈述角度、知识点应有差异，避免仅改否定词或同义反复。",
    "short_answer": "【本批换题】仍为纯简答题：题目数不超过 10；各题设问切入点、作答篇幅与得分点结构应有差异，避免多题几乎雷同。",
    "calculation": "【本批换题】仍为纯计算题：题目数不超过 10；各题涉及的公式链、未知量类型与数值情境宜多样化，避免本批几乎全部同一套路或仅改数字的重复题。",
    "mixed": "【本批换题】用户选择为**混合题型**：本批仍须满足**至少 1 道 mcq、1 道 tf、1 道 short_answer、1 道 calculation**，总题数 4～10 道且不超过 10，四类数量宜均衡；设问须与上一批明显不同。",
}

SYSTEM_PROMPT = """你是面向留学生的课程复习助教。用户会提供课程讲义摘录（语言任意）以及练习题型要求（在用户消息开头）。
你必须全程使用简体中文作答，并只输出一个 JSON 对象，不要 Markdown 围栏、不要前后说明文字。
JSON 的键必须严格为：
- "core_points": 数组。每一项必须是对象，且恰好包含两个键：
  - "point": 考点表述，格式严格为「中文考点（English keyword）」：先写简练的中文，紧跟一对中文全角括号，括号内为该考点在学术/课程中常用的英文关键词或短语（首字母大小写符合英语习惯，勿再套一层括号）。示例："知识产权的定义（Definition of Intellectual Property）"、"邻接权（Related Rights）"。不得只写中文不写英文，也不得只有英文没有中文。
  - "star_rating": 整数 1～5，表示该考点在考试中的相对重要程度（须与下列星级含义一致，且与讲义内容匹配）：
    5 = 核心必考，每次考试几乎必出现；4 = 高频考点，应重点掌握；3 = 中等重要，偶尔出题；2 = 了解即可，较少出题；1 = 背景知识，基本不考。
  【五星考点数量】star_rating 为 5 的条目总数**硬性不得超过 5 条**；信息量正常时建议 3～5 条五星；严禁把大半考点都标为 5 星；其余考点应诚实降为 4～1 星。
  【复习提纲优先】若用户消息中同时包含「【复习提纲（教师提供）】」与「【课程讲义】」所引导的两段材料：判断 star_rating 时**必须优先依据复习提纲**中体现的重点、结构与表述；提纲中反复出现、明示「重点」「必考」「掌握」或占篇幅显著的内容，应**倾向标注更高星级**（仍须遵守五星考点总数上限等全部硬性规则）。课程讲义用于穷尽知识点与核对事实；若提纲与讲义在「考频判断」上侧重不同，**以提纲为准**确定星级，但不得编造与讲义明显矛盾的事实。
- "concept_explanations": 数组，长度必须与 "core_points" 完全一致且顺序一一对应：第 i 条详解对应第 i 个考点。每一项必须是对象，且恰好包含三个字符串键（均用简体中文书写）：
  - "what_it_is": 用最直白、最简单的语言说明「这个概念到底是什么、在讲什么」，假设读者完全没听过课也能读懂；避免堆砌术语，必要时用类比。
  - "formulas_notes": 若该考点涉及公式或符号，请写出公式（可用 LaTeX 风格或纯文本如 E=mc²），并逐项说明每个符号/变量代表什么、常用单位是什么、在题目或应用中如何代入使用；若本考点基本无公式，则写一两句话说明「本概念以定性理解为主」或「无常用公式」即可，勿留空键。
  - "life_example": 举一个贴近日常生活的短例子或小故事，帮助读者建立直觉（一两段即可）。
- "difficult_analysis": 单个字符串，写本课**主干重难点解析**（条理清晰，可含多段，使用 \\n 换行）；勿把下列误区与对比混写进本字段。
- "common_misconceptions": **字符串数组**，**恰好 2～3 条**（不得少于 2、不得多于 3），每条用一句话概括学生**最容易犯的典型错误**或**理解误区**（表述简洁、可独立阅读）。
- "concept_comparison": 单个字符串，写 **1～2 组**本课中**易混淆的相似概念、术语或情形**之间的**核心区别**（可分点、可换行）；若讲义中可对比内容不足，也需基于材料做合理归纳，勿留空。
- "practice_questions": 数组，每一项必须是对象，且必须包含以下键：
  - "question": 题干（格式须符合用户消息中的题型要求；MCQ 须在题干后换行给出 A. B. C. D. 四行选项；判断题 tf 仅写陈述不写 True/False 选项行）；
  - "reference_answer": 参考答案（MCQ/tf 可附简短文字说明）；
  - "solution_approach": 解题思路（关键步骤、知识点、理由；可含多段，用 \\n 换行）；
  - "question_format": 取值为 "mcq"、"tf"、"short_answer" 或 "calculation"（"mcq"=四选一；"tf"=判断题；"short_answer"=简答；"calculation"=计算）；
  - "correct_option": 当 question_format 为 "mcq" 时，必须是 "A"、"B"、"C"、"D" 之一；为 "tf" 时必须是 "TRUE" 或 "FALSE"；为 "short_answer" 或 "calculation" 时必须为空字符串 ""。
  - "related_core_point_indices": **整数数组**（可为空但不得省略键）：本题考查到的「core_points」中的考点**下标**（从 0 开始），须与题干内容一致，可含多个下标；下标必须在 0 ～ len(core_points)-1 范围内。每道题必须显式列出，便于前端展示「考察知识点」。

【练习题语言（强制）】practice_questions 中每一题的 question、reference_answer、solution_approach 必须使用**简体中文**进行出题与解析（国际通用计量单位符号如 m、s、N 等可保留）。**禁止**用整句、整段英文来出题或写解析；**仅允许**在确有必要时用「中文术语（English term）」这一对中文全角括号的形式夹注专业英文名词，且括号外主体仍为中文。选择题 A.–D. 各选项的正文同样以中文为主，英文仅限括注术语。化学式、公认数学/物理符号、公式本身不受「必须中文」限制，但其前后文字说明仍须为中文。

**练习题数量（硬性）**：practice_questions 数组长度**不得超过 10**；非混合题型时宜 3～10 道。若用户消息中的题型为**混合（mixed）**，则还须满足混合说明中的**四类各至少 1 道、总数 4～10 道**等全部硬性要求；其他单一题型时须符合该题型说明中的数量与结构要求。

确保 JSON 合法，字符串内的换行和引号需正确转义。"""

REGENERATE_PRACTICE_PROMPT = """你是面向留学生的课程复习助教。
用户会再次提供**同一份**讲义摘录，且刚刚已完成过一批练习题；你的任务是根据该讲义**重新生成一整批全新的练习题**（设问角度、情境、选项或数值须与常见套路有明显变化，**禁止**仅对上一批题目做同义改写或微调）。
只输出**一个** JSON 对象，不要 Markdown 代码围栏、不要任何前后说明文字。该对象**只包含一个键**：
- "practice_questions": 数组。每一项的结构与主分析接口中的 practice_questions **完全一致**：须含 question、reference_answer、solution_approach、question_format、correct_option、**related_core_point_indices**（整数数组，考点在 core_points 中的下标）；题型须符合用户消息开头的题型说明。

【练习题语言（强制）】须与主分析接口相同：question、reference_answer、solution_approach 一律以**简体中文**出题与解析；英文仅允许「中文（English term）」括注专业术语；禁止整段英文叙述。

【换一批时的题型分布】本批题目须**严格服从**用户消息开头的题型说明；**不得**为求新而偏离用户所选题型。**题目总数不得超过 10 道。**若用户选择**混合题型**：本批仍须含**至少 1 道 mcq、1 道 tf、1 道 short_answer、1 道 calculation**，四类宜均衡。若用户选择**单一题型**：本批须全部为该题型，且设问角度多样化。

内容具体可作答。

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
    """输出每项含 point、star_rating（字符串 "1"～"5"）。兼容旧版 importance。"""
    out: list[dict[str, str]] = []
    five_star = 0
    for i, item in enumerate(items):
        if isinstance(item, str):
            t = item.strip()
            if t:
                out.append({"point": t, "star_rating": "3"})
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

        star: int | None = None
        sr = item.get("star_rating")
        if sr is not None and str(sr).strip() != "":
            try:
                star = int(float(str(sr).strip()))
            except (TypeError, ValueError):
                star = None
        if star is None:
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
            star = {"必考": 5, "一般": 3, "了解": 2}[imp_clean]
        if star < 1:
            star = 1
        if star > 5:
            star = 5
        if star == 5:
            five_star += 1
            if five_star > 5:
                star = 4

        out.append({"point": point, "star_rating": str(star)})
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


def _normalize_common_misconceptions(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [x.strip() for x in re.split(r"[\n；;]+", raw) if x.strip()]
        return parts[:5]
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()][:5]


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

    if "common_misconceptions" not in data:
        data["common_misconceptions"] = []
    elif not isinstance(data["common_misconceptions"], list):
        raise DeepSeekError("common_misconceptions 应为数组")
    if "concept_comparison" not in data:
        data["concept_comparison"] = ""
    elif not isinstance(data["concept_comparison"], str):
        data["concept_comparison"] = str(data["concept_comparison"])

    data["core_points"] = _normalize_core_points(data["core_points"])
    data["concept_explanations"] = _normalize_concept_explanations(
        len(data["core_points"]),
        data["concept_explanations"],
    )
    data["difficult_analysis"] = str(data["difficult_analysis"]).strip()
    data["common_misconceptions"] = _normalize_common_misconceptions(
        data.get("common_misconceptions")
    )
    data["concept_comparison"] = str(
        data.get("concept_comparison") or data.get("concept_contrast") or ""
    ).strip()
    hint = normalize_practice_type(practice_type_hint)
    data["practice_questions"] = _normalize_practice_questions(
        data["practice_questions"], hint, len(data["core_points"])
    )
    return data


def _normalize_related_indices(raw: Any, core_points_len: int, i: int) -> str:
    """返回逗号分隔的下标字符串；core_points_len<=0 时仅做非负过滤与去重。"""
    idxs: list[int] = []
    if isinstance(raw, list):
        for x in raw:
            try:
                idxs.append(int(x))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, str) and raw.strip():
        for p in re.split(r"[,，\s]+", raw.strip()):
            try:
                idxs.append(int(p))
            except ValueError:
                continue
    out: list[int] = []
    seen: set[int] = set()
    upper = core_points_len - 1 if core_points_len > 0 else 10**6
    for j in idxs:
        if j < 0:
            continue
        if core_points_len > 0 and j > upper:
            continue
        if j not in seen:
            seen.add(j)
            out.append(j)
    return ",".join(str(x) for x in out)


def _normalize_practice_questions(
    items: list[Any], practice_type_hint: str = "mixed", core_points_len: int = 0
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
                        "question_format": "short_answer",
                        "correct_option": "",
                        "related_core_point_indices": "",
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
        if fmt_raw in ("true_false", "tf", "判断", "判断题"):
            fmt_raw = "tf"
        if fmt_raw in ("short", "shortanswer", "short_answer", "简答"):
            fmt_raw = "short_answer"
        if fmt_raw in ("calc", "calculation", "计算"):
            fmt_raw = "calculation"
        if fmt_raw == "written":
            fmt_raw = "written"  # resolve below

        if fmt_raw not in ("mcq", "tf", "short_answer", "calculation", "written"):
            if practice_type_hint == "mcq":
                fmt_raw = "mcq"
            elif practice_type_hint == "true_false":
                fmt_raw = "tf"
            elif practice_type_hint == "short_answer":
                fmt_raw = "short_answer"
            elif practice_type_hint == "calculation":
                fmt_raw = "calculation"
            else:
                fmt_raw = "written"

        if fmt_raw == "written":
            if practice_type_hint == "calculation":
                fmt_raw = "calculation"
            elif practice_type_hint == "short_answer":
                fmt_raw = "short_answer"
            elif practice_type_hint == "mixed":
                kind = str(
                    item.get("exercise_kind")
                    or item.get("sub_type")
                    or item.get("题型")
                    or ""
                ).strip().lower()
                if kind in ("calc", "calculation", "计算", "计算题"):
                    fmt_raw = "calculation"
                elif kind in ("short", "short_answer", "简答", "简答题"):
                    fmt_raw = "short_answer"
                else:
                    fmt_raw = "short_answer"
            else:
                fmt_raw = "short_answer"

        correct = str(item.get("correct_option") or item.get("correct") or "").strip()
        correct_upper = correct.upper()

        if fmt_raw == "mcq":
            cu = correct_upper
            if cu not in ("A", "B", "C", "D"):
                m = re.search(r"\b([ABCD])\b", ans, re.I)
                cu = m.group(1).upper() if m else ""
            if cu not in ("A", "B", "C", "D"):
                fmt_raw = "short_answer"
                correct = ""
            else:
                correct = cu
        elif fmt_raw == "tf":
            cu = correct_upper.replace(" ", "")
            if cu in ("T", "TRUE", "对", "是"):
                correct = "TRUE"
            elif cu in ("F", "FALSE", "错", "否"):
                correct = "FALSE"
            else:
                m = re.search(r"\b(TRUE|FALSE)\b", ans, re.I)
                correct = m.group(1).upper() if m else ""
            if correct not in ("TRUE", "FALSE"):
                fmt_raw = "short_answer"
                correct = ""
        else:
            correct = ""

        rel = _normalize_related_indices(
            item.get("related_core_point_indices")
            or item.get("related_points")
            or item.get("core_point_indices"),
            core_points_len,
            i,
        )

        out.append(
            {
                "question": q,
                "reference_answer": ans,
                "solution_approach": sol,
                "question_format": fmt_raw,
                "correct_option": correct,
                "related_core_point_indices": rel,
            }
        )

    _validate_practice_count(out)
    if practice_type_hint == "mixed":
        _validate_mixed_practice_questions(out)
    return out


def parse_practice_questions_only(
    content: str, practice_type_hint: str, core_points_len: int = 0
) -> list[dict[str, str]]:
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
    return _normalize_practice_questions(data["practice_questions"], hint, core_points_len)


def regenerate_practice_questions(
    document_text: str, practice_type: str = "mixed", core_points_count: int = 0
) -> list[dict[str, str]]:
    """同一讲义 + 同一题型下，重新生成一批练习题。document_text 可为已在路由层合并提纲后的全文。"""
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

    idx_hint = ""
    if isinstance(core_points_count, int) and core_points_count > 0:
        idx_hint = (
            f"\n【考点下标】与当前分析一致的核心考点共 **{core_points_count}** 条，下标范围为 **0～{core_points_count - 1}**。"
            "每道题必须输出 **related_core_point_indices**（整数数组），且每个下标均须落在上述范围内。\n"
        )

    if "【复习提纲" in trimmed:
        mat_label = (
            "材料全文（含复习提纲与课程讲义；可能已截断至约 "
            f"{MAX_INPUT_CHARS} 字）"
        )
    else:
        mat_label = f"讲义文本（可能已截断至约 {MAX_INPUT_CHARS} 字）"

    user_content = (
        f"{type_block}\n\n"
        "【任务】上一批练习题已完成。请仅根据下列同一讲义内容，输出**全新一批**练习题（JSON 仅含 practice_questions 键）。\n\n"
        f"{reroll_append}{idx_hint}\n"
        f"{mat_label}：\n\n{trimmed}"
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

    return parse_practice_questions_only(
        choice, practice_type_hint=ptype, core_points_len=max(0, int(core_points_count or 0))
    )


def normalize_practice_type(value: Any) -> str:
    if not isinstance(value, str):
        return "mixed"
    v = value.strip().lower()
    # 兼容前端可能传的大写
    aliases = {
        "mcq": "mcq",
        "true_false": "true_false",
        "truefalse": "true_false",
        "tf": "true_false",
        "判断": "true_false",
        "判断题": "true_false",
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


def _validate_practice_count(items: list[dict[str, str]]) -> None:
    if len(items) > 10:
        raise DeepSeekError("练习题超过 10 道上限，请重试。")


def _validate_mixed_practice_questions(items: list[dict[str, str]]) -> None:
    n_mcq = n_tf = n_short = n_calc = 0
    for it in items:
        f = str(it.get("question_format", "")).strip().lower()
        if f == "mcq":
            n_mcq += 1
        elif f == "tf":
            n_tf += 1
        elif f == "short_answer":
            n_short += 1
        elif f == "calculation":
            n_calc += 1
    if n_mcq < 1 or n_tf < 1 or n_short < 1 or n_calc < 1:
        raise DeepSeekError(
            "混合题型须至少包含 1 道 mcq、1 道 tf、1 道 short_answer、1 道 calculation；"
            "当前批次不满足，请重试。"
        )


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

    if "【复习提纲" in trimmed:
        intro = (
            f"以下为用户提供的材料（含「复习提纲」与「课程讲义」两段；可能已截断至约 {MAX_INPUT_CHARS} 字）："
        )
    else:
        intro = f"以下是从课程讲义中提取的文本（可能已截断至约 {MAX_INPUT_CHARS} 字）："

    user_content = f"{type_block}\n\n{intro}\n\n{trimmed}"

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
