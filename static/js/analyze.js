/**
 * 浏览器内提取 PDF / PPTX 文字，仅将纯文本 POST 到后端（适配 Vercel 等小请求体限制）。
 */
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";
import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.mjs";

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function decodeXmlText(s) {
  if (!s) return "";
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number.parseInt(n, 10)))
    .replace(/&#x([\da-fA-F]+);/g, (_, h) => String.fromCharCode(Number.parseInt(h, 16)));
}

async function extractPdfText(arrayBuffer) {
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const parts = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const line = content.items.map((it) => it.str).join("");
    parts.push(line);
  }
  return parts.join("\n\n").trim();
}

async function extractPptxText(arrayBuffer) {
  const zip = await JSZip.loadAsync(arrayBuffer);
  const names = Object.keys(zip.files).filter((p) =>
    /^ppt\/slides\/slide\d+\.xml$/i.test(p)
  );
  names.sort((a, b) => {
    const na = parseInt(a.match(/slide(\d+)/i)?.[1] || "0", 10);
    const nb = parseInt(b.match(/slide(\d+)/i)?.[1] || "0", 10);
    return na - nb;
  });
  const chunks = [];
  for (const path of names) {
    const xml = await zip.file(path).async("string");
    const re = /<a:t>([^<]*)<\/a:t>/g;
    let m;
    while ((m = re.exec(xml)) !== null) {
      const t = decodeXmlText(m[1]).trim();
      if (t) chunks.push(t);
    }
  }
  return chunks.join("\n").trim();
}

function extOf(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

function showFlash(category, message) {
  const ul = document.getElementById("flash-list");
  if (!ul) return;
  ul.innerHTML = "";
  const li = document.createElement("li");
  li.className = `flash flash--${category}`;
  li.textContent = message;
  ul.appendChild(li);
}

function clearFlash() {
  const ul = document.getElementById("flash-list");
  if (ul) ul.innerHTML = "";
}

const HISTORY_STORAGE_KEY = "exam-review-helper-history-v1";
const HISTORY_MAX_ITEMS = 80;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const data = JSON.parse(raw);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
}

function appendHistoryRecord({ source_name, practice_type, text_truncated, analysis }) {
  if (!analysis || typeof analysis !== "object") return;
  const list = loadHistory();
  const entry = {
    id:
      typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
    savedAt: new Date().toISOString(),
    source_name: source_name || "（未命名）",
    practice_type: practice_type || "mixed",
    text_truncated: !!text_truncated,
    analysis: JSON.parse(JSON.stringify(analysis)),
  };
  list.unshift(entry);
  saveHistory(list.slice(0, HISTORY_MAX_ITEMS));
}

function deleteHistoryItem(id) {
  saveHistory(loadHistory().filter((x) => x.id !== id));
}

function clearAllHistory() {
  localStorage.removeItem(HISTORY_STORAGE_KEY);
}

function openHistoryPanel() {
  const p = document.getElementById("history-panel");
  if (!p) return;
  p.hidden = false;
  p.setAttribute("aria-hidden", "false");
  renderHistoryList();
}

function closeHistoryPanel() {
  const p = document.getElementById("history-panel");
  if (!p) return;
  p.hidden = true;
  p.setAttribute("aria-hidden", "true");
}

function renderHistoryList() {
  const ul = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  if (!ul || !empty) return;
  const items = loadHistory();
  ul.innerHTML = "";
  empty.hidden = items.length > 0;
  for (const item of items) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.dataset.id = item.id;
    const t = new Date(item.savedAt);
    const timeStr = Number.isNaN(t.getTime())
      ? ""
      : t.toLocaleString("zh-CN", { dateStyle: "short", timeStyle: "short" });
    li.innerHTML =
      `<button type="button" class="history-item__main" data-action="open-history">` +
      `<span class="history-item__name">${escapeHtml(item.source_name || "（未命名）")}</span>` +
      `<span class="history-item__time">${escapeHtml(timeStr)}</span>` +
      `</button>` +
      `<button type="button" class="history-item__delete" data-action="delete-history" aria-label="删除此条">删除</button>`;
    ul.appendChild(li);
  }
}

function normalizePracticeRow(raw) {
  if (typeof raw === "string") {
    return {
      question: raw,
      reference_answer: "",
      solution_approach: "",
      question_format: "written",
      correct_option: "",
    };
  }
  return {
    question: raw.question || "",
    reference_answer: raw.reference_answer || "",
    solution_approach: raw.solution_approach || "",
    question_format: String(raw.question_format || "written").toLowerCase(),
    correct_option: raw.correct_option || "",
  };
}

/** 从题干+选项文本中解析 MCQ；失败返回 null */
function parseMcqBlocks(questionText) {
  const lines = questionText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const options = [];
  const stemLines = [];
  for (const line of lines) {
    const m = line.match(/^([A-Da-d])\s*[\.\．、:：\)]\s*(.+)$/);
    if (m) {
      options.push({ key: m[1].toUpperCase(), text: m[2].trim() });
    } else if (options.length === 0) {
      stemLines.push(line);
    }
  }
  if (options.length < 2) return null;
  return { stem: stemLines.join("\n"), options };
}

function normalizeCorrectOption(referenceAnswer, correctOption) {
  let c = String(correctOption || "").trim().toUpperCase();
  if (/^[ABCD]$/.test(c)) return c;
  const m = String(referenceAnswer || "").match(/\b([ABCD])\b/i);
  return m ? m[1].toUpperCase() : "";
}

function shouldUseMcqInteractive(row, practiceType) {
  const correct = normalizeCorrectOption(row.reference_answer, row.correct_option);
  const parsed = parseMcqBlocks(row.question || "");
  if (!correct || !parsed) return false;
  const fmt = (row.question_format || "").toLowerCase();
  if (fmt === "written") return false;
  if (fmt === "mcq") return true;
  return practiceType === "mcq";
}

function buildPracticeItemHtml(row, practiceType) {
  const ansHtml = row.reference_answer
    ? escapeHtml(row.reference_answer).replace(/\n/g, "<br />")
    : "（暂无）";
  const solHtml = row.solution_approach
    ? escapeHtml(row.solution_approach).replace(/\n/g, "<br />")
    : "（暂无）";

  if (!shouldUseMcqInteractive(row, practiceType)) {
    return (
      `<li class="practice-item practice-item--written">` +
      `<div class="practice-item__stem">${escapeHtml(row.question)}</div>` +
      `<button type="button" class="btn btn--answer js-toggle-answer" aria-expanded="false">查看答案</button>` +
      `<div class="practice-item__detail" hidden>` +
      `<p class="practice-item__label">参考答案</p>` +
      `<div class="practice-item__body">${ansHtml}</div>` +
      `<p class="practice-item__label">解题思路</p>` +
      `<div class="practice-item__body">${solHtml}</div>` +
      `</div></li>`
    );
  }

  const correct = normalizeCorrectOption(row.reference_answer, row.correct_option);
  const parsed = parseMcqBlocks(row.question);
  const optsHtml = parsed.options
    .map(
      (o) =>
        `<li><button type="button" class="mcq-option" data-key="${o.key}" aria-pressed="false">` +
        `<span class="mcq-option__key">${o.key}.</span>` +
        `<span class="mcq-option__text">${escapeHtml(o.text)}</span>` +
        `</button></li>`
    )
    .join("");

  return (
    `<li class="practice-item practice-item--mcq" data-correct="${correct}">` +
    `<div class="practice-item__stem">${escapeHtml(parsed.stem)}</div>` +
    `<ul class="mcq-options" role="list">${optsHtml}</ul>` +
    `<button type="button" class="btn btn--confirm js-mcq-confirm">确认答案</button>` +
    `<div class="practice-item__mcq-feedback" hidden>` +
    `<p class="mcq-feedback__summary"></p>` +
    `<p class="practice-item__label">参考答案</p>` +
    `<div class="practice-item__body">${ansHtml}</div>` +
    `<p class="practice-item__label">解题思路</p>` +
    `<div class="practice-item__body">${solHtml}</div>` +
    `</div></li>`
  );
}

function resetToUploadState() {
  clearFlash();
  const toolbar = document.getElementById("result-toolbar");
  const resultSection = document.getElementById("result-section");
  const nameEl = document.getElementById("source-filename");
  const analysisSection = document.getElementById("analysis-section");
  const input = document.getElementById("material-file");
  const form = document.getElementById("analyze-form");

  if (toolbar) toolbar.hidden = true;
  if (resultSection) resultSection.hidden = true;
  if (nameEl) nameEl.textContent = "";
  if (analysisSection) {
    analysisSection.innerHTML = "";
    analysisSection.hidden = true;
  }
  if (input) input.value = "";
  form?.querySelectorAll('input[name="practice_type"]').forEach((el) => {
    el.checked = el.value === "mixed";
  });

  if (form) {
    form.scrollIntoView({ behavior: "smooth", block: "start" });
    input?.focus();
  }
}

function buildConceptExplanationsSection(analysis) {
  const points = analysis.core_points || [];
  const items = analysis.concept_explanations;
  if (!Array.isArray(items) || items.length === 0) {
    return (
      `<h2 class="analysis__heading">概念详解</h2>` +
      `<p class="analysis__note">暂无概念详解数据（可能为旧版历史记录）。</p>`
    );
  }
  let html = `<h2 class="analysis__heading">概念详解</h2><div class="concept-detail-list">`;
  for (let i = 0; i < items.length; i++) {
    const it = items[i] || {};
    const rawPoint = points[i];
    const title =
      typeof rawPoint === "string"
        ? rawPoint
        : rawPoint?.point || `考点 ${i + 1}`;
    const w = it.what_it_is || it.plain_explanation || "";
    const f = it.formulas_notes || it.formulas || "";
    const ex = it.life_example || "";
    const wHtml = w ? escapeHtml(w).replace(/\n/g, "<br />") : "（暂无）";
    const fHtml = f ? escapeHtml(f).replace(/\n/g, "<br />") : "（暂无）";
    const exHtml = ex ? escapeHtml(ex).replace(/\n/g, "<br />") : "（暂无）";
    html += `<article class="concept-detail-card">`;
    html += `<h3 class="concept-detail-card__title">${escapeHtml(title)}</h3>`;
    html += `<section class="concept-detail-block">`;
    html += `<h4 class="concept-detail-block__label">是什么（零基础版）</h4>`;
    html += `<div class="concept-detail-block__body">${wHtml}</div>`;
    html += `</section>`;
    html += `<section class="concept-detail-block">`;
    html += `<h4 class="concept-detail-block__label">公式与符号</h4>`;
    html += `<div class="concept-detail-block__body">${fHtml}</div>`;
    html += `</section>`;
    html += `<section class="concept-detail-block">`;
    html += `<h4 class="concept-detail-block__label">生活化例子</h4>`;
    html += `<div class="concept-detail-block__body">${exHtml}</div>`;
    html += `</section>`;
    html += `</article>`;
  }
  html += `</div>`;
  return html;
}

function renderAnalysis(data, textTruncated, maxChars, practiceType = "mixed") {
  const section = document.getElementById("analysis-section");
  const resultSection = document.getElementById("result-section");
  const nameEl = document.getElementById("source-filename");
  const toolbar = document.getElementById("result-toolbar");
  if (!section || !resultSection || !nameEl) return;

  if (toolbar) toolbar.hidden = false;

  nameEl.textContent = data.source_name || "（未命名）";
  resultSection.hidden = false;

  let html = "";
  if (textTruncated) {
    html += `<p class="analysis__note">讲义较长，仅前 ${maxChars} 个字符已参与本次分析。</p>`;
  }
  const impClass = {
    必考: "importance--must",
    一般: "importance--normal",
    了解: "importance--light",
  };
  html += `<h2 class="analysis__heading">核心考点列表</h2><ol class="analysis-list core-points-list">`;
  for (const raw of data.analysis.core_points || []) {
    let point;
    let imp;
    if (typeof raw === "string") {
      point = raw;
      imp = "一般";
    } else {
      point = raw.point || "";
      imp = raw.importance || "一般";
    }
    if (!impClass[imp]) imp = "一般";
    const ic = impClass[imp];
    html += `<li class="core-point">`;
    html += `<span class="core-point__tag ${ic}">【${escapeHtml(imp)}】</span>`;
    html += `<span class="core-point__text">${escapeHtml(point)}</span>`;
    html += `</li>`;
  }
  html += `</ol>`;
  html += buildConceptExplanationsSection(data.analysis);
  html += `<h2 class="analysis__heading">难点解析</h2>`;
  html += `<div class="analysis__prose">${escapeHtml(String(data.analysis.difficult_analysis ?? "")).replace(/\n/g, "<br />")}</div>`;
  html += `<h2 class="analysis__heading">同类型练习题</h2><ol class="analysis-list practice-list">`;
  const questions = data.analysis.practice_questions || [];
  for (const raw of questions) {
    html += buildPracticeItemHtml(normalizePracticeRow(raw), practiceType);
  }
  html += `</ol>`;

  section.innerHTML = html;
  section.hidden = false;
  (toolbar || resultSection).scrollIntoView({ behavior: "smooth", block: "start" });
}

async function extractLocalText(file) {
  const ext = extOf(file.name);
  const buf = await file.arrayBuffer();
  if (ext === "pdf") {
    return extractPdfText(buf);
  }
  if (ext === "pptx") {
    return extractPptxText(buf);
  }
  if (ext === "ppt") {
    throw new Error("浏览器无法读取旧版 .ppt，请在 PowerPoint / WPS 中另存为 .pptx 或导出为 PDF 后再选文件。");
  }
  throw new Error("请选择 .pdf 或 .pptx 文件。");
}

function init() {
  const form = document.getElementById("analyze-form");
  const maxChars = parseInt(form?.dataset.maxChars || "120000", 10);
  const apiUrl = form?.dataset.apiUrl || "/api/analyze";

  if (!form) return;

  document.getElementById("btn-history-open")?.addEventListener("click", openHistoryPanel);
  document.getElementById("btn-history-close")?.addEventListener("click", closeHistoryPanel);
  document.getElementById("history-panel-backdrop")?.addEventListener("click", closeHistoryPanel);
  document.getElementById("btn-history-clear")?.addEventListener("click", () => {
    if (!confirm("确定清空全部历史记录？此操作不可恢复。")) return;
    clearAllHistory();
    renderHistoryList();
  });

  document.getElementById("history-list")?.addEventListener("click", (e) => {
    const del = e.target.closest("[data-action='delete-history']");
    if (del) {
      e.preventDefault();
      e.stopPropagation();
      const li = del.closest(".history-item");
      const id = li?.dataset.id;
      if (id) {
        deleteHistoryItem(id);
        renderHistoryList();
      }
      return;
    }
    const main = e.target.closest("[data-action='open-history']");
    if (main) {
      const li = main.closest(".history-item");
      const id = li?.dataset.id;
      const rec = loadHistory().find((x) => x.id === id);
      if (!rec?.analysis) return;
      closeHistoryPanel();
      clearFlash();
      renderAnalysis(
        { source_name: rec.source_name, analysis: rec.analysis },
        rec.text_truncated,
        maxChars,
        rec.practice_type || "mixed"
      );
      showFlash("success", "已载入历史记录。");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    const p = document.getElementById("history-panel");
    if (p && !p.hidden) closeHistoryPanel();
  });

  document.getElementById("analysis-section")?.addEventListener("click", (e) => {
    const mcqBtn = e.target.closest(".mcq-option");
    if (mcqBtn) {
      const item = mcqBtn.closest(".practice-item--mcq");
      if (!item || item.classList.contains("is-locked")) return;
      item.querySelectorAll(".mcq-option").forEach((b) => {
        b.classList.remove("is-selected");
        b.setAttribute("aria-pressed", "false");
      });
      mcqBtn.classList.add("is-selected");
      mcqBtn.setAttribute("aria-pressed", "true");
      return;
    }

    const confirmMcq = e.target.closest(".js-mcq-confirm");
    if (confirmMcq) {
      const item = confirmMcq.closest(".practice-item--mcq");
      if (!item || item.classList.contains("is-locked")) return;
      const correct = item.getAttribute("data-correct") || "";
      const selected = item.querySelector(".mcq-option.is-selected");
      if (!selected) {
        showFlash("error", "请先选择一个选项。");
        return;
      }
      const pick = selected.getAttribute("data-key") || "";
      item.classList.add("is-locked");

      const summary = item.querySelector(".mcq-feedback__summary");
      const feedback = item.querySelector(".practice-item__mcq-feedback");
      const correctSafe = escapeHtml(correct);

      if (pick === correct) {
        item.classList.add("mcq-result--correct");
        item.querySelectorAll(".mcq-option").forEach((b) => {
          b.disabled = true;
          if (b === selected) b.classList.add("is-correct");
        });
        if (summary) {
          summary.innerHTML = `回答正确！正确选项为 <strong>${correctSafe}</strong>。以下为参考答案与解题思路。`;
        }
      } else {
        item.classList.add("mcq-result--wrong");
        selected.classList.add("is-wrong");
        item.querySelectorAll(".mcq-option").forEach((b) => {
          b.disabled = true;
          if (b.getAttribute("data-key") === correct) b.classList.add("is-correct");
        });
        if (summary) {
          summary.innerHTML = `回答错误。正确选项为 <strong>${correctSafe}</strong>。以下为参考答案与解题思路。`;
        }
      }
      feedback?.removeAttribute("hidden");
      confirmMcq.disabled = true;
      return;
    }

    const btn = e.target.closest(".js-toggle-answer");
    if (!btn) return;
    const item = btn.closest(".practice-item");
    const detail = item?.querySelector(".practice-item__detail");
    if (!detail) return;
    const isHidden = detail.hasAttribute("hidden");
    if (isHidden) {
      detail.removeAttribute("hidden");
      btn.textContent = "收起答案";
      btn.setAttribute("aria-expanded", "true");
    } else {
      detail.setAttribute("hidden", "");
      btn.textContent = "查看答案";
      btn.setAttribute("aria-expanded", "false");
    }
  });

  document.getElementById("btn-reset-upload")?.addEventListener("click", () => {
    resetToUploadState();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearFlash();

    const input = document.getElementById("material-file");
    const file = input?.files?.[0];
    if (!file) {
      showFlash("error", "请选择要分析的 PDF 或 PPTX 文件。");
      return;
    }

    const btn = form.querySelector('button[type="submit"]');
    const analysisSection = document.getElementById("analysis-section");
    const resultSection = document.getElementById("result-section");
    const toolbar = document.getElementById("result-toolbar");
    if (toolbar) toolbar.hidden = true;
    if (analysisSection) {
      analysisSection.innerHTML = "";
      analysisSection.hidden = true;
    }
    if (resultSection) resultSection.hidden = true;

    btn.disabled = true;
    const oldLabel = btn.textContent;
    btn.textContent = "正在提取文字…";

    let text;
    try {
      text = await extractLocalText(file);
    } catch (err) {
      showFlash("error", err.message || String(err));
      btn.disabled = false;
      btn.textContent = oldLabel;
      return;
    }

    if (!text) {
      showFlash("error", "未能从文件中读出文字，可能是扫描版 PDF 或空白演示文稿。");
      btn.disabled = false;
      btn.textContent = oldLabel;
      return;
    }

    let textTruncated = false;
    let sendText = text;
    if (sendText.length > maxChars) {
      sendText = sendText.slice(0, maxChars);
      textTruncated = true;
    }

    btn.textContent = "正在调用 DeepSeek 分析…";

    const practiceType =
      form.querySelector('input[name="practice_type"]:checked')?.value || "mixed";

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: sendText,
          source_name: file.name,
          practice_type: practiceType,
        }),
      });
      const payload = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg = payload.error || `请求失败（${res.status}）`;
        showFlash("error", msg);
        btn.disabled = false;
        btn.textContent = oldLabel;
        return;
      }

      showFlash("success", "分析完成。");
      renderAnalysis(
        { source_name: payload.source_name, analysis: payload.analysis },
        textTruncated,
        maxChars,
        payload.practice_type || practiceType
      );
      try {
        appendHistoryRecord({
          source_name: payload.source_name || file.name,
          practice_type: payload.practice_type || practiceType,
          text_truncated: textTruncated,
          analysis: payload.analysis,
        });
      } catch (err) {
        console.warn("历史记录保存失败", err);
      }
    } catch (err) {
      showFlash("error", err.message || "网络错误，请稍后重试。");
    } finally {
      btn.disabled = false;
      btn.textContent = oldLabel;
    }
  });
}

init();
