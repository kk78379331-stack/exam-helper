/**
 * 浏览器内提取 PDF / PPTX 文字，仅将纯文本 POST 到后端（适配 Vercel 等小请求体限制）。
 */
import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.mjs";
import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/pdf.worker.mjs";

/** 最近一次「文件分析」的讲义文本与题型，用于换一批练习题（历史记录载入后为空） */
let lastPracticeContext = null;

/** 待分析的本地文件队列（可多选累加，可从列表移除） */
let materialFileQueue = [];

/** 历史记录面板中正在编辑显示名称的记录 id */
let historyEditId = null;

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
  if (historyEditId === id) historyEditId = null;
  saveHistory(loadHistory().filter((x) => x.id !== id));
}

function historyDisplayName(item) {
  const lines = String(item.source_name || "")
    .split(/\r?\n/)
    .map((s) => s.trim())
    .filter(Boolean);
  return lines.length ? lines.join("、") : "（未命名）";
}

function commitHistoryRename(id, rawValue) {
  historyEditId = null;
  const trimmed = String(rawValue || "").trim();
  const list = loadHistory();
  const idx = list.findIndex((x) => x.id === id);
  if (idx < 0) {
    renderHistoryList();
    return;
  }
  list[idx].source_name = trimmed || "（未命名）";
  saveHistory(list);
  renderHistoryList();
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

    if (historyEditId === item.id) {
      li.classList.add("history-item--editing");
      const row = document.createElement("div");
      row.className = "history-item__edit-row";
      const inp = document.createElement("input");
      inp.type = "text";
      inp.className = "history-item__name-input";
      inp.setAttribute("aria-label", "记录显示名称");
      inp.value = String(item.source_name || "").replace(/\r?\n/g, "、");
      const cancelBtn = document.createElement("button");
      cancelBtn.type = "button";
      cancelBtn.className = "history-item__edit-cancel btn btn--secondary btn--small";
      cancelBtn.dataset.action = "cancel-edit-history";
      cancelBtn.textContent = "取消";
      row.appendChild(inp);
      row.appendChild(cancelBtn);
      const hint = document.createElement("p");
      hint.className = "history-item__edit-hint";
      hint.textContent = "回车保存；Esc 取消";
      li.appendChild(row);
      li.appendChild(hint);
      ul.appendChild(li);
      requestAnimationFrame(() => {
        inp.focus();
        inp.select();
      });
      continue;
    }

    li.innerHTML =
      `<button type="button" class="history-item__main" data-action="open-history">` +
      `<span class="history-item__name">${escapeHtml(historyDisplayName(item))}</span>` +
      `<span class="history-item__time">${escapeHtml(timeStr)}</span>` +
      `</button>` +
      `<button type="button" class="history-item__edit" data-action="edit-history" aria-label="编辑名称">编辑</button>` +
      `<button type="button" class="history-item__delete" data-action="delete-history" aria-label="删除此条">删除</button>`;
    ul.appendChild(li);
  }
}

function renderMaterialFileQueue() {
  const listEl = document.getElementById("file-queue-list");
  const emptyEl = document.getElementById("file-queue-empty");
  if (!listEl || !emptyEl) return;
  listEl.innerHTML = "";
  const n = materialFileQueue.length;
  emptyEl.hidden = n > 0;
  listEl.hidden = n === 0;
  for (let i = 0; i < n; i++) {
    const f = materialFileQueue[i];
    const li = document.createElement("li");
    li.className = "file-queue-item";
    li.innerHTML =
      `<span class="file-queue-item__name">${escapeHtml(f.name)}</span>` +
      `<button type="button" class="file-queue-item__remove btn btn--secondary btn--small" data-action="remove-queued-file" data-index="${i}" aria-label="从列表移除此文件">移除</button>`;
    listEl.appendChild(li);
  }
}

function setSourceFilenameDisplay(sourceName) {
  const el = document.getElementById("source-filename");
  if (!el) return;
  const raw = String(sourceName || "").trim();
  if (!raw) {
    el.textContent = "（未命名）";
    return;
  }
  const lines = raw.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  if (lines.length <= 1) {
    el.textContent = lines[0] || "（未命名）";
    return;
  }
  el.innerHTML = lines
    .map((name) => `<div class="result__source-line">${escapeHtml(name)}</div>`)
    .join("");
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

function buildAccordionItem(title, panelHtml, expanded) {
  const expandedCls = expanded ? " is-expanded" : "";
  const ae = expanded ? "true" : "false";
  return (
    `<section class="accordion__item${expandedCls}">` +
    `<button type="button" class="accordion__header" aria-expanded="${ae}">` +
    `<span class="accordion__title">${escapeHtml(title)}</span>` +
    `<span class="accordion__chevron" aria-hidden="true"></span>` +
    `</button>` +
    `<div class="accordion__panel">${panelHtml}</div>` +
    `</section>`
  );
}

/** 一级区块内的二级折叠：默认折叠，sectionExtraClass 如 concept-detail-nest / difficult-nest */
function buildNestedAccordionSection(title, panelHtml, expanded, sectionExtraClass = "") {
  const expandedCls = expanded ? " is-expanded" : "";
  const ae = expanded ? "true" : "false";
  const extra = sectionExtraClass ? ` ${sectionExtraClass}` : "";
  return (
    `<section class="accordion__item accordion__item--nested${extra}${expandedCls}">` +
    `<button type="button" class="accordion__header accordion__header--nested" aria-expanded="${ae}">` +
    `<span class="accordion__title">${escapeHtml(title)}</span>` +
    `<span class="accordion__chevron" aria-hidden="true"></span>` +
    `</button>` +
    `<div class="accordion__panel accordion__panel--nested">${panelHtml}</div>` +
    `</section>`
  );
}

function buildCorePointsInnerHtml(analysis, textTruncated, maxChars) {
  let inner = "";
  if (textTruncated) {
    inner += `<p class="analysis__note">讲义较长，仅前 ${maxChars} 个字符已参与本次分析。</p>`;
  }
  const impClass = {
    必考: "importance--must",
    一般: "importance--normal",
    了解: "importance--light",
  };
  inner += `<ol class="analysis-list core-points-list">`;
  for (const raw of analysis.core_points || []) {
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
    inner += `<li class="core-point">`;
    inner += `<span class="core-point__tag ${ic}">【${escapeHtml(imp)}】</span>`;
    inner += `<span class="core-point__text">${escapeHtml(point)}</span>`;
    inner += `</li>`;
  }
  inner += `</ol>`;
  return inner;
}

function buildConceptDetailInnerHtml(analysis) {
  const points = analysis.core_points || [];
  const items = analysis.concept_explanations;
  if (!Array.isArray(items) || items.length === 0) {
    return `<p class="analysis__note">暂无概念详解数据（可能为旧版历史记录）。</p>`;
  }
  let html = `<div class="concept-detail-list">`;
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
    let panel = `<section class="concept-detail-block">`;
    panel += `<h4 class="concept-detail-block__label">是什么（零基础版）</h4>`;
    panel += `<div class="concept-detail-block__body">${wHtml}</div>`;
    panel += `</section>`;
    panel += `<section class="concept-detail-block">`;
    panel += `<h4 class="concept-detail-block__label">公式与符号</h4>`;
    panel += `<div class="concept-detail-block__body">${fHtml}</div>`;
    panel += `</section>`;
    panel += `<section class="concept-detail-block">`;
    panel += `<h4 class="concept-detail-block__label">生活化例子</h4>`;
    panel += `<div class="concept-detail-block__body">${exHtml}</div>`;
    panel += `</section>`;
    html += buildNestedAccordionSection(title, panel, false, "concept-detail-nest");
  }
  html += `</div>`;
  return html;
}

function buildDifficultInnerHtml(analysis) {
  const main = String(analysis.difficult_analysis ?? "");
  const myths = analysis.common_misconceptions;
  const cc = String(analysis.concept_comparison ?? "");
  let mythsPanel;
  if (Array.isArray(myths) && myths.length) {
    mythsPanel = `<ol class="difficult-myths analysis-list">`;
    for (const m of myths) {
      mythsPanel += `<li>${escapeHtml(String(m)).replace(/\n/g, "<br />")}</li>`;
    }
    mythsPanel += `</ol>`;
  } else {
    mythsPanel = `<p class="analysis__note analysis__note--tight">暂无。</p>`;
  }
  const mainPanel = `<div class="analysis__prose">${escapeHtml(main).replace(/\n/g, "<br />")}</div>`;
  const ccPanel = `<div class="analysis__prose">${cc ? escapeHtml(cc).replace(/\n/g, "<br />") : "暂无。"}</div>`;
  let html = `<div class="difficult-stack">`;
  html += buildNestedAccordionSection("难点说明", mainPanel, false, "difficult-nest");
  html += buildNestedAccordionSection("常见误区", mythsPanel, false, "difficult-nest");
  html += buildNestedAccordionSection("概念对比", ccPanel, false, "difficult-nest");
  html += `</div>`;
  return html;
}

function rebalancePracticeBatchAccordions() {
  const batchesEl = document.getElementById("practice-batches");
  if (!batchesEl) return;
  const sections = batchesEl.querySelectorAll(".practice-batch");
  sections.forEach((sec, i) => {
    const isLast = i === sections.length - 1;
    sec.classList.toggle("is-expanded", isLast);
    const hdr = sec.querySelector(":scope > .accordion__header");
    if (hdr) hdr.setAttribute("aria-expanded", isLast ? "true" : "false");
  });
}

function buildPracticeOlInnerHtml(questions, practiceType) {
  let html = "";
  for (const raw of questions || []) {
    html += buildPracticeItemHtml(normalizePracticeRow(raw), practiceType);
  }
  return html;
}

function buildPracticeBatchSectionHtml(batchIndex, questions, practiceType, expandedBatch) {
  const expandedCls = expandedBatch ? " is-expanded" : "";
  const ae = expandedBatch ? "true" : "false";
  return (
    `<section class="accordion__item practice-batch${expandedCls}" data-batch="${batchIndex}">` +
    `<button type="button" class="accordion__header practice-batch__header" aria-expanded="${ae}">` +
    `<span class="accordion__title">第${batchIndex}批练习题</span>` +
    `<span class="accordion__chevron" aria-hidden="true"></span>` +
    `</button>` +
    `<div class="accordion__panel">` +
    `<ol class="analysis-list practice-list">` +
    buildPracticeOlInnerHtml(questions, practiceType) +
    `</ol></div></section>`
  );
}

function buildPracticeInnerHtml(analysis, practiceType, showReroll) {
  let html = `<div id="practice-root"><div id="practice-batches">`;
  html += buildPracticeBatchSectionHtml(1, analysis.practice_questions, practiceType, true);
  html += `</div>`;
  if (showReroll) {
    html +=
      `<div class="practice-reroll-bar">` +
      `<button type="button" class="btn btn--secondary js-reroll-practice" id="btn-reroll-practice">换一批题目</button>` +
      `</div>`;
  }
  html += `</div>`;
  return html;
}

function resetToUploadState() {
  lastPracticeContext = null;
  materialFileQueue = [];
  renderMaterialFileQueue();
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

function renderAnalysis(data, textTruncated, maxChars, practiceType = "mixed", options = {}) {
  const showPracticeReroll = options.showPracticeReroll === true;
  const section = document.getElementById("analysis-section");
  const resultSection = document.getElementById("result-section");
  const nameEl = document.getElementById("source-filename");
  const toolbar = document.getElementById("result-toolbar");
  if (!section || !resultSection || !nameEl) return;

  if (toolbar) toolbar.hidden = false;

  setSourceFilenameDisplay(data.source_name);
  resultSection.hidden = false;

  let html = `<div class="analysis-accordions">`;
  html += buildAccordionItem(
    "核心考点列表",
    buildCorePointsInnerHtml(data.analysis, textTruncated, maxChars),
    true
  );
  html += buildAccordionItem("概念详解", buildConceptDetailInnerHtml(data.analysis), false);
  html += buildAccordionItem("难点解析", buildDifficultInnerHtml(data.analysis), false);
  html += buildAccordionItem(
    "同类型练习题",
    buildPracticeInnerHtml(data.analysis, practiceType, showPracticeReroll),
    true
  );
  html += `</div>`;

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

/** 将多份已抽取文本按顺序合并，便于模型综合多文件 */
function buildMergedMaterialText(parts) {
  const multi = parts.length > 1;
  let body = "";
  if (multi) {
    body +=
      "【说明】以下文本由多个讲义文件按用户所选顺序依次抽取并拼接；每段前有文件名标记。请综合全部材料，输出统一的考点、概念详解、难点解析与练习题（勿只针对某一个文件）。\n\n";
  }
  for (const { name, text } of parts) {
    body += `────────\n《${name}》\n────────\n\n${text}\n\n`;
  }
  return body.trimEnd();
}

function formatSourceNamesForPayload(files) {
  return files.map((f) => f.name).join("\n");
}

function onMaterialFileInputChange(e) {
  const input = e.target;
  const picked = Array.from(input.files || []);
  for (const f of picked) {
    const ext = extOf(f.name);
    if (ext !== "pdf" && ext !== "pptx") {
      showFlash("error", `已跳过不支持的文件：${f.name}（仅支持 .pdf / .pptx）`);
      continue;
    }
    const dup = materialFileQueue.some(
      (x) => x.name === f.name && x.size === f.size && x.lastModified === f.lastModified
    );
    if (!dup) materialFileQueue.push(f);
  }
  input.value = "";
  renderMaterialFileQueue();
}

function init() {
  const form = document.getElementById("analyze-form");
  const maxChars = parseInt(form?.dataset.maxChars || "120000", 10);
  const apiUrl = form?.dataset.apiUrl || "/api/analyze";

  if (!form) return;

  renderMaterialFileQueue();
  document.getElementById("material-file")?.addEventListener("change", onMaterialFileInputChange);
  document.getElementById("file-queue-list")?.addEventListener("click", (e) => {
    const rm = e.target.closest("[data-action='remove-queued-file']");
    if (!rm) return;
    const i = Number.parseInt(rm.getAttribute("data-index") || "", 10);
    if (Number.isNaN(i) || i < 0 || i >= materialFileQueue.length) return;
    materialFileQueue.splice(i, 1);
    renderMaterialFileQueue();
  });

  document.getElementById("btn-history-open")?.addEventListener("click", openHistoryPanel);
  document.getElementById("btn-history-close")?.addEventListener("click", closeHistoryPanel);
  document.getElementById("history-panel-backdrop")?.addEventListener("click", closeHistoryPanel);
  document.getElementById("btn-history-clear")?.addEventListener("click", () => {
    if (!confirm("确定清空全部历史记录？此操作不可恢复。")) return;
    clearAllHistory();
    renderHistoryList();
  });

  document.getElementById("history-list")?.addEventListener("keydown", (e) => {
    const inp = e.target.closest(".history-item__name-input");
    if (!inp) return;
    const li = inp.closest(".history-item");
    const id = li?.dataset.id;
    if (!id) return;
    if (e.key === "Enter") {
      e.preventDefault();
      e.stopPropagation();
      commitHistoryRename(id, inp.value);
    } else if (e.key === "Escape") {
      e.preventDefault();
      e.stopPropagation();
      historyEditId = null;
      renderHistoryList();
    }
  });

  document.getElementById("history-list")?.addEventListener("click", (e) => {
    const cancelEd = e.target.closest("[data-action='cancel-edit-history']");
    if (cancelEd) {
      e.preventDefault();
      historyEditId = null;
      renderHistoryList();
      return;
    }
    const editBtn = e.target.closest("[data-action='edit-history']");
    if (editBtn) {
      e.preventDefault();
      e.stopPropagation();
      const li = editBtn.closest(".history-item");
      const id = li?.dataset.id;
      if (id) {
        historyEditId = id;
        renderHistoryList();
      }
      return;
    }
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
      historyEditId = null;
      closeHistoryPanel();
      clearFlash();
      lastPracticeContext = null;
      renderAnalysis(
        { source_name: rec.source_name, analysis: rec.analysis },
        rec.text_truncated,
        maxChars,
        rec.practice_type || "mixed",
        { showPracticeReroll: false }
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
    const accHeader = e.target.closest(".accordion__header");
    if (accHeader) {
      const item = accHeader.closest(".accordion__item");
      if (item) {
        item.classList.toggle("is-expanded");
        const expanded = item.classList.contains("is-expanded");
        accHeader.setAttribute("aria-expanded", expanded ? "true" : "false");
        return;
      }
    }

    const rerollBtn = e.target.closest(".js-reroll-practice");
    if (rerollBtn) {
      e.preventDefault();
      if (!lastPracticeContext?.text) {
        showFlash("error", "请先完成一次文件分析后再换一批题目；从历史记录打开时无法换题。");
        return;
      }
      rerollBtn.disabled = true;
      const prevLabel = rerollBtn.textContent;
      rerollBtn.textContent = "正在生成…";
      void (async () => {
        try {
          const res = await fetch(lastPracticeContext.regenerateUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: lastPracticeContext.text,
              practice_type: lastPracticeContext.practiceType,
            }),
          });
          const payload = await res.json().catch(() => ({}));
          if (!res.ok) {
            showFlash("error", payload.error || `换题失败（${res.status}）`);
            return;
          }
          const batchesEl = document.getElementById("practice-batches");
          const pt = payload.practice_type || lastPracticeContext.practiceType;
          if (!batchesEl) {
            showFlash("error", "页面结构异常，请重新分析后再试。");
            return;
          }
          const existing = batchesEl.querySelectorAll(".practice-batch").length;
          const nextBatch = existing + 1;
          batchesEl.insertAdjacentHTML(
            "beforeend",
            buildPracticeBatchSectionHtml(nextBatch, payload.practice_questions, pt, true)
          );
          rebalancePracticeBatchAccordions();
          batchesEl
            .querySelector(`.practice-batch[data-batch="${nextBatch}"]`)
            ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
          showFlash("success", `已追加第 ${nextBatch} 批练习题。`);
        } catch (err) {
          showFlash("error", err.message || "网络错误，请稍后重试。");
        } finally {
          const b = document.getElementById("btn-reroll-practice");
          if (b) {
            b.disabled = false;
            b.textContent = "换一批题目";
          }
        }
      })();
      return;
    }

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

    if (materialFileQueue.length === 0) {
      showFlash("error", "请先选择至少一个 PDF 或 PPTX 文件（可多选，也可分多次加入列表）。");
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
    const queueSnapshot = materialFileQueue.slice();
    const sourceNamePayload = formatSourceNamesForPayload(queueSnapshot);

    const extractedParts = [];
    try {
      for (let i = 0; i < queueSnapshot.length; i++) {
        const file = queueSnapshot[i];
        btn.textContent = `正在提取 ${i + 1}/${queueSnapshot.length}：${file.name}`;
        const piece = await extractLocalText(file);
        if (!piece || !piece.trim()) {
          showFlash(
            "error",
            `未能从「${file.name}」中读出有效文字，可能是扫描版 PDF 或空白演示文稿。请移除或替换该文件后重试。`
          );
          btn.disabled = false;
          btn.textContent = oldLabel;
          return;
        }
        extractedParts.push({ name: file.name, text: piece.trim() });
      }
    } catch (err) {
      showFlash("error", err.message || String(err));
      btn.disabled = false;
      btn.textContent = oldLabel;
      return;
    }

    const merged = buildMergedMaterialText(extractedParts);
    let textTruncated = false;
    let sendText = merged;
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
          source_name: sourceNamePayload,
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
      lastPracticeContext = {
        text: sendText,
        practiceType: payload.practice_type || practiceType,
        regenerateUrl: form.dataset.regenerateUrl || "/api/regenerate-practice",
      };
      const displaySource =
        typeof payload.source_name === "string" && payload.source_name.trim()
          ? payload.source_name
          : sourceNamePayload;
      renderAnalysis(
        { source_name: displaySource, analysis: payload.analysis },
        textTruncated,
        maxChars,
        payload.practice_type || practiceType,
        { showPracticeReroll: true }
      );
      try {
        appendHistoryRecord({
          source_name: displaySource,
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
