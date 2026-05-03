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

function renderAnalysis(data, textTruncated, maxChars) {
  const section = document.getElementById("analysis-section");
  const resultSection = document.getElementById("result-section");
  const nameEl = document.getElementById("source-filename");
  if (!section || !resultSection || !nameEl) return;

  nameEl.textContent = data.source_name || "（未命名）";
  resultSection.hidden = false;

  let html = "";
  if (textTruncated) {
    html += `<p class="analysis__note">讲义较长，仅前 ${maxChars} 个字符已参与本次分析。</p>`;
  }
  html += `<h2 class="analysis__heading">核心考点列表</h2><ol class="analysis-list">`;
  for (const item of data.analysis.core_points) {
    html += `<li>${escapeHtml(item)}</li>`;
  }
  html += `</ol><h2 class="analysis__heading">难点解析</h2>`;
  html += `<div class="analysis__prose">${escapeHtml(data.analysis.difficult_analysis).replace(/\n/g, "<br />")}</div>`;
  html += `<h2 class="analysis__heading">同类型练习题</h2><ol class="analysis-list">`;
  for (const item of data.analysis.practice_questions) {
    html += `<li>${escapeHtml(item)}</li>`;
  }
  html += `</ol>`;

  section.innerHTML = html;
  section.hidden = false;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
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

    try {
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: sendText,
          source_name: file.name,
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
        maxChars
      );
    } catch (err) {
      showFlash("error", err.message || "网络错误，请稍后重试。");
    } finally {
      btn.disabled = false;
      btn.textContent = oldLabel;
    }
  });
}

init();
