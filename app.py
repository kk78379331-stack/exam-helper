from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# 必须从项目根目录加载 .env；无参数 load_dotenv 只查「当前工作目录」，换目录运行会读不到 Key。
load_dotenv(BASE_DIR / ".env")

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from deepseek_client import DeepSeekError, MAX_INPUT_CHARS, analyze_course_text
from text_extract import ExtractionError, extract_text
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"pdf", "ppt", "pptx"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage) -> tuple[str, str]:
    """返回 (展示用原始文件名, 磁盘上的安全文件名)。"""
    original = file_storage.filename or ""
    ext = original.rsplit(".", 1)[1].lower()
    safe = secure_filename(original)
    if not safe or safe == f".{ext}" or safe == ext:
        safe = f"upload_{uuid.uuid4().hex}.{ext}"
    dest = UPLOAD_FOLDER / safe
    file_storage.save(dest)
    return original, safe


@app.route("/", methods=["GET", "POST"])
def index():
    last_filename = None
    analysis = None
    text_truncated = False

    if request.method == "GET":
        return render_template(
            "index.html",
            last_filename=None,
            analysis=None,
            text_truncated=False,
            max_chars=MAX_INPUT_CHARS,
        )

    if "material_file" not in request.files:
        flash("请选择要上传的文件（PDF 或 PPT）。", "error")
        return redirect(url_for("index"))

    file = request.files["material_file"]
    if file.filename == "":
        flash("请选择要上传的文件（PDF 或 PPT）。", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("仅支持 .pdf、.ppt 或 .pptx 格式。", "error")
        return redirect(url_for("index"))

    display_name, safe_name = save_upload(file)
    last_filename = display_name
    disk_path = UPLOAD_FOLDER / safe_name

    try:
        doc_text = extract_text(disk_path)
    except ExtractionError as exc:
        flash(str(exc), "error")
        return render_template(
            "index.html",
            last_filename=last_filename,
            analysis=None,
            text_truncated=False,
            max_chars=MAX_INPUT_CHARS,
        )

    if len(doc_text) > MAX_INPUT_CHARS:
        text_truncated = True

    try:
        analysis = analyze_course_text(doc_text)
    except DeepSeekError as exc:
        flash(str(exc), "error")
        return render_template(
            "index.html",
            last_filename=last_filename,
            analysis=None,
            text_truncated=text_truncated,
            max_chars=MAX_INPUT_CHARS,
        )

    flash("分析完成。", "success")

    return render_template(
        "index.html",
        last_filename=last_filename,
        analysis=analysis,
        text_truncated=text_truncated,
        max_chars=MAX_INPUT_CHARS,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
