from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from flask import Flask, jsonify, render_template, request

from deepseek_client import MAX_INPUT_CHARS, DeepSeekError, analyze_course_text

# Vercel 等对请求体有限制：仅接收 JSON 文本，限制在 2MB 以内足够覆盖截断后的讲义文本
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-in-production")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        max_chars=MAX_INPUT_CHARS,
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not request.is_json:
        return jsonify({"error": "请使用 application/json 发送请求。"}), 415

    data = request.get_json(silent=True) or {}
    text = data.get("text")
    source_name = data.get("source_name")

    if not isinstance(text, str):
        return jsonify({"error": "缺少字段 text 或类型错误。"}), 400

    text = text.strip()
    if not text:
        return jsonify({"error": "text 为空，无法分析。"}), 400

    text_truncated = False
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        text_truncated = True

    if isinstance(source_name, str):
        source_name = source_name.strip() or None
    else:
        source_name = None

    try:
        analysis = analyze_course_text(text)
    except DeepSeekError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(
        {
            "ok": True,
            "source_name": source_name,
            "analysis": analysis,
            "text_truncated": text_truncated,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
