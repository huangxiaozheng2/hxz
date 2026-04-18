
Copy

"""
index.py  –  生活文件一键读懂：Vercel Serverless 后端
路由：
  POST /analyze   接收 OCR 文本，返回结构化解析结果
  POST /chat      接收 OCR 原文 + 对话历史 + 新问题，返回 AI 回答
"""
 
from http.server import BaseHTTPRequestHandler
import json, os
from anthropic import Anthropic
 
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
 
# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────
 
def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Content-Type": "application/json",
    }
 
 
def send_json(handler, status: int, body: dict):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    for k, v in cors_headers().items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
 
 
def read_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)
    return json.loads(raw)
 
 
# ────────────────────────────────────────────────
# /analyze  –  结构化解析
# ────────────────────────────────────────────────
 
ANALYZE_SYSTEM = """你是一个专门帮助普通人读懂生活文件的 AI 助手。
用户会发给你从文件中识别出的文字（OCR 结果），可能包含租房合同、电费账单、医疗单据、学校通知等。
请严格按照以下 JSON 格式返回，不要输出任何其他内容：
{
  "success": true,
  "document_type": "文件类型（账单/合同/通知/医疗/其他）",
  "one_line_summary": "一句话说清楚这份文件是什么",
  "key_points": ["核心要点1", "核心要点2"],
  "key_info": [
    {"label": "关键字段名", "value": "对应的值"}
  ],
  "next_steps": ["建议用户做的事情1", "建议2"],
  "risks": ["风险提醒1（如果有）"]
}
用中文回答。risks 为空数组表示无风险。"""
 
 
def handle_analyze(handler):
    try:
        body = read_body(handler)
        ocr_text = body.get("ocr_text", "").strip()
        if not ocr_text:
            send_json(handler, 400, {"success": False, "error": "ocr_text 不能为空"})
            return
 
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1500,
            system=ANALYZE_SYSTEM,
            messages=[{"role": "user", "content": f"请分析以下文件内容：\n\n{ocr_text}"}],
        )
 
        text = message.content[0].text.strip()
        # 清理可能的 markdown 代码块
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        send_json(handler, 200, result)
 
    except json.JSONDecodeError as e:
        send_json(handler, 500, {"success": False, "error": f"AI 返回格式错误：{str(e)}"})
    except Exception as e:
        send_json(handler, 500, {"success": False, "error": str(e)})
 
 
# ────────────────────────────────────────────────
# /chat  –  追问对话
# ────────────────────────────────────────────────
 
CHAT_SYSTEM_TEMPLATE = """你是一个专门帮助普通人读懂生活文件的 AI 助手。
用户之前上传了一份文件，OCR 识别内容如下：
 
--- 文件原文开始 ---
{ocr_text}
--- 文件原文结束 ---
 
请基于以上文件内容，用简洁、友好的中文回答用户的问题。
如果问题与文件无关，也可以正常回答，但优先结合文件内容作答。
回答要简明扼要，避免重复文件原文，直接给出结论和建议。"""
 
 
def handle_chat(handler):
    try:
        body = read_body(handler)
        ocr_text = body.get("ocr_text", "").strip()
        # history: [{"role": "user"/"assistant", "content": "..."}]
        history  = body.get("history", [])
        question = body.get("question", "").strip()
 
        if not question:
            send_json(handler, 400, {"success": False, "error": "question 不能为空"})
            return
 
        # 构造带文件上下文的 system prompt
        system_prompt = CHAT_SYSTEM_TEMPLATE.format(
            ocr_text=ocr_text if ocr_text else "（用户未提供文件原文）"
        )
 
        # 拼接历史 + 新问题
        messages = list(history) + [{"role": "user", "content": question}]
 
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            system=system_prompt,
            messages=messages,
        )
 
        answer = message.content[0].text.strip()
        send_json(handler, 200, {"success": True, "answer": answer})
 
    except Exception as e:
        send_json(handler, 500, {"success": False, "error": str(e)})
 
 
# ────────────────────────────────────────────────
# Vercel Handler
# ────────────────────────────────────────────────
 
class handler(BaseHTTPRequestHandler):
 
    def log_message(self, format, *args):
        pass  # 关闭默认日志，避免 Vercel 日志噪音
 
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in cors_headers().items():
            self.send_header(k, v)
        self.end_headers()
 
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/analyze":
            handle_analyze(self)
        elif path == "/chat":
            handle_chat(self)
        else:
            send_json(self, 404, {"success": False, "error": f"未知路由：{path}"})
        
