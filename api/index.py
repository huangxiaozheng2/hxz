from http.server import BaseHTTPRequestHandler
import json, os
import dashscope

# 设置 API Key (确保你在 Vercel 环境变量里填了 DASHSCOPE_API_KEY)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

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

# 路由分发逻辑
class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in cors_headers().items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/api/analyze" or path == "/analyze":
            self.handle_analyze()
        elif path == "/api/chat" or path == "/chat":
            self.handle_chat()
        else:
            send_json(self, 404, {"success": False, "error": f"未知路由: {path}"})

    # --- 1. 结构化分析 ---
    def handle_analyze(self):
        try:
            body = read_body(self)
            ocr_text = body.get("ocr_text", "").strip()
            
            response = dashscope.Generation.call(
                model="qwen-max",
                prompt=f"你是一个文件助手。请将以下OCR文字转为JSON格式返回：\n{ocr_text}",
                result_format='message'
            )
            
            if response.status_code == 200:
                # 这里建议根据你的 iOS 前端需求微调返回的 JSON 结构
                send_json(self, 200, {"success": True, "data": response.output.choices[0].message.content})
            else:
                send_json(self, 500, {"success": False, "error": response.message})
        except Exception as e:
            send_json(self, 500, {"success": False, "error": str(e)})

    # --- 2. 追问对话 ---
    def handle_chat(self):
        try:
            body = read_body(self)
            ocr_text = body.get("ocr_text", "")
            history = body.get("history", [])
            question = body.get("question", "")

            messages = [{"role": "system", "content": f"基于此文件内容回答：{ocr_text}"}]
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": question})

            response = dashscope.Generation.call(
                model="qwen-max",
                messages=messages,
                result_format='message'
            )

            if response.status_code == 200:
                answer = response.output.choices[0].message.content
                send_json(self, 200, {"success": True, "answer": answer})
            else:
                send_json(self, 500, {"success": False, "error": response.message})
        except Exception as e:
            send_json(self, 500, {"success": False, "error": str(e)})
