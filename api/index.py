from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import dashscope
import os
import json

app = FastAPI()

# 跨域设置，确保 iOS App 能连上
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 填入你的通义千问 Key
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

@app.get("/")
async def root():
    return {"status": "Backend is running!"}

# 1. 解析接口
@app.post("/analyze")
@app.post("/api/analyze")
async def analyze(request: Request):
    try:
        body = await request.json()
        ocr_text = body.get("ocr_text", "").strip()
        
        response = dashscope.Generation.call(
            model="qwen-max",
            prompt=f"你是一个文件助手。请将以下文字转为JSON返回：\n{ocr_text}",
            result_format='message'
        )
        return {"success": True, "data": response.output.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}

# 2. 追问接口
@app.post("/chat")
@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        ocr_text = body.get("ocr_text", "")
        history = body.get("history", [])
        question = body.get("question", "")

        messages = [{"role": "system", "content": f"基于此文件回答：{ocr_text}"}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        response = dashscope.Generation.call(
            model="qwen-max",
            messages=messages,
            result_format='message'
        )
        return {"success": True, "answer": response.output.choices[0].message.content}
    except Exception as e:
        return {"success": False, "error": str(e)}
