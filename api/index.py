# api/index.py
# 生活文件一键读懂 — 后端服务
#
# 运行环境：Vercel Serverless（Python 3.11）
# 框架：FastAPI + Mangum（ASGI → AWS Lambda handler 适配器，Vercel 复用同一套）
# AI：通义千问 qwen-turbo（dashscope SDK）
#
# 业务流程：
#   iOS 端 OCR 文本 → POST /analyze → 千问 API → JSON 结构化结果 → iOS 端展示

import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mangum import Mangum
import dashscope
from dashscope import Generation

# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 从环境变量读取 API Key
# 在 Vercel Dashboard → Settings → Environment Variables 中配置
# 变量名：DASHSCOPE_API_KEY
# ─────────────────────────────────────────────
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
if not DASHSCOPE_API_KEY:
    logger.warning("⚠️  未检测到 DASHSCOPE_API_KEY，请在 Vercel 环境变量中配置。")

dashscope.api_key = DASHSCOPE_API_KEY

# ─────────────────────────────────────────────
# FastAPI 实例
# ─────────────────────────────────────────────
app = FastAPI(
    title="生活文件一键读懂 API",
    description="接收 OCR 文本，调用通义千问，返回结构化分析结果",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# CORS 配置
# 生产环境请将 allow_origins 改为你的域名
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO: 上线后改为 iOS App 实际域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 请求 / 响应数据模型
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """iOS 端传入的请求体"""
    ocr_text: str = Field(
        ...,
        min_length=1,
        max_length=8000,         # 限制单次分析长度，控制 token 成本
        description="OCR 识别后的原始文本"
    )

class KeyInfo(BaseModel):
    """关键信息条目"""
    label: str                   # 例如："截止日期"
    value: str                   # 例如："2025年4月15日"

class AnalyzeResponse(BaseModel):
    """返回给 iOS 端的结构化结果"""
    success: bool
    one_line_summary: str        # 一句话总结
    key_points: list[str]        # 核心摘要（2-3 条）
    key_info: list[KeyInfo]      # 关键信息：日期、金额、联系人等
    next_steps: list[str]        # 下一步建议
    risks: list[str]             # 风险提醒（空列表 = 无风险）
    document_type: str           # 文件类型判断：账单/合同/通知/医疗/其他

class ErrorResponse(BaseModel):
    """统一错误响应"""
    success: bool = False
    error_code: str
    message: str

# ─────────────────────────────────────────────
# Prompt 构造
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个专业的生活文件分析助手，帮助普通用户快速理解账单、合同、通知、医疗单据等生活文件。

你的任务：
1. 判断文件类型（账单/合同/通知/医疗/学校/租房/其他）
2. 用一句话总结文件核心内容
3. 提取 2-3 条最重要的核心摘要
4. 提取关键信息：日期、金额、联系人、地点、电话等
5. 给出 1-3 条用户下一步应该做的事
6. 识别潜在风险（逾期、罚款、自动续费、需补材料等），无风险则返回空列表

重要规则：
- 仅供参考，不提供法律/医疗/财务建议
- 措辞平实，适合普通用户理解
- 严格按照 JSON 格式返回，不要输出任何其他内容

返回格式（严格 JSON，不加 markdown 代码块）：
{
  "document_type": "账单",
  "one_line_summary": "这是一份2025年3月的电费账单，共需缴纳328.5元。",
  "key_points": [
    "本月用电费用合计 ¥328.50",
    "缴费截止日期为 2025年4月15日",
    "逾期将产生滞纳金"
  ],
  "key_info": [
    {"label": "缴费金额", "value": "¥328.50"},
    {"label": "截止日期", "value": "2025年4月15日"},
    {"label": "账单周期", "value": "2025年3月1日-3月31日"}
  ],
  "next_steps": [
    "在4月15日前通过网上国网App或银行完成缴费",
    "缴费后保存凭证备用"
  ],
  "risks": [
    "⚠️ 距截止日期仅剩数天，请尽快缴费，逾期将产生违约金"
  ]
}"""


def build_user_prompt(ocr_text: str) -> str:
    return f"""请分析以下从生活文件中识别出的文字内容：

---
{ocr_text}
---

请按照系统提示的 JSON 格式返回分析结果，不要输出任何其他内容。"""


# ─────────────────────────────────────────────
# 调用千问 API
# ─────────────────────────────────────────────

def call_qwen(ocr_text: str) -> dict:
    """
    调用通义千问 qwen-turbo，返回解析后的 dict。
    使用 qwen-turbo 平衡速度与成本；如需更高准确率可改用 qwen-plus / qwen-max。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(ocr_text)},
    ]

    response = Generation.call(
        model="qwen-turbo",         # 可选: qwen-plus / qwen-max
        messages=messages,
        result_format="message",    # 返回 message 格式
        temperature=0.1,            # 低温度，确保输出稳定
        max_tokens=1500,
    )

    # 检查返回状态
    if response.status_code != 200:
        logger.error(f"千问 API 错误: {response.code} - {response.message}")
        raise ValueError(f"千问 API 调用失败: {response.message}")

    # 提取文本内容
    raw_content = response.output.choices[0].message.content.strip()
    logger.info(f"千问原始返回: {raw_content[:200]}...")  # 只记录前 200 字符

    # 清理可能的 markdown 代码块包装（```json ... ```）
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        raw_content = "\n".join(lines[1:-1])  # 去掉首尾行

    # 解析 JSON
    result = json.loads(raw_content)
    return result


# ─────────────────────────────────────────────
# API 路由
# ─────────────────────────────────────────────

@app.get("/")
def health_check():
    """健康检查接口，用于 Vercel 部署验证"""
    return {"status": "ok", "service": "生活文件一键读懂 API", "version": "1.0.0"}


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="分析文件文本",
    description="接收 OCR 识别后的文本，返回结构化分析结果（一句话总结、核心摘要、风险提醒等）"
)
async def analyze_document(request: AnalyzeRequest):
    """
    主分析接口
    - 输入：OCR 文本（纯文本，不含原始图片）
    - 输出：结构化 JSON（文件类型、摘要、关键信息、下一步、风险）
    """

    # 检查 API Key 是否已配置
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error_code": "API_KEY_MISSING",
                "message": "服务未配置 AI API Key，请联系管理员。"
            }
        )

    logger.info(f"收到分析请求，文本长度: {len(request.ocr_text)} 字符")

    try:
        # 调用千问
        result = call_qwen(request.ocr_text)

        # 组装响应（做防御性处理，字段缺失时给默认值）
        return AnalyzeResponse(
            success=True,
            document_type=result.get("document_type", "其他"),
            one_line_summary=result.get("one_line_summary", "无法生成总结，请重试。"),
            key_points=result.get("key_points", []),
            key_info=[
                KeyInfo(label=item["label"], value=item["value"])
                for item in result.get("key_info", [])
                if "label" in item and "value" in item
            ],
            next_steps=result.get("next_steps", []),
            risks=result.get("risks", []),
        )

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "success": False,
                "error_code": "PARSE_ERROR",
                "message": "AI 返回格式异常，请重试。"
            }
        )
    except ValueError as e:
        logger.error(f"千问调用失败: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "success": False,
                "error_code": "AI_ERROR",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error(f"未知错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "message": "服务器内部错误，请稍后重试。"
            }
        )


# ─────────────────────────────────────────────
# Vercel Serverless 入口
# Mangum 将 ASGI（FastAPI）适配为 Vercel 的 handler
# ─────────────────────────────────────────────
handler = Mangum(app, lifespan="off")
