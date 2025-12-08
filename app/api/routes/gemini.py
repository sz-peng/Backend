"""
Gemini兼容的API端点
支持Gemini API格式的图片生成 (/v1beta/models/{model}:generateContent)
支持图生图功能和SSE流式响应（心跳保活）
"""
from typing import Optional
import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps_flexible import get_user_flexible
from app.api.deps import get_plugin_api_service
from app.models.user import User
from app.services.plugin_api_service import PluginAPIService
from app.schemas.plugin_api import GenerateContentRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1beta", tags=["Gemini兼容API"])


@router.post(
    "/models/{model}:generateContent",
    summary="图片生成",
    description="使用Gemini模型生成图片，支持文生图和图生图。支持JWT token或API key认证。响应使用SSE格式（心跳保活）"
)
async def generate_content(
    model: str,
    request: GenerateContentRequest,
    current_user: User = Depends(get_user_flexible),
    service: PluginAPIService = Depends(get_plugin_api_service)
):
    """
    图片生成API（Gemini格式）
    
    支持两种认证方式：
    1. JWT Token: Authorization: Bearer <jwt_token> - 用于网页端调用
    2. API Key: Authorization: Bearer <api_key> - API key 以 sk- 开头，用于程序调用
    
    参数说明:
    - model (必需): 模型名称，例如 gemini-2.5-flash-image 或 gemini-3-pro-image
    - contents (必需): 包含提示词和/或图片的消息数组
    - generationConfig.imageConfig (可选): 图片生成配置
      - aspectRatio: 宽高比。支持的值：1:1、2:3、3:2、3:4、4:3、9:16、16:9、21:9。
                     如果未指定，模型将根据提供的任何参考图片选择默认宽高比。
      - imageSize: 图片尺寸。支持的值为 1K、2K、4K。如果未指定，模型将使用默认值 1K。
    
    **文生图请求示例:**
    ```json
    {
      "contents": [
        {
          "role": "user",
          "parts": [
            {
              "text": "生成一只可爱的猫"
            }
          ]
        }
      ],
      "generationConfig": {
        "imageConfig": {
          "aspectRatio": "1:1",
          "imageSize": "1K"
        }
      }
    }
    ```
    
    **图生图请求示例:**
    ```json
    {
      "contents": [
        {
          "role": "user",
          "parts": [
            {
              "text": "将这张图片转换为油画风格"
            },
            {
              "inlineData": {
                "mimeType": "image/jpeg",
                "data": "base64编码的图片数据..."
              }
            }
          ]
        }
      ],
      "generationConfig": {
        "imageConfig": {
          "aspectRatio": "16:9"
        }
      }
    }
    ```
    
    **响应格式（SSE流式）:**
    响应使用 Server-Sent Events (SSE) 格式，包含心跳保活机制（每30秒发送心跳）。
    
    SSE事件类型:
    - `heartbeat`: 心跳事件，用于保持连接活跃
    - `result`: 最终结果，包含生成的图片数据
    - `error`: 错误事件
    
    最终结果格式:
    ```json
    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "inlineData": {
                  "mimeType": "image/jpeg",
                  "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
                }
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP"
        }
      ]
    }
    ```
    
    响应字段说明:
    - candidates[0].content.parts[0].inlineData.data: Base64 编码的图片数据
    - candidates[0].content.parts[0].inlineData.mimeType: 图片 MIME 类型，例如 image/jpeg
    """
    try:
        # 获取 config_type（通过 API key 认证时会设置）
        config_type = getattr(current_user, '_config_type', None)
        
        # 使用流式请求以支持SSE心跳保活
        async def generate():
            async for chunk in service.generate_content_stream(
                user_id=current_user.id,
                model=model,
                request_data=request.model_dump(),
                config_type=config_type
            ):
                yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except httpx.HTTPStatusError as e:
        # 透传上游API的错误响应
        error_data = getattr(e, 'response_data', {"detail": str(e)})
        if isinstance(error_data, dict) and 'detail' in error_data:
            detail = error_data['detail']
        else:
            detail = error_data
        raise HTTPException(
            status_code=e.response.status_code,
            detail=detail
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"图片生成失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图片生成失败: {str(e)}"
        )