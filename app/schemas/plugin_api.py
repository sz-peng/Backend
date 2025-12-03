"""
Plug-in API相关的数据模式
"""
from typing import Optional, Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field


# ==================== Plug-in API密钥相关 ====================

class PluginAPIKeyCreate(BaseModel):
    """创建Plug-in API密钥请求"""
    api_key: str = Field(..., description="用户在plug-in-api系统中的API密钥")


class PluginAPIKeyResponse(BaseModel):
    """Plug-in API密钥响应"""
    id: int
    user_id: int
    plugin_user_id: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


class PluginAPIKeyUpdate(BaseModel):
    """更新Plug-in API密钥"""
    api_key: Optional[str] = Field(None, description="新的API密钥")
    is_active: Optional[bool] = Field(None, description="是否激活")


# ==================== Plug-in API代理请求 ====================

class CreatePluginUserRequest(BaseModel):
    """创建plug-in-api用户请求（管理员操作）"""
    name: Optional[str] = Field(None, description="用户名称")
    prefer_shared: int = Field(0, description="Cookie优先级，0=专属优先，1=共享优先")


class CreatePluginUserResponse(BaseModel):
    """创建plug-in-api用户响应"""
    success: bool
    message: str
    data: Dict[str, Any]


class OAuthAuthorizeRequest(BaseModel):
    """获取OAuth授权URL请求"""
    is_shared: int = Field(0, description="0=专属cookie，1=共享cookie")


class OAuthAuthorizeResponse(BaseModel):
    """OAuth授权响应"""
    success: bool
    data: Dict[str, Any]


class OAuthCallbackRequest(BaseModel):
    """手动提交OAuth回调"""
    callback_url: str = Field(..., description="完整的回调URL")


class UpdateCookiePreferenceRequest(BaseModel):
    """更新Cookie优先级"""
    prefer_shared: int = Field(..., description="0=专属优先，1=共享优先")


class UpdateAccountStatusRequest(BaseModel):
    """更新账号状态"""
    status: int = Field(..., description="0=禁用，1=启用")


class UpdateAccountNameRequest(BaseModel):
    """更新账号名称"""
    name: str = Field(..., description="账号名称")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求（支持多模态）"""
    model: str = Field(..., description="模型名称")
    messages: List[Dict[str, Any]] = Field(..., description="消息列表，支持文本和多模态内容")
    stream: bool = Field(True, description="是否流式输出")
    temperature: Optional[float] = Field(1.0, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具调用配置")
    
    model_config = {"extra": "allow"}  # 允许额外字段，支持OpenAI的所有参数


class QuotaConsumptionQuery(BaseModel):
    """配额消耗查询参数"""
    limit: Optional[int] = Field(None, description="限制返回数量")
    start_date: Optional[str] = Field(None, description="开始日期")
    end_date: Optional[str] = Field(None, description="结束日期")


# ==================== 通用响应 ====================

class PluginAPIResponse(BaseModel):
    """通用Plug-in API响应"""
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None