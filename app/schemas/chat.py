from datetime import datetime
from pydantic import BaseModel, Field
from typing import List


class SourceReference(BaseModel):
    """搜索结果中的单个引用来源。"""
    chunk_uid: str = Field(..., description="分块ID")
    document_uid: str = Field(..., description="文档ID")
    filename: str = Field(default="", description="文件名")
    content_preview: str = Field(default="", description="内容片段（前200字）")
    section_title: str = Field(default="", description="节标题")
    page_no: int = Field(default=0, description="页码")
    score: float = Field(default=0, description="RRF 融合分数")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="会话ID",
    )
    web_search: bool = Field(
        default=False,
        description="是否强制启用联网搜索",
    )
    agent_model_config: dict | None = Field(
        default=None,
        description="客户端自定义模型配置 (Issue 2: Multi-Agent API 设置)",
    )


class ChatResponse(BaseModel):
    conversation_id: str = Field(..., description="会话ID")
    answer: str = Field(..., description="模型回答")
    messages_saved: int = Field(..., description="本次保存的消息条数")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="本次搜索引用的文档来源",
    )

class ConversationSummary(BaseModel):
    conversation_id: str = Field(..., description="会话ID")
    title: str | None = Field(default=None, description="会话标题")
    updated_at: datetime = Field(..., description="更新时间")


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = Field(
        default_factory=list,
        description="最近会话列表"
    )


class MessageItem(BaseModel):
    id: int = Field(..., description="消息ID")
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    created_at: datetime = Field(..., description="创建时间")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="该消息关联的检索来源（仅 assistant）",
    )
    metadata: dict | None = Field(
        default=None,
        description="来源使用情况 (web/工具/通用知识等, Issue 2)",
    )


class ConversationMessagesResponse(BaseModel):
    conversation_id: str = Field(..., description="会话ID")
    title: str | None = Field(default=None, description="会话标题")
    messages: list[MessageItem] = Field(
        default_factory=list,
        description="消息列表"
    )

class DeleteMessagesRequest(BaseModel):
    message_ids: List[int]


class EditMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, description="编辑后的用户问题")


class SimpleOKResponse(BaseModel):
    ok: bool
    message: str = ""

class RegenerateResponse(BaseModel):
    conversation_id: str
    answer: str