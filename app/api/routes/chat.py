from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    DeleteMessagesRequest,
    EditMessageRequest,
)
from app.services.chat_service import ChatService
from fastapi.responses import StreamingResponse
import json

router = APIRouter(tags=["chat"])
service = ChatService()


def _check_guest_question_limit(user: User) -> str | None:
    """检查游客提问次数，返回错误信息或 None。"""
    if user.role != "guest":
        return None
    if user.question_limit and user.question_limit > 0:
        if user.question_count >= user.question_limit:
            return f"您的问题次数已用完，请联系管理员获取正式账号：{settings.contact_admin_email}"
    return None


def _increment_question_count(db: Session, user: User):
    """游客提问计数+1。"""
    if user.role != "guest":
        return
    if user.question_limit and user.question_limit > 0:
        user.question_count += 1
        db.commit()
        db.refresh(user)


def _to_json(obj):
    def _convert(o):
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, ensure_ascii=False, default=_convert)


def _stream_generator(stream_iter):
    """统一的 SSE 生成器：处理 dict 事件 + 字符串 token。"""
    try:
        for event in stream_iter:
            if isinstance(event, dict):
                yield f"data: {_to_json(event)}\n\n"
            else:
                yield f"data: {_to_json({'type': 'token', 'content': event})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    limit_error = _check_guest_question_limit(current_user)
    if limit_error:
        raise HTTPException(status_code=403, detail=limit_error)
    try:
        _increment_question_count(db, current_user)
        return service.chat(
            db=db,
            message=req.message,
            conversation_id=req.conversation_id,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"调用失败：{exc}") from exc


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
def conversation_messages(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationMessagesResponse:
    try:
        return service.get_conversation_messages(
            db=db,
            conversation_id=conversation_id,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"读取会话消息失败：{exc}",
        ) from exc


@router.get("/conversations/recent", response_model=ConversationListResponse)
def recent_conversations(
        limit: int = Query(default=50, ge=1, le=200),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
) -> ConversationListResponse:
    try:
        return service.get_recent_conversations(
            db=db, limit=limit, user=current_user,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"读取最近会话失败：{exc}"
        ) from exc


@router.post("/chat/stream")
def chat_stream(
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    limit_error = _check_guest_question_limit(current_user)
    if limit_error:
        raise HTTPException(status_code=403, detail=limit_error)
    _increment_question_count(db, current_user)
    effective_web_search = bool(req.web_search and current_user.role != "guest")
    return StreamingResponse(
        _stream_generator(service.stream_chat(
            db=db,
            message=req.message,
            conversation_id=req.conversation_id,
            user=current_user,
            web_search=effective_web_search,
            model_config=req.agent_model_config,
        )),
        media_type="text/event-stream",
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
        conversation_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
):
    try:
        return service.delete_conversation(db, conversation_id, user=current_user)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除会话失败：{exc}") from exc


@router.post("/conversations/{conversation_id}/messages/delete")
def delete_messages(
    conversation_id: str,
    req: DeleteMessagesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.delete_messages(
            db=db,
            conversation_id=conversation_id,
            message_ids=req.message_ids,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除消息失败：{exc}") from exc


@router.post("/conversations/{conversation_id}/regenerate")
def regenerate_last_answer(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.regenerate_last_answer(
            db=db,
            conversation_id=conversation_id,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"重新生成失败：{exc}") from exc


@router.put("/conversations/{conversation_id}/last-user-message")
def edit_last_user_and_regenerate(
    conversation_id: str,
    req: EditMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.edit_last_user_and_regenerate(
            db=db,
            conversation_id=conversation_id,
            new_content=req.content,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"编辑并重新生成失败：{exc}") from exc


@router.post("/conversations/{conversation_id}/regenerate/stream")
def regenerate_last_answer_stream(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        _stream_generator(service.stream_regenerate_last_answer(
            db=db,
            conversation_id=conversation_id,
            user=current_user,
        )),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/conversations/{conversation_id}/last-user-message/stream")
def edit_last_user_and_regenerate_stream(
    conversation_id: str,
    req: EditMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        _stream_generator(service.stream_edit_last_user_and_regenerate(
            db=db,
            conversation_id=conversation_id,
            new_content=req.content,
            user=current_user,
        )),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.put("/conversations/{conversation_id}/messages/{message_id}")
def edit_message_and_regenerate(
    conversation_id: str,
    message_id: int,
    req: EditMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        return service.edit_message_and_regenerate(
            db=db,
            conversation_id=conversation_id,
            message_id=message_id,
            new_content=req.content,
            user=current_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"编辑并重新生成失败：{exc}") from exc


@router.post("/conversations/{conversation_id}/messages/{message_id}/stream")
def edit_message_and_regenerate_stream(
    conversation_id: str,
    message_id: int,
    req: EditMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return StreamingResponse(
        _stream_generator(service.stream_edit_message_and_regenerate(
            db=db,
            conversation_id=conversation_id,
            message_id=message_id,
            new_content=req.content,
            user=current_user,
        )),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


from app.repository.chunk_repo import list_chunks_by_document
@router.get("/documents/{document_uid}/chunks")
def document_chunks(
    document_uid: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    chunks = list_chunks_by_document(
        db=db,
        document_uid=document_uid,
        limit=limit,
    )

    return {
        "document_uid": document_uid,
        "count": len(chunks),
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "block_type": c.block_type,
                "token_count": c.token_count,
                "content": c.content[:500],
            }
            for c in chunks
        ],
    }
