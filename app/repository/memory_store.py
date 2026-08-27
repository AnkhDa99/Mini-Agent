from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List
from uuid import uuid4


@dataclass
class ConversationState:
    conversation_id: str
    messages: List[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._store: Dict[str, ConversationState] = {}
        self._lock = Lock()

    def create_conversation(self) -> str:
        with self._lock:
            conversation_id = f"conv_{uuid4().hex[:12]}"
            self._store[conversation_id] = ConversationState(conversation_id=conversation_id)
            return conversation_id

    def ensure_conversation(self, conversation_id: str | None) -> str:
        if conversation_id and conversation_id in self._store:
            return conversation_id
        return self.create_conversation()

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            convo = self._store.setdefault(conversation_id, ConversationState(conversation_id=conversation_id))
            convo.messages.append({"role": role, "content": content})
            convo.updated_at = datetime.utcnow()

    def get_messages(self, conversation_id: str) -> list[dict]:
        convo = self._store.get(conversation_id)
        if not convo:
            return []
        return list(convo.messages)


conversation_store = InMemoryConversationStore()
