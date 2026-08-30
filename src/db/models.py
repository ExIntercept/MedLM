"""SQLModel schemas for API users, consultations, and message histories."""
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ApiUser(SQLModel, table=True):
    __tablename__ = "api_users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, nullable=False)
    email: Optional[str] = Field(default=None, index=True, unique=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=_utcnow)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="api_users.id", index=True, nullable=False)
    title: Optional[str] = None
    patient_profile: Optional[str] = None  # JSON-serialized snapshot of the intake form
    created_at: datetime = Field(default_factory=_utcnow)


class ConversationMessage(SQLModel, table=True):
    __tablename__ = "conversation_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversations.id", index=True, nullable=False)
    role: str
    content: str
    meta: Optional[str] = None  # JSON-serialized (faithfulness score, status, etc.)
    created_at: datetime = Field(default_factory=_utcnow)
