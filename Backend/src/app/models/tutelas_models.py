"""
Modelos SQLAlchemy para dominio de Tutelas.

Define todos los modelos de base de datos para tutelas, usuarios, conversaciones,
mensajes, asignaciones y feedback usando SQLAlchemy ORM.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    UUID,
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# =========================
# ======= ENUMS ===========
# =========================



class ChatRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"



# =========================
# ===== APP MODELS =======
# =========================



class ChatSession(Base):
    __tablename__ = "app_chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"), nullable=False)
    tutela_id = Column(UUID(as_uuid=True), ForeignKey("app_tutelas.id"))
    title = Column(String)
    is_active = Column(Boolean, nullable=False, default=True)
    meta_data = Column(JSONB)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    owner_user = relationship("User", back_populates="chat_sessions")
    tutela = relationship("Tutela", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")
    document_versions = relationship("DocumentVersion", back_populates="chat_session")
    feedback = relationship("Feedback", back_populates="chat_session")


class ChatMessage(Base):
    __tablename__ = "app_chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    session_id = Column(UUID(as_uuid=True), ForeignKey("app_chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # chat_role enum
    content = Column(Text, nullable=False)
    content_embedding = Column(Vector(1024))  # pgvector
    meta_data = Column(JSONB)
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    # Indexes
    __table_args__ = (Index("idx_chat_messages_session_created", "session_id", "created_at"),)

