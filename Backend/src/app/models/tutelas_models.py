"""
Modelos SQLAlchemy alineados con el esquema limpio de conversaciones/mensajes.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import UUID, Column, DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class ChatRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AppUser(Base):
    __tablename__ = "app_users"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    email = Column(CITEXT, unique=True)
    role = Column(Text)
    created_at = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    conversations = relationship("Conversation", back_populates="user")


class Conversation(Base):
    __tablename__ = "app_conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("app_users.id"), nullable=True)
    title = Column(Text)
    status = Column(Text, nullable=False, server_default=text("'open'"))
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    last_message_at = Column(DateTime)
    meta_data = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    user = relationship("AppUser", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

    __table_args__ = (
        Index("ix_app_conversations_metadata_gin", "metadata", postgresql_using="gin"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("app_conversations.id"), nullable=False
    )
    sender = Column(Text, nullable=False)
    role = Column(Text)
    content = Column(Text)
    model = Column(Text)
    meta_data = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    embedding = Column(Vector(384))
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index("ix_messages_metadata_gin", "metadata", postgresql_using="gin"),
        Index(
            "ix_messages_embedding_ivfflat_cosine",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": 100},
        ),
    )

