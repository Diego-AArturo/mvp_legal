-- ============================================================================
-- SCHEMA: Conversaciones + Mensajes (PgVector) para MVP
-- Target: PostgreSQL 14+ (pgvector, citext, pg_trgm, uuid-ossp)
--
-- Objetivo:
-- - Guardar conversaciones y mensajes (incluyendo documentos generados por IA).
-- - Soportar búsqueda semántica via pgvector sobre `public.messages.embedding`.
-- - Eliminar todo lo relacionado a normografía/sincronización Postgres<->Neo4j.
-- ============================================================================

SET client_min_messages = warning;

-- ============================================================================
-- EXTENSIONS
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- SCHEMA
-- ============================================================================
CREATE SCHEMA IF NOT EXISTS public;

-- ============================================================================
-- UTILITY FUNCTION
-- ============================================================================
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

-- ============================================================================
-- ENUM TYPES
-- ============================================================================
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_role') THEN
    CREATE TYPE public.chat_role AS ENUM ('user','assistant','system');
  END IF;
END$$;

-- ============================================================================
-- TABLES
-- ============================================================================

-- Usuarios mínimos (solo lo necesario para asignar owner a conversaciones)
CREATE TABLE IF NOT EXISTS public.app_users (
  id         uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
  email      citext      UNIQUE,
  role       text,
  created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Conversaciones
CREATE TABLE IF NOT EXISTS public.app_conversations (
  id              uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         uuid        REFERENCES public.app_users(id) ON DELETE SET NULL,
  title           text,
  status          text        NOT NULL DEFAULT 'open',
  started_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  last_message_at timestamptz,
  metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

DROP TRIGGER IF EXISTS trg_app_conversations_updated_at ON public.app_conversations;
CREATE TRIGGER trg_app_conversations_updated_at
BEFORE UPDATE ON public.app_conversations
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Mensajes (incluye embeddings y metadata)
CREATE TABLE IF NOT EXISTS public.messages (
  id               uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
  conversation_id  uuid        NOT NULL REFERENCES public.app_conversations(id) ON DELETE CASCADE,
  sender           text        NOT NULL,
  role             public.chat_role,
  content          text,
  model            text,
  metadata         jsonb       NOT NULL DEFAULT '{}'::jsonb,
  embedding        vector(384),
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);

DROP TRIGGER IF EXISTS trg_messages_updated_at ON public.messages;
CREATE TRIGGER trg_messages_updated_at
BEFORE UPDATE ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS ix_messages_conversation_id_created_at
  ON public.messages (conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS ix_app_conversations_metadata_gin
  ON public.app_conversations USING gin (metadata);

CREATE INDEX IF NOT EXISTS ix_messages_metadata_gin
  ON public.messages USING gin (metadata);

-- ANN index for embeddings (pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS ix_messages_embedding_ivfflat_cosine
  ON public.messages USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON COLUMN public.messages.embedding IS 'Sentence embedding (dimension 384).';
