-- ============================================================================
-- SCHEMA: Tutelas app + Normografia
-- Target: PostgreSQL 14+ (pgvector, citext, pg_trgm, uuid-ossp, pgcrypto)
-- ============================================================================

-- Safety options (optional)
SET client_min_messages = warning;

-- ============================================================================
-- EXTENSIONS
--  - uuid-ossp        : uuid_generate_v4()
--  - pgcrypto         : gen_random_uuid() (used in normografia.node default)
--  - citext           : case-insensitive text for email/username
--  - vector           : pgvector for embeddings
--  - pg_trgm          : trigram index for fast ILIKE searches
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================================
-- SCHEMAS
-- ============================================================================
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS normografia;

-- ============================================================================
-- UTILITY FUNCTIONS & TRIGGERS
-- ============================================================================
-- Generic "updated_at" touch trigger (expects column 'updated_at')
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

-- "last_updated" touch trigger (expects column 'last_updated')
CREATE OR REPLACE FUNCTION public.set_last_updated()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.last_updated := now();
  RETURN NEW;
END;
$$;

-- ============================================================================
-- ENUM TYPES (derived from your model)
-- ============================================================================
-- Roles for app users
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
    CREATE TYPE public.user_role AS ENUM ('abogado','administrador','auditor');
  END IF;
END$$;

-- Tutela status
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tutela_estado') THEN
    CREATE TYPE public.tutela_estado AS ENUM ('pendiente','respondida');
  END IF;
END$$;

-- Assignment status
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'assign_estado') THEN
    CREATE TYPE public.assign_estado AS ENUM ('active','completed');
  END IF;
END$$;

-- Chat message role
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_role') THEN
    CREATE TYPE public.chat_role AS ENUM ('user','assistant','system');
  END IF;
END$$;

-- Agent execution status (obs_agent_logs)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_execution_status') THEN
    CREATE TYPE public.agent_execution_status AS ENUM ('running','success','failed','cancelled');
  END IF;
END$$;

-- Normografia relationship types
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'normografia_rel_type') THEN
    CREATE TYPE public.normografia_rel_type AS ENUM ('CONTAINS','MENTIONS','DESCRIBES','REFERS_TO','PERTENECE_A');
  END IF;
END$$;

-- ============================================================================
-- TABLES
-- ============================================================================

-- ===== Users (email citext)
CREATE TABLE IF NOT EXISTS public.app_users (
  id         uuid         PRIMARY KEY DEFAULT uuid_generate_v4(),
  username   citext       NOT NULL UNIQUE,     -- case-insensitive
  email      citext       NOT NULL UNIQUE,     -- requires citext extension
  full_name  varchar(255) NOT NULL,
  cedula     varchar(50)  NOT NULL UNIQUE,
  role       public.user_role NOT NULL,
  created_at timestamptz  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE public.app_users IS 'Primary user table';

-- ===== Conversations
CREATE TABLE IF NOT EXISTS public.app_conversations (
  id              uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         uuid        REFERENCES public.app_users(id) ON DELETE SET NULL,
  title           text,
  status          text        NOT NULL DEFAULT 'open', -- optional enum in the future
  started_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  last_message_at timestamptz,
  metadata        jsonb       NOT NULL DEFAULT '{}'::jsonb
);

-- Auto-update 'updated_at'
DROP TRIGGER IF EXISTS trg_app_conversations_updated_at ON public.app_conversations;
CREATE TRIGGER trg_app_conversations_updated_at
BEFORE UPDATE ON public.app_conversations
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ===== Messages (embedding vector, cascade)
CREATE TABLE IF NOT EXISTS public.messages (
  id               uuid        PRIMARY KEY DEFAULT uuid_generate_v4(),
  conversation_id  uuid        NOT NULL REFERENCES public.app_conversations(id) ON DELETE CASCADE,
  sender           text        NOT NULL, -- alternatively: sender_user_id uuid
  content          text,
  model            text,
  minio_bucket     text,
  minio_key        text,
  metadata         jsonb       NOT NULL DEFAULT '{}'::jsonb,
  created_at       timestamptz NOT NULL DEFAULT now(),
  message_id       uuid UNIQUE,            -- idempotency token (unique, NULL allowed)
  role             public.chat_role,
  embedding        vector(384),
  liked            boolean     NOT NULL DEFAULT false, -- Indica si el mensaje tiene like
  updated_at       timestamptz NOT NULL DEFAULT now()
);

-- Ensure "unique when not null" already satisfied via UNIQUE on nullable column in PG.
-- If you prefer a partial index instead of UNIQUE constraint, use the following:
-- CREATE UNIQUE INDEX IF NOT EXISTS ux_messages_message_id_notnull ON public.messages (message_id) WHERE message_id IS NOT NULL;

-- Auto-update 'updated_at'
DROP TRIGGER IF EXISTS trg_messages_updated_at ON public.messages;
CREATE TRIGGER trg_messages_updated_at
BEFORE UPDATE ON public.messages
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Optional ANN index for embeddings (requires pgvector >= 0.5)
-- Adjust lists per data scale; choose cosine/inner/Euclidean as needed.
CREATE INDEX IF NOT EXISTS ix_messages_embedding_ivfflat_cosine
ON public.messages USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ===== Tutelas (defaults & checks)
CREATE TABLE IF NOT EXISTS public.app_tutelas (
  id                  uuid            PRIMARY KEY DEFAULT uuid_generate_v4(),
  title               varchar(500)    NOT NULL,
  content             text            NOT NULL,
  estado              public.tutela_estado NOT NULL DEFAULT 'pendiente',
  categoria           varchar(100),
  solicitante_nombre  varchar(255),
  entidad_demandada   varchar(255),
  fecha_llegada       date,
  urgencia_score      numeric(5,2),
  creado_por_id       uuid            NOT NULL REFERENCES public.app_users(id) ON DELETE RESTRICT,
  asignado_a_id       uuid            REFERENCES public.app_users(id) ON DELETE SET NULL,
  conversation_id     uuid            REFERENCES public.app_conversations(id) ON DELETE SET NULL,
  metadata            jsonb           NOT NULL DEFAULT '{}'::jsonb,
  created_at          timestamptz     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at          timestamptz     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  responded_at        timestamptz,
  deadline_at         timestamptz
);

COMMENT ON TABLE public.app_tutelas IS 'Primary tutelas table';

-- Auto-update 'updated_at'
DROP TRIGGER IF EXISTS trg_app_tutelas_updated_at ON public.app_tutelas;
CREATE TRIGGER trg_app_tutelas_updated_at
BEFORE UPDATE ON public.app_tutelas
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ===== Asignaciones (1 active per tutela)
CREATE TABLE IF NOT EXISTS public.app_asignaciones (
  id            uuid            PRIMARY KEY DEFAULT uuid_generate_v4(),
  tutela_id     uuid            NOT NULL REFERENCES public.app_tutelas(id) ON DELETE CASCADE,
  asignado_a_id uuid            NOT NULL REFERENCES public.app_users(id) ON DELETE RESTRICT,
  estado        public.assign_estado NOT NULL DEFAULT 'active',
  assigned_at   timestamptz     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Enforce "1 active per tutela"
CREATE UNIQUE INDEX IF NOT EXISTS ux_app_asignaciones_one_active_per_tutela
ON public.app_asignaciones (tutela_id)
WHERE estado = 'active';

-- ===== Chat sessions
CREATE TABLE IF NOT EXISTS public.app_chat_sessions (
  id             uuid          PRIMARY KEY DEFAULT uuid_generate_v4(),
  owner_user_id  uuid          NOT NULL REFERENCES public.app_users(id) ON DELETE RESTRICT,
  tutela_id      uuid          REFERENCES public.app_tutelas(id) ON DELETE SET NULL,
  title          varchar(500),
  is_active      boolean       NOT NULL DEFAULT true,
  metadata       jsonb         NOT NULL DEFAULT '{}'::jsonb,
  created_at     timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Auto-update 'updated_at'
DROP TRIGGER IF EXISTS trg_app_chat_sessions_updated_at ON public.app_chat_sessions;
CREATE TRIGGER trg_app_chat_sessions_updated_at
BEFORE UPDATE ON public.app_chat_sessions
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ===== Document versions (MinIO bucket/key)
CREATE TABLE IF NOT EXISTS public.app_document_versions (
  id               uuid          PRIMARY KEY DEFAULT uuid_generate_v4(),
  tutela_id        uuid          NOT NULL REFERENCES public.app_tutelas(id) ON DELETE RESTRICT,
  chat_session_id  uuid          REFERENCES public.app_chat_sessions(id) ON DELETE SET NULL,
  version_number   int           NOT NULL,
  title            varchar(500)  NOT NULL,
  change_summary   text,
  created_by_id    uuid          NOT NULL REFERENCES public.app_users(id) ON DELETE RESTRICT,
  bucket           text          NOT NULL,
  object_key       text          NOT NULL,
  object_sha256    varchar(64),
  object_bytes     int,
  storage_meta     jsonb,
  created_at       timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Unique (tutela_id, version_number)
CREATE UNIQUE INDEX IF NOT EXISTS ux_app_document_versions_tutela_version
ON public.app_document_versions (tutela_id, version_number);

-- ===== Feedback (rating with check)
CREATE TABLE IF NOT EXISTS public.app_feedback (
  id               uuid          PRIMARY KEY DEFAULT uuid_generate_v4(),
  tutela_id        uuid          REFERENCES public.app_tutelas(id) ON DELETE SET NULL,
  user_id          uuid          REFERENCES public.app_users(id) ON DELETE SET NULL,
  chat_session_id  uuid          REFERENCES public.app_chat_sessions(id) ON DELETE SET NULL,
  rating           int,
  created_at       timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT chk_app_feedback_rating_range
    CHECK (rating IS NULL OR (rating BETWEEN 1 AND 5))
);

COMMENT ON TABLE public.app_feedback IS 'User feedback (rating 1..5)';

-- ===== Audit logs
CREATE TABLE IF NOT EXISTS public.obs_audit_logs (
  id           bigserial     PRIMARY KEY,
  user_id      uuid          REFERENCES public.app_users(id) ON DELETE SET NULL,
  user_email   citext,
  user_role    public.user_role,
  level        varchar(20),
  action       varchar(255)  NOT NULL,
  message      text,
  entity_type  varchar(255),
  entity_id    varchar(255),
  ip_address   inet,
  user_agent   text,
  metadata     jsonb,
  occurred_at  timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ===== Agent logs (enum added)
CREATE TABLE IF NOT EXISTS public.obs_agent_logs (
  id                 bigserial     PRIMARY KEY,
  conversation_id    uuid          REFERENCES public.app_conversations(id) ON DELETE SET NULL,
  trace_id           varchar(255)  NOT NULL,
  parent_log_id      bigint        REFERENCES public.obs_agent_logs(id) ON DELETE SET NULL,
  agent_name         varchar(100)  NOT NULL,
  agent_phase        varchar(50),
  operation          varchar(100),
  status             public.agent_execution_status NOT NULL,
  attempt_number     int           NOT NULL DEFAULT 1,
  started_at         timestamptz   NOT NULL,
  completed_at       timestamptz,
  execution_time_ms  int,
  input_summary      jsonb         NOT NULL DEFAULT '{}'::jsonb,
  output_summary     jsonb         NOT NULL DEFAULT '{}'::jsonb,
  error_type         varchar(255),
  error_message      text,
  error_stacktrace   text,
  metadata           jsonb         NOT NULL DEFAULT '{}'::jsonb,
  created_at         timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         timestamptz   NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Auto-update 'updated_at'
DROP TRIGGER IF EXISTS trg_obs_agent_logs_updated_at ON public.obs_agent_logs;
CREATE TRIGGER trg_obs_agent_logs_updated_at
BEFORE UPDATE ON public.obs_agent_logs
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- ============================================================================
-- NORMOGRAFIA (Graph-like model)
-- ============================================================================

-- ---- Table: normografia.node ----
CREATE TABLE IF NOT EXISTS normografia."node" (
  node_uid         uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
  element_id       text         UNIQUE,                 -- Neo4j elementId(n)
  label            text         NOT NULL,               -- Law/Section/Article/Chunk/Document
  graph_id         text         NOT NULL UNIQUE,        -- Logical ID in Neo4j

  -- Legal content
  nombre           text,
  texto            text,
  numero_original  text,
  tipo             text,
  source           text,
  processing_status text,
  total_chunks     int,
  hash             text,
  tematica         text,
  resumen_tematica text,

  -- Semantic vector
  embedding        vector(384),                         -- requires pgvector

  -- Metadata
  created_at       timestamptz  DEFAULT now(),
  last_updated     timestamptz,
  enabled          boolean      NOT NULL DEFAULT true,
  properties       jsonb,
  parent           uuid         REFERENCES normografia."node"(node_uid) ON DELETE SET NULL
);

COMMENT ON TABLE normografia."node" IS 'Norms/documents nodes. Indexes on (label), (enabled), trigram on nombre.';

-- Auto-update 'last_updated'
DROP TRIGGER IF EXISTS trg_normografia_node_last_updated ON normografia."node";
CREATE TRIGGER trg_normografia_node_last_updated
BEFORE UPDATE ON normografia."node"
FOR EACH ROW EXECUTE FUNCTION public.set_last_updated();

-- Helpful indexes for normografia.node
CREATE INDEX IF NOT EXISTS ix_normografia_node_label ON normografia."node"(label);
CREATE INDEX IF NOT EXISTS ix_normografia_node_enabled ON normografia."node"(enabled);
CREATE INDEX IF NOT EXISTS ix_normografia_node_nombre_trgm ON normografia."node" USING gin (nombre gin_trgm_ops);

-- Optional ANN index for embeddings (pgvector)
CREATE INDEX IF NOT EXISTS ix_normografia_node_embedding_ivfflat_cosine
ON normografia."node" USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- ---- Table: normografia.edge ----
CREATE TABLE IF NOT EXISTS normografia."edge" (
  edge_id     bigserial                    PRIMARY KEY,
  rel_type    public.normografia_rel_type  NOT NULL,
  start_node  uuid                         NOT NULL REFERENCES normografia."node"(node_uid) ON DELETE CASCADE, -- origin
  end_node    uuid                         NOT NULL REFERENCES normografia."node"(node_uid) ON DELETE CASCADE, -- target
  properties  jsonb,
  created_at  timestamptz                  DEFAULT now(),
  enabled     boolean                      NOT NULL DEFAULT true,
  CONSTRAINT ux_normografia_edge_unique UNIQUE (rel_type, start_node, end_node)
);

COMMENT ON TABLE normografia."edge" IS 'Relationships between nodes. Unique (rel_type, start_node, end_node).';

-- Useful indexes
CREATE INDEX IF NOT EXISTS ix_normografia_edge_start ON normografia."edge"(start_node);
CREATE INDEX IF NOT EXISTS ix_normografia_edge_end   ON normografia."edge"(end_node);
CREATE INDEX IF NOT EXISTS ix_normografia_edge_enabled ON normografia."edge"(enabled);

-- ============================================================================
-- OPTIONAL: PERFORMANCE/SEARCH INDEXES
-- ============================================================================
-- JSONB GIN indexes commonly used for metadata queries
CREATE INDEX IF NOT EXISTS ix_app_conversations_metadata_gin ON public.app_conversations USING gin (metadata);
CREATE INDEX IF NOT EXISTS ix_messages_metadata_gin            ON public.messages         USING gin (metadata);
CREATE INDEX IF NOT EXISTS ix_app_tutelas_metadata_gin         ON public.app_tutelas      USING gin (metadata);
CREATE INDEX IF NOT EXISTS ix_obs_agent_logs_metadata_gin      ON public.obs_agent_logs   USING gin (metadata);
CREATE INDEX IF NOT EXISTS ix_normografia_node_properties_gin  ON normografia."node"      USING gin (properties);

-- Convenience indexes
CREATE INDEX IF NOT EXISTS ix_messages_conversation_id_created_at ON public.messages (conversation_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_app_tutelas_estado_created_at        ON public.app_tutelas (estado, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_app_chat_sessions_active_updated     ON public.app_chat_sessions (is_active, updated_at DESC);

-- ============================================================================
-- COMMENTS (documentation)
-- ============================================================================
COMMENT ON COLUMN public.messages.embedding IS 'Sentence embedding (dimension 384).';
COMMENT ON COLUMN normografia."node".embedding IS 'Semantic embedding for the node (dimension 384).';
COMMENT ON COLUMN public.obs_agent_logs.status IS 'Execution lifecycle of an agent step.';
COMMENT ON COLUMN public.app_asignaciones.estado IS 'Only one "active" assignment per tutela enforced via partial unique index.';
