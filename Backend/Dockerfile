# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

# Build arguments with safe defaults so docker-compose can override them
ARG APP_USER=app

# Evita bytecode y buffers, mejora logs en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=off \
    PIP_DEFAULT_TIMEOUT=100 \
    # El código vive en `/app/src` (se copia y/o se monta vía docker-compose).
    # Para que `uvicorn app.main:app` funcione, `app` debe ser importable.
    PYTHONPATH=/app/src

WORKDIR /app

# Instala dependencias de Python primero para cachear capas
COPY requirements.txt /app/requirements.txt

# Nota performance:
# - `torch`, `transformers`, `spacy`, `datasets`, etc. hacen el build muy pesado (GBs).
# - En Windows/WSL el paso "exporting/unpacking" puede tardar mucho escribiendo capas grandes a disco.
# - Estos mounts permiten cachear APT y pip entre builds sin meter la caché dentro de la imagen.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
        build-essential \
        gcc \
        git \
        libpq-dev && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove \
        build-essential \
        gcc \
        git \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copia el código fuente
COPY src /app/src


# Comando por defecto: sirve la API FastAPI con uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
