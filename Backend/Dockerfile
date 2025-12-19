FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencias del sistema mínimas necesarias
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libpq-dev \
        curl \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias
COPY requirements.txt /app/requirements.txt

# IMPORTANTE: cache pip persistente + no wheel compilation
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -m spacy download es_core_news_sm

# Copiar app
COPY src /app/src

# Crear usuario seguro
RUN useradd --create-home --shell /bin/sh appuser
USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini","--"]

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]
