FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencias del sistema realmente necesarias
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libpq-dev \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copiar código
COPY src /app/src

# Seguridad: no correr como root
RUN useradd -m appuser
USER appuser

# Puertos
EXPOSE 8000


ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
