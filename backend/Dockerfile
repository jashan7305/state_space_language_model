
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# Install dependencies using the existing lockfile
COPY backend/pyproject.toml backend/uv.lock /app/backend/

WORKDIR /app/backend

RUN uv sync --frozen --no-dev --no-install-project

# Copy backend application
COPY backend/main.py /app/backend/main.py
COPY backend/sslm_chat.py /app/backend/sslm_chat.py
COPY backend/dumbasslm_chat.py /app/backend/dumbasslm_chat.py
COPY backend/sslm /app/backend/sslm
COPY backend/dumbasslm /app/backend/dumbasslm

# Copy static frontend
COPY frontend /app/frontend

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
