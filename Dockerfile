# Caesar's Rome: FastAPI app + graph. Runs locally with Docker and on Vercel's container runtime.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

# dependencies first, so this slow layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# only what runs (the allowlist in .dockerignore decides what exists in the build context)
RUN useradd --create-home --uid 10001 app
COPY --chown=app:app api.py ask.py ./
COPY --chown=app:app graph/ ./graph/
COPY --chown=app:app web/ ./web/
COPY --chown=app:app data/processed/graph.json ./data/processed/graph.json
USER app

EXPOSE 8000

# docker's own check; platforms may ignore it
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'))"]

# `exec` so SIGTERM reaches uvicorn and it shuts down cleanly (Vercel gives 30s grace).
# Secrets (ANTHROPIC_API_KEY, GITHUB_DISPATCH_TOKEN) come from the environment, never the image.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
