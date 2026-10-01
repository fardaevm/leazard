# syntax=docker/dockerfile:1
# Leazard: FastAPI + LangGraph lease analyzer.
# Build:  docker build -t leazard .
# Run:    see README "Run with Docker".

ARG PYTHON_IMAGE=python:3.12-slim-trixie

# ── Builder: resolve dependencies from poetry.lock only ───────────────────────
FROM ${PYTHON_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Poetry lives in its own venv so it never ends up in the runtime image.
RUN python -m venv /opt/poetry \
 && /opt/poetry/bin/pip install "poetry==2.3.2" "poetry-plugin-export==1.10.1"

WORKDIR /build
COPY pyproject.toml poetry.lock ./
# Main dependencies only (no dev group), with hashes; installed into /install
# and later copied onto /usr/local, so the runtime image has no virtualenv.
RUN /opt/poetry/bin/poetry export --only main --format requirements.txt --output requirements.txt \
 && pip install --no-deps --require-hashes --prefix=/install -r requirements.txt

# ── Runtime ────────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

# Thread settings mirror the defaults main.py sets. HOME points at the tmpfs so
# libraries that want a cache dir work with a read-only root filesystem.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    TOKENIZERS_PARALLELISM=false \
    HOME=/tmp \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    PORT=8000 \
    DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/leaze.db \
    UPLOAD_DIR=/data/uploads \
    RAG_STORE_DIR=/data/rag_store \
    LAW_DIR=/app/data/law

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --home-dir /tmp --no-create-home --shell /usr/sbin/nologin app \
 && mkdir -p /data/uploads /data/rag_store \
 && chown -R 10001:10001 /data

COPY --from=builder /install /usr/local

# OpenAIEmbeddings tokenizes with tiktoken, which otherwise downloads this file on first use.
RUN python -c "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-small')" \
 && chmod -R a+rX /opt/tiktoken

COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

WORKDIR /app
# Law corpus changes rarely; code last for better layer caching.
COPY data/law/ data/law/
COPY ui/ ui/
COPY utils/ utils/
COPY rag/ rag/
COPY main.py agent.py db.py auth.py ./

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"]

# Starts as root only long enough to chown /data (Fly.io mounts volumes root-owned),
# then drops to UID 10001 with setpriv. Already non-root (compose `user:`) → just execs.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Exactly ONE worker: background jobs run in an in-process ThreadPoolExecutor with
# in-process admission locking, so extra workers would each run their own job pool,
# break the MAX_CONCURRENT_JOBS limit, and mark each other's jobs as interrupted on start.
# The port comes from $PORT (exported as UVICORN_PORT by the entrypoint).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "1"]
