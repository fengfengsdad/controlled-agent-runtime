FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    CORPUS_DIR=/app/data/corpus \
    CHECKPOINT_DB=/app/data/checkpoints.db \
    AUDIT_DB=/app/data/audit.db \
    VECTOR_DIR=/app/data/vector_store \
    LLM_PROVIDER=stub \
    EMBEDDING_PROVIDER=stub

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/corpus ./data/corpus
COPY eval ./eval

RUN pip install --no-cache-dir -e .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["uvicorn", "agent_runtime.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
