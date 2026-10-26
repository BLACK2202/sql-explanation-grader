# SQL Explanation Grader: RAG Enhanced (API + dashboard)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    SQLRAG_STORE_DIR=/app/rag_store

WORKDIR /app

# CPU-only PyTorch keeps the image small; the LLM runs in the Ollama container.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container works offline.
ARG PRELOAD_EMBEDDINGS=1
RUN if [ "$PRELOAD_EMBEDDINGS" = "1" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"; \
    fi

COPY sqlrag ./sqlrag
COPY harness ./harness
COPY dashboard ./dashboard
COPY web ./web
COPY prompts ./prompts
COPY data ./data
COPY corpus ./corpus
COPY results ./results
COPY reports ./reports
COPY docker/entrypoint.sh /entrypoint.sh

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/rag_store /app/corpus/uploads /app/data/labels \
    && chown -R app:app /app /entrypoint.sh \
    && chmod +x /entrypoint.sh
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8).status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
