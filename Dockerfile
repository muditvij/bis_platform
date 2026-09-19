FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ ./backend/
COPY data/ ./data/
COPY frontend/ ./frontend/
COPY api/ ./api/

# Pre-download FastEmbed BAAI/bge-small-en-v1.5 ONNX model at build time to eliminate runtime latency
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Pre-migrate knowledge base to skill markdown files
RUN python -m backend.migrate_kb

# Pre-populate SQLite and pre-compute embeddings/edges
RUN python -m backend.skill_loader

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Single worker fits 1GB RAM constraint
CMD ["uvicorn", "backend.app:api", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
