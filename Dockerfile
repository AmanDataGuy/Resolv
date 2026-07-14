# Container image for the Groq-backed pipeline + Streamlit demo (CPU only).
#
# RAG runs sentence-transformers on CPU; the detector/drafter call Groq over the
# API, so no GPU is needed. Deploy to AWS App Runner / ECS Fargate.
#
# The fine-tuned local backend (agents/drafter_local.py) is intentionally NOT served
# from this image — it needs a GPU + the adapter weights. In the container, keep the
# default DRAFTER_BACKEND=groq; serve the fine-tuned model separately (e.g. a
# SageMaker endpoint) if you want it live.
FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer caches unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + the small mock fixtures (.dockerignore keeps raw data / models out).
COPY . .

# GROQ_API_KEY is injected at runtime (App Runner secret / `docker run -e`), never baked in.
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
