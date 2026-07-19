# Container for the Streamlit split-screen demo (app.py). CPU-only, no GPU, no model weights:
# every LLM call goes out to a hosted provider (Groq / OpenRouter / Gemini) over the API, so the
# image is just Python + the deps in requirements.txt. Deploy to Cloud Run / App Runner / Fargate.
#
# The extractor fine-tune runs on Kaggle, not here — the tuned adapter is a training artifact, not
# something this image serves. The demo drives the same pipeline the API (api/main.py) exposes.
FROM python:3.11-slim

WORKDIR /app

# Deps first so this layer caches unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source + the small versioned order DB (data/db/orders.json). .dockerignore keeps the raw
# Kaggle downloads, training data, and caches out of the build context.
COPY . .

# The provider key is injected at runtime (Cloud Run secret / `docker run -e`), never baked in.
# Cloud Run sets $PORT (default 8080); shell-form CMD so it expands. Streamlit binds all
# interfaces so the platform can reach it.
EXPOSE 8080
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
