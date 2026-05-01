# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.12 slim — minimal OS footprint
FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
# gcc needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker caches this layer
# Only changes when requirements_deploy.txt changes
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# ── NLTK data ─────────────────────────────────────────────────────────────────
# Download stopwords + wordnet at build time so container doesn't
# need internet access at runtime
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# ── Copy project files ────────────────────────────────────────────────────────
# src/ — inference modules
COPY src/ ./src/

# flask_app/ — templates, static, routes, app.py
COPY flask_app/ ./flask_app/

# models/ — pkl files (tfidf + cnn_lstm only, no BERT)
COPY models/tfidf_augmented_15.pkl ./models/
COPY models/CNN_LSTM_augmented_mtsample.pkl ./models/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# gunicorn = production WSGI server (not Flask dev server)
# --workers 1 because F1 free tier is single-core + low RAM
# --timeout 120 for slow first inference
# app:app means flask_app/app.py → app object
CMD ["gunicorn", "--chdir", "flask_app", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:8000", "app:app"]