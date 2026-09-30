# FROM
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# WORKDIR
WORKDIR /app

# COPY AND RUN
# Install dependencies first so Docker can cache this layer between code changes.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Copy only what the API needs at runtime (includes app/model/model.pkl).
COPY app ./app

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# PORT
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# COMMAND
# Hosts like Render provide the port via $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
