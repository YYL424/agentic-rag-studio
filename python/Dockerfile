FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-sim poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements*.txt ./
ARG REQUIREMENTS_FILE=requirements-core.txt
RUN pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

COPY . .

RUN useradd --create-home --uid 10001 agenthub && \
    mkdir -p /app/uploads /app/data && \
    chown -R agenthub:agenthub /app

USER agenthub

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health/live', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
