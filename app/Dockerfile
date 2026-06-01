FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies for sentence-transformers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (layer cached when requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (volume-mounted in dev, included for completeness)
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
