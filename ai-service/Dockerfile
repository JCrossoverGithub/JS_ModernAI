FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.6 support BEFORE Poetry resolves other deps,
# so Poetry sees an already-satisfied torch and skips the CPU wheel from PyPI.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Install Poetry
RUN pip install --no-cache-dir "poetry==1.8.3"

# Copy dependency manifests only (enables layer caching on code changes)
COPY pyproject.toml poetry.lock* ./

# Install all non-dev dependencies without creating a virtualenv inside the container
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev --no-root

COPY . .

EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
