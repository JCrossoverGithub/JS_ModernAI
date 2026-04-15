FROM python:3.11-slim

WORKDIR /app

# Install system deps for torch/CUDA (CPU fallback in container)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch with CUDA support (still works on CPU if EMBED_DEVICE=cpu)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]
