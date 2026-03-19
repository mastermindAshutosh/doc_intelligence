FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ✅ Install only required system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# ✅ Upgrade pip
RUN pip install --upgrade pip

# ✅ Install CPU-only PyTorch FIRST (critical fix)
RUN pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# ✅ Copy dependency file
COPY pyproject.toml /app/

# ✅ Install project deps (without reinstalling torch)
RUN pip install .

# 🔥 Preload model (keeps cold start lower)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# ✅ Copy source code
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
