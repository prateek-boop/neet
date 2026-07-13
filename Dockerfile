# GPU image — run with the NVIDIA Container Toolkit installed on the host
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
#
# Ubuntu 24.04 ships Python 3.12 — required by the pinned deps (numpy 2.5 /
# matplotlib 3.11 no longer support the 3.10 that Ubuntu 22.04 provides).
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# poppler-utils + tesseract: PDF page rendering / OCR fallback (subprocess calls)
# libgl1 + libglib2.0-0: required by opencv-python (pulled in by docling)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        poppler-utils tesseract-ocr tesseract-ocr-eng \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 blocks pip installs into the system Python (PEP 668) — use a venv.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first so `docker compose up --build` after a code-only change
# doesn't reinstall the (large) ML stack.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY embeddings.py rag.py server.py ./

RUN mkdir -p /app/index_store /app/uploads /app/data /app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
