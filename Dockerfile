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
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user. Override APP_UID/APP_GID at build time to match the
# host account that owns the bind mounts (./index_store, ./uploads) so the
# entrypoint's chown is a no-op instead of a rewrite:
#   docker compose build --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g)
ARG APP_UID=1000
ARG APP_GID=1000
# Ubuntu 24.04 ships a default 'ubuntu' user/group at UID/GID 1000 that collides
# with our default target ids — free whatever occupies them before creating app.
RUN existing_user="$(getent passwd ${APP_UID} | cut -d: -f1)"; \
    [ -n "$existing_user" ] && userdel -r "$existing_user" 2>/dev/null; \
    existing_group="$(getent group ${APP_GID} | cut -d: -f1)"; \
    [ -n "$existing_group" ] && groupdel "$existing_group" 2>/dev/null; \
    groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app

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
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Writable runtime dirs owned by the app user. (The entrypoint re-chowns these
# at boot too, to cover volumes/bind mounts that arrive root-owned.)
RUN mkdir -p /app/index_store /app/uploads /app/data /app/.cache/huggingface \
    && chown -R app:app /app

EXPOSE 8000

# Entrypoint starts as root only to fix mount ownership, then drops to the
# unprivileged `app` user (via gosu) before running this CMD.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
