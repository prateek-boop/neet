"""
FastAPI backend wrapping embeddings.py (ingestion) and rag.py (paper generation)
behind an HTTP API, so the whole pipeline can run as one long-lived service
instead of one-shot CLI invocations.

Design notes:
  - The LLM + embedding model are loaded once at startup and kept warm in
    memory (loading a multi-GB model per request would be far too slow).
  - Ingestion and generation both use the GPU/CPU heavily, so they're
    serialized through a single-worker background queue rather than run
    concurrently per-request.
  - Long-running work (ingest, generate, full mock) is submitted as a job
    and polled via /jobs/{job_id} instead of blocking the HTTP request,
    since paper generation can take minutes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
from secrets import compare_digest
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

import rag
from embeddings import PipelineConfig, run_pipeline
from rag import (
    ConfigError,
    ModelLoadError,
    PaperVault,
    RAGConfig,
    VectorStoreError,
    check_system_memory,
    generate_full_neet_mock,
    generate_question_paper,
    load_data,
    load_embedding_model,
    load_llm,
)

log = logging.getLogger("server")
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)

UPLOAD_DIR = Path("./uploads")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "512"))

# Optional shared secret for the answer-key time-lock override. If unset,
# the override is disabled entirely — never silently open.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

# PaperVault ids look like "paper_1720900000_a1b2c3" — anything else is rejected
# before it can touch the filesystem.
_PAPER_ID_RE = re.compile(r"^paper_\d+_[0-9a-f]{6}$")

# Cap on finished jobs kept for polling; running/queued jobs are never pruned.
MAX_FINISHED_JOBS = 200


# ─────────────────────────────────────────────────────────────────────────────
# APP STATE
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self) -> None:
        self.cfg: RAGConfig | None = None
        self.vault: PaperVault | None = None

        self.tokenizer: Any = None
        self.llm: Any = None
        self.emb_model: Any = None
        self.milvus_index: Any = None
        self.bm25: Any = None
        self.pages_and_chunks: list[dict] = []
        self.is_multimodal: bool = False
        self.model_id: str | None = None

        self.status: Literal["initializing", "ready", "no_data", "error"] = "initializing"
        self.status_detail: str = "Starting up…"
        self.error: str | None = None

        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, dict[str, Any]] = {}

    def data_loaded(self) -> bool:
        return self.milvus_index is not None and self.bm25 is not None


state = AppState()


def _prune_jobs() -> None:
    finished = [
        (j["created_at"], j["id"])
        for j in state.jobs.values() if j["status"] in ("done", "failed")
    ]
    if len(finished) > MAX_FINISHED_JOBS:
        finished.sort()
        for _, job_id in finished[: len(finished) - MAX_FINISHED_JOBS]:
            state.jobs.pop(job_id, None)


def _new_job(kind: str) -> str:
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    state.jobs[job_id] = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "created_at": time.time(),
        "finished_at": None,
        "result": None,
        "error": None,
    }
    return job_id


def _run_job(job_id: str, fn, *args, **kwargs) -> None:
    job = state.jobs[job_id]
    job["status"] = "running"
    try:
        job["result"] = fn(*args, **kwargs)
        job["status"] = "done"
    except Exception as exc:
        log.error("Job %s failed:\n%s", job_id, traceback.format_exc())
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        job["finished_at"] = time.time()


def _submit(kind: str, fn, *args, **kwargs) -> str:
    job_id = _new_job(kind)
    state.executor.submit(_run_job, job_id, fn, *args, **kwargs)
    return job_id


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def _reload_index() -> None:
    """(Re)connects to Milvus and rebuilds the in-memory BM25 index."""
    milvus_index, bm25, chunks = load_data(state.cfg)
    with state.lock:
        state.milvus_index = milvus_index
        state.bm25 = bm25
        state.pages_and_chunks = chunks


def _initialize() -> None:
    try:
        state.status_detail = "Validating config…"
        cfg = RAGConfig()
        cfg.validate()
        state.cfg = cfg
        state.vault = PaperVault(cfg.vault_dir)

        rag._hf_login()

        state.status_detail = "Checking for local study material to auto-ingest…"
        rag.auto_ingest_data(cfg)

        state.status_detail = "Loading Milvus index + BM25…"
        try:
            _reload_index()
        except VectorStoreError as exc:
            log.warning("No data indexed yet: %s", exc)
            state.status = "no_data"
            state.status_detail = str(exc)

        state.status_detail = "Selecting LLM for available hardware…"
        model_id, torch_dtype, is_multimodal = check_system_memory(cfg)
        state.model_id = model_id
        state.is_multimodal = is_multimodal

        state.status_detail = f"Loading LLM ({model_id})… this can take a while on first run"
        tokenizer, llm = load_llm(model_id, torch_dtype, cfg, is_multimodal)
        state.tokenizer = tokenizer
        state.llm = llm

        state.status_detail = "Loading embedding model…"
        state.emb_model = load_embedding_model()

        if state.status != "no_data":
            state.status = "ready"
        state.status_detail = "Ready"
        log.info("Startup complete — status=%s, model=%s, multimodal=%s", state.status, model_id, is_multimodal)

    except (ConfigError, ModelLoadError) as exc:
        state.status = "error"
        state.error = str(exc)
        log.error("Startup failed: %s", exc)
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        log.error("Startup failed:\n%s", traceback.format_exc())


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Runs on the same background executor so it's serialized with any
    # ingest/generate request that arrives right after startup.
    state.executor.submit(_initialize)
    yield
    state.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="AI Teacher — NEET Paper Generator API", lifespan=lifespan)
# Comma-separated allowlist; defaults to "*" (open) so local/dev is unchanged.
# Set CORS_ORIGINS="https://app.example.com,https://admin.example.com" to lock down.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready() -> RAGConfig:
    if state.status == "error":
        raise HTTPException(503, f"Backend failed to start: {state.error}")
    if state.status == "initializing":
        raise HTTPException(503, f"Backend still starting up: {state.status_detail}")
    if not state.data_loaded():
        raise HTTPException(
            409,
            "No study material has been ingested yet. POST /ingest first.",
        )
    return state.cfg  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)


class FullMockRequest(BaseModel):
    subjects: list[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    body = {
        "status": state.status,
        "detail": state.status_detail,
        "error": state.error,
        "model_id": state.model_id,
        "is_multimodal": state.is_multimodal,
        "data_loaded": state.data_loaded(),
        "chunks_loaded": len(state.pages_and_chunks),
    }
    # 503 on a failed startup so container healthchecks flag the service;
    # "initializing" stays 200 — first boot legitimately takes minutes.
    return JSONResponse(body, status_code=503 if state.status == "error" else 200)


# ─────────────────────────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def _do_ingest(pdf_paths: list[str], catalog_paths: list[str], faq_paths: list[str], urls: list[str]) -> dict:
    cfg = PipelineConfig(
        pdfs=pdf_paths,
        catalogs=catalog_paths,
        faqs=faq_paths,
        urls=urls,
        output_dir=state.cfg.index_dir,
        milvus_uri_override=state.cfg.milvus_uri,
        milvus_collection_override=state.cfg.collection_name,
        milvus_visual_collection_override=state.cfg.visual_collection_name,
        incremental=True,
    )
    cfg.validate()
    metrics = run_pipeline(cfg)
    _reload_index()
    with state.lock:
        if state.status == "no_data":
            state.status = "ready"
    return {
        "sources_processed": metrics.sources_processed,
        "sources_skipped": metrics.sources_skipped,
        "sources_failed": metrics.sources_failed,
        "total_chunks": metrics.total_chunks,
        "total_images": metrics.total_images,
        "total_tables": metrics.total_tables,
        "chunks_loaded_after_reload": len(state.pages_and_chunks),
    }


_EXT_TO_KIND = {".pdf": "pdf", ".csv": "catalog", ".json": "faq"}


def _safe_upload_name(filename: str | None) -> str:
    """Client filenames are untrusted — reduce to a safe basename. Keeping the
    (sanitized) original name makes re-uploads of the same document update it
    in place via the incremental hash cache, instead of duplicating it."""
    name = Path(filename or "").name          # strips any directory components
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).lstrip(".")
    return name or uuid.uuid4().hex


def _save_upload(f: UploadFile, dest: Path) -> None:
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := f.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds MAX_UPLOAD_MB={MAX_UPLOAD_MB}")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


@app.post("/ingest")
async def ingest(
    files: list[UploadFile] | None = File(default=None),
    urls: str = Form(default=""),
):
    """
    Upload PDF / CSV(catalog) / JSON(FAQ) files and/or pass comma-separated
    URLs to scrape, and (re)build the Milvus index. Runs as a background job.
    """
    if state.cfg is None:
        if state.status == "error":
            raise HTTPException(503, f"Backend failed to start: {state.error}")
        raise HTTPException(503, f"Backend still starting up: {state.status_detail}")

    by_kind: dict[str, list[str]] = {"pdf": [], "catalog": [], "faq": []}
    for f in files or []:
        name = _safe_upload_name(f.filename)
        kind = _EXT_TO_KIND.get(Path(name).suffix.lower())
        if kind is None:
            raise HTTPException(400, f"Unsupported file type: {f.filename} (need .pdf/.csv/.json)")
        dest = UPLOAD_DIR / name
        _save_upload(f, dest)
        by_kind[kind].append(str(dest))

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    for u in url_list:
        if not u.startswith(("http://", "https://")):
            raise HTTPException(400, f"Invalid URL: {u}")

    if not (any(by_kind.values()) or url_list):
        raise HTTPException(400, "Provide at least one file or url.")

    job_id = _submit("ingest", _do_ingest, by_kind["pdf"], by_kind["catalog"], by_kind["faq"], url_list)
    return {"job_id": job_id}


@app.post("/reload")
def reload_index():
    if state.cfg is None:
        if state.status == "error":
            raise HTTPException(503, f"Backend failed to start: {state.error}")
        raise HTTPException(503, f"Backend still starting up: {state.status_detail}")
    job_id = _submit("reload", _reload_index)
    return {"job_id": job_id}


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _do_generate(request_text: str) -> dict:
    with state.lock:
        milvus_index, bm25, chunks = state.milvus_index, state.bm25, state.pages_and_chunks
    summary, paper_id = generate_question_paper(
        request_text,
        state.tokenizer, state.llm, state.emb_model,
        milvus_index, bm25, chunks,
        state.cfg, state.vault,
        is_multimodal=state.is_multimodal,
    )
    if paper_id is None:
        raise RuntimeError(f"Generation failed to produce a valid paper for: {summary}")
    return {"paper_id": paper_id, "summary": summary}


def _do_fullmock(subjects: list[str] | None) -> dict:
    with state.lock:
        milvus_index, bm25, chunks = state.milvus_index, state.bm25, state.pages_and_chunks
    summary, paper_id = generate_full_neet_mock(
        state.tokenizer, state.llm, state.emb_model,
        milvus_index, bm25, chunks,
        state.cfg, state.vault,
        is_multimodal=state.is_multimodal,
        subjects=subjects,
    )
    if paper_id is None:
        raise RuntimeError(f"Full mock generation failed: {summary}")
    return {"paper_id": paper_id, "summary": summary}


@app.post("/generate")
def generate(body: GenerateRequest):
    _require_ready()
    job_id = _submit("generate", _do_generate, body.request)
    return {"job_id": job_id}


@app.post("/generate/fullmock")
def generate_fullmock(body: FullMockRequest | None = None):
    _require_ready()
    subjects = body.subjects if body else None
    job_id = _submit("fullmock", _do_fullmock, subjects)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job


# ─────────────────────────────────────────────────────────────────────────────
# PAPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_vault() -> PaperVault:
    if state.vault is None:
        raise HTTPException(503, "Backend not ready")
    return state.vault


def _check_paper_id(paper_id: str) -> str:
    if not _PAPER_ID_RE.match(paper_id):
        raise HTTPException(404, f"No paper found with id '{paper_id}'")
    return paper_id


@app.get("/papers")
def list_papers():
    if state.vault is None:
        return []
    out = []
    for p in sorted(state.vault.path.glob("paper_*.json")):
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "paper_id": record.get("paper_id"),
            "paper_title": record.get("paper", {}).get("paper_title"),
            "subject": record.get("paper", {}).get("subject"),
            "generated_at": record.get("generated_at"),
            "unlock_at": record.get("unlock_at"),
        })
    return sorted(out, key=lambda r: r["generated_at"] or 0, reverse=True)


@app.get("/papers/{paper_id}")
def get_paper(paper_id: str):
    vault = _require_vault()
    path = vault._record_path(_check_paper_id(paper_id))
    if not path.exists():
        raise HTTPException(404, f"No paper found with id '{paper_id}'")
    record = json.loads(path.read_text(encoding="utf-8"))
    record.pop("answer_key", None)
    return record


@app.get("/papers/{paper_id}/html")
def get_paper_html(paper_id: str):
    vault = _require_vault()
    html_path = vault.path / f"{_check_paper_id(paper_id)}.html"
    if not html_path.exists():
        raise HTTPException(404, f"No HTML export found for paper '{paper_id}'")
    return FileResponse(html_path, media_type="text/html")


@app.get("/papers/{paper_id}/answerkey")
def get_answer_key(paper_id: str, x_admin_token: str | None = Header(default=None)):
    """
    Returns the answer key once the paper's time-lock expires (423 before that).
    The lock can be overridden by sending an X-Admin-Token header matching the
    ADMIN_TOKEN env var; if ADMIN_TOKEN is unset the override is disabled.
    """
    vault = _require_vault()
    force = False
    if x_admin_token is not None:
        if not (ADMIN_TOKEN and compare_digest(x_admin_token, ADMIN_TOKEN)):
            raise HTTPException(403, "Invalid admin token")
        force = True

    unlocked, payload = vault.get_answer_key(_check_paper_id(paper_id), force=force)
    if unlocked:
        return {"unlocked": True, "answer_key": payload}
    if isinstance(payload, (int, float)):
        return JSONResponse({"unlocked": False, "seconds_remaining": int(payload)}, status_code=423)
    raise HTTPException(404, str(payload))
