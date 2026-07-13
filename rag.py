from __future__ import annotations
import argparse
import base64
import html
import json
import logging
import os
import re
import secrets
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(format=fmt, datefmt="%Y-%m-%d %H:%M:%S", level=level.upper())

    for logger_name in ["httpx", "httpcore", "sentence_transformers", "huggingface_hub", "transformers", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    return logging.getLogger("teacher.rag")


log = _setup_logging(os.environ.get("LOG_LEVEL", "INFO"))


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipelineError(RuntimeError):
    """Base error for this module."""

class ModelLoadError(RAGPipelineError):
    """Raised when a model cannot be loaded."""

class VectorStoreError(RAGPipelineError):
    """Raised when the Milvus collection cannot be reached or read."""

class ConfigError(RAGPipelineError):
    """Raised for invalid configuration."""


# ─────────────────────────────────────────────────────────────────────────────
# ENV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    # Index
    index_dir:   Path  = field(default_factory=lambda: Path(os.environ.get("INDEX_DIR", "./index_store")))

    # Retrieval — a full paper needs more grounding material than a single fact-check
    top_k:       int   = field(default_factory=lambda: int(os.environ.get("RAG_TOP_K",       "8")))
    min_score:   float = field(default_factory=lambda: float(os.environ.get("RAG_MIN_SCORE", "0.25")))

    # Floor only — _estimate_max_new_tokens() scales this up per-request for large
    # question counts; an explicit --max-tokens always wins if higher than the estimate.
    max_new_tokens: int  = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "4096")))
    quantize_4bit:  bool = field(default_factory=lambda: os.environ.get("QUANTIZE", "4bit").lower() == "4bit")

    # check_system_memory() picks the strongest multimodal model that fits
    # available VRAM/RAM, so a retrieved figure can be genuinely looked at, not just cited.
    enable_vision: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_VISION", "true").lower() not in ("0", "false", "no")
    )
    max_context_images: int = field(
        default_factory=lambda: int(os.environ.get("MAX_CONTEXT_IMAGES", "4"))
    )

    # Failed attempts are regenerated with the violations fed back into the prompt,
    # up to this many EXTRA tries (total attempts = max_generation_retries + 1).
    max_generation_retries: int = field(
        default_factory=lambda: int(os.environ.get("RAG_MAX_RETRIES", "2"))
    )

    # Milvus — override either to point at a remote server / Zilliz Cloud later
    milvus_uri_override:        str = field(default_factory=lambda: os.environ.get("MILVUS_URI", ""))
    milvus_collection_override: str = field(default_factory=lambda: os.environ.get("MILVUS_COLLECTION", ""))
    visual_collection_override: str = field(default_factory=lambda: os.environ.get("MILVUS_VISUAL_COLLECTION", ""))

    def validate(self) -> None:
        if self.top_k < 1:
            raise ConfigError("top_k must be ≥ 1")
        if not (0.0 <= self.min_score <= 1.0):
            raise ConfigError("min_score must be in [0.0, 1.0]")
        if self.max_new_tokens < 64:
            raise ConfigError("max_new_tokens must be ≥ 64")
        if self.max_generation_retries < 0:
            raise ConfigError("max_generation_retries must be ≥ 0")
        if self.max_context_images < 0:
            raise ConfigError("max_context_images must be ≥ 0")

    @property
    def milvus_uri(self) -> str:
        """Must match the .db file (or server URI) embeddings.py wrote to."""
        if self.milvus_uri_override:
            return self.milvus_uri_override
        return str(self.index_dir / "milvus.db")

    @property
    def collection_name(self) -> str:
        return self.milvus_collection_override or "multimodal_text"

    @property
    def visual_collection_name(self) -> str:
        return self.visual_collection_override or "multimodal_visual"

    @property
    def vault_dir(self) -> Path:
        """Where generated papers + locked answer keys are stored."""
        return self.index_dir / "papers"

    @property
    def generated_diagrams_dir(self) -> Path:
        """Rendered output of deterministic diagram generation — see render_generated_diagram()."""
        return self.index_dir / "generated_diagrams"


@dataclass
class MilvusIndex:
    """Thin handle bundling the connected client + the collection it talks to."""
    client: MilvusClient
    collection_name: str
    visual_collection_name: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# PAPER VAULT  (time-locked answer keys)
# ─────────────────────────────────────────────────────────────────────────────

class PaperVault:
    """
    Persists generated papers + their answer keys, enforcing a time-lock
    before the answer key can be revealed.

    Unlock rule: the answer key becomes visible as soon as EITHER condition
    is met — the exam's stated duration has elapsed since generation, OR the
    calendar day has rolled over since generation — whichever comes first.
    """

    DEFAULT_DURATION_MIN = 180   # typical NEET/JEE Main duration as a fallback

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def _record_path(self, paper_id: str) -> Path:
        return self.path / f"{paper_id}.json"

    @staticmethod
    def _compute_unlock_at(generated_at: float, duration_minutes: int) -> float:
        by_duration = generated_at + duration_minutes * 60

        gen_dt = datetime.fromtimestamp(generated_at)
        next_midnight = datetime(gen_dt.year, gen_dt.month, gen_dt.day) + timedelta(days=1)
        by_next_day = next_midnight.timestamp()

        return min(by_duration, by_next_day)

    def save(self, parsed: dict) -> dict:
        """
        Strips the answer key out of `parsed` (mutates it in place so the
        caller can safely print/display the rest), stores both halves to
        disk, and returns the vault record (paper_id, unlock_at, etc.).
        """
        try:
            duration = int(parsed.get("duration_minutes", self.DEFAULT_DURATION_MIN))
        except (TypeError, ValueError):
            duration = self.DEFAULT_DURATION_MIN

        answer_key = parsed.pop("answer_key", [])
        validation_warnings = parsed.pop("_validation_warnings", [])

        generated_at = time.time()
        paper_id = f"paper_{int(generated_at)}_{secrets.token_hex(3)}"
        unlock_at = self._compute_unlock_at(generated_at, duration)

        record = {
            "paper_id":            paper_id,
            "generated_at":        generated_at,
            "duration_minutes":    duration,
            "unlock_at":           unlock_at,
            "paper":               parsed,
            "answer_key":          answer_key,
            "validation_warnings": validation_warnings,
        }
        self._record_path(paper_id).write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def get_answer_key(self, paper_id: str, force: bool = False) -> tuple[bool, Any]:
        """
        Returns (unlocked: bool, payload). If unlocked, payload is the
        answer_key list. If still locked, payload is seconds remaining.
        `force=True` is an explicit admin override — not used by any
        default code path, only via a deliberate CLI flag or command.
        """
        path = self._record_path(paper_id)
        if not path.exists():
            return False, f"No paper found with id '{paper_id}'."

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"Could not read paper record: {exc}"

        if force or time.time() >= record["unlock_at"]:
            return True, record["answer_key"]

        return False, record["unlock_at"] - time.time()


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MEMORY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_system_memory(cfg: RAGConfig) -> tuple[str, torch.dtype, bool]:
    """
    Inspect GPU VRAM (primary signal on a discrete-GPU machine) and system RAM
    (CPU fallback), and select the strongest model that fits, gated by cfg.enable_vision:
      - CUDA, >=10 GB VRAM  -> Qwen2.5-VL-7B-Instruct, 4-bit (purpose-built for
        document/chart/table comprehension)
      - CUDA, 6-10 GB VRAM  -> gemma-3-4b-it, 4-bit (multimodal, lighter)
      - No GPU, RAM>=16GB   -> gemma-3-4b-it, float16 on CPU (slow but works)
      - Otherwise           -> gemma-3-1b-it (text-only, no vision_config at all)

    Returns (model_id, torch_dtype, is_multimodal).
    """
    if not cfg.enable_vision:
        log.info("Vision disabled (--no-vision) → gemma-3-1b-it (text-only)")
        return "google/gemma-3-1b-it", torch.float16, False

    vram_gb = 0.0
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            log.info("GPU detected: %s (%.1f GB VRAM)", props.name, vram_gb)
        except Exception as exc:
            log.warning("Could not query GPU VRAM: %s", exc)

    if vram_gb >= 10:
        log.info(
            "%.1f GB VRAM → using Qwen2.5-VL-7B-Instruct (multimodal, 4-bit) — "
            "best available here for genuine diagram + dense-table comprehension",
            vram_gb,
        )
        return "Qwen/Qwen2.5-VL-7B-Instruct", torch.bfloat16, True

    if vram_gb >= 6:
        log.info("%.1f GB VRAM → using gemma-3-4b-it (multimodal, 4-bit)", vram_gb)
        return "google/gemma-3-4b-it", torch.float16, True

    # No capable GPU — fall back to the original RAM-based CPU selection.
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024 ** 3))
        log.info("System RAM — total: %d GB  available: %d GB", total_gb, round(vm.available / (1024 ** 3)))
    except ImportError:
        log.warning("psutil not installed — cannot detect RAM. Defaulting to 1b text-only model.")
        total_gb = 0

    if total_gb >= 16:
        log.info("No capable GPU, 16 GB+ RAM → using gemma-3-4b-it (multimodal, float16 CPU — will be slow)")
        return "google/gemma-3-4b-it", torch.float16, True

    log.warning(
        "No capable GPU and <16 GB RAM detected — falling back to gemma-3-1b-it (text-only): "
        "diagram-based questions will be citation-only, not genuine visual comprehension. "
        "Pass --no-vision to silence this warning."
    )
    return "google/gemma-3-1b-it", torch.float16, False


# ─────────────────────────────────────────────────────────────────────────────
# HF AUTH
# ─────────────────────────────────────────────────────────────────────────────

def _hf_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token") or os.environ.get("hf_token")

    if not token or token == "YOUR_HF_TOKEN_HERE":
        log.warning("HF_TOKEN not found in environment or .env file.")
        return
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token)
        log.info("Hugging Face login successful.")
    except Exception as exc:
        log.warning("HF login failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_QWEN_VL_MIN_PIXELS = 256 * 28 * 28    # processor resolution floor
_QWEN_VL_MAX_PIXELS = 1024 * 28 * 28   # cap — bounds vision-token count (and VRAM) per image


def _is_qwen_vl(model_id: str) -> bool:
    return "qwen" in model_id.lower() and "-vl" in model_id.lower()


def load_llm(model_id: str, torch_dtype: torch.dtype, cfg: RAGConfig, is_multimodal: bool) -> tuple:
    """
    Load processor/tokenizer + LLM with optional 4-bit quantization, falling back
    through 4-bit CUDA -> float16/bf16 CUDA -> CPU. Multimodal models load via
    AutoProcessor + AutoModelForImageTextToText (the class that wires up the vision
    tower); text-only models use the plain AutoTokenizer + AutoModelForCausalLM path.
    min_pixels/max_pixels bound Qwen2.5-VL's per-image vision-token cost.
    """
    if is_multimodal:
        log.info("Loading processor (multimodal): %s", model_id)
        processor_kwargs = (
            dict(min_pixels=_QWEN_VL_MIN_PIXELS, max_pixels=_QWEN_VL_MAX_PIXELS)
            if _is_qwen_vl(model_id) else {}
        )
        try:
            processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        except Exception as exc:
            raise ModelLoadError(f"Processor load failed for '{model_id}': {exc}") from exc
        model_cls = AutoModelForImageTextToText
    else:
        log.info("Loading tokenizer: %s", model_id)
        try:
            processor = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except Exception as exc:
            raise ModelLoadError(f"Tokenizer load failed for '{model_id}': {exc}") from exc
        model_cls = AutoModelForCausalLM

    def _try_load(kwargs: dict) -> Any:
        return model_cls.from_pretrained(model_id, **kwargs)

    attempts = []

    if cfg.quantize_4bit and torch.cuda.is_available():
        attempts.append((f"4-bit quantized (CUDA, {torch_dtype})", dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            trust_remote_code=True,
        )))

    if torch.cuda.is_available():
        attempts.append((f"{torch_dtype} (CUDA)", dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="cuda",
        )))

    attempts.append((f"{torch_dtype} (CPU fallback)", dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )))

    for label, kwargs in attempts:
        try:
            log.info("Attempting model load: %s …", label)
            llm = _try_load(kwargs)
            log.info("Model loaded via: %s", label)
            return processor, llm
        except Exception as exc:
            log.warning("%s load failed: %s — trying next option.", label, exc)

    raise ModelLoadError(
        f"All load strategies failed for '{model_id}'. "
        "Check HF_TOKEN, internet access, and available RAM."
    )


def load_embedding_model() -> SentenceTransformer:
    model_name = "all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading embedding model: %s on %s", model_name, device)
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise ModelLoadError(f"Embedding model load failed: {exc}") from exc


_visual_embedding_model: SentenceTransformer | None = None


def load_visual_embedding_model() -> SentenceTransformer:
    """Load CLIP lazily only when a linked visual collection exists."""
    global _visual_embedding_model
    if _visual_embedding_model is None:
        model_name = "sentence-transformers/clip-ViT-B-32"
        log.info("Loading visual embedding model: %s on cpu", model_name)
        try:
            _visual_embedding_model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            raise ModelLoadError(f"Visual embedding model load failed: {exc}") from exc
    return _visual_embedding_model


# ─────────────────────────────────────────────────────────────────────────────
# INDEX LOADING  (Milvus collection + BM25 corpus)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(cfg: RAGConfig) -> tuple[MilvusIndex, BM25Okapi, list[dict]]:
    """
    Connects to the Milvus collection built by embeddings.py, pulls
    every chunk into memory (paginated) to build a BM25 keyword index
    alongside the vector store, and returns a handle to both.
    """
    try:
        client = MilvusClient(uri=cfg.milvus_uri)
    except Exception as exc:
        raise VectorStoreError(f"Could not open Milvus at '{cfg.milvus_uri}': {exc}") from exc

    if not client.has_collection(cfg.collection_name):
        raise VectorStoreError(
            f"Collection '{cfg.collection_name}' not found at {cfg.milvus_uri}. "
            "Run embeddings.py first to ingest study material."
        )

    visual_collection = (
        cfg.visual_collection_name
        if client.has_collection(cfg.visual_collection_name)
        else None
    )
    milvus_index = MilvusIndex(
        client=client,
        collection_name=cfg.collection_name,
        visual_collection_name=visual_collection,
    )
    client.load_collection(cfg.collection_name)
    if visual_collection:
        client.load_collection(visual_collection)

    log.info("Loading chunks from Milvus collection '%s' …", cfg.collection_name)
    all_chunks: list[dict] = []
    batch_size, offset = 1000, 0

    while True:
        try:
            batch = client.query(
                collection_name=cfg.collection_name,
                filter="id >= 0",   # catch-all: our stable IDs are always non-negative
                output_fields=[
                    "id", "chunk", "source", "source_type", "word_start", "word_end",
                    "page_no", "element_type", "record_level", "bbox", "asset_path",
                    "page_image_path", "element_ids", "relation_ids", "section_path",
                ],
                limit=batch_size,
                offset=offset,
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to read chunks from Milvus: {exc}") from exc

        if not batch:
            break
        all_chunks.extend(batch)
        offset += len(batch)
        if len(batch) < batch_size:
            break

    if not all_chunks:
        raise VectorStoreError(
            f"Collection '{cfg.collection_name}' is empty. "
            "Run embeddings.py first to ingest study material."
        )

    log.info("Loaded %d chunks from Milvus.", len(all_chunks))

    log.info("Building BM25 index (keyword search) …")
    tokenized_corpus = [c.get("chunk", "").lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return milvus_index, bm25, all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    request:          str,
    emb_model:        SentenceTransformer,
    milvus_index:     MilvusIndex,
    bm25:             BM25Okapi,
    pages_and_chunks: list[dict],
    cfg:              RAGConfig,
) -> list[dict]:
    """
    Hybrid Search: combines Dense (Milvus) and Sparse (BM25) results,
    deduplicated by each chunk's stable Milvus id (not list position —
    Milvus row order isn't guaranteed to match pages_and_chunks order).
    """
    merged: dict[int, dict] = {}
    output_fields = [
        "chunk", "source", "source_type", "word_start", "word_end", "page_no",
        "element_type", "record_level", "bbox", "asset_path", "page_image_path",
        "element_ids", "relation_ids", "section_path",
    ]

    # 1. Dense Retrieval (Milvus)
    query_vec = emb_model.encode([request]).tolist()
    try:
        results = milvus_index.client.search(
            collection_name=milvus_index.collection_name,
            data=query_vec,
            limit=cfg.top_k,
            output_fields=output_fields,
        )
        for hit in results[0] if results else []:
            score = float(hit.get("distance", 0.0))
            if score < cfg.min_score:
                continue
            key = hit.get("id")
            entity = hit.get("entity", {})
            merged[key] = {
                **entity,
                "id": key,
                "_score": score,
                "_type": "dense",
                "_rank_score": 1.0,
            }
    except Exception as exc:
        log.warning("Milvus dense search failed: %s", exc)

    # 2. Cross-modal retrieval: CLIP text query against pages and visual regions.
    if milvus_index.visual_collection_name:
        try:
            visual_model = load_visual_embedding_model()
            visual_vector = visual_model.encode([request], normalize_embeddings=True).tolist()
            visual_results = milvus_index.client.search(
                collection_name=milvus_index.visual_collection_name,
                data=visual_vector,
                limit=cfg.top_k,
                output_fields=output_fields,
            )
            for rank, hit in enumerate(visual_results[0] if visual_results else [], 1):
                key = hit.get("id")
                entity = hit.get("entity", {})
                score = float(hit.get("distance", 0.0))
                if key in merged:
                    merged[key]["_type"] = "dense+visual"
                    merged[key]["_visual_score"] = score
                    merged[key]["_rank_score"] += 1.0 / rank
                else:
                    merged[key] = {
                        **entity,
                        "id": key,
                        "_score": score,
                        "_visual_score": score,
                        "_type": "visual",
                        "_rank_score": 1.0 / rank,
                    }
        except Exception as exc:
            log.warning("Milvus visual search failed: %s", exc)

    # 3. Sparse Retrieval (BM25)
    tokenized_query = request.lower().split()
    s_scores = bm25.get_scores(tokenized_query)
    s_indices = np.argsort(s_scores)[::-1][:cfg.top_k]

    for idx in s_indices:
        score = float(s_scores[idx])
        if score <= 0 or not (0 <= idx < len(pages_and_chunks)):
            continue
        chunk = pages_and_chunks[idx]
        key = chunk.get("id")
        if key in merged:
            merged[key]["_type"] = "hybrid"
            merged[key]["_rank_score"] = merged[key].get("_rank_score", 0.0) + 0.25
        else:
            merged[key] = {
                **chunk,
                "_score": score,
                "_type": "sparse",
                "_rank_score": 0.25,
            }

    results = sorted(
        merged.values(),
        key=lambda item: (item.get("_rank_score", 0.0), item.get("_score", 0.0)),
        reverse=True,
    )[: cfg.top_k * 2]
    log.debug("Hybrid Retrieval: %d unique chunks found.", len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

_ASSERTION_REASON_OPTIONS = [
    "A) Both Assertion and Reason are true, and Reason is the correct explanation of Assertion",
    "B) Both Assertion and Reason are true, but Reason is NOT the correct explanation of Assertion",
    "C) Assertion is true, but Reason is false",
    "D) Both Assertion and Reason are false",
]

# Real NTA-NEET per-subject structure: Section A is compulsory, Section B
# offers extra questions of which only a subset must be attempted.
NEET_SECTION_A_COUNT = 35
NEET_SECTION_B_COUNT = 15
NEET_SECTION_B_ATTEMPT = 10


def _infer_paper_scale(request: str) -> tuple[bool, int]:
    """
    Decide whether this request wants the full NEET per-subject structure
    (Section A: 35 compulsory + Section B: 15, attempt any 10) or a smaller
    custom count (single compulsory section, still MCQ-only / +4-per-1).

    Looks for an explicit question count in the request (e.g. "20 questions",
    "15 MCQs"); requests mentioning "full"/"complete"/"mock" or with no
    explicit count default to the full structure.
    """
    match = re.search(r"\b(\d{1,3})\s*(?:questions?|mcqs?|ques\.?)\b", request, re.IGNORECASE)
    if match:
        count = int(match.group(1))
        if count < NEET_SECTION_A_COUNT:
            return False, count
    return True, NEET_SECTION_A_COUNT + NEET_SECTION_B_COUNT


# Per-question token budget (stem + 4 options + metadata + answer_key entry, with
# headroom) plus a fixed overhead for title/instructions/JSON structure. A full
# 50-question NEET paper needs roughly 300*50+800 ~= 15,800 tokens.
_TOKENS_PER_QUESTION = 300
_TOKENS_FIXED_OVERHEAD = 800


def _estimate_max_new_tokens(target_count: int, cfg: RAGConfig) -> int:
    """Dynamic output-token budget for `target_count` questions.
    Never goes below cfg.max_new_tokens, so an explicit override acts as a floor."""
    estimated = _TOKENS_PER_QUESTION * target_count + _TOKENS_FIXED_OVERHEAD
    return max(cfg.max_new_tokens, estimated)


def build_exam_prompt(
    request: str,
    context_items: list[dict],
    full_structure: bool,
    target_count: int,
    history: list[dict] | None = None,
    violations: list[str] | None = None,
    images_attached: bool = False,
) -> str:
    """
    Build the prompt for a strict NEET-pattern paper grounded in the retrieved
    study material (MCQ/Assertion-Reason only, +4/-1 marking). `full_structure`/
    `target_count` come from _infer_paper_scale(request). `violations` carries
    rule failures from a previous validate_neet_paper() attempt, fed back so a
    retry can self-correct. `images_attached` is true only when running the
    multimodal model with real images loaded for this request.
    """
    context_blocks = []
    for item in context_items:
        chunk = item.get("chunk", "").strip()
        if not chunk:
            continue
        locator = (
            f"SOURCE PAGE: {item.get('page_no', 'unknown')} | "
            f"TYPE: {item.get('element_type', item.get('source_type', 'text'))} | "
            f"ASSET: {item.get('asset_path', '')} | BBOX: {item.get('bbox', '')}"
        )
        context_blocks.append(f"{locator}\n{chunk}")
    context_str = "\n---\n".join(context_blocks)
    if not context_str:
        context_str = "No reference material found for this topic in the local dataset."

    history_str = ""
    if history:
        history_str = "\nRECENT REQUESTS (for follow-up context, e.g. 'make it harder'):\n" + "\n".join(
            f"- {h['request']} → {h['summary']}" for h in history
        )

    if full_structure:
        structure_str = f"""Use the REAL NTA-NEET two-section structure for this subject:
- Section A: EXACTLY {NEET_SECTION_A_COUNT} questions, numbered 1-{NEET_SECTION_A_COUNT}, ALL compulsory.
- Section B: EXACTLY {NEET_SECTION_B_COUNT} questions, numbered 1-{NEET_SECTION_B_COUNT}, of which the
  candidate attempts any {NEET_SECTION_B_ATTEMPT}. State this in that section's "section_name"
  (e.g. "Section B — Attempt any {NEET_SECTION_B_ATTEMPT} of {NEET_SECTION_B_COUNT}")."""
    else:
        structure_str = f"""Use a SINGLE compulsory section of EXACTLY {target_count} questions,
numbered 1-{target_count}. Do not split into Section A/B for a custom-sized request like this."""

    violations_str = ""
    if violations:
        violations_str = (
            "\nYOUR PREVIOUS ATTEMPT VIOLATED THESE STRICT RULES — FIX EVERY ONE:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n"
        )

    if images_attached:
        image_rule = (
            "4. Relevant images ARE ATTACHED to this message directly (in the order their ASSET\n"
            "   paths appear in the reference material below). Actually LOOK at each attached image\n"
            "   before writing a question grounded in it — describe/interpret genuine visual content\n"
            "   (identify a structure, read a graph, compare labelled parts) rather than a generic\n"
            "   restatement of its caption. Copy that item's exact ASSET path into the question's\n"
            "   \"figure_ref\" field as a citation. If no image is relevant, \"figure_ref\" is \"\"."
        )
    else:
        image_rule = (
            "4. No images are attached to this message (text-only mode) — you only have each image\n"
            "   element's caption/citation, not its actual visual content. Do NOT invent what a figure\n"
            "   shows. Only reference an image (copying its ASSET path into \"figure_ref\") when its\n"
            "   caption alone gives you enough to write a grounded question; otherwise leave \"figure_ref\" \"\"."
        )

    return f"""You are an experienced NTA-NEET exam setter. Your job is to design a rigorous,
NEET-pattern question paper STRICTLY grounded in the REFERENCE MATERIAL below.

NEET IS SINGLE-CORRECT MCQ ONLY. There is NO short-answer, long-answer, or numerical-value
question type in this paper — every single question is one of exactly two kinds:
  1. STANDARD  — a stem with EXACTLY 4 options ("A) ...", "B) ...", "C) ...", "D) ...").
  2. ASSERTION_REASON — an "Assertion (A): ..." and "Reason (R): ..." pair as the question
     stem, paired with EXACTLY these 4 options, verbatim, in this exact order:
     {json.dumps(_ASSERTION_REASON_OPTIONS)}

MARKING: every question is worth EXACTLY 4 marks, marking_scheme is EXACTLY "+4 / -1"
(one mark deducted per wrong attempt, zero for unattempted). No other marks value or
marking scheme is valid anywhere in this paper.

{structure_str}

GUIDELINES:
1. Base every question on facts present in the REFERENCE MATERIAL. Do not invent facts
   not supported by it.
2. If the material is insufficient for the target question count, generate fewer
   high-quality questions rather than padding with unsupported content — note the
   shortfall in "coverage_note".
3. Tag each question with its topic and difficulty (Easy / Moderate / Difficult).
{image_rule}
5. If a question is grounded in a table element (TYPE: table in the reference material —
   its full markdown table is given directly in the reference text, not just a caption),
   you may build a genuine data-interpretation question from it: copy that table's EXACT
   markdown (the "| ... |" rows, unmodified) into the question's "table_data" field so the
   student sees the same table. Otherwise "table_data" is "". Never invent numbers or rows
   that are not literally present in the retrieved table.
6. If a question needs a diagram that ISN'T in the reference material at all (no suitable
   TYPE: image item), you may request ONE deterministically-rendered diagram via
   "generated_diagram" — but ONLY these two exact, verifiable kinds:
     - {{"kind": "molecule", "smiles": "<valid SMILES>", "compound_name": "<name>"}} — a
       chemical structure, rendered exactly as the SMILES specifies. The SMILES must be
       chemically correct for the named compound — this is checked and rejected if invalid.
     - {{"kind": "plot", "chart_type": "line" | "bar", "title": "...", "x_label": "...",
       "y_label": "...", "series": [{{"label": "...", "x": [<numbers>], "y": [<numbers>]}}]}}
       — ONLY plotting numeric values that are literally present in a retrieved TYPE: table
       item. NEVER invent data points for a plot.
   A question can have EITHER "figure_ref" (a real retrieved image) OR "generated_diagram"
   (rendered from scratch), never both. If neither applies, "generated_diagram" is null.
7. Provide the answer key SEPARATELY in "answer_key" — NEVER leak answers inside the
   question text or options themselves. "answer" is ALWAYS a single option letter (A/B/C/D).
8. Output ONLY valid JSON. No markdown fences, no commentary outside the JSON.
{violations_str}
REQUEST: {request}
{history_str}

REFERENCE MATERIAL:
{context_str}

JSON OUTPUT SCHEMA:
{{
  "paper_title": "<e.g. NEET Mock Test — Human Physiology>",
  "subject": "<Physics | Chemistry | Botany | Zoology | Biology>",
  "exam_pattern": "NTA-NEET",
  "total_marks": <int>,
  "duration_minutes": <int>,
  "instructions": ["All questions in Section A are compulsory.", "..."],
  "coverage_note": "<brief note ONLY if reference material was limited, else empty string>",
  "sections": [
    {{
      "section_name": "Section A",
      "question_type": "MCQ",
      "marking_scheme": "+4 / -1",
      "questions": [
        {{
          "number": 1,
          "question_type": "STANDARD",
          "question": "...",
          "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
          "marks": 4,
          "topic": "...",
          "difficulty": "Easy | Moderate | Difficult",
          "figure_ref": "",
          "table_data": "",
          "generated_diagram": null
        }},
        {{
          "number": 2,
          "question_type": "ASSERTION_REASON",
          "question": "Assertion (A): ... Reason (R): ...",
          "options": {json.dumps(_ASSERTION_REASON_OPTIONS)},
          "marks": 4,
          "topic": "...",
          "difficulty": "Moderate",
          "figure_ref": "",
          "table_data": "",
          "generated_diagram": {{"kind": "molecule", "smiles": "CCO", "compound_name": "Ethanol"}}
        }}
      ]
    }}
  ],
  "answer_key": [
    {{ "section": "Section A", "number": 1, "answer": "C", "explanation": "<short>" }},
    {{ "section": "Section A", "number": 2, "answer": "A", "explanation": "<short>" }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    prompt:        str,
    processor:     Any,
    llm:           Any,
    cfg:           RAGConfig,
    is_multimodal: bool = False,
    images:        list[Image.Image] | None = None,
    temperature:   float = 0.3,
) -> str:
    """
    Run the LLM and return the raw generated text (JSON extraction happens later in
    clean_and_parse()). `temperature` is annealed downward across retry attempts.
    When is_multimodal, `images` are attached via the processor's chat template so
    the model genuinely looks at them instead of only reading a text citation.
    """
    device = next(llm.parameters()).device

    if is_multimodal:
        content: list[dict] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": img} for img in (images or []))
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        eos_id = processor.tokenizer.eos_token_id
        decode = processor.tokenizer.decode
    else:
        inputs = processor(prompt, return_tensors="pt").to(device)
        eos_id = processor.eos_token_id
        decode = processor.decode

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    elapsed = time.perf_counter() - t0
    log.debug("LLM generation: %.1f s", elapsed)

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    return decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSER / CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_first_json_object(text: str) -> str:
    """
    Returns the substring spanning the first complete top-level JSON object,
    tracking brace depth and string state so nested objects/arrays (e.g. a
    "sections" list full of question objects) don't trip up extraction.
    A naive regex or "split on first }" breaks badly on this schema.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text[start:]   # unterminated — let json.loads raise a clear error


def clean_and_parse(raw: str) -> dict:
    text = re.sub(r"```(?:json)?", "", raw).strip()
    json_str = _extract_first_json_object(text)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Trailing-comma fix, for both object and array endings
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(json_str)


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC DIAGRAM GENERATION  (RDKit molecules, matplotlib plots)
# ─────────────────────────────────────────────────────────────────────────────
#
# Deliberately not a diffusion/image-generation model: those hallucinate
# plausible-looking pixels with no guarantee of scientific correctness. RDKit
# and matplotlib are deterministic instead — given a valid SMILES, RDKit draws
# the actual structure; given real retrieved numbers, matplotlib plots the
# actual curve. validate_neet_paper() gates both before any rendering happens.

def _is_valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return bool(smiles) and Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def render_molecule_diagram(smiles: str, out_path: Path) -> bool:
    """Render a chemical structure from SMILES via RDKit. Returns False (and logs
    a warning) rather than raising, since this is a best-effort post-processing step."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning("Could not render molecule — invalid SMILES: %r", smiles)
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Draw.MolToFile(mol, str(out_path), size=(450, 450))
        return True
    except Exception as exc:
        log.warning("Molecule rendering failed for SMILES %r: %s", smiles, exc)
        return False


def render_plot_diagram(spec: dict, out_path: Path) -> bool:
    """Render a simple line/bar plot via matplotlib. `spec["series"]` values are
    expected to already be grounded in retrieved data (validate_neet_paper checks this)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = spec.get("series") or []
        if not series:
            return False

        fig, ax = plt.subplots(figsize=(5, 4))
        chart_type = spec.get("chart_type", "line")
        for s in series:
            x, y, label = s.get("x", []), s.get("y", []), s.get("label", "")
            if chart_type == "bar":
                ax.bar(x, y, label=label)
            else:
                ax.plot(x, y, marker="o", label=label)
        ax.set_title(spec.get("title", ""))
        ax.set_xlabel(spec.get("x_label", ""))
        ax.set_ylabel(spec.get("y_label", ""))
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return True
    except Exception as exc:
        log.warning("Plot rendering failed: %s", exc)
        return False


def render_generated_diagram(spec: dict, out_path: Path) -> bool:
    """Dispatch to the right deterministic renderer based on spec['kind']."""
    kind = spec.get("kind")
    if kind == "molecule":
        return render_molecule_diagram(str(spec.get("smiles", "")), out_path)
    if kind == "plot":
        return render_plot_diagram(spec, out_path)
    log.warning("Unknown generated_diagram kind: %r", kind)
    return False


def render_all_generated_diagrams(parsed: dict, out_dir: Path) -> None:
    """Renders every question's generated_diagram spec (if any) to a real PNG and sets
    figure_ref to that file, reusing the existing figure_ref display machinery."""
    for section in parsed.get("sections", []):
        for q in section.get("questions", []):
            spec = q.get("generated_diagram")
            if not spec or q.get("figure_ref"):
                continue
            num = q.get("number", "x")
            out_path = out_dir / f"q{num}_{spec.get('kind', 'diagram')}_{uuid.uuid4().hex[:8]}.png"
            if render_generated_diagram(spec, out_path):
                q["figure_ref"] = str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# STRICT NEET-PATTERN VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

_OPTION_LETTER_RE = re.compile(r"^\s*([A-D])[\)\.]")

_MAX_VIOLATIONS_REPORTED = 15   # keep retry-feedback prompt bounded


def _option_letter(option: str) -> str | None:
    m = _OPTION_LETTER_RE.match(option or "")
    return m.group(1) if m else None


def validate_neet_paper(parsed: dict, context_items: list[dict] | None = None) -> list[str]:
    """
    Hard structural gate for NEET-pattern conformance; violations drive an automatic
    regenerate-with-feedback retry in generate_question_paper. Checks: MCQ-only, exactly
    4 lettered options, Assertion-Reason template, uniform 4 marks / "+4 / -1", complete
    answer_key with valid letters, sequential numbering, figure_ref exists on disk, and
    (when context_items is supplied) figure_ref/table_data genuinely match what was
    retrieved rather than being fabricated citations. Returns a list of violations
    (empty == valid).
    """
    violations: list[str] = []

    real_image_paths = {
        c["asset_path"] for c in (context_items or [])
        if c.get("element_type") == "image" and c.get("asset_path")
    }
    real_table_chunks = [
        str(c.get("chunk", "")) for c in (context_items or [])
        if c.get("element_type") == "table"
    ]
    # Grounding checks only fire when context_items was actually passed.
    check_grounding = context_items is not None

    sections = parsed.get("sections")
    if not isinstance(sections, list) or not sections:
        return ["Paper has no 'sections' — cannot validate or use it."]

    exam_pattern = str(parsed.get("exam_pattern", ""))
    if "neet" not in exam_pattern.lower():
        violations.append(
            f"exam_pattern is '{exam_pattern}' — must clearly identify this as NEET pattern (e.g. 'NTA-NEET')."
        )

    answer_key = parsed.get("answer_key")
    if not isinstance(answer_key, list):
        violations.append("answer_key is missing or not a list.")
        answer_key = []
    answer_lookup = {
        (str(a.get("section", "")).strip(), a.get("number")): a
        for a in answer_key if isinstance(a, dict)
    }

    for section in sections:
        sec_name = str(section.get("section_name", "")).strip()
        q_type = str(section.get("question_type", ""))
        if q_type != "MCQ":
            violations.append(
                f"Section '{sec_name}': question_type is '{q_type}' — NEET is MCQ-only, must be 'MCQ'."
            )

        marking = str(section.get("marking_scheme", "")).strip()
        if marking != "+4 / -1":
            violations.append(
                f"Section '{sec_name}': marking_scheme is '{marking}' — must be exactly '+4 / -1'."
            )

        questions = section.get("questions")
        if not isinstance(questions, list) or not questions:
            violations.append(f"Section '{sec_name}': has no questions.")
            continue

        expected_numbers = set(range(1, len(questions) + 1))
        seen_numbers = set()

        for q in questions:
            num = q.get("number")
            seen_numbers.add(num)
            loc = f"Section '{sec_name}' Q{num}"

            if not str(q.get("question", "")).strip():
                violations.append(f"{loc}: empty question stem.")

            if q.get("marks") != 4:
                violations.append(f"{loc}: marks is {q.get('marks')!r} — must be exactly 4.")

            options = q.get("options")
            if not isinstance(options, list) or len(options) != 4:
                violations.append(
                    f"{loc}: has {len(options) if isinstance(options, list) else 0} option(s) — must be exactly 4."
                )
                options = []

            letters = [_option_letter(o) for o in options]
            if letters != ["A", "B", "C", "D"]:
                violations.append(f"{loc}: options must be lettered 'A)'..'D)' in order.")

            q_subtype = str(q.get("question_type", "STANDARD"))
            if q_subtype == "ASSERTION_REASON":
                stem = str(q.get("question", ""))
                if "assertion" not in stem.lower() or "reason" not in stem.lower():
                    violations.append(f"{loc}: ASSERTION_REASON stem must contain both an Assertion and a Reason.")
                expected_bodies = [o.split(")", 1)[1].strip() for o in _ASSERTION_REASON_OPTIONS]
                actual_bodies = [o.split(")", 1)[1].strip() if ")" in o else o for o in options]
                if actual_bodies != expected_bodies:
                    violations.append(f"{loc}: ASSERTION_REASON options must match the fixed 4-option template exactly.")
            elif q_subtype != "STANDARD":
                violations.append(f"{loc}: question_type '{q_subtype}' invalid — must be 'STANDARD' or 'ASSERTION_REASON'.")

            table_data = str(q.get("table_data", "") or "")
            if table_data and "|" not in table_data:
                violations.append(f"{loc}: table_data is set but doesn't look like a markdown table (no '|' found).")
            elif table_data and check_grounding:
                # Header row must genuinely appear in some retrieved table's content.
                header_line = next((ln.strip() for ln in table_data.splitlines() if ln.strip()), "")
                if header_line and not any(header_line in chunk for chunk in real_table_chunks):
                    violations.append(f"{loc}: table_data doesn't match any retrieved table — looks fabricated.")

            figure_ref = str(q.get("figure_ref", "") or "")
            if figure_ref:
                if not Path(figure_ref).exists():
                    violations.append(f"{loc}: figure_ref '{figure_ref}' does not exist on disk — hallucinated citation.")
                elif check_grounding and figure_ref not in real_image_paths:
                    violations.append(f"{loc}: figure_ref '{figure_ref}' was not among the retrieved images for this request.")

            generated_diagram = q.get("generated_diagram")
            if generated_diagram is not None:
                if figure_ref:
                    violations.append(f"{loc}: has both figure_ref and generated_diagram set — use only one.")
                elif not isinstance(generated_diagram, dict):
                    violations.append(f"{loc}: generated_diagram must be an object with a 'kind' field.")
                else:
                    gd_kind = generated_diagram.get("kind")
                    if gd_kind == "molecule":
                        smiles = str(generated_diagram.get("smiles", ""))
                        if not _is_valid_smiles(smiles):
                            violations.append(
                                f"{loc}: generated_diagram SMILES {smiles!r} is not valid — RDKit could not parse it."
                            )
                    elif gd_kind == "plot":
                        series = generated_diagram.get("series")
                        if not isinstance(series, list) or not series:
                            violations.append(f"{loc}: generated_diagram plot has no data series.")
                        elif check_grounding:
                            # Every plotted number must trace back to real retrieved table content.
                            values = [
                                v for s in series if isinstance(s, dict)
                                for v in list(s.get("x", []) or []) + list(s.get("y", []) or [])
                            ]
                            ungrounded = [v for v in values if not any(str(v) in chunk for chunk in real_table_chunks)]
                            if ungrounded:
                                violations.append(
                                    f"{loc}: generated_diagram plot has data not found in any retrieved "
                                    f"table (e.g. {ungrounded[0]!r}) — looks fabricated."
                                )
                    else:
                        violations.append(f"{loc}: generated_diagram kind {gd_kind!r} invalid — must be 'molecule' or 'plot'.")

            key = (sec_name, num)
            if key not in answer_lookup:
                violations.append(f"{loc}: missing from answer_key.")
            else:
                ans = str(answer_lookup[key].get("answer", "")).strip()
                if ans not in ("A", "B", "C", "D"):
                    violations.append(f"{loc}: answer_key answer '{ans}' is not a single option letter (A-D).")

        if seen_numbers != expected_numbers:
            violations.append(
                f"Section '{sec_name}': question numbers {sorted(n for n in seen_numbers if n is not None)} "
                f"are not sequential 1..{len(questions)}."
            )

    return violations[:_MAX_VIOLATIONS_REPORTED]


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-READABLE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int, indent: str = "    ") -> None:
    words, line = text.split(), ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            print(indent + line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        print(indent + line)


def print_question_paper(
    paper: dict,
    paper_id: str,
    unlock_at: float,
    validation_warnings: list[str] | None = None,
) -> None:
    try:
        W = min(os.get_terminal_size().columns, 80)
    except OSError:
        W = 80

    SEP, TSEP = "─" * W, "═" * W

    print(f"\n{TSEP}")
    print(f"  {paper.get('paper_title', 'Question Paper')}")
    print(TSEP)
    print(f"  Subject   : {paper.get('subject', 'N/A')}   |   Pattern: {paper.get('exam_pattern', 'NTA')}")
    print(f"  Marks     : {paper.get('total_marks', 'N/A')}   |   Duration: {paper.get('duration_minutes', 'N/A')} min")
    print(SEP)

    if validation_warnings:
        print("  NEET-FORMAT VALIDATION FAILED — this paper does NOT fully conform:")
        for v in validation_warnings:
            _wrap(f"-  {v}", W - 6, "     ")
        print()

    instructions = paper.get("instructions") or []
    if instructions:
        print("  INSTRUCTIONS")
        for ins in instructions:
            _wrap(f"-  {ins}", W - 6, "     ")
        print()

    coverage_note = paper.get("coverage_note", "")
    if coverage_note:
        print("  COVERAGE NOTE")
        _wrap(coverage_note, W - 6, "     ")
        print()

    for section in paper.get("sections", []):
        print(SEP)
        print(f"  {section.get('section_name', 'Section')}   "
              f"[{section.get('question_type', '')} · {section.get('marking_scheme', '')}]")
        print(SEP)
        for q in section.get("questions", []):
            qtype = q.get("question_type", "STANDARD")
            tag = "  [Assertion-Reason]" if qtype == "ASSERTION_REASON" else ""
            print(f"\n  Q{q.get('number', '?')}.  ({q.get('marks', '?')} marks)  {q.get('difficulty', '')}{tag}")
            _wrap(q.get("question", ""), W - 6, "      ")
            table_data = q.get("table_data", "")
            if table_data:
                print("        Table:")
                for line in table_data.splitlines():
                    print(f"        {line}")
            for opt in q.get("options", []):
                print(f"        {opt}")
            figure_ref = q.get("figure_ref", "")
            if figure_ref:
                label = "Generated diagram" if q.get("generated_diagram") else "Figure"
                print(f"        {label}: {Path(figure_ref).name}")
        print()

    print(TSEP)
    unlock_str = datetime.fromtimestamp(unlock_at).strftime("%Y-%m-%d %I:%M %p")
    print(f"  Answer key locked until {unlock_str}")
    print(f"     (unlocks once the exam duration elapses, or by the next day — whichever is first)")
    print(f"     Paper ID: {paper_id}   →   check later with: /answerkey {paper_id}")
    print(TSEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# HTML EXPORT  (real <table> + <img> rendering; the actual deliverable to view/print)
# ─────────────────────────────────────────────────────────────────────────────

_PAPER_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
.meta { color: #444; margin-bottom: 1rem; }
.warning { background: #fff3cd; border: 1px solid #e0a800; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1.5rem; }
.warning ul { margin: 0.5rem 0 0 1rem; }
h2 { border-bottom: 2px solid #ddd; padding-bottom: 0.25rem; margin-top: 2rem; }
.question { margin: 1.5rem 0; padding: 1rem; border: 1px solid #e0e0e0; border-radius: 8px; }
.question .qhead { font-weight: 600; margin-bottom: 0.5rem; }
.options { list-style: none; padding-left: 0.5rem; }
.options li { margin: 0.25rem 0; }
.q-table { border-collapse: collapse; margin: 0.75rem 0; font-size: 0.92rem; }
.q-table th, .q-table td { border: 1px solid #bbb; padding: 4px 8px; text-align: left; }
.q-table th { background: #f2f2f2; }
.q-figure { max-width: 100%; margin: 0.75rem 0; border: 1px solid #ddd; border-radius: 4px; }
.figure-missing { color: #b00020; font-style: italic; }
.figure-caption { color: #666; font-size: 0.85rem; margin-top: -0.5rem; }
"""


def _markdown_table_to_html(md: str) -> str:
    """Convert a "| a | b |\\n|---|---|\\n| c | d |" markdown table into a real HTML <table>."""
    lines = [ln for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return f"<p>{html.escape(md)}</p>"

    def _cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = _cells(lines[0])
    rows = [_cells(ln) for ln in lines[2:]] if len(lines) > 2 else []

    parts = ["<table class='q-table'><thead><tr>"]
    parts += [f"<th>{html.escape(c)}</th>" for c in header]
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _image_to_data_uri(path: str) -> str | None:
    """Base64-inline a figure so the exported HTML is a single portable file."""
    try:
        data = Path(path).read_bytes()
        return f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"
    except Exception as exc:
        log.warning("Could not embed image '%s' into HTML export: %s", path, exc)
        return None


def export_paper_html(
    paper: dict,
    paper_id: str,
    out_dir: Path,
    validation_warnings: list[str] | None = None,
) -> Path:
    """
    Render the generated paper as a single self-contained HTML file: table_data
    becomes a real <table>, figure_ref becomes a real embedded <img> (base64
    data URI) — not a filename string. Saved as <out_dir>/<paper_id>.html.
    """
    blocks = [f"<h1>{html.escape(paper.get('paper_title', 'Question Paper'))}</h1>"]
    blocks.append(
        "<p class='meta'>"
        f"<b>Subject:</b> {html.escape(str(paper.get('subject', 'N/A')))} &nbsp;|&nbsp; "
        f"<b>Pattern:</b> {html.escape(str(paper.get('exam_pattern', 'NTA')))} &nbsp;|&nbsp; "
        f"<b>Marks:</b> {html.escape(str(paper.get('total_marks', 'N/A')))} &nbsp;|&nbsp; "
        f"<b>Duration:</b> {html.escape(str(paper.get('duration_minutes', 'N/A')))} min"
        "</p>"
    )

    if validation_warnings:
        blocks.append(
            "<div class='warning'><b>NEET-format validation failed — this paper does not fully conform:</b><ul>"
            + "".join(f"<li>{html.escape(v)}</li>" for v in validation_warnings)
            + "</ul></div>"
        )

    instructions = paper.get("instructions") or []
    if instructions:
        blocks.append("<p><b>Instructions:</b></p><ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in instructions) + "</ul>")

    coverage_note = paper.get("coverage_note", "")
    if coverage_note:
        blocks.append(f"<p><i>{html.escape(coverage_note)}</i></p>")

    for section in paper.get("sections", []):
        blocks.append(
            f"<h2>{html.escape(section.get('section_name', 'Section'))} "
            f"({html.escape(section.get('question_type', ''))} · {html.escape(section.get('marking_scheme', ''))})</h2>"
        )
        for q in section.get("questions", []):
            qtype = q.get("question_type", "STANDARD")
            tag = " [Assertion-Reason]" if qtype == "ASSERTION_REASON" else ""
            blocks.append("<div class='question'>")
            blocks.append(
                f"<div class='qhead'>Q{q.get('number', '?')}. "
                f"({q.get('marks', '?')} marks) {html.escape(q.get('difficulty', ''))}{tag}</div>"
            )
            blocks.append(f"<p>{html.escape(q.get('question', ''))}</p>")

            table_data = q.get("table_data", "")
            if table_data:
                blocks.append(_markdown_table_to_html(table_data))

            figure_ref = q.get("figure_ref", "")
            if figure_ref:
                uri = _image_to_data_uri(figure_ref)
                q_number = q.get("number", "?")
                caption = "Generated diagram" if q.get("generated_diagram") else "Figure from source material"
                if uri:
                    blocks.append(f"<img class='q-figure' src='{uri}' alt='Figure for Q{q_number}'/>")
                    blocks.append(f"<p class='figure-caption'><i>{html.escape(caption)}</i></p>")
                else:
                    blocks.append(f"<p class='figure-missing'>[Figure not available: {html.escape(figure_ref)}]</p>")

            options = q.get("options", [])
            blocks.append("<ul class='options'>" + "".join(f"<li>{html.escape(o)}</li>" for o in options) + "</ul>")
            blocks.append("</div>")

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(paper.get('paper_title', 'Question Paper'))}</title>"
        f"<style>{_PAPER_CSS}</style></head><body>{''.join(blocks)}</body></html>"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper_id}.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def print_answer_key(paper_id: str, answer_key: list[dict], forced: bool = False) -> None:
    try:
        W = min(os.get_terminal_size().columns, 80)
    except OSError:
        W = 80
    SEP = "─" * W

    label = "ANSWER KEY" + ("  (admin override)" if forced else "")
    print(f"\n{label}  —  {paper_id}")
    print(SEP)
    for item in answer_key:
        sec = item.get("section", "")
        num = item.get("number", "?")
        ans = item.get("answer", "")
        exp = item.get("explanation", "")
        print(f"  [{sec}]  Q{num}:  {ans}")
        if exp:
            _wrap(f"↳ {exp}", W - 8, "        ")
    print(SEP + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION-PAPER GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _load_context_images(context_items: list[dict], cfg: RAGConfig) -> list[Image.Image]:
    """Load the PNG files behind the highest-ranked retrieved image elements, for genuine
    attachment to the multimodal LLM's prompt. Capped at cfg.max_context_images."""
    candidates = [
        c for c in context_items
        if c.get("element_type") == "image" and c.get("asset_path")
    ]
    candidates.sort(key=lambda c: c.get("_rank_score", 0.0), reverse=True)

    images: list[Image.Image] = []
    for c in candidates[: cfg.max_context_images]:
        path = c["asset_path"]
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as exc:
            log.warning("Could not load context image '%s' for attachment: %s", path, exc)
    return images


def _generate_and_validate_one_subject(
    request:          str,
    tokenizer:        Any,
    llm:              Any,
    emb_model:        SentenceTransformer,
    milvus_index:     MilvusIndex,
    bm25:             BM25Okapi,
    pages_and_chunks: list[dict],
    cfg:              RAGConfig,
    history:          list[dict] | None = None,
    is_multimodal:    bool = False,
) -> tuple[dict | None, list[str], str]:
    """
    Retrieval -> generation -> validation -> regenerate-on-violation retry loop for ONE
    subject/request — shared by generate_question_paper and generate_full_neet_mock
    (which calls this once per subject). Retrieval runs once; only generation retries.
    Returns (parsed_paper_or_None, violations, raw_last_answer_for_fallback_display).
    """
    full_structure, target_count = _infer_paper_scale(request)
    subject_cfg = replace(cfg, max_new_tokens=_estimate_max_new_tokens(target_count, cfg))

    context_items = retrieve(request, emb_model, milvus_index, bm25, pages_and_chunks, subject_cfg)
    log.debug("Retrieved %d context chunks.", len(context_items))

    images = _load_context_images(context_items, subject_cfg) if is_multimodal else []
    images_attached = bool(images)
    if is_multimodal and not images:
        log.debug("Multimodal model loaded, but no image elements were retrieved for this request.")

    violations: list[str] = []
    parsed: dict | None = None
    answer = ""
    max_attempts = subject_cfg.max_generation_retries + 1

    for attempt in range(1, max_attempts + 1):
        # Anneal temperature down after a validation failure to converge on stricter format adherence.
        temperature = max(0.1, 0.3 - 0.1 * (attempt - 1))
        prompt = build_exam_prompt(
            request, context_items, full_structure, target_count,
            history=history, violations=violations, images_attached=images_attached,
        )
        answer = generate_answer(
            prompt, tokenizer, llm, subject_cfg, temperature=temperature,
            is_multimodal=is_multimodal, images=images,
        )

        try:
            candidate = clean_and_parse(answer)
        except Exception as parse_exc:
            log.warning("Attempt %d/%d: JSON parse failed (%s).", attempt, max_attempts, parse_exc)
            violations = [f"Previous attempt was not valid JSON: {parse_exc}"]
            continue

        violations = validate_neet_paper(candidate, context_items=context_items)
        if not violations:
            parsed = candidate
            break

        log.warning(
            "Attempt %d/%d: %d NEET-format violation(s) — %s",
            attempt, max_attempts, len(violations), "; ".join(violations[:3]),
        )
        parsed = candidate   # keep the best-effort candidate in case every attempt fails

    return parsed, violations, answer


def generate_question_paper(
    request:          str,
    tokenizer:        Any,
    llm:              Any,
    emb_model:        SentenceTransformer,
    milvus_index:     MilvusIndex,
    bm25:             BM25Okapi,
    pages_and_chunks: list[dict],
    cfg:              RAGConfig,
    vault:            PaperVault,
    history:          list[dict] | None = None,
    is_multimodal:    bool = False,
) -> tuple[str, str | None]:
    """
    Single-request paper generation (one subject, or a custom-sized request)
    → vault storage → terminal + HTML display. Returns
    (summary_for_history, paper_id_or_None).
    """
    t0 = time.perf_counter()

    parsed, violations, answer = _generate_and_validate_one_subject(
        request, tokenizer, llm, emb_model, milvus_index, bm25, pages_and_chunks,
        cfg, history=history, is_multimodal=is_multimodal,
    )

    if parsed is None:
        log.warning("No JSON could be parsed after %d attempt(s) — showing raw output.", cfg.max_generation_retries + 1)
        print("\n--- Raw Model Response ---")
        for ln in answer.splitlines():
            s = ln.strip()
            if s and s not in ("```", "```json"):
                print(" ", s)
        print("--------------------------")
        log.debug("Request processed in %.1f s (raw fallback)", time.perf_counter() - t0)
        return request, None

    if violations:
        log.error(
            "Paper still has %d NEET-format violation(s) — showing it anyway with a visible warning banner.",
            len(violations),
        )

    parsed["_validation_warnings"] = violations

    render_all_generated_diagrams(parsed, cfg.generated_diagrams_dir)

    record = vault.save(parsed)
    print_question_paper(parsed, record["paper_id"], record["unlock_at"], validation_warnings=violations)
    html_path = export_paper_html(parsed, record["paper_id"], vault.path, validation_warnings=violations)
    print(f"  Full paper with rendered tables/figures → {html_path}\n")

    log.debug("Paper generated in %.1f s", time.perf_counter() - t0)
    return parsed.get("paper_title", request), record["paper_id"]


NEET_FULL_MOCK_SUBJECTS = ["Physics", "Chemistry", "Botany", "Zoology"]
NEET_FULL_MOCK_DURATION_MINUTES = 200   # real NEET's fixed total duration, all subjects combined


def generate_full_neet_mock(
    tokenizer:        Any,
    llm:              Any,
    emb_model:        SentenceTransformer,
    milvus_index:     MilvusIndex,
    bm25:             BM25Okapi,
    pages_and_chunks: list[dict],
    cfg:              RAGConfig,
    vault:            PaperVault,
    is_multimodal:    bool = False,
    subjects:         list[str] | None = None,
) -> tuple[str, str | None]:
    """
    Generates a full multi-subject NEET mock: one 35+15-question paper per subject
    (see NEET_FULL_MOCK_SUBJECTS), merged into a single paper with one combined
    answer key — mirroring the real exam (4 subjects x 45 attempted questions =
    720 marks across 200 minutes). Each subject is generated independently via
    _generate_and_validate_one_subject; sections/answer_key are concatenated with
    the subject prefixed onto section_name, so the merged dict has the exact same
    shape as a single-subject paper and every downstream function handles it
    unchanged. A subject with no ingested source material honestly reflects thin
    coverage via coverage_note rather than inventing content.
    """
    t0 = time.perf_counter()
    subjects = subjects or NEET_FULL_MOCK_SUBJECTS

    merged_sections: list[dict] = []
    merged_answer_key: list[dict] = []
    merged_violations: list[str] = []
    coverage_notes: list[str] = []
    total_marks = 0
    any_subject_succeeded = False

    for subject in subjects:
        log.info("── Generating %s (full NEET paper: %d + %d questions) ──",
                  subject, NEET_SECTION_A_COUNT, NEET_SECTION_B_COUNT)
        request = (
            f"Generate a full NEET {subject} paper: Section A ({NEET_SECTION_A_COUNT} compulsory "
            f"questions) and Section B ({NEET_SECTION_B_COUNT} questions, attempt any "
            f"{NEET_SECTION_B_ATTEMPT}), strictly from the {subject} reference material."
        )
        parsed, violations, _ = _generate_and_validate_one_subject(
            request, tokenizer, llm, emb_model, milvus_index, bm25, pages_and_chunks,
            cfg, is_multimodal=is_multimodal,
        )

        if parsed is None:
            log.error("%s: no valid JSON after all attempts — subject OMITTED from the mock.", subject)
            coverage_notes.append(f"{subject}: generation failed entirely (invalid JSON) — omitted.")
            continue

        any_subject_succeeded = True
        total_marks += int(parsed.get("total_marks", 0) or 0)
        note = str(parsed.get("coverage_note", "") or "").strip()
        if note:
            coverage_notes.append(f"{subject}: {note}")
        if violations:
            merged_violations.extend(f"[{subject}] {v}" for v in violations)

        for section in parsed.get("sections", []):
            section = dict(section)
            section["section_name"] = f"{subject} — {section.get('section_name', 'Section')}"
            merged_sections.append(section)

        for entry in parsed.get("answer_key", []):
            entry = dict(entry)
            entry["section"] = f"{subject} — {entry.get('section', '')}"
            merged_answer_key.append(entry)

    if not any_subject_succeeded:
        log.error("Full NEET mock failed: every subject failed to generate valid JSON.")
        return "Full NEET Mock (failed)", None

    merged_paper = {
        "paper_title": "NEET Full Mock Test",
        "subject": ", ".join(subjects),
        "exam_pattern": "NTA-NEET",
        "total_marks": total_marks,
        "duration_minutes": NEET_FULL_MOCK_DURATION_MINUTES,
        "instructions": [
            "All questions in each subject's Section A are compulsory.",
            "In each subject's Section B, attempt any "
            f"{NEET_SECTION_B_ATTEMPT} of {NEET_SECTION_B_COUNT} questions.",
            "Each question carries 4 marks; 1 mark is deducted for every wrong attempt.",
        ],
        "coverage_note": " | ".join(coverage_notes),
        "sections": merged_sections,
        "answer_key": merged_answer_key,
        "_validation_warnings": merged_violations,
    }

    render_all_generated_diagrams(merged_paper, cfg.generated_diagrams_dir)

    record = vault.save(merged_paper)
    print_question_paper(merged_paper, record["paper_id"], record["unlock_at"], validation_warnings=merged_violations)

    html_path = export_paper_html(merged_paper, record["paper_id"], vault.path, validation_warnings=merged_violations)
    print(f"  Full NEET mock with rendered tables/figures → {html_path}\n")

    log.info("Full NEET mock generated in %.1f s across %d subject(s).", time.perf_counter() - t0, len(subjects))
    return merged_paper["paper_title"], record["paper_id"]


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_signal(sig: int, _frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print("\n\n[Signal received — shutting down after current request …]\n")


def auto_ingest_data(cfg: RAGConfig) -> None:
    """Discovers local PDFs/CSVs and ingests them into the same Milvus collections
    this pipeline reads from, explicitly passing URI/collection names.

    Scans the working dir, $DATA_DIR (default ./data — the Docker mount point for
    study material), and ./uploads (files POSTed to the API in earlier runs — the
    incremental hash cache makes re-checking them cheap)."""
    try:
        from embeddings import run_pipeline, PipelineConfig

        scan_dirs, seen = [], set()
        for d in (Path("."), Path(os.environ.get("DATA_DIR", "data")), Path("uploads")):
            if d.is_dir() and (r := d.resolve()) not in seen:
                seen.add(r)
                scan_dirs.append(d)

        pdfs = [str(p) for d in scan_dirs for p in sorted(d.glob("*.pdf"))]
        csvs = [str(p) for d in scan_dirs for p in sorted(d.glob("*.csv"))]

        if not pdfs and not csvs:
            log.info("No local PDF or CSV files found for auto-ingestion.")
            return

        log.info("Checking for new/updated study material: %d PDFs, %d CSVs", len(pdfs), len(csvs))

        embed_cfg = PipelineConfig(
            pdfs=pdfs,
            catalogs=csvs,
            output_dir=cfg.index_dir,
            milvus_uri_override=cfg.milvus_uri,
            milvus_collection_override=cfg.collection_name,
            milvus_visual_collection_override=cfg.visual_collection_name,
            incremental=True,
        )

        run_pipeline(embed_cfg)
        log.info("Auto-ingestion check complete.")

    except ImportError:
        log.warning("embeddings.py not found. Skipping auto-ingestion.")
    except Exception as e:
        log.error("Auto-ingestion failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive(
    tokenizer:        Any,
    llm:              Any,
    emb_model:        SentenceTransformer,
    milvus_index:     MilvusIndex,
    bm25:             BM25Okapi,
    pages_and_chunks: list[dict],
    cfg:              RAGConfig,
    vault:            PaperVault,
    is_multimodal:    bool = False,
) -> None:
    try:
        W = min(os.get_terminal_size().columns, 80)
    except OSError:
        W = 80

    vision_str = "vision ON — diagrams genuinely understood" if is_multimodal else "vision OFF — text-only"
    print("\n" + "═" * W)
    print(f"  AI Teacher · NTA Question Paper Generator   |  collection: {cfg.collection_name}  |  {vision_str}")
    print("═" * W)
    print("  Describe the paper you want (subject, topic, # of questions, duration).")
    print("  Commands: exit | quit | /reload | /answerkey <paper_id> | /fullmock")
    print("═" * W + "\n")

    history: list[dict] = []

    while not _shutdown_requested:
        try:
            request = input(">  Teacher: ").strip()
        except EOFError:
            break

        if not request:
            continue
        if request.lower() in ("exit", "quit"):
            break

        if request == "/reload":
            log.info("Reloading from Milvus collection '%s' …", cfg.collection_name)
            try:
                new_index, new_bm25, new_chunks = load_data(cfg)
                milvus_index = new_index
                bm25 = new_bm25
                pages_and_chunks[:] = new_chunks
                log.info("Index reloaded: %d chunks.", len(pages_and_chunks))
            except VectorStoreError as exc:
                log.error("Reload failed: %s", exc)
            continue

        if request == "/fullmock":
            print(f"\n[Setting the full NEET mock — {len(NEET_FULL_MOCK_SUBJECTS)} subjects, "
                  f"{NEET_SECTION_A_COUNT + NEET_SECTION_B_COUNT} questions each, this will take a while …]")
            try:
                generate_full_neet_mock(
                    tokenizer, llm, emb_model, milvus_index, bm25,
                    pages_and_chunks, cfg, vault, is_multimodal=is_multimodal,
                )
            except KeyboardInterrupt:
                print("\n[Full mock interrupted]\n")
            except Exception:
                log.error("Unexpected error during full mock generation:\n%s", traceback.format_exc())
            continue

        if request.startswith("/answerkey"):
            parts = request.split()
            if len(parts) < 2:
                print("Usage: /answerkey <paper_id>")
                continue
            paper_id = parts[1]
            unlocked, payload = vault.get_answer_key(paper_id)
            if unlocked:
                print_answer_key(paper_id, payload)
            else:
                if isinstance(payload, (int, float)):
                    mins_left = int(payload // 60) + 1
                    print(f"\nAnswer key for '{paper_id}' is still locked. "
                          f"Unlocks in ~{mins_left} minute(s).\n")
                else:
                    print(f"\n{payload}\n")
            continue

        print("\n[Setting the paper …]")
        try:
            summary, paper_id = generate_question_paper(
                request, tokenizer, llm, emb_model, milvus_index, bm25,
                pages_and_chunks, cfg, vault, history=history, is_multimodal=is_multimodal,
            )
            history.append({"request": request, "summary": summary})
            if len(history) > 5:
                history.pop(0)
        except KeyboardInterrupt:
            print("\n[Request interrupted]\n")
        except Exception:
            log.error("Unexpected error during request:\n%s", traceback.format_exc())

    print("\nGoodbye! — AI Teacher shutting down.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag_pipeline",
        description="AI Teacher: Hybrid RAG (Milvus + BM25) NTA-pattern question paper generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--index-dir",      metavar="DIR",   default=None,  help="Local dir holding milvus.db / papers (default: ./index_store)")
    p.add_argument("--milvus-uri",     metavar="URI",   default=None,  help="Local .db path or remote Milvus/Zilliz URI (default: <index-dir>/milvus.db)")
    p.add_argument("--collection",     metavar="NAME",  default=None,  help="Text collection name (default: multimodal_text)")
    p.add_argument("--visual-collection", metavar="NAME", default=None, help="Visual collection name (default: multimodal_visual)")
    p.add_argument("--request",        metavar="TEXT",  default=None,  help="Generate one paper from this request and exit")
    p.add_argument("--full-mock",      action="store_true",            help="Generate a full multi-subject NEET mock (Physics/Chemistry/Botany/Zoology, 35+15 each) and exit")
    p.add_argument("--reveal-answer-key",       metavar="PAPER_ID", default=None, help="Show answer key if its lock has expired")
    p.add_argument("--admin-reveal-answer-key", metavar="PAPER_ID", default=None, help="Force-show an answer key, bypassing the lock (explicit admin override)")
    p.add_argument("--top-k",          type=int,        default=None,  help="Chunks to retrieve per request (default: 8)")
    p.add_argument("--min-score",      type=float,      default=None,  help="Min cosine similarity threshold (default: 0.25)")
    p.add_argument("--max-tokens",     type=int,        default=None,  help="Max LLM output tokens floor (default: 4096; auto-scaled up for large question counts)")
    p.add_argument("--no-4bit",        action="store_true",            help="Disable 4-bit quantization")
    p.add_argument("--no-vision",      action="store_true",            help="Force the text-only 1b model even with enough RAM for the multimodal 4b model")
    p.add_argument("--max-context-images", type=int,     default=None,  help="Max real images attached per request in multimodal mode (default: 4)")
    p.add_argument("--log-level",      default="INFO",  choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    global log
    log = _setup_logging(args.log_level)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        cfg = RAGConfig()
        if args.index_dir:  cfg.index_dir      = Path(args.index_dir)
        if args.milvus_uri: cfg.milvus_uri_override        = args.milvus_uri
        if args.collection: cfg.milvus_collection_override = args.collection
        if args.visual_collection: cfg.visual_collection_override = args.visual_collection
        if args.top_k:       cfg.top_k          = args.top_k
        if args.min_score:   cfg.min_score      = args.min_score
        if args.max_tokens:  cfg.max_new_tokens = args.max_tokens
        if args.no_4bit:     cfg.quantize_4bit  = False
        if args.no_vision:   cfg.enable_vision  = False
        if args.max_context_images: cfg.max_context_images = args.max_context_images
        cfg.validate()

        vault = PaperVault(cfg.vault_dir)

        # Answer-key checks don't need the LLM loaded at all — handle early.
        if args.reveal_answer_key:
            unlocked, payload = vault.get_answer_key(args.reveal_answer_key)
            if unlocked:
                print_answer_key(args.reveal_answer_key, payload)
            elif isinstance(payload, (int, float)):
                print(f"Locked. Unlocks in ~{int(payload // 60) + 1} minute(s).")
            else:
                print(payload)
            return 0

        if args.admin_reveal_answer_key:
            unlocked, payload = vault.get_answer_key(args.admin_reveal_answer_key, force=True)
            if unlocked:
                print_answer_key(args.admin_reveal_answer_key, payload, forced=True)
            else:
                print(payload)
            return 0

        _hf_login()

        log.info("=" * 60)
        log.info("AI Teacher — NTA Question Paper Generator  (Milvus + BM25)")
        log.info("  milvus_uri : %s", cfg.milvus_uri)
        log.info("  collection : %s", cfg.collection_name)
        log.info("  top_k      : %d  min_score: %.2f", cfg.top_k, cfg.min_score)
        log.info("  max_tokens : %d", cfg.max_new_tokens)
        log.info("  4-bit quant: %s", cfg.quantize_4bit)
        log.info("  vision     : %s", cfg.enable_vision)
        log.info("=" * 60)

        auto_ingest_data(cfg)

        milvus_index, bm25, pages_and_chunks = load_data(cfg)

        model_id, torch_dtype, is_multimodal = check_system_memory(cfg)
        tokenizer, llm         = load_llm(model_id, torch_dtype, cfg, is_multimodal)
        emb_model              = load_embedding_model()

        if args.full_mock:
            print(f"\n[Setting the full NEET mock — {len(NEET_FULL_MOCK_SUBJECTS)} subjects, "
                  f"{NEET_SECTION_A_COUNT + NEET_SECTION_B_COUNT} questions each, this will take a while …]")
            generate_full_neet_mock(
                tokenizer, llm, emb_model, milvus_index, bm25,
                pages_and_chunks, cfg, vault, is_multimodal=is_multimodal,
            )
            return 0

        if args.request:
            print("\n[Setting the paper …]")
            generate_question_paper(
                args.request, tokenizer, llm, emb_model, milvus_index, bm25,
                pages_and_chunks, cfg, vault, is_multimodal=is_multimodal,
            )
            return 0

        run_interactive(
            tokenizer, llm, emb_model, milvus_index, bm25, pages_and_chunks, cfg, vault,
            is_multimodal=is_multimodal,
        )
        return 0

    except (ConfigError, VectorStoreError, ModelLoadError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception:
        log.critical("Unexpected error:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
