from __future__ import annotations
import csv
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient
from pypdf import PdfReader
from PIL import Image
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(format=fmt, datefmt="%Y-%m-%d %H:%M:%S", level=level.upper())
    return logging.getLogger("emb.pipeline")


log = _setup_logging(os.environ.get("LOG_LEVEL", "INFO"))


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingPipelineError(RuntimeError):
    """Base error for this module."""

class LoaderError(EmbeddingPipelineError):
    """Raised when a data source cannot be loaded."""

class VectorStoreError(EmbeddingPipelineError):
    """Raised when the Milvus collection cannot be created, written to, or queried."""

class ConfigError(EmbeddingPipelineError):
    """Raised for invalid configuration."""


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    All runtime settings — overridable via env vars, YAML, or CLI.

    Priority (highest → lowest):
        CLI flags  >  YAML file  >  environment variables  >  coded defaults
    """

    # ── Model ────────────────────────────────────────────────────────────────
    embed_model: str = field(
        default_factory=lambda: os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size:    int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE",    "180")))
    chunk_overlap: int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "35")))

    min_chunk_len: int = field(default_factory=lambda: int(os.environ.get("MIN_CHUNK_LEN", "50")))

    # ── Embedding ────────────────────────────────────────────────────────────
    batch_size: int = 32

    # ── Visual (image) embedding ─────────────────────────────────────────────
    # Same CLIP checkpoint rag.py loads for cross-modal (text→image) search,
    # so query-time and index-time vectors live in the same embedding space.
    visual_embed_model: str = field(
        default_factory=lambda: os.environ.get("VISUAL_EMBED_MODEL", "sentence-transformers/clip-ViT-B-32")
    )
    extract_images: bool = field(
        default_factory=lambda: os.environ.get("EXTRACT_IMAGES", "true").lower() not in ("0", "false", "no")
    )
    extract_tables: bool = field(
        default_factory=lambda: os.environ.get("EXTRACT_TABLES", "true").lower() not in ("0", "false", "no")
    )
    image_scale:   float = field(default_factory=lambda: float(os.environ.get("IMAGE_SCALE",   "1.5")))
    min_image_dim: int   = field(default_factory=lambda: int(os.environ.get("MIN_IMAGE_DIM", "40")))
    # Optional cap for dev/testing on huge PDFs; None (default) processes every page.
    max_pdf_pages: int | None = field(
        default_factory=lambda: (int(v) if (v := os.environ.get("MAX_PDF_PAGES")) else None)
    )

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "./index_store"))
    )
    run_id: str = ""   # namespaces the Milvus collections, e.g. "v2" → "multimodal_text_v2"

    # ── Milvus ───────────────────────────────────────────────────────────────
    # Override either to point at a remote server / Zilliz Cloud later.
    # Base names match rag.py's RAGConfig defaults so the query side finds
    # exactly what this pipeline writes.
    milvus_uri_override:        str = field(default_factory=lambda: os.environ.get("MILVUS_URI", ""))
    milvus_collection_override: str = field(default_factory=lambda: os.environ.get("MILVUS_COLLECTION", ""))
    milvus_visual_collection_override: str = field(
        default_factory=lambda: os.environ.get("MILVUS_VISUAL_COLLECTION", "")
    )

    # ── Sources ──────────────────────────────────────────────────────────────
    pdfs:     list[str] = field(default_factory=list)
    faqs:     list[str] = field(default_factory=list)
    catalogs: list[str] = field(default_factory=list)
    urls:     list[str] = field(default_factory=list)

    # ── Incremental mode ─────────────────────────────────────────────────────
    # When True, sources whose content hash hasn't changed are skipped.
    incremental: bool = True

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.batch_size < 1:
            raise ConfigError("batch_size must be ≥ 1")
        if self.min_chunk_len < 1:
            raise ConfigError("min_chunk_len must be ≥ 1")
        if self.min_image_dim < 1:
            raise ConfigError("min_image_dim must be ≥ 1")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.collection_name):
            raise ConfigError(f"Invalid Milvus collection name: '{self.collection_name}'")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.visual_collection_name):
            raise ConfigError(f"Invalid Milvus visual collection name: '{self.visual_collection_name}'")

    @property
    def milvus_uri(self) -> str:
        """Local Milvus Lite file by default; set MILVUS_URI to point at a remote server."""
        if self.milvus_uri_override:
            return self.milvus_uri_override
        return str(self.output_dir / "milvus.db")

    @property
    def collection_name(self) -> str:
        """Text-chunk collection; base name matches rag.py's RAGConfig default."""
        if self.milvus_collection_override:
            return self.milvus_collection_override
        return f"multimodal_text_{self.run_id}" if self.run_id else "multimodal_text"

    @property
    def visual_collection_name(self) -> str:
        """Image-region collection (CLIP vectors); base name matches rag.py's default."""
        if self.milvus_visual_collection_override:
            return self.milvus_visual_collection_override
        return f"multimodal_visual_{self.run_id}" if self.run_id else "multimodal_visual"

    @property
    def assets_dir(self) -> Path:
        """Cropped figure images, saved so asset_path can be opened/rendered later."""
        return self.output_dir / "assets"

    @property
    def page_images_dir(self) -> Path:
        """Full-page renders, saved so page_image_path can be opened/rendered later."""
        return self.output_dir / "page_images"

    @property
    def chunks_backup_path(self) -> Path:
        """JSON audit file for this run's chunks only — Milvus is the source of truth."""
        suffix = f"_{self.run_id}" if self.run_id else ""
        return self.output_dir / f"pdf_chunks{suffix}.json"

    @property
    def hash_cache_path(self) -> Path:
        return self.output_dir / "source_hashes.json"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from a YAML file. Unknown keys emit a warning instead of being silently ignored."""
        try:
            import yaml  # optional dependency
        except ImportError:
            raise ConfigError("PyYAML is required for --config. Install with: pip install pyyaml")
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"Config file not found: {path}")

        obj = cls()
        known_keys = {f.name for f in obj.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        for k, v in data.items():
            if k in known_keys:
                setattr(obj, k, v)
            else:
                log.warning("YAML config: unrecognised key '%s' — check for typos. Ignoring.", k)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    sources_processed: int   = 0
    sources_skipped:   int   = 0   # unchanged since last run
    sources_failed:    int   = 0
    total_chunks:      int   = 0
    total_images:      int   = 0   # extracted figure/diagram count
    total_tables:      int   = 0   # extracted table count
    vectors_deleted:   int   = 0   # stale vectors purged for changed sources
    embed_time_s:      float = 0.0
    image_embed_time_s: float = 0.0
    insert_time_s:     float = 0.0
    total_time_s:      float = 0.0

    def report(self) -> None:
        log.info("─" * 60)
        log.info("Run Metrics")
        log.info("  Sources processed    : %d", self.sources_processed)
        log.info("  Sources skipped      : %d (incremental / no change)", self.sources_skipped)
        log.info("  Sources failed       : %d", self.sources_failed)
        log.info("  Total chunks         : %d", self.total_chunks)
        log.info("  Total images         : %d", self.total_images)
        log.info("  Total tables         : %d", self.total_tables)
        log.info("  Stale vectors purged : %d", self.vectors_deleted)
        log.info("  Embed time (text)    : %.1f s", self.embed_time_s)
        log.info("  Embed time (images)  : %.1f s", self.image_embed_time_s)
        log.info("  Milvus insert time   : %.1f s", self.insert_time_s)
        log.info("  Total time           : %.1f s", self.total_time_s)
        log.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL HASH CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _file_md5(path: str, chunk_bytes: int = 65536) -> str:
    """Fast MD5 of a file's content — used for incremental change detection."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while block := f.read(chunk_bytes):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _content_md5(content: str) -> str:
    """MD5 of a string — used to hash fetched URL content so page changes are detected."""
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


class HashCache:
    """
    Persists MD5 hashes of each source between pipeline runs.
    On the next run, if a source's hash matches what's stored here, it is
    skipped — saving embedding time and avoiding unnecessary Milvus writes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("Hash cache corrupted — rebuilding from scratch.")

    def is_unchanged(self, key: str, current_hash: str) -> bool:
        return self._cache.get(key) == current_hash

    def update(self, key: str, current_hash: str) -> None:
        self._cache[key] = current_hash

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_text_quality(text: str) -> float:
    """Return a conservative 0..1 score for whether extracted text is readable."""
    if not text or len(text.strip()) < 100:
        return 0.0
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    printable = sum(char.isprintable() for char in chars) / len(chars)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in chars) / len(chars)
    controls = sum(not char.isprintable() for char in chars) / len(chars)
    # Broken Type 3 fonts commonly decode mostly as NULs, symbols, or one letter.
    return max(0.0, min(1.0, printable * 0.45 + ascii_letters * 0.75 - controls * 4.0))


def _ocr_pdf_page(path: str, page_number: int, dpi: int = 150) -> tuple[int, str]:
    """Render and OCR one PDF page using Poppler and Tesseract."""
    with tempfile.TemporaryDirectory(prefix="emb-ocr-") as tmp_dir:
        image_root = Path(tmp_dir) / "page"
        render = subprocess.run(
            [
                "pdftoppm", "-f", str(page_number), "-l", str(page_number),
                "-r", str(dpi), "-jpeg", "-jpegopt", "quality=90",
                "-singlefile", path, str(image_root),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        if render.returncode != 0:
            raise LoaderError(f"Could not render PDF page {page_number}")
        ocr_env = os.environ.copy()
        ocr_env["OMP_THREAD_LIMIT"] = "1"
        ocr = subprocess.run(
            ["tesseract", f"{image_root}.jpg", "stdout", "-l", "eng", "--psm", "3"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=ocr_env,
        )
        if ocr.returncode != 0:
            raise LoaderError(f"OCR failed on PDF page {page_number}: {ocr.stderr.strip()}")
        return page_number, ocr.stdout.strip()


def _ocr_pdf(path: str, page_count: int) -> str:
    cache_path = Path(f"{path}.ocr.txt")
    source_mtime = Path(path).stat().st_mtime
    if cache_path.exists() and cache_path.stat().st_mtime >= source_mtime:
        cached = cache_path.read_text(encoding="utf-8")
        if _pdf_text_quality(cached) >= 0.70:
            log.info("  Reusing validated OCR cache: %s", cache_path)
            return cached

    workers = min(4, max(1, os.cpu_count() or 1))
    log.warning(
        "Native PDF text is garbled; running Tesseract OCR on %d pages (%d workers) ...",
        page_count, workers,
    )
    pages: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ocr_pdf_page, path, page): page for page in range(1, page_count + 1)}
        for completed, future in enumerate(as_completed(futures), 1):
            page_number, text = future.result()
            pages[page_number] = text
            if completed % 20 == 0 or completed == page_count:
                log.info("  OCR progress: %d/%d pages", completed, page_count)
    result = "\n\n".join(
        f"[Page {page}]\n{pages[page]}" for page in range(1, page_count + 1) if pages.get(page)
    )
    if _pdf_text_quality(result) < 0.70:
        raise LoaderError("OCR output still failed text-quality validation")
    try:
        cache_path.write_text(result, encoding="utf-8")
        log.info("  OCR text cache saved: %s", cache_path)
    except OSError as exc:
        # The source may live on a read-only mount (e.g. Docker bind mount) —
        # losing the cache is fine, losing the OCR result is not.
        log.warning("  Could not save OCR cache (%s) — continuing without it.", exc)
    return result


def _load_pdf(path: str) -> str:
    """
    Extract PDF text natively, falling back to OCR when font decoding is garbled.
    """
    try:
        reader = PdfReader(path)
        sample_count = min(8, len(reader.pages))
        sample_text = "\n\n".join(
            text for page in reader.pages[:sample_count] if (text := page.extract_text())
        )
        quality = _pdf_text_quality(sample_text)
        log.info("  Native PDF extraction quality: %.2f", quality)
        if quality < 0.70:
            return _ocr_pdf(path, len(reader.pages))
        pages = [text for page in reader.pages if (text := page.extract_text())]
        return "\n\n".join(pages)
    except Exception as exc:
        raise LoaderError(f"PDF load failed [{path}]: {exc}") from exc


def _slug(path: str) -> str:
    """Filesystem-safe stem for namespacing saved asset/page-image files."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).stem)


def _docling_convert(path: str, cfg: PipelineConfig):
    """Shared docling conversion pass, used by both _extract_pdf_images and _extract_pdf_tables
    so a source is only ever converted once regardless of how many visual-element types are enabled."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = cfg.image_scale
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = cfg.extract_tables

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    page_range = (1, cfg.max_pdf_pages) if cfg.max_pdf_pages else (1, sys.maxsize)

    try:
        result = converter.convert(path, page_range=page_range)
        return result.document
    except Exception as exc:
        raise LoaderError(f"Docling conversion failed [{path}]: {exc}") from exc


def _save_page_image(doc, page_no: int, pages_dir: Path, saved_pages: dict[int, str]) -> str:
    """Save (once per page, cached in saved_pages) the full-page render used for citation."""
    if page_no not in saved_pages:
        page = doc.pages.get(page_no)
        if page and page.image and page.image.pil_image:
            page_path = pages_dir / f"p{page_no:04d}.png"
            page.image.pil_image.convert("RGB").save(page_path)
            saved_pages[page_no] = str(page_path)
        else:
            saved_pages[page_no] = ""
    return saved_pages[page_no]


def _extract_pdf_images(doc, path: str, cfg: PipelineConfig) -> list[dict]:
    """
    Extract figures/diagrams from an already-converted docling document, cropping each
    to a PNG plus a full-page render for citation, and pulling a best-effort caption.
    Independent of _load_pdf's native/OCR text path. Returns one dict per image with a
    transient "_pil_image" key (consumed by embed_images(), stripped before persistence).
    Decorative images smaller than cfg.min_image_dim are skipped.
    """
    assets_dir = cfg.assets_dir / _slug(path)
    pages_dir  = cfg.page_images_dir / _slug(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    saved_pages: dict[int, str] = {}
    elements: list[dict] = []

    for idx, pic in enumerate(doc.pictures):
        if not pic.prov:
            continue
        prov = pic.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox

        img = pic.get_image(doc)
        if img is None:
            continue
        img = img.convert("RGB")
        if img.width < cfg.min_image_dim or img.height < cfg.min_image_dim:
            continue

        asset_path = assets_dir / f"p{page_no:04d}_i{idx:03d}.png"
        img.save(asset_path)
        page_image_path = _save_page_image(doc, page_no, pages_dir, saved_pages)

        caption_raw = pic.caption_text(doc) or ""
        caption = clean_text(caption_raw)
        # Same broken-font garbling that triggers OCR fallback for text can
        # also corrupt captions — drop low-quality captions instead of
        # embedding/storing garbage.
        if not caption or _pdf_text_quality(caption) < 0.70:
            caption = f"Figure from {Path(path).name}, page {page_no} (no caption extracted)."

        elements.append({
            "source":          path,
            "source_type":     "pdf",
            "element_type":    "image",
            "record_level":    "element",
            "page_no":         page_no,
            "bbox":            {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b,
                                 "coord_origin": str(bbox.coord_origin)},
            "chunk":           caption,
            "asset_path":      str(asset_path),
            "page_image_path": page_image_path,
            "element_ids":     [],
            "relation_ids":    [],
            "section_path":    "",
            "word_start":      -1,
            "word_end":        -1,
            "_docling_index":  idx,   # stable across re-runs; used for the Milvus primary key
            "_pil_image":      img,
        })

    return elements


def _ocr_table_cell(crop: Image.Image, psm: str = "6") -> str:
    """OCR a single table-cell crop. psm=6 (uniform text block) handles both
    single-line and multi-line wrapped cell content."""
    with tempfile.TemporaryDirectory(prefix="emb-cell-ocr-") as tmp_dir:
        p = Path(tmp_dir) / "cell.png"
        crop.save(p)
        result = subprocess.run(
            ["tesseract", str(p), "stdout", "-l", "eng", "--psm", psm],
            capture_output=True, text=True, check=False, timeout=15,
        )
        return result.stdout.strip()


def _reconstruct_table_via_ocr(tbl, doc, page_no: int, table_crop: Image.Image) -> str:
    """
    Recover a table's text via per-cell OCR when docling's native (font-decode) text is
    garbled, using docling's cell bounding boxes (from the layout model, independent of
    the broken font-decode) as ground truth for where to crop. Cell/table bboxes can use
    different coordinate origins (TOPLEFT vs BOTTOMLEFT) — both normalised via
    to_top_left_origin() before cropping. OCR runs per-cell in parallel, then cells are
    placed into a num_rows x num_cols grid and rendered as markdown.
    """
    page_height = doc.pages[page_no].size.height
    table_bbox = tbl.prov[0].bbox.to_top_left_origin(page_height)
    scale_x = table_crop.width / table_bbox.width
    scale_y = table_crop.height / table_bbox.height

    data = tbl.data
    grid: list[list[str]] = [["" for _ in range(data.num_cols)] for _ in range(data.num_rows)]

    def _cell_text(cell) -> tuple:
        cb = cell.bbox.to_top_left_origin(page_height)
        left   = (cb.l - table_bbox.l) * scale_x
        right  = (cb.r - table_bbox.l) * scale_x
        top    = (cb.t - table_bbox.t) * scale_y
        bottom = (cb.b - table_bbox.t) * scale_y
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        box = (
            max(0, int(left) - 3), max(0, int(top) - 3),
            min(table_crop.width, int(right) + 3), min(table_crop.height, int(bottom) + 3),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return cell, ""
        text = clean_text(_ocr_table_cell(table_crop.crop(box)))
        return cell, text

    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 1))) as pool:
        for cell, text in pool.map(_cell_text, data.table_cells):
            for r in range(cell.start_row_offset_idx, min(cell.end_row_offset_idx, data.num_rows)):
                for c in range(cell.start_col_offset_idx, min(cell.end_col_offset_idx, data.num_cols)):
                    grid[r][c] = text

    if not grid:
        return ""
    lines = ["| " + " | ".join(grid[0]) + " |", "|" + "|".join(["---"] * len(grid[0])) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in grid[1:])
    return "\n".join(lines)


def _extract_pdf_tables(doc, path: str, cfg: PipelineConfig) -> list[dict]:
    """
    Extract tables from an already-converted docling document as clean markdown text,
    inserted into the TEXT collection (not the visual one) since a table's value is its
    factual content, which the exam LLM needs to read, not a raster crop CLIP could only
    shallowly embed. Falls back to _reconstruct_table_via_ocr() when native text quality
    is below threshold. A cropped PNG is still saved to asset_path for citation/display.
    """
    assets_dir = cfg.assets_dir / _slug(path)
    pages_dir  = cfg.page_images_dir / _slug(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    saved_pages: dict[int, str] = {}
    elements: list[dict] = []

    for idx, tbl in enumerate(doc.tables):
        if not tbl.prov:
            continue
        prov = tbl.prov[0]
        page_no = prov.page_no
        bbox = prov.bbox

        crop = tbl.get_image(doc)
        if crop is None:
            continue
        crop = crop.convert("RGB")

        native_md = tbl.export_to_markdown(doc)
        if _pdf_text_quality(native_md) >= 0.70:
            table_md = native_md
        else:
            try:
                table_md = _reconstruct_table_via_ocr(tbl, doc, page_no, crop)
            except Exception as exc:
                log.warning("  Table OCR reconstruction failed (page %d): %s", page_no, exc)
                continue
            if _pdf_text_quality(table_md) < 0.50:
                log.warning("  Table on page %d unreadable even after OCR — skipping.", page_no)
                continue

        asset_path = assets_dir / f"p{page_no:04d}_t{idx:03d}.png"
        crop.save(asset_path)
        page_image_path = _save_page_image(doc, page_no, pages_dir, saved_pages)

        caption_raw = tbl.caption_text(doc) or ""
        caption = clean_text(caption_raw)
        if not caption or _pdf_text_quality(caption) < 0.70:
            caption = f"Table from {Path(path).name}, page {page_no}."

        elements.append({
            "source":          path,
            "source_type":     "pdf",
            "element_type":    "table",
            "record_level":    "element",
            "page_no":         page_no,
            "bbox":            {"l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b,
                                 "coord_origin": str(bbox.coord_origin)},
            "chunk":           f"{caption}\n\n{table_md}",
            "asset_path":      str(asset_path),
            "page_image_path": page_image_path,
            "element_ids":     [],
            "relation_ids":    [],
            "section_path":    "",
            "word_start":      -1,
            "word_end":        -1,
            "_docling_index":  idx,
        })

    return elements


def _load_faq(path: str) -> str:
    """
    Load a FAQ JSON file — expected format: list of {question, answer} objects.
    Each entry is formatted as "Q: ...\nA: ..." for downstream chunking.
    """
    try:
        with open(path, encoding="utf-8") as f:
            faqs = json.load(f)
        if not isinstance(faqs, list):
            raise LoaderError(f"FAQ JSON must be a list of {{question, answer}} objects: {path}")
        blocks = [
            f"Q: {item['question'].strip()}\nA: {item['answer'].strip()}"
            for item in faqs
            if item.get("question") and item.get("answer")
        ]
        return "\n\n".join(blocks)
    except (json.JSONDecodeError, KeyError) as exc:
        raise LoaderError(f"FAQ JSON parse failed [{path}]: {exc}") from exc


def _load_catalog(path: str) -> str:
    """
    Load a CSV product catalog.
    Each row is serialised as "col1: val1 | col2: val2 | ..." for chunking.
    """
    try:
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                line = " | ".join(f"{k}: {v}" for k, v in row.items() if v and str(v).strip())
                if line:
                    rows.append(line)
        if not rows:
            raise LoaderError(f"CSV catalog produced no rows: {path}")
        return "\n".join(rows)
    except Exception as exc:
        raise LoaderError(f"CSV load failed [{path}]: {exc}") from exc


def _scrape_url(url: str, timeout: int = 15, retries: int = 3) -> str:
    """
    Download and extract visible text from a URL using requests + BeautifulSoup.
    Strips script/style/nav/footer tags before extracting text.
    Retries up to `retries` times with exponential back-off on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise LoaderError("requests and beautifulsoup4 are required for URL scraping.")

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "EMB-Builder/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "head", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            if not text.strip():
                raise LoaderError(f"URL returned empty content: {url}")
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2 ** attempt
                log.warning(
                    "URL fetch attempt %d/%d failed (%s) — retrying in %ds",
                    attempt, retries, exc, wait
                )
                time.sleep(wait)

    raise LoaderError(
        f"URL scrape failed after {retries} attempts [{url}]: {last_exc}"
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

_RE_NON_ASCII  = re.compile(r"[^\x00-\x7F]+")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalise raw text before chunking:
      - Strip non-ASCII characters (e.g. smart quotes, ligatures)
      - Collapse all whitespace to single spaces
    """
    text = _RE_NON_ASCII.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    source_type: str,
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Split cleaned text into overlapping word-window chunks (window=chunk_size,
    step=chunk_size-chunk_overlap), discarding windows shorter than min_chunk_len.
    `local_index` is position within this source only, not the Milvus primary key
    (see _stable_chunk_id). Chunks carry the same schema fields as image/table
    elements (page_no, element_type, bbox, asset_path, etc.), filled with neutral
    defaults here, so rag.py can query both collections uniformly.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[dict] = []
    step = cfg.chunk_size - cfg.chunk_overlap
    if step < 1:
        step = 1

    for local_index, i in enumerate(range(0, len(words), step)):
        window = " ".join(words[i : i + cfg.chunk_size])
        if len(window.strip()) < cfg.min_chunk_len:
            continue
        chunks.append({
            "local_index": local_index,
            "chunk":       window,
            "source":      source,
            "source_type": source_type,
            "word_start":  i,
            "word_end":    min(i + cfg.chunk_size, len(words)),
            "page_no":         -1,
            "element_type":    "text",
            "record_level":    "chunk",
            "bbox":            {},
            "asset_path":      "",
            "page_image_path": "",
            "element_ids":     [],
            "relation_ids":    [],
            "section_path":    "",
        })

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """Encode chunk text into float32 embedding vectors. Milvus's COSINE metric computes
    cosine similarity natively from raw vectors, so manual L2 normalisation isn't needed."""
    texts = [c["chunk"] for c in chunks]
    log.info("Embedding %d chunks (batch=%d) …", len(texts), cfg.batch_size)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    elapsed = time.perf_counter() - t0

    log.info(
        "Embedding complete in %.1f s  (%.0f chunks/s)",
        elapsed,
        len(texts) / max(elapsed, 1e-9),
    )
    return embeddings, elapsed


def embed_images(
    images: list[dict],
    visual_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """Encode extracted figure/diagram crops into CLIP embedding vectors.
    normalize_embeddings=True matches rag.py's cross-modal query encoding."""
    pil_images = [img["_pil_image"] for img in images]
    log.info("Embedding %d image(s) with CLIP (batch=%d) …", len(pil_images), cfg.batch_size)

    t0 = time.perf_counter()
    embeddings = visual_model.encode(
        pil_images,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    elapsed = time.perf_counter() - t0

    log.info(
        "Image embedding complete in %.1f s  (%.0f images/s)",
        elapsed,
        len(pil_images) / max(elapsed, 1e-9),
    )
    return embeddings, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# STABLE PRIMARY KEYS
# ─────────────────────────────────────────────────────────────────────────────

def _stable_chunk_id(source: str, word_start: int) -> int:
    """Deterministic int64-safe primary key derived from (source, word_start).
    Re-running the same source produces the same IDs, so re-inserts are idempotent."""
    digest = hashlib.md5(f"{source}:{word_start}".encode()).digest()   # 16 raw bytes
    full_int = int.from_bytes(digest, byteorder="big")
    return full_int % (2 ** 63)   # clamp to signed int64 range


def _stable_image_id(source: str, page_no: int, index: int) -> int:
    """Deterministic int64-safe primary key for an image element, namespaced "img" so it
    can never collide with a text chunk's id. `index` is the picture's position in
    doc.pictures, stable across re-runs as long as the source PDF doesn't change."""
    digest = hashlib.md5(f"{source}:img:{page_no}:{index}".encode()).digest()
    full_int = int.from_bytes(digest, byteorder="big")
    return full_int % (2 ** 63)


def _stable_table_id(source: str, page_no: int, index: int) -> int:
    """Deterministic int64-safe primary key for a table element, namespaced "tbl" so it
    can't collide with a text chunk's id even though both land in the same collection."""
    digest = hashlib.md5(f"{source}:tbl:{page_no}:{index}".encode()).digest()
    full_int = int.from_bytes(digest, byteorder="big")
    return full_int % (2 ** 63)


def _to_milvus_records(chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Zip text chunks and their embedding vectors into Milvus-ready record dicts."""
    if len(chunks) != len(embeddings):
        raise EmbeddingPipelineError(
            f"Chunk/embedding count mismatch: {len(chunks)} chunks vs "
            f"{len(embeddings)} embeddings — this is a bug, please report it."
        )

    return [
        {
            "id":              _stable_chunk_id(c["source"], c["word_start"]),
            "vector":          vec.tolist(),
            "chunk":           c["chunk"],
            "source":          c["source"],
            "source_type":     c["source_type"],
            "word_start":      c["word_start"],
            "word_end":        c["word_end"],
            "page_no":         c.get("page_no", -1),
            "element_type":    c.get("element_type", "text"),
            "record_level":    c.get("record_level", "chunk"),
            "bbox":            c.get("bbox", {}),
            "asset_path":      c.get("asset_path", ""),
            "page_image_path": c.get("page_image_path", ""),
            "element_ids":     c.get("element_ids", []),
            "relation_ids":    c.get("relation_ids", []),
            "section_path":    c.get("section_path", ""),
        }
        for c, vec in zip(chunks, embeddings)
    ]


def _to_milvus_table_records(tables: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Zip table elements and their TEXT-model vectors into Milvus-ready record dicts.
    Same field shape as _to_milvus_records, keyed by _stable_table_id."""
    if len(tables) != len(embeddings):
        raise EmbeddingPipelineError(
            f"Table/embedding count mismatch: {len(tables)} tables vs "
            f"{len(embeddings)} embeddings — this is a bug, please report it."
        )

    return [
        {
            "id":              _stable_table_id(t["source"], t["page_no"], t["_docling_index"]),
            "vector":          vec.tolist(),
            "chunk":           t["chunk"],
            "source":          t["source"],
            "source_type":     t["source_type"],
            "word_start":      t["word_start"],
            "word_end":        t["word_end"],
            "page_no":         t["page_no"],
            "element_type":    t["element_type"],
            "record_level":    t["record_level"],
            "bbox":            t["bbox"],
            "asset_path":      t["asset_path"],
            "page_image_path": t["page_image_path"],
            "element_ids":     t["element_ids"],
            "relation_ids":    t["relation_ids"],
            "section_path":    t["section_path"],
        }
        for t, vec in zip(tables, embeddings)
    ]


def _to_milvus_image_records(images: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Zip image elements and their CLIP vectors into Milvus-ready record dicts.
    Mirrors _to_milvus_records' field shape (minus the transient "_pil_image" key)."""
    if len(images) != len(embeddings):
        raise EmbeddingPipelineError(
            f"Image/embedding count mismatch: {len(images)} images vs "
            f"{len(embeddings)} embeddings — this is a bug, please report it."
        )

    return [
        {
            "id":              _stable_image_id(img["source"], img["page_no"], img["_docling_index"]),
            "vector":          vec.tolist(),
            "chunk":           img["chunk"],
            "source":          img["source"],
            "source_type":     img["source_type"],
            "word_start":      img["word_start"],
            "word_end":        img["word_end"],
            "page_no":         img["page_no"],
            "element_type":    img["element_type"],
            "record_level":    img["record_level"],
            "bbox":            img["bbox"],
            "asset_path":      img["asset_path"],
            "page_image_path": img["page_image_path"],
            "element_ids":     img["element_ids"],
            "relation_ids":    img["relation_ids"],
            "section_path":    img["section_path"],
        }
        for img, vec in zip(images, embeddings)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MILVUS COLLECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def get_milvus_client(cfg: PipelineConfig) -> MilvusClient:
    """Open (or create) a Milvus connection at the configured URI."""
    log.info("Connecting to Milvus: %s", cfg.milvus_uri)
    try:
        return MilvusClient(uri=cfg.milvus_uri)
    except Exception as exc:
        raise VectorStoreError(
            f"Could not open Milvus at '{cfg.milvus_uri}': {exc}"
        ) from exc


def ensure_collection(client: MilvusClient, collection_name: str, dim: int) -> None:
    """
    Create the Milvus collection if it doesn't already exist.
    Uses COSINE metric so raw (un-normalised) vectors can be compared directly.
    """
    try:
        if client.has_collection(collection_name):
            log.info("Collection '%s' already exists — reusing.", collection_name)
            return
        log.info(
            "Creating collection '%s' (dim=%d, metric=COSINE) …",
            collection_name, dim
        )
        client.create_collection(
            collection_name=collection_name,
            dimension=dim,
            metric_type="COSINE",
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Could not create collection '{collection_name}': {exc}"
        ) from exc


def delete_existing_source(
    client: MilvusClient,
    collection_name: str,
    source: str,
) -> int:
    """
    Purge all previously-inserted vectors for a source before re-inserting its updated
    chunks — chunk boundaries shift when source content changes, so without purging
    first, stale chunks would linger alongside fresh ones. The visual collection is
    created lazily, so not existing yet is a no-op here, not an error.
    """
    if not client.has_collection(collection_name):
        return 0
    try:
        result = client.delete(
            collection_name=collection_name,
            filter=f'source == "{source}"',
        )
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        return count
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to purge existing vectors for source '{source}' in collection "
            f"'{collection_name}'. Aborting to prevent duplicate/stale data. "
            f"Original error: {exc}"
        ) from exc


def insert_chunks(
    client: MilvusClient,
    collection_name: str,
    records: list[dict],
    retries: int = 3,
) -> float:
    """Insert embedding records into Milvus with exponential-backoff retries."""
    t0 = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            client.insert(collection_name=collection_name, data=records)
            return time.perf_counter() - t0
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2 ** attempt
                log.warning(
                    "Milvus insert attempt %d/%d failed (%s) — retrying in %ds",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)

    raise VectorStoreError(
        f"Insert into '{collection_name}' failed after {retries} attempts: {last_exc}"
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    chunks: list[dict],
    cfg: PipelineConfig,
    skipped_sources: list[str],
    images: list[dict] | None = None,
    tables: list[dict] | None = None,
) -> None:
    """
    Write a JSON audit backup for this run's chunks/images/tables. Milvus is the
    durable source of truth; the metadata header records whether the run was
    incremental and which sources were skipped, so this file is self-documenting
    and isn't mistaken for a complete index dump. Transient keys ("_pil_image",
    "_docling_index") are stripped since they aren't needed once persisted.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    def _strip_transient(items: list[dict] | None) -> list[dict]:
        return [
            {k: v for k, v in item.items() if k not in ("_pil_image", "_docling_index")}
            for item in (items or [])
        ]
    output = {
        "_metadata": {
            "WARNING": (
                "This file contains only chunks/images/tables processed in THIS run. "
                "Skipped (unchanged) sources are in Milvus but NOT here."
            ),
            "incremental": cfg.incremental,
            "skipped_sources": skipped_sources,
            "milvus_uri": cfg.milvus_uri,
            "collection": cfg.collection_name,
            "visual_collection": cfg.visual_collection_name,
        },
        "chunks": chunks,
        "images": _strip_transient(images),
        "tables": _strip_transient(tables),
    }
    cfg.chunks_backup_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Chunk/image/table audit backup (this run only) → %s", cfg.chunks_backup_path)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test(
    client: MilvusClient,
    collection_name: str,
    model: SentenceTransformer,
    query: str | None = None,
    top_k: int = 3,
    visual_collection_name: str | None = None,
    visual_model: SentenceTransformer | None = None,
) -> None:
    """
    Run a quick sanity-check search against the just-populated collection(s). When no
    query is given, one is derived from the first indexed chunk so the check is always
    relevant to the actual indexed content. When visual_collection_name/visual_model are
    supplied, also runs a cross-modal check mirroring rag.py's retrieve().
    """
    log.info("── Smoke test ──────────────────────────────────────")
    try:
        client.load_collection(collection_name)
    except Exception as exc:
        log.warning("Could not load collection for smoke test: %s", exc)
        return

    if not query:
        try:
            sample = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["chunk"],
                limit=1,
            )
            if sample:
                query = " ".join(sample[0]["chunk"].split()[:20])
                log.info("Auto-derived smoke-test query from first chunk: %s …", query)
            else:
                log.warning("Collection appears empty — skipping smoke test.")
                return
        except Exception as exc:
            log.warning("Could not retrieve sample chunk for smoke test: %s", exc)
            return

    log.info("Query: %s", query)
    q_vec = model.encode([query]).tolist()
    try:
        results = client.search(
            collection_name=collection_name,
            data=q_vec,
            limit=top_k,
            output_fields=["chunk"],
        )
        for rank, hit in enumerate(results[0], 1):
            score   = hit.get("distance", 0.0)
            entity  = hit.get("entity", {})
            preview = str(entity.get("chunk", ""))[:100].replace("\n", " ")
            log.info("  [%d] score=%.4f  %s …", rank, score, preview)
    except Exception as exc:
        log.warning("Smoke test search failed: %s", exc)

    if visual_collection_name and visual_model:
        try:
            client.load_collection(visual_collection_name)
            visual_query_vec = visual_model.encode([query], normalize_embeddings=True).tolist()
            visual_results = client.search(
                collection_name=visual_collection_name,
                data=visual_query_vec,
                limit=top_k,
                output_fields=["chunk", "asset_path", "page_no"],
            )
            log.info("  Cross-modal (CLIP) check against '%s':", visual_collection_name)
            for rank, hit in enumerate(visual_results[0], 1):
                score  = hit.get("distance", 0.0)
                entity = hit.get("entity", {})
                log.info(
                    "    [%d] score=%.4f  page=%s  %s",
                    rank, score, entity.get("page_no"), entity.get("asset_path", ""),
                )
        except Exception as exc:
            log.warning("Visual smoke test search failed: %s", exc)

    log.info("────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> RunMetrics:
    """
    Orchestrates the full embedding pipeline: validate config, load the text model,
    ensure the text collection exists, then for each source compute its content hash
    (skipping if unchanged), load/clean/chunk it, extract figures and tables for PDFs,
    purge stale vectors, and accumulate for batch embedding. Chunks, tables, and images
    are then embedded and inserted, followed by an audit backup, hash-cache persistence,
    and a smoke test.
    """
    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    metrics         = RunMetrics()
    hash_cache      = HashCache(cfg.hash_cache_path) if cfg.incremental else None
    skipped_sources: list[str] = []
    t_total         = time.perf_counter()

    # ── 1. Load embedding model ──────────────────────────────────
    log.info("Loading embedding model: %s", cfg.embed_model)
    try:
        model = SentenceTransformer(cfg.embed_model)
    except Exception as exc:
        raise EmbeddingPipelineError(
            f"Cannot load embedding model '{cfg.embed_model}': {exc}"
        ) from exc

    dim = model.get_embedding_dimension()

    # ── 2. Connect to Milvus & ensure text collection ────────────
    client = get_milvus_client(cfg)
    ensure_collection(client, cfg.collection_name, dim)

    # ── 3. Ingest all sources ────────────────────────────────────
    log.info("Ingesting sources …")
    all_chunks: list[dict] = []
    all_images: list[dict] = []
    all_tables: list[dict] = []

    source_list: list[tuple[str, str]] = (
        [("pdf",     p) for p in cfg.pdfs]
        + [("faq",   p) for p in cfg.faqs]
        + [("catalog", p) for p in cfg.catalogs]
        + [("url",   u) for u in cfg.urls]
    )

    if not source_list:
        raise ConfigError(
            "No data sources specified. Use --pdf, --faq, --catalog, or --url."
        )

    for source_type, path in source_list:
        # URL sources must be fetched first so the hash covers page content, not just the URL.
        if source_type == "url":
            try:
                raw_content = _scrape_url(path)
            except LoaderError as exc:
                log.error("  FAIL  %s", exc)
                metrics.sources_failed += 1
                continue
            current_hash = _content_md5(raw_content)
        else:
            raw_content  = None        # loaded below after skip check
            current_hash = _file_md5(path)

        # ── Incremental skip check ───────────────────────────────
        if hash_cache and hash_cache.is_unchanged(path, current_hash):
            log.info(
                "  SKIP  [%s]  %s  (unchanged since last run)",
                source_type.upper(), path,
            )
            metrics.sources_skipped += 1
            skipped_sources.append(path)
            continue

        log.info("  LOAD  [%s]  %s", source_type.upper(), path)
        try:
            # For non-URL sources, load now (after confirming it's not skipped)
            if raw_content is None:
                if source_type == "pdf":
                    raw_content = _load_pdf(path)
                elif source_type == "faq":
                    raw_content = _load_faq(path)
                elif source_type == "catalog":
                    raw_content = _load_catalog(path)
                else:
                    log.warning("Unknown source type '%s' — skipping.", source_type)
                    metrics.sources_failed += 1
                    continue

            cleaned = clean_text(raw_content)
            chunks  = chunk_text(cleaned, source=path, source_type=source_type, cfg=cfg)
            log.info("  → %d chunks created", len(chunks))

            images: list[dict] = []
            tables: list[dict] = []
            if source_type == "pdf" and (cfg.extract_images or cfg.extract_tables):
                try:
                    # One shared conversion pass for both — table-structure
                    # detection roughly triples per-page cost, so this only
                    # pays that price when cfg.extract_tables is actually on.
                    doc = _docling_convert(path, cfg)
                    if cfg.extract_images:
                        images = _extract_pdf_images(doc, path, cfg)
                        log.info("  → %d image(s) extracted", len(images))
                    if cfg.extract_tables:
                        tables = _extract_pdf_tables(doc, path, cfg)
                        log.info("  → %d table(s) extracted", len(tables))
                except LoaderError as exc:
                    # Image/table extraction is additive — a failure here
                    # shouldn't sink the (already-successful) text extraction.
                    log.warning("  Visual extraction failed (continuing with text only): %s", exc)

            # Purge stale vectors before inserting fresh ones — both collections,
            # since a source's images live in the visual one, its text in the text one.
            deleted = delete_existing_source(client, cfg.collection_name, path)
            deleted += delete_existing_source(client, cfg.visual_collection_name, path)
            if deleted:
                log.info("  Purged %d stale vector(s) for this source.", deleted)
                metrics.vectors_deleted += deleted

            all_chunks.extend(chunks)
            all_images.extend(images)
            all_tables.extend(tables)
            metrics.sources_processed += 1

            if hash_cache:
                hash_cache.update(path, current_hash)

        except LoaderError as exc:
            log.error("  FAIL  %s", exc)
            metrics.sources_failed += 1

    log.info(
        "Ingestion summary: %d processed, %d skipped (in Milvus, unchanged), %d failed",
        metrics.sources_processed,
        metrics.sources_skipped,
        metrics.sources_failed,
    )
    if skipped_sources:
        log.info(
            "Skipped sources (existing vectors retained in Milvus): %s",
            ", ".join(skipped_sources),
        )

    if not all_chunks and not all_images and not all_tables:
        if metrics.sources_skipped > 0:
            log.info("All sources skipped (unchanged). No new chunks/images/tables to process.")
            metrics.total_time_s = time.perf_counter() - t_total
            return metrics

        raise EmbeddingPipelineError(
            "No chunks, images, or tables were produced. Check your source files and paths."
        )

    insert_time = 0.0

    # ── 4. Embed + insert text chunks + tables (same text model + collection) ──
    # Tables are embedded with the TEXT model, not CLIP — a table's value is
    # its factual content (which the exam LLM needs to read), not a raster
    # crop CLIP could only shallowly embed. They're combined into one insert
    # here since both land in cfg.collection_name.
    if all_chunks or all_tables:
        log.info("Total new chunks to embed: %d  (%d table(s) among them)", len(all_chunks) + len(all_tables), len(all_tables))
        metrics.total_chunks = len(all_chunks)
        metrics.total_tables = len(all_tables)

        records: list[dict] = []
        embed_time = 0.0

        if all_chunks:
            chunk_embeddings, t = embed_chunks(all_chunks, model, cfg)
            embed_time += t
            records.extend(_to_milvus_records(all_chunks, chunk_embeddings))

        if all_tables:
            table_embeddings, t = embed_chunks(all_tables, model, cfg)
            embed_time += t
            records.extend(_to_milvus_table_records(all_tables, table_embeddings))

        metrics.embed_time_s = embed_time

        insert_time += insert_chunks(client, cfg.collection_name, records)
        log.info(
            "Inserted %d text vector(s) (chunks + tables) into '%s'",
            len(records), cfg.collection_name,
        )

    # ── 5. Embed + insert images (CLIP) ───────────────────────────
    visual_model: SentenceTransformer | None = None
    if all_images:
        log.info("Total new images to embed: %d", len(all_images))
        metrics.total_images = len(all_images)

        log.info("Loading visual embedding model: %s", cfg.visual_embed_model)
        try:
            visual_model = SentenceTransformer(cfg.visual_embed_model)
        except Exception as exc:
            raise EmbeddingPipelineError(
                f"Cannot load visual embedding model '{cfg.visual_embed_model}': {exc}"
            ) from exc

        image_embeddings, image_embed_time = embed_images(all_images, visual_model, cfg)
        metrics.image_embed_time_s = image_embed_time

        image_records = _to_milvus_image_records(all_images, image_embeddings)

        # Created lazily — only once we actually have an image to store —
        # so a text-only run never leaves behind an empty visual collection.
        visual_dim = visual_model.get_embedding_dimension()
        ensure_collection(client, cfg.visual_collection_name, visual_dim)

        insert_time += insert_chunks(client, cfg.visual_collection_name, image_records)
        log.info(
            "Inserted %d image vector(s) into '%s'",
            len(image_records), cfg.visual_collection_name,
        )

    metrics.insert_time_s = insert_time

    save_outputs(all_chunks, cfg, skipped_sources, images=all_images, tables=all_tables)

    if hash_cache:
        hash_cache.save()

    smoke_test(
        client, cfg.collection_name, model,
        visual_collection_name=cfg.visual_collection_name if visual_model else None,
        visual_model=visual_model,
    )

    metrics.total_time_s = time.perf_counter() - t_total
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embeddings_builder",
        description="Build/update a Milvus vector collection from multiple data sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",        metavar="FILE", help="YAML config file (overrides all flags)")
    p.add_argument("--pdf",           metavar="FILE", action="append", default=[], dest="pdfs",     help="PDF file(s) to ingest")
    p.add_argument("--faq",           metavar="FILE", action="append", default=[], dest="faqs",     help="FAQ JSON file(s)")
    p.add_argument("--catalog",       metavar="FILE", action="append", default=[], dest="catalogs", help="Product catalog CSV(s)")
    p.add_argument("--url",           metavar="URL",  action="append", default=[], dest="urls",     help="URL(s) to scrape")
    p.add_argument("--run-id",        metavar="ID",   default="",                                   help="Namespaces the collections, e.g. v2 → 'multimodal_text_v2'")
    p.add_argument("--output-dir",    metavar="DIR",  default="./index_store",                      help="Output directory for .db + JSON backup + saved images (default: ./index_store)")
    p.add_argument("--model",         metavar="NAME", default=None,                                 help="SentenceTransformer text-embedding model name")
    p.add_argument("--milvus-uri",    metavar="URI",  default=None,                                 help="Local .db path or remote Milvus/Zilliz URI")
    p.add_argument("--collection",    metavar="NAME", default=None,                                 help="Milvus text collection name (default: multimodal_text[_<run-id>])")
    p.add_argument("--chunk-size",    type=int,       default=None)
    p.add_argument("--chunk-overlap", type=int,       default=None)
    p.add_argument("--min-chunk-len", type=int,       default=None,                                 help="Minimum chunk length in characters (default: 50)")
    p.add_argument("--no-incremental", action="store_true",                                         help="Re-process all sources even if unchanged")
    p.add_argument("--no-images",     action="store_true",                                          help="Skip figure/diagram extraction from PDFs (text-only, faster)")
    p.add_argument("--no-tables",     action="store_true",                                          help="Skip table extraction from PDFs (table-structure detection is the slowest stage)")
    p.add_argument("--visual-model",  metavar="NAME", default=None,                                 help="CLIP model for image embeddings (default: sentence-transformers/clip-ViT-B-32)")
    p.add_argument("--visual-collection", metavar="NAME", default=None,                              help="Milvus image collection name (default: multimodal_visual[_<run-id>])")
    p.add_argument("--max-pdf-pages", type=int,       default=None,                                 help="Cap pages scanned for images per PDF (dev/testing on large files)")
    p.add_argument("--log-level",     default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    global log
    log = _setup_logging(args.log_level)

    try:
        if args.config:
            cfg = PipelineConfig.from_yaml(args.config)
        else:
            cfg = PipelineConfig()

        # CLI flags override config file / env vars
        if args.pdfs:           cfg.pdfs      = args.pdfs
        if args.faqs:           cfg.faqs      = args.faqs
        if args.catalogs:       cfg.catalogs  = args.catalogs
        if args.urls:           cfg.urls      = args.urls
        if args.run_id:         cfg.run_id    = args.run_id
        if args.output_dir:     cfg.output_dir = Path(args.output_dir)
        if args.model:          cfg.embed_model = args.model
        if args.milvus_uri:     cfg.milvus_uri_override        = args.milvus_uri
        if args.collection:     cfg.milvus_collection_override = args.collection
        if args.chunk_size:     cfg.chunk_size    = args.chunk_size
        if args.chunk_overlap:  cfg.chunk_overlap = args.chunk_overlap
        if args.min_chunk_len:  cfg.min_chunk_len = args.min_chunk_len
        if args.no_incremental: cfg.incremental   = False
        if args.no_images:      cfg.extract_images = False
        if args.no_tables:      cfg.extract_tables = False
        if args.visual_model:   cfg.visual_embed_model = args.visual_model
        if args.visual_collection: cfg.milvus_visual_collection_override = args.visual_collection
        if args.max_pdf_pages:  cfg.max_pdf_pages = args.max_pdf_pages

        log.info("=" * 60)
        log.info("EMB-Builder  |  Production Multimodal Embeddings Pipeline  (Milvus backend)")
        log.info("  text model    : %s", cfg.embed_model)
        log.info("  chunk_size    : %d  overlap: %d  min_len: %d",
                 cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_len)
        log.info("  output_dir    : %s", cfg.output_dir)
        log.info("  milvus_uri    : %s", cfg.milvus_uri)
        log.info("  text collection   : %s", cfg.collection_name)
        log.info("  extract_images    : %s", cfg.extract_images)
        log.info("  extract_tables    : %s", cfg.extract_tables)
        if cfg.extract_images:
            log.info("  visual model      : %s", cfg.visual_embed_model)
            log.info("  visual collection : %s", cfg.visual_collection_name)
        log.info("  incremental   : %s", cfg.incremental)
        log.info("=" * 60)

        metrics = run_pipeline(cfg)
        metrics.report()

        log.info("Build complete.")
        return 0

    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return 2
    except EmbeddingPipelineError as exc:
        log.error("Pipeline error: %s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return 130
    except Exception:
        log.critical("Unexpected error:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
