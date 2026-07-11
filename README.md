# NEET Multimodal Exam Paper Generator

A local, two-stage pipeline that builds a multimodal (text + images + tables) search
index from your study material (PDFs, FAQs, catalogs, URLs) and generates strict
NTA-NEET-pattern question papers grounded in it, with a real local LLM.

- `embeddings.py` — builds the index: extracts text, figures/diagrams, and tables
  from your sources and stores them in Milvus.
- `rag.py` — generates papers: retrieves relevant material and asks a local LLM to
  write a NEET-format paper strictly grounded in what was retrieved.

---

## 1. Prerequisites

- Python 3.10+
- System packages (used for PDF rendering / OCR fallback):
  - `poppler-utils` (provides `pdftoppm`)
  - `tesseract-ocr`

  Debian/Ubuntu:
  ```bash
  sudo apt install poppler-utils tesseract-ocr
  ```
  Arch:
  ```bash
  sudo pacman -S poppler tesseract tesseract-data-eng
  ```
  macOS:
  ```bash
  brew install poppler tesseract
  ```

- A GPU is strongly recommended for `rag.py`. The exam-generation LLM is
  automatically selected based on what's available:

  | Hardware | Model used |
  |---|---|
  | CUDA GPU, 10 GB+ VRAM | Qwen2.5-VL-7B-Instruct (4-bit) — best, genuine diagram + table comprehension |
  | CUDA GPU, 6-10 GB VRAM | gemma-3-4b-it (4-bit) — multimodal, lighter |
  | No GPU, 16 GB+ RAM | gemma-3-4b-it (float16, CPU) — works, slow |
  | Otherwise | gemma-3-1b-it (text-only) — no diagram comprehension |

  `embeddings.py`'s CLIP model and SentenceTransformer text model are small and run
  fine on CPU regardless.

---

## 2. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional: create a `.env` file for a Hugging Face token if any model you use is
gated:

```
HF_TOKEN=hf_...
```

---

## 3. Build the index

Point `embeddings.py` at your source material:

```bash
python3 embeddings.py --pdf your_textbook.pdf
```

This creates `./index_store/`:
- `milvus.db` — two collections: `multimodal_text` (text chunks + tables, as
  readable markdown) and `multimodal_visual` (CLIP embeddings of figures/diagrams)
- `assets/`, `page_images/` — cropped figures/tables and full-page renders
- `pdf_chunks.json` — a JSON audit backup of this run
- `source_hashes.json` — incremental-build cache (re-running skips unchanged sources)

Other source types: `--faq faq.json`, `--catalog products.csv`, `--url https://...`
(repeatable — pass as many as you like).

Useful flags:
```
--no-images              skip figure/diagram extraction (faster)
--no-tables               skip table extraction (table-structure detection is the slowest stage)
--max-pdf-pages N         cap pages scanned for images/tables (useful for testing on a large PDF)
--run-id NAME             namespace collections, e.g. --run-id v2 -> multimodal_text_v2
--collection / --visual-collection   override collection names
--milvus-uri              point at a remote Milvus/Zilliz server instead of a local file
```

Run `python3 embeddings.py --help` for the full list.

---

## 4. Generate papers

```bash
python3 rag.py
```

This opens an interactive prompt:

```
Describe the paper you want (subject, topic, # of questions, duration).
Commands: exit | quit | /reload | /answerkey <paper_id> | /fullmock
```

Examples:
```
20 questions on Solid State
Full Chemistry paper on Solutions and Electrochemistry
```

`/fullmock` generates a complete 4-subject NEET mock (Physics, Chemistry, Botany,
Zoology — 35+15 questions each, 720 marks, 200 minutes), pulling from whichever
subjects' material has actually been ingested.

One-shot (non-interactive) usage:
```bash
python3 rag.py --request "20 questions on Solid State"
python3 rag.py --full-mock
```

Each generated paper produces:
- A terminal preview (questions, options, tables inline, figure citations)
- A self-contained HTML file (`index_store/papers/<paper_id>.html`) with real
  rendered `<table>` elements and embedded images — this is the actual file to
  open/print, since the terminal can't display images
- A locked answer key, revealed with `/answerkey <paper_id>` once the exam
  duration has elapsed (or the next calendar day, whichever is first)

Useful flags:
```
--no-vision                     force the text-only model even if a GPU could run the multimodal one
--no-4bit                       disable 4-bit quantization
--top-k / --min-score           retrieval tuning
--max-tokens                    output-token floor (auto-scaled up for large papers)
--max-context-images            cap how many real images are attached per request (default 4)
```

Run `python3 rag.py --help` for the full list.

---

## 5. How it works

**Ingestion (`embeddings.py`):**
1. PDF text is extracted natively, falling back to Tesseract OCR when native
   extraction quality is poor (e.g. broken/Type-3 fonts).
2. Figures and diagrams are located with `docling`'s layout model, cropped to PNG,
   captioned, and embedded with CLIP into `multimodal_visual`.
3. Tables are located with `docling`'s table-structure model. When the native
   per-cell text is unreadable (same broken-font issue as body text), each cell is
   individually re-OCR'd using the table's own (font-independent) cell geometry, and
   reassembled into clean markdown — stored as text in `multimodal_text`, since a
   table's value is its factual content, which the LLM needs to actually read.

**Generation (`rag.py`):**
1. **Retrieval**: hybrid search — dense vector search + BM25 keyword search over
   text/tables, plus CLIP cross-modal search over images — merged and ranked.
2. **Prompt**: instructs the model to write a strict NTA-NEET paper (MCQ and
   Assertion-Reason only, no short/long answer, uniform +4/-1 marking, real
   35-compulsory + 15-attempt-any-10 section structure) grounded only in what was
   retrieved.
3. **Vision**: on a multimodal model, retrieved images are attached as real
   pixels the model can look at — not just cited by filename.
4. **Validation**: every generated paper is checked against NEET's structural
   rules (option counts, marking scheme, Assertion-Reason template, answer-key
   completeness) *and* grounding rules (a cited figure must be a real file that
   was actually retrieved; a table shown in a question must genuinely match a
   retrieved table's content). Any violation triggers an automatic regenerate
   with the exact violations fed back to the model, up to a configurable retry
   limit — and if it still fails, the paper is shown anyway with a visible
   warning banner, never silently passed off as conformant.
5. **Diagrams with no source figure**: if a question needs a diagram that isn't
   in the retrieved material, the model can request one be *rendered*, not
   generated by an image model — a chemical structure via RDKit (from a SMILES
   string, checked to actually parse) or a plot via matplotlib (only of numbers
   that are literally present in a retrieved table). This is deliberate: an
   image-generation model would hallucinate plausible-looking but potentially
   wrong diagrams (wrong bond counts, invented data), which is exactly what the
   rest of this pipeline's grounding rules exist to prevent.

---

## 6. Configuration reference

Both scripts read from environment variables (or a `.env` file) as defaults,
overridable by CLI flags.

**`embeddings.py`** (`PipelineConfig`):

| Env var | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Text embedding model |
| `VISUAL_EMBED_MODEL` | `sentence-transformers/clip-ViT-B-32` | Image embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `180` / `35` | Text chunking window |
| `MIN_CHUNK_LEN` | `50` | Minimum chunk length (chars) |
| `EXTRACT_IMAGES` / `EXTRACT_TABLES` | `true` | Toggle figure/table extraction |
| `MIN_IMAGE_DIM` | `40` | Skip decorative images smaller than this (px) |
| `MILVUS_URI` / `MILVUS_COLLECTION` / `MILVUS_VISUAL_COLLECTION` | local file / `multimodal_text` / `multimodal_visual` | Storage location and collection names |
| `OUTPUT_DIR` | `./index_store` | Where everything is written |

**`rag.py`** (`RAGConfig`):

| Env var | Default | Meaning |
|---|---|---|
| `INDEX_DIR` | `./index_store` | Must match `embeddings.py`'s `OUTPUT_DIR` |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `8` / `0.25` | Retrieval tuning |
| `LLM_MAX_TOKENS` | `4096` | Output-token floor (auto-scaled up per request) |
| `QUANTIZE` | `4bit` | Set to anything else to disable 4-bit quantization |
| `ENABLE_VISION` | `true` | Set `false` to force the text-only model |
| `MAX_CONTEXT_IMAGES` | `4` | Cap on real images attached per request |
| `RAG_MAX_RETRIES` | `2` | Extra regenerate attempts on validation failure |
| `HF_TOKEN` | — | Hugging Face token, for gated models |

---

## 7. Known limitations

Being direct about what this does *not* guarantee:

- **Answer keys are format-validated, not fact-checked.** The validator confirms
  the marked answer is a well-formed option letter — it does not verify that
  letter is actually the scientifically correct answer. Spot-check before
  treating a paper as authoritative.
- **A full multi-subject mock (`/fullmock`) needs all 4 subjects' source material
  ingested.** A subject with nothing ingested will honestly show a thin/empty
  `coverage_note` rather than invent content for it.
- **Table/figure extraction quality depends on the source PDF.** Books with
  broken/Type-3 fonts are recovered via OCR, which is a real improvement over raw
  garbled text but not pixel-perfect — expect occasional row misalignment on
  dense data tables.
- **Small local models can still drift** in ways the structural validator
  doesn't catch (question difficulty calibration, subject-matter nuance) — the
  validator enforces NEET *format*, not pedagogical quality.
