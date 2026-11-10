"""Regulatory knowledge base: extract sections from IRDAI/Ombudsman PDFs, embed, retrieve.

Phase 1: extraction + index building + retrieval primitives.
Phase 2: wiring into match.py / verify.py / app.py.

Embeddings: one section per call, one shared SDK client, checkpoint every 5 sections.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
REG_DIR = ROOT / "data" / "regulations"
INDEX_PATH = REG_DIR / "index.npz"
PARTIAL_PATH = REG_DIR / "index.partial.npz"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")

# ---------------------------------------------------------
# REPLACE THIS URL WITH YOUR GITHUB RELEASE URL
# ---------------------------------------------------------
INDEX_URL = "https://github.com/Harsh-KSh/ClaimCourt/releases/download/v1.0.0/index.npz"

MIN_SECTION_CHARS = 80
MAX_SECTION_CHARS = 3000

EMBED_SLEEP_S = 1.2
SAVE_EVERY = 5
MAX_429_RETRIES = 5
RETRY_SLEEP_S = 60.0
CLIENT_RESET_SLEEP_S = 3.0

_shared_client: genai.Client | None = None


@dataclass(frozen=True)
class Section:
    source_name: str
    section_id: str
    page: int
    text: str


# ---------- SDK client ----------

def _client() -> genai.Client:
    global _shared_client
    if _shared_client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing in .env")
        _shared_client = genai.Client(api_key=key)
    return _shared_client


def _reset_client() -> None:
    global _shared_client
    _shared_client = None


# ---------- extraction ----------

_RE_OMB_RULE = re.compile(r"^\s*(\d+[A-Z]?)\.\s+(.+)$")
_RE_MC_CLAUSE = re.compile(r"^\s*(\d+\.\d+)\.?\s+(.+)$")


def _pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text)
            out.append((i, text))
    return out


def _split_by_regex(
    pages: list[tuple[int, str]],
    pattern: re.Pattern,
    prefix: str,
) -> list[Section]:
    sections: list[Section] = []
    cur_id: str | None = None
    cur_page = 0
    buf: list[str] = []

    def flush():
        nonlocal cur_id, cur_page, buf
        if cur_id and buf:
            text = " ".join(buf).strip()
            if len(text) >= MIN_SECTION_CHARS:
                for chunk in _chunk_text(text):
                    sections.append(Section("", f"{prefix}{cur_id}", cur_page, chunk))
        buf = []

    for page_no, text in pages:
        for line in text.splitlines():
            m = pattern.match(line)
            if m:
                flush()
                cur_id = m.group(1)
                cur_page = page_no
                buf = [line[m.end(1):].strip() or ""]
            elif cur_id is not None:
                buf.append(line)
    flush()
    return sections


def _chunk_text(text: str) -> list[str]:
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    chunks, cur = [], ""
    for sentence in re.split(r"(?<=[.!?।])\s+", text):
        if len(cur) + len(sentence) > MAX_SECTION_CHARS and cur:
            chunks.append(cur.strip())
            cur = sentence
        else:
            cur += " " + sentence
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def extract_sections(pdf_path: Path, source_name: str) -> list[Section]:
    pages = _pdf_pages(pdf_path)
    name = pdf_path.name.lower()
    if "ombudsman" in name:
        secs = _split_by_regex(pages, _RE_OMB_RULE, "Rule ")
    elif "health insurance business" in name:
        secs = _split_by_regex(pages, re.compile(r"^\s*(\d+)\)\s+(.+)$"), "Para ")
    else:
        secs = _split_by_regex(pages, _RE_MC_CLAUSE, "Clause ")
    return [Section(source_name=source_name, section_id=s.section_id, page=s.page, text=s.text) for s in secs]


# ---------- embedding ----------

def _embed_one(section: Section) -> np.ndarray:
    content = (
        f"task: search result | title: {section.source_name} §{section.section_id} | "
        f"text: {section.text}"
    )
    resp = _client().models.embed_content(model=EMBED_MODEL, contents=[content])
    if not resp.embeddings:
        raise RuntimeError(
            f"API returned no embeddings for {section.source_name} § {section.section_id}"
        )
    values = getattr(resp.embeddings[0], "values", None)
    if values is None:
        raise RuntimeError(
            f"API returned empty values for {section.source_name} § {section.section_id}"
        )
    vec = np.asarray(values, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        raise RuntimeError(
            f"API returned zero-length vector for {section.source_name} § {section.section_id}"
        )
    return vec


def _embed_with_retry(section: Section) -> np.ndarray:
    for attempt in range(1, MAX_429_RETRIES + 1):
        try:
            return _embed_one(section)
        except ClientError as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                print(
                    f"  [429] {section.section_id}: sleeping {RETRY_SLEEP_S:.0f}s "
                    f"(retry {attempt}/{MAX_429_RETRIES})"
                )
                time.sleep(RETRY_SLEEP_S)
                continue
            raise
        except RuntimeError as exc:
            msg = str(exc).lower()
            if any(w in msg for w in ("closed", "connection", "timeout", "transport")):
                print(
                    f"  [client reset] {section.section_id}: {exc}; "
                    f"sleeping {CLIENT_RESET_SLEEP_S:.0f}s (retry {attempt}/{MAX_429_RETRIES})"
                )
                _reset_client()
                time.sleep(CLIENT_RESET_SLEEP_S)
                continue
            raise
    raise RuntimeError(
        f"Embedding failed after {MAX_429_RETRIES} retries for "
        f"{section.source_name} § {section.section_id}"
    )


def _probe_client() -> None:
    print("[probe] testing embedding client...")
    try:
        probe = Section(source_name="probe", section_id="probe", page=0, text="health insurance")
        vec = _embed_one(probe)
        print(f"[probe] OK (vector dim {vec.shape[0]})")
    except Exception as exc:
        print(f"[probe] FAILED: {type(exc).__name__}: {exc}")
        print("  If this is 'client has been closed' or similar, try in the venv:")
        print("    pip install --upgrade --force-reinstall google-genai httpx")
        print("  Then re-run. If it persists, the SDK version is incompatible with this Python.")
        raise


# ---------- checkpoint helpers ----------

def _file_hashes(pdfs: list[Path]) -> np.ndarray:
    return np.array([hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs])


def _load_partial(expected_hashes: np.ndarray) -> tuple[list[np.ndarray], list[Section]]:
    if not PARTIAL_PATH.exists():
        return [], []
    try:
        data = np.load(PARTIAL_PATH, allow_pickle=False)
        if not np.array_equal(data["source_hashes"], expected_hashes):
            print("[partial] source PDFs changed; discarding old checkpoint")
            PARTIAL_PATH.unlink(missing_ok=True)
            return [], []
        vecs_arr = data["vectors"]
        secs_arr = data["sections"]
        if vecs_arr.ndim != 2 or secs_arr.ndim != 1 or vecs_arr.shape[0] != secs_arr.shape[0]:
            print(
                f"[partial] inconsistent checkpoint "
                f"(vectors {getattr(vecs_arr, 'shape', '?')}, "
                f"sections {getattr(secs_arr, 'shape', '?')}); discarding"
            )
            PARTIAL_PATH.unlink(missing_ok=True)
            return [], []
        vecs = [vecs_arr[i].astype(np.float32, copy=False) for i in range(vecs_arr.shape[0])]
        secs = [Section(**json.loads(str(s))) for s in secs_arr]
        return vecs, secs
    except Exception as exc:
        print(f"[partial] could not read checkpoint ({exc}); discarding")
        PARTIAL_PATH.unlink(missing_ok=True)
        return [], []


def _save_partial(vecs: list[np.ndarray], secs: list[Section], hashes: np.ndarray) -> None:
    if not vecs:
        return
    np.savez_compressed(
        PARTIAL_PATH,
        vectors=np.vstack(vecs),
        sections=np.array([json.dumps(asdict(s)) for s in secs]),
        source_hashes=hashes,
    )


# ---------- build ----------

def build_index() -> None:
    if not REG_DIR.exists():
        raise RuntimeError(f"Missing folder: {REG_DIR}")

    pdfs = sorted(REG_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs in {REG_DIR}")

    hashes = _file_hashes(pdfs)

    all_sections: list[Section] = []
    for pdf_path in pdfs:
        print(f"[extract] {pdf_path.name}")
        source_name = pdf_path.stem.replace("_", " ")
        secs = extract_sections(pdf_path, source_name)
        print(f"  -> {len(secs)} sections")
        all_sections.extend(secs)

    if not all_sections:
        raise RuntimeError("No sections extracted. Check filenames match the plan.")

    _probe_client()

    done_vecs, done_secs = _load_partial(hashes)
    start_at = len(done_secs)
    if done_secs:
        print(f"[resume] {start_at}/{len(all_sections)} sections already embedded")
    remaining = all_sections[start_at:]

    print(f"[total] {len(all_sections)} sections; {len(remaining)} remaining")

    vecs: list[np.ndarray] = list(done_vecs)
    secs_done: list[Section] = list(done_secs)

    for i, section in enumerate(remaining, start=1):
        vec = _embed_with_retry(section)
        vecs.append(vec)
        secs_done.append(section)
        if len(secs_done) % SAVE_EVERY == 0 or len(secs_done) == len(all_sections):
            _save_partial(vecs, secs_done, hashes)
        print(f"  [{start_at + i}/{len(all_sections)}] {section.source_name} § {section.section_id} (p{section.page})")
        if i < len(remaining):
            time.sleep(EMBED_SLEEP_S)

    if len(vecs) != len(all_sections):
        raise RuntimeError(
            f"Internal error: {len(vecs)} vectors for {len(all_sections)} sections. "
            f"Per-section embedding should never miss. Check the log for skipped sections."
        )

    vectors = np.vstack(vecs)
    if vectors.shape[0] != len(all_sections):
        raise RuntimeError(
            f"Internal error after stacking: got {vectors.shape[0]} rows for {len(all_sections)} sections."
        )

    np.savez_compressed(
        INDEX_PATH,
        vectors=vectors,
        sections=np.array([json.dumps(asdict(s)) for s in all_sections]),
        source_hashes=hashes,
    )
    PARTIAL_PATH.unlink(missing_ok=True)
    print(f"[done] wrote {INDEX_PATH} ({vectors.shape[0]} vectors, {vectors.shape[1]} dim)")


# ---------- retrieval ----------

@lru_cache(maxsize=1)
def ensure_index() -> None:
    """Downloads the index from GitHub Releases if it does not exist locally.
    Cached so it only ever runs once per Python process."""
    if INDEX_PATH.exists():
        return
    if not INDEX_URL or "YOUR_USERNAME" in INDEX_URL:
        raise RuntimeError(
            "Please replace INDEX_URL in pipeline/reg_rules.py with your GitHub Release URL."
        )
    print(f"[download] fetching index from {INDEX_URL}")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(INDEX_URL, INDEX_PATH)
        print("[download] success")
    except Exception as exc:
        raise RuntimeError(f"Failed to download index from {INDEX_URL}: {exc}")


def load_index(path: Path | None = None) -> tuple[np.ndarray, list[Section]]:
    if path is None:
        ensure_index()
        path = INDEX_PATH
    if not path.exists():
        raise RuntimeError(f"Index not found: {path}. Run `python -m pipeline.reg_rules` first.")
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"]
    sections = [Section(**json.loads(str(s))) for s in data["sections"]]
    return vectors, sections


def retrieve(query: str, top_k: int = 3) -> list[Section]:
    if not query.strip():
        return []
    vectors, sections = load_index()
    resp = _client().models.embed_content(
        model=EMBED_MODEL,
        contents=[f"task: search result | query: {query}"],
    )
    q = np.asarray(resp.embeddings[0].values, dtype=np.float32).reshape(-1)
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(sims)[-top_k:][::-1]
    return [sections[i] for i in order]


if __name__ == "__main__":
    build_index()