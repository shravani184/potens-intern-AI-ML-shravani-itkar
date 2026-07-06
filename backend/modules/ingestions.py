from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from backend.core.exceptions import IngestionError
from backend.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Sentence:

    text: str
    source: str
    page: int


@dataclass
class PageContent:

    source: str
    page: int
    text: str

_NLTK_READY: bool | None = None

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "no", "vol", "inc", "ltd", "co", "corp", "u.s", "u.k",
}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _ensure_nltk() -> bool:
    """Try to make NLTK's sentence tokenizer available. Returns success flag."""
    global _NLTK_READY
    if _NLTK_READY is not None:
        return _NLTK_READY
    try:
        import nltk

        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
                _NLTK_READY = True
                return True
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                    nltk.data.find(f"tokenizers/{resource}")
                    _NLTK_READY = True
                    return True
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        pass
    logger.warning("NLTK Punkt unavailable; using regex sentence splitter.")
    _NLTK_READY = False
    return False


def _regex_split(text: str) -> List[str]:
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = re.sub(
            rf"\b({re.escape(abbr)})\.", r"\1<DOT>", protected, flags=re.IGNORECASE
        )
    parts = _SENTENCE_RE.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def split_into_sentences(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    if _ensure_nltk():
        try:
            import nltk

            sentences = nltk.sent_tokenize(text)
        except Exception:  # noqa: BLE001
            sentences = _regex_split(text)
    else:
        sentences = _regex_split(text)

    cleaned: List[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 2 and re.search(r"[A-Za-z0-9]", s):
            cleaned.append(s)
    return cleaned


def _clean_text(text: str) -> str:

    if not text:
        return ""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r", "\n")

    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_pdf(path: str | Path) -> List[PageContent]:
    
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise IngestionError(f"Unsupported file type (expected .pdf): {path.name}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise IngestionError("pypdf is not installed.") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Could not open PDF '{path.name}': {exc}") from exc

    pages: List[PageContent] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to extract page %d of %s: %s", page_index, path.name, exc)
            raw = ""
        cleaned = _clean_text(raw)
        if cleaned:
            pages.append(PageContent(source=path.name, page=page_index, text=cleaned))

    if not pages:
        raise IngestionError(
            f"No extractable text found in '{path.name}'. "
            "It may be a scanned document requiring OCR."
        )
    logger.info("Loaded '%s' (%d pages with text).", path.name, len(pages))
    return pages


def extract_sentences(path: str | Path) -> List[Sentence]:

    sentences: List[Sentence] = []
    for page in load_pdf(path):
        for sent in split_into_sentences(page.text):
            sentences.append(Sentence(text=sent, source=page.source, page=page.page))
    logger.info("Extracted %d sentences from '%s'.", len(sentences), Path(path).name)
    return sentences


def list_pdf_files(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.pdf") if p.is_file())