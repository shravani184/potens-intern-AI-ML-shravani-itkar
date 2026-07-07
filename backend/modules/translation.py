from __future__ import annotations

from dataclasses import dataclass

from backend.core.logging_config import get_logger

logger = get_logger(__name__)

# Make langdetect deterministic across runs.
try:  # pragma: no cover - import guard
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except Exception:  # noqa: BLE001
    _LANGDETECT_OK = False


@dataclass
class Translation:

    text: str
    source_lang: str
    target_lang: str


def detect_language(text: str) -> str:
    text = (text or "").strip()
    if not text or not _LANGDETECT_OK:
        return "en"
    try:
        return detect(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language detection failed (%s); defaulting to 'en'.", exc)
        return "en"


def _translate(text: str, source: str, target: str) -> str:
    if not text.strip() or source == target:
        return text
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Translation %s->%s failed (%s); returning original text.",
            source,
            target,
            exc,
        )
        return text


def to_english(text: str, source_lang: str | None = None) -> Translation:
    source = source_lang or detect_language(text)
    if source == "en":
        return Translation(text=text, source_lang="en", target_lang="en")
    translated = _translate(text, source="auto", target="en")
    return Translation(text=translated, source_lang=source, target_lang="en")


def from_english(text: str, target_lang: str) -> str:
    if not target_lang or target_lang == "en":
        return text
    return _translate(text, source="en", target=target_lang)
