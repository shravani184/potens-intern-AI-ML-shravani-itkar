from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from config import settings  # noqa: E402

SAMPLE_DOCS = [
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
    ),
    (
        "bert.pdf",
        "https://arxiv.org/pdf/1810.04805",
    ),
    (
        "gpt3_language_models_few_shot.pdf",
        "https://arxiv.org/pdf/2005.14165",
    ),
    (
        "llama2.pdf",
        "https://arxiv.org/pdf/2307.09288",
    ),
    (
        "retrieval_augmented_generation.pdf",
        "https://arxiv.org/pdf/2005.11401",
    ),
]


def download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Failed to download {dest.name}: {exc}")
        return False


def main() -> int:
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(SAMPLE_DOCS)} sample PDFs into {docs_dir} …\n")

    ok = 0
    for filename, url in SAMPLE_DOCS:
        dest = docs_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  • {filename} already present, skipping.")
            ok += 1
            continue
        print(f"  ↓ {filename} <- {url}")
        if download(url, dest):
            ok += 1

    print(f"\nDownloaded {ok}/{len(SAMPLE_DOCS)} documents.")
    print("Next: python -m scripts.ingest_documents")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
