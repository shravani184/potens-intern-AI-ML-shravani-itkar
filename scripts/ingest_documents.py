from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.logging_config import configure_logging, get_logger  # noqa: E402
from backend.modules.pipeline import ingest_directory, ingest_file  # noqa: E402
from backend.modules.vectorstore import get_vector_store  # noqa: E402
from config import settings  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG index.")
    parser.add_argument("files", nargs="*", help="Specific PDF files to ingest.")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the collection before ingesting."
    )
    args = parser.parse_args()

    configure_logging()

    if args.reset:
        logger.info("Resetting collection '%s'.", settings.collection_name)
        get_vector_store().reset()

    total = 0
    if args.files:
        for f in args.files:
            try:
                result = ingest_file(f, replace=True)
                total += result.chunks
                print(f"  ✓ {result.source}: {result.chunks} chunks")
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {f}: {exc}")
    else:
        print(f"Ingesting all PDFs from: {settings.documents_dir}")
        try:
            results = ingest_directory(settings.documents_dir, replace=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}")
            return 1
        for r in results:
            total += r.chunks
            print(f"  ✓ {r.source}: {r.chunks} chunks")

    print(f"\nDone. Total chunks indexed: {total}")
    print(f"Collection now holds {get_vector_store().count()} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
