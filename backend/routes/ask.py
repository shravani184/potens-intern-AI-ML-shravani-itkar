from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.core.exceptions import RAGError
from backend.core.logging_config import get_logger
from backend.models.schemas import AskRequest, AskResponse
from backend.modules.ask_service import answer_question

logger = get_logger(__name__)
router = APIRouter(tags=["QA"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return answer_question(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except RAGError as exc:
        logger.error("Ask failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail="Internal error.") from exc
