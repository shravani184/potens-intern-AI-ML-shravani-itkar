from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.core.exceptions import RAGError
from backend.core.logging_config import get_logger
from backend.models.schemas import ContradictRequest, ContradictResponse
from backend.modules.contradiction import check_contradiction

logger = get_logger(__name__)
router = APIRouter(tags=["Contradiction"])


@router.post("/contradict", response_model=ContradictResponse)
def contradict(request: ContradictRequest) -> ContradictResponse:
    try:
        return check_contradiction(
            document_1=request.document_1,
            document_2=request.document_2,
            topic=request.topic,
        )
    except RAGError as exc:
        logger.error("Contradict failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /contradict")
        raise HTTPException(status_code=500, detail="Internal error.") from exc
