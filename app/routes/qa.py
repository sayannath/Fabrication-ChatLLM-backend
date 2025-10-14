import structlog
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.modules import ChainOfThought
from app.models import QARequest, QAResponse, RetrievedSource

logger = structlog.get_logger()
router = APIRouter()
cot = ChainOfThought()


@router.post("/qa", response_model=QAResponse)
async def generate_answer(payload: QARequest) -> QAResponse:
    try:
        prediction = cot(question=payload.question)
        logger.info("qa.success", question=payload.question[:200])
    except Exception as exc:
        logger.error("qa.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not hasattr(prediction, "answer") or prediction.answer is None:
        logger.error("qa.missing_answer")
        raise HTTPException(status_code=500, detail="Model did not return an answer.")

    contexts = [str(value) for value in getattr(prediction, "contexts", [])]
    raw_sources = getattr(prediction, "sources", [])
    sources: list[RetrievedSource] = []
    for item in raw_sources:
        if isinstance(item, RetrievedSource):
            sources.append(item)
            continue
        if isinstance(item, dict):
            try:
                sources.append(RetrievedSource(**item))
            except ValidationError:
                logger.warning("qa.source_validation_failed", source=item)
        else:
            logger.warning("qa.source_unexpected_type", type=str(type(item)))

    return QAResponse(answer=prediction.answer, contexts=contexts, sources=sources)
