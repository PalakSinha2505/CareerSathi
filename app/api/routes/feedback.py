from fastapi import APIRouter
from pydantic import BaseModel

from ..services.feedback_service import generate_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    analysis: dict
    feedback_mode: str = "harsh"


@router.post("/generate")
def generate_feedback_route(data: FeedbackRequest):
    return generate_feedback(
        question=data.question,
        answer=data.answer,
        analysis=data.analysis,
        feedback_mode=data.feedback_mode
    )
