from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.api.services.analyzer_service import analyze_answer
from app.api.services.feedback_service import generate_feedback
from app.api.services.scoring_service import calculate_overall_score

router = APIRouter()


class AnswerInput(BaseModel):
    question: str
    answer: str
    role: str
    experience_level: str
    feedback_mode: str = "harsh"


class InterviewRequest(BaseModel):
    responses: List[AnswerInput]


@router.post("/evaluate")
def evaluate_interview(data: InterviewRequest):
    analysis_results = []
    detailed_feedback = []

    for item in data.responses:
        analysis = analyze_answer(
            question=item.question,
            answer=item.answer,
            role=item.role,
            experience_level=item.experience_level
        )

        feedback = generate_feedback(
            question=item.question,
            answer=item.answer,
            analysis=analysis,
            feedback_mode=item.feedback_mode
        )

        analysis_results.append(analysis)

        detailed_feedback.append({
            "question": item.question,
            "analysis": analysis,
            "feedback": feedback
        })

    overall_score = calculate_overall_score(analysis_results)

    return {
        "overall": overall_score,
        "responses": detailed_feedback
    }
