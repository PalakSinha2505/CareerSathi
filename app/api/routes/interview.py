from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from slowapi.util import get_remote_address
from slowapi import Limiter
from pydantic import BaseModel
from typing import List, Dict

from app.db.database import get_db
from app.db.models import Interview, User, QuestionAnswer
from app.auth.dependencies import get_current_user
from app.api.schemas import (
    InterviewRequest,
    InterviewResponse,
    AnswerResponse,
    InterviewHistoryResponse,
)
from app.api.services.analyzer_service import analyze_answer
from app.api.services.feedback_service import generate_feedback
from app.api.services.scoring_service import calculate_overall_score
from app.api.services.interview_service import generate_question


router = APIRouter(prefix="/interview", tags=["interview"])

# ---------------------------------
# Rate Limiter
# ---------------------------------
limiter = Limiter(key_func=get_remote_address)


# =====================================================
# 1️⃣  Generate Next AI Question (ADAPTIVE)
# =====================================================

class NextQuestionRequest(BaseModel):
    role: str
    experience_level: str
    history: List[Dict] = []


@router.post("/next-question")
@limiter.limit("10/minute")
def get_next_question(
    request: Request,
    data: NextQuestionRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        question = generate_question(
            role=data.role,
            experience_level=data.experience_level,
            history=data.history,
        )

        return {"question": question}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate interview question",
        )


# =====================================================
# 2️⃣  Evaluate Full Interview
# =====================================================

@router.post("/evaluate", response_model=InterviewResponse)
@limiter.limit("5/minute")
def evaluate_interview(
    request: Request,
    data: InterviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not data.responses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No responses provided",
        )

    analysis_results = []

    role = data.responses[0].role
    level = data.responses[0].experience_level

    interview = Interview(
        role=role,
        level=level,
        score=0,
        user_id=current_user.id,
    )

    db.add(interview)
    db.flush()

    # Save each Q&A
    for item in data.responses:
        analysis = analyze_answer(
            question=item.question,
            answer=item.answer,
            role=item.role,
            experience_level=item.experience_level,
        )

        feedback = generate_feedback(
            question=item.question,
            answer=item.answer,
            analysis=analysis,
            feedback_mode=item.feedback_mode,
        )

        analysis_results.append(analysis)

        qa = QuestionAnswer(
            interview_id=interview.id,
            question=item.question,
            answer=item.answer,
            analysis=analysis,
            feedback=feedback,
        )

        db.add(qa)

    overall_score = calculate_overall_score(analysis_results)

    interview.score = overall_score

    db.commit()
    db.refresh(interview)

    responses = [
        AnswerResponse(
            question=qa.question,
            answer=qa.answer,
            analysis=qa.analysis,
            feedback=qa.feedback,
        )
        for qa in interview.answers
    ]

    return InterviewResponse(
        id=interview.id,
        role=interview.role,
        level=interview.level,
        score=interview.score,
        created_at=str(interview.created_at),
        responses=responses,
    )


# =====================================================
# 3️⃣  Interview History
# =====================================================

@router.get("/history", response_model=InterviewHistoryResponse)
def get_interview_history(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    interviews = (
        db.query(Interview)
        .filter(Interview.user_id == current_user.id)
        .order_by(Interview.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []

    for interview in interviews:
        result.append(
            InterviewResponse(
                id=interview.id,
                role=interview.role,
                level=interview.level,
                score=interview.score,
                created_at=str(interview.created_at),
                responses=[
                    AnswerResponse(
                        question=qa.question,
                        answer=qa.answer,
                        analysis=qa.analysis,
                        feedback=qa.feedback,
                    )
                    for qa in interview.answers
                ],
            )
        )

    return InterviewHistoryResponse(total=len(interviews), interviews=result)


# =====================================================
# 4️⃣  Analytics
# =====================================================

@router.get("/analytics")
def get_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    interviews = (
        db.query(Interview)
        .filter(Interview.user_id == current_user.id)
        .order_by(Interview.created_at.desc())
        .all()
    )

    if not interviews:
        return {
            "total_interviews": 0,
            "average_score": 0,
            "highest_score": 0,
            "lowest_score": 0,
            "roles_breakdown": {},
            "questions_breakdown": {},
        }

    scores = [i.score["overall_score"] for i in interviews if i.score]

    average_score = sum(scores) / len(scores) if scores else 0
    highest_score = max(scores) if scores else 0
    lowest_score = min(scores) if scores else 0

    # Role breakdown
    roles_breakdown = {}

    for interview in interviews:
        if interview.role not in roles_breakdown:
            roles_breakdown[interview.role] = {
                "attempts": 0,
                "total_score": 0,
                "average_score": 0,
            }

        if interview.score:
            roles_breakdown[interview.role]["attempts"] += 1
            roles_breakdown[interview.role]["total_score"] += interview.score["overall_score"]

    for role, stats in roles_breakdown.items():
        if stats["attempts"] > 0:
            stats["average_score"] = round(
                stats["total_score"] / stats["attempts"], 2
            )
        else:
            stats["average_score"] = 0

        del stats["total_score"]

    # Question breakdown
    questions_breakdown = {}

    for interview in interviews:
        for qa in interview.answers:
            if qa.question not in questions_breakdown:
                questions_breakdown[qa.question] = {
                    "attempts": 0,
                    "feedback_samples": [],
                }

            questions_breakdown[qa.question]["attempts"] += 1

            if qa.feedback:
                questions_breakdown[qa.question]["feedback_samples"].append(
                    qa.feedback
                )
                questions_breakdown[qa.question]["feedback_samples"] = (
                    questions_breakdown[qa.question]["feedback_samples"][-3:]
                )

    return {
        "total_interviews": len(interviews),
        "average_score": round(average_score, 2),
        "highest_score": highest_score,
        "lowest_score": lowest_score,
        "roles_breakdown": roles_breakdown,
        "questions_breakdown": questions_breakdown,
    }
