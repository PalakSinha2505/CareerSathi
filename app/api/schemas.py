from pydantic import BaseModel, Field
from typing import List, Optional, Annotated

# ----------------------------
# Request Schemas
# ----------------------------

class AnswerInput(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=300)]
    answer: Annotated[str, Field(min_length=1, max_length=500)]
    role: Annotated[str, Field(min_length=1, max_length=50)]
    experience_level: Annotated[str, Field(min_length=1, max_length=50)]
    feedback_mode: Annotated[str, Field(max_length=20)] = "harsh"


class InterviewRequest(BaseModel):
    responses: Annotated[List[AnswerInput], Field(min_items=1, max_items=20)]


# ----------------------------
# Response Schemas
# ----------------------------

class AnswerResponse(BaseModel):
    question: str
    answer: str
    analysis: Optional[str] = None
    feedback: Optional[str] = None


class InterviewResponse(BaseModel):
    id: int
    role: str
    level: str
    score: Optional[int] = None
    created_at: Optional[str] = None
    responses: List[AnswerResponse] = Field(default_factory=list)


class InterviewHistoryResponse(BaseModel):
    total: int
    interviews: List[InterviewResponse] = Field(default_factory=list)
