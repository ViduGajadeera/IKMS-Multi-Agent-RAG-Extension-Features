from pydantic import BaseModel
from typing import Optional, Dict


class QuestionRequest(BaseModel):
    """Request body for the `/qa` endpoint.

    The PRD specifies a single field named `question` that contains
    the user's natural language question about the vector databases paper.
    """

    question: str


class QAResponse(BaseModel):
    """Response body for the `/qa` endpoint."""

    answer: str
    context: Optional[str] = None
    citations: Optional[Dict[str, dict]] = None
