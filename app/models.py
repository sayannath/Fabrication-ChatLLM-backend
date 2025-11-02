from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str


class RetrievedSource(BaseModel):
    paper: str
    snippet: str
    score: float
    metadata: Optional[Dict[str, str]] = None


class QAResponse(BaseModel):
    answer: str
    contexts: List[str] = Field(default_factory=list)
    sources: List[RetrievedSource] = Field(default_factory=list)
