from typing import Literal

from pydantic import BaseModel, Field


class Explanation(BaseModel):
    explanation: str = Field(
        ...,
        min_length=10,
        description="Plain-language explanation of what the SQL query returns.",
    )


class Verdict(BaseModel):
    """Structured judge verdict for one explanation."""

    grade: Literal["good", "bad"]
    correctness: bool = Field(..., description="No incorrect claims about query semantics or result.")
    completeness: bool = Field(..., description="Covers all SQL operations that are actually present and material.")
    hallucination_free: bool = Field(..., description="Does not invent tables, columns, filters, joins, or behavior.")
    clarity: bool = Field(..., description="Clear plain-language explanation understandable by a non-expert.")
    reason: str = Field(..., min_length=5, description="Concise evidence-based reason for the verdict.")
