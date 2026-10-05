from typing import Literal

from pydantic import BaseModel, Field


class Explanation(BaseModel):
    explanation: str = Field(
        ...,
        min_length=10,
        description="Plain-language explanation of what the SQL query returns.",
    )


class Verdict(BaseModel):
    """Structured, dimension-level judge verdict."""

    grade: Literal["good", "bad"]
    correctness: bool = Field(..., description="No material semantic error about what the SQL returns.")
    completeness: bool = Field(..., description="Covers every material SQL operation that is actually present.")
    hallucination_free: bool = Field(..., description="Does not invent schema, predicates, joins, ordering, grouping, or business meaning.")
    clarity: bool = Field(..., description="Clear enough for a non-expert reader to follow.")
    reason: str = Field(..., min_length=5, description="Short evidence-based reason tied to the supplied SQL.")
