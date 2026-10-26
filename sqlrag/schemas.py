"""Pydantic schemas for structured LLM output.

`Explanation` and `Verdict` are re-exported from the original harness/schema.py
contract so older result files stay readable.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Explanation(BaseModel):
    """Baseline (prompt-only) generator output; identical to harness/schema.py."""

    explanation: str = Field(..., min_length=10, description="Plain-language explanation of what the SQL query returns.")


class Citation(BaseModel):
    source_id: str = Field(..., max_length=8, description="Marker of a provided source, e.g. S1.")
    supports: str = Field("", max_length=160, description="At most 15 words on what the source supports; do not copy the passage.")


class RAGExplanation(BaseModel):
    """Retrieval-augmented generator output."""

    explanation: str = Field(..., min_length=10, description="One plain-language paragraph with [S#] source markers.")
    citations: list[Citation] = Field(default_factory=list)
    evidence_sufficient: bool = Field(True, description="False when the sources do not explain a feature the query relies on.")
    missing_evidence: str = Field("", description="Which feature lacked evidence, if any.")


class Verdict(BaseModel):
    """Legacy binary judge output (harness/schema.py)."""

    grade: Literal["good", "bad"]
    reason: str = Field(..., min_length=5, description="Short reason for the grade.")


def _clamp_score(v) -> int:
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, v))


class RubricVerdict(BaseModel):
    """Schema-aware rubric judge output. `grade` keeps the legacy good/bad metric."""

    correctness: int = Field(..., description="1-5")
    completeness: int = Field(..., description="1-5")
    clarity: int = Field(..., description="1-5")
    groundedness: int = Field(..., description="1-5")
    unsupported_claims: list[str] = Field(default_factory=list)
    grade: Literal["good", "bad"]
    reason: str = Field(..., min_length=5)

    @field_validator("correctness", "completeness", "clarity", "groundedness", mode="before")
    @classmethod
    def _clamp(cls, v):
        return _clamp_score(v)

    @property
    def rubric_pass(self) -> bool:
        """The rubric's own pass rule, used to audit the grade for consistency."""
        return self.correctness >= 4 and self.completeness >= 4 and self.clarity >= 3 and self.groundedness >= 4
