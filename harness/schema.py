from pydantic import BaseModel, Field
from typing import Literal


class Explanation(BaseModel):
    explanation: str = Field(..., min_length=10, description="Plain-language explanation of what the SQL query returns.")


class Verdict(BaseModel):
    grade: Literal["good", "bad"]
    reason: str = Field(..., min_length=5, description="Short reason for the grade.")