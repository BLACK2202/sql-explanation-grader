from pydantic import BaseModel
from typing import Literal

class Explanation(BaseModel):
    explanation: str

class Verdict(BaseModel):
    grade: Literal["good", "bad"]
    reason: str