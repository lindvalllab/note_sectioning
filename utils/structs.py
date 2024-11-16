from pydantic import BaseModel
from typing import Optional

class IntervalHistoryOutput(BaseModel):
    start_string: Optional[str] = None
    end_string: Optional[str] = None

class AssessmentAndPlan(BaseModel):
    start_string: Optional[str] = None
    end_string: Optional[str] = None