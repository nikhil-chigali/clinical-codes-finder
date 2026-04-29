import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import CodeResult, SystemName


class PlannerOutput(BaseModel):
    selected_systems: list[SystemName]
    search_terms: dict[SystemName, str]
    rationale: str
