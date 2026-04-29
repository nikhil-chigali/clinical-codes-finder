from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class SystemName(StrEnum):
    ICD10CM = "ICD10CM"
    LOINC = "LOINC"
    RXNORM = "RXNORM"
    HCPCS = "HCPCS"
    UCUM = "UCUM"
    HPO = "HPO"


class CodeResult(BaseModel):
    system: SystemName
    code: str
    display: str
    score: float  # rank-derived confidence on [0.0, 1.0]; 1.0 = top result
    raw: dict[str, Any]
