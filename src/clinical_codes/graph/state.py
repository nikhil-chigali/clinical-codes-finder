import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import CodeResult, SystemName
