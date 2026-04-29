import operator
from typing import get_args, get_type_hints

import pytest
from pydantic import ValidationError

from clinical_codes.config import MAX_ITERATIONS
from clinical_codes.schemas import SystemName


# ── MAX_ITERATIONS ────────────────────────────────────────────────────────────

def test_max_iterations_value() -> None:
    assert MAX_ITERATIONS == 2


def test_max_iterations_type() -> None:
    assert isinstance(MAX_ITERATIONS, int)
