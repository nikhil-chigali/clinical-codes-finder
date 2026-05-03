# RxNorm Dose-String Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Override `RxNormClient.search()` so that a dose-string query like `"lisinopril 20 mg"` — which returns zero results from the API — automatically retries with just the drug name and returns per-strength `CodeResult`s ranked with the matching dose first.

**Architecture:** Two files change — `src/clinical_codes/tools/rxnorm.py` and `tests/tools/test_rxnorm.py`. A module-level `_DOSE_RE` detects dose strings; the `search()` override strips the dose and reuses the existing `_fetch_with_retry`/`_build_params` infrastructure; `_parse_strengths` expands one `CodeResult` per CUI/strength pair from the API row, bucketing matching-dose entries to the top. All other files (`base.py`, nodes, prompts, etc.) are untouched.

**Tech Stack:** `re` (stdlib), `httpx`, existing `ClinicalTablesClient` / `_fetch_with_retry` from `tools/base.py`, `settings.fetch_results` from `clinical_codes.config`

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `tests/tools/test_rxnorm.py` | 4 new integration tests for the dose-string fallback |
| Modify | `src/clinical_codes/tools/rxnorm.py` | `_DOSE_RE`, `search()` override, `_parse_strengths` method |

---

## Task 1: Add 4 failing integration tests

**Files:**
- Modify: `tests/tools/test_rxnorm.py`

- [ ] **Step 1: Append 4 test functions to `tests/tools/test_rxnorm.py`**

Add these four functions at the end of the file, after `test_display_contains_drug_name`:

```python
async def test_dose_string_returns_results() -> None:
    async with RxNormClient() as client:
        results = await client.search("lisinopril 20 mg", count=10)
    assert len(results) > 0


async def test_dose_match_ranks_first() -> None:
    async with RxNormClient() as client:
        results = await client.search("lisinopril 20 mg", count=10)
    assert results
    assert "20" in results[0].display


async def test_fallback_display_contains_strength() -> None:
    async with RxNormClient() as client:
        results = await client.search("metformin 500 mg", count=5)
    assert results
    assert " — " in results[0].display


async def test_no_dose_no_fallback() -> None:
    async with RxNormClient() as client:
        results = await client.search("zzznomatchzzz", count=5)
    assert results == []
```

- [ ] **Step 2: Run the 4 new tests to verify 3 fail**

```bash
uv run pytest tests/tools/test_rxnorm.py::test_dose_string_returns_results tests/tools/test_rxnorm.py::test_dose_match_ranks_first tests/tools/test_rxnorm.py::test_fallback_display_contains_strength tests/tools/test_rxnorm.py::test_no_dose_no_fallback -v
```

Expected: 3 FAILED (`test_dose_string_returns_results`, `test_dose_match_ranks_first`, `test_fallback_display_contains_strength`), 1 PASSED (`test_no_dose_no_fallback` — nonsense query already returns `[]` without any override)

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/tools/test_rxnorm.py
git commit -m "test: add failing tests for rxnorm dose-string fallback"
```

---

## Task 2: Implement the dose-string fallback

**Files:**
- Modify: `src/clinical_codes/tools/rxnorm.py`

- [ ] **Step 1: Replace the entire contents of `src/clinical_codes/tools/rxnorm.py`**

```python
import re
from typing import Any

import httpx

from clinical_codes.config import settings
from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.base import ClinicalTablesClient, _fetch_with_retry

_DOSE_RE = re.compile(r'\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b', re.IGNORECASE)


class RxNormClient(ClinicalTablesClient):
    system = SystemName.RXNORM

    def _endpoint(self) -> str:
        return "rxterms/v3/search"

    def _build_params(self, query: str, count: int) -> dict[str, str]:
        return {
            "terms": query,
            "count": str(count),
            "df": "DISPLAY_NAME,RXCUIS,STRENGTHS_AND_FORMS",
        }

    async def search(self, query: str, count: int | None = None) -> list[CodeResult]:
        results = await super().search(query, count)
        if results:
            return results

        dose_match = _DOSE_RE.search(query)
        if not dose_match:
            return []

        n = count if count is not None else settings.fetch_results
        drug_name = _DOSE_RE.sub("", query).strip()
        try:
            data = await _fetch_with_retry(
                self._client, self._endpoint(), self._build_params(drug_name, n)
            )
        except httpx.HTTPError:
            return []
        return self._parse_strengths(data, n, dose_hint=dose_match.group(0).strip())

    def _parse_strengths(self, data: Any, count: int, dose_hint: str) -> list[CodeResult]:
        rows: list[list[str]] = data[3] or []
        dose_norm = dose_hint.strip().lower()

        matching: list[tuple[str, str, dict]] = []
        others: list[tuple[str, str, dict]] = []

        for row in rows:
            if len(row) < 3:
                continue
            drug = row[0]
            cuis = [c.strip() for c in row[1].split(",") if c.strip()]
            strengths = [s.strip() for s in row[2].split(",") if s.strip()]
            for cui, strength in zip(cuis, strengths):
                if not cui:
                    continue
                display = f"{drug} — {strength}"
                raw = {"code": cui, "display": display, "drug": drug, "strength": strength, "row": row}
                bucket = matching if dose_norm in strength.lower() else others
                bucket.append((cui, display, raw))

        ordered = (matching + others)[:count]
        codes = [e[0] for e in ordered]
        displays = [e[1] for e in ordered]
        raws = [e[2] for e in ordered]
        return self._make_results(codes, displays, raws)

    def _parse_response(self, data: Any, count: int) -> list[CodeResult]:
        # response[3][i] = [DISPLAY_NAME, RXCUIS_csv, STRENGTHS_AND_FORMS_csv]
        # One CodeResult per row (drug+form group). Code = first CUI in the row.
        rows: list[list[str]] = data[3] or []
        codes, displays, raws = [], [], []

        for row in rows[:count]:
            if len(row) < 2:
                continue
            drug = row[0]
            cuis = [c.strip() for c in row[1].split(",") if c.strip()]
            code = cuis[0] if cuis else ""
            if not code:
                continue
            codes.append(code)
            displays.append(drug)
            raws.append({"code": code, "display": drug, "row": row})

        return self._make_results(codes, displays, raws)
```

- [ ] **Step 2: Run the 4 new tests to verify they all pass**

```bash
uv run pytest tests/tools/test_rxnorm.py::test_dose_string_returns_results tests/tools/test_rxnorm.py::test_dose_match_ranks_first tests/tools/test_rxnorm.py::test_fallback_display_contains_strength tests/tools/test_rxnorm.py::test_no_dose_no_fallback -v
```

Expected: 4 PASSED

- [ ] **Step 3: Run the full rxnorm test file to confirm no regressions**

```bash
uv run pytest tests/tools/test_rxnorm.py -v
```

Expected: 8 PASSED (4 original + 4 new)

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest
```

Expected: all tests pass, 0 failures

- [ ] **Step 5: Commit the implementation**

```bash
git add src/clinical_codes/tools/rxnorm.py
git commit -m "feat: add dose-string fallback to RxNormClient"
```
