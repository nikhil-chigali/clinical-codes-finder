# RxNorm Dose-String Fallback — Design

## Problem

The RxTerms v3 API (`/rxterms/v3/search`) does a prefix match on `DISPLAY_NAME`. Drug display names look like `"lisinopril (Oral Pill)"` — they never start with a dose string. Queries like `"lisinopril 20 mg"` or `"metformin 500 mg"` return zero results, which is the root cause of the multi-system top-3 recall failure on those gold queries.

The API is designed as a two-step UI flow: search by drug name → pick a strength from a dropdown. The second step's data (`RXCUIS`, `STRENGTHS_AND_FORMS`) is already present in the first response — it just isn't used by the current `_parse_response`.

## Goal

When a RxNorm search returns 0 results and the query contains a dose string, automatically retry with just the drug name and return per-strength `CodeResult`s ranked so the matching dose is first. Transparent to all callers.

---

## Architecture

**Two files change. Nothing else.**

| Action | Path |
|---|---|
| Modify | `src/clinical_codes/tools/rxnorm.py` |
| Modify | `tests/tools/test_rxnorm.py` |

`base.py`, the executor, prompts, and all other tools are untouched.

---

## Call flow

```
search("lisinopril 20 mg")
  │
  ├─ super().search("lisinopril 20 mg")      ← existing path, returns []
  │
  ├─ empty?       yes
  ├─ dose found?  yes  (_DOSE_RE matches "20 mg")
  │
  ├─ drug_name = "lisinopril"
  ├─ _fetch_with_retry(self._client, endpoint, params("lisinopril", count))
  └─ _parse_strengths(data, count, dose_hint="20 mg")
       → per-strength CodeResults, "20 MG" entries ranked first

search("lisinopril")
  └─ super().search("lisinopril")            ← non-empty, return as-is (unchanged)

search("zzznomatch")
  ├─ super().search("zzznomatch")            ← returns []
  ├─ empty?       yes
  ├─ dose found?  no
  └─ return []                               (unchanged)
```

`_fetch_with_retry` is imported from `base.py` alongside `ClinicalTablesClient` — intra-package import, no new dependency.

---

## Dose detection

Module-level compiled regex:

```python
_DOSE_RE = re.compile(r'\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b', re.IGNORECASE)
```

Matches: `"20 mg"`, `"500mg"`, `"1.5 mcg"`, `"10 ml"`, `"5 units"`. Does not match bare numbers.

**Detection:** `_DOSE_RE.search(query)` — match object or `None`.

**Drug name extraction:** `_DOSE_RE.sub("", query).strip()`

```
"lisinopril 20 mg"  →  "lisinopril"
"metformin 500 mg"  →  "metformin"
"insulin 5 units"   →  "insulin"
```

**Dose hint** passed to `_parse_strengths`: `match.group(0).strip().lower()` — used as a substring to match against `STRENGTHS_AND_FORMS` entries.

---

## `search()` override

```python
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
```

---

## `_parse_strengths`

Expands each API row into one `CodeResult` per strength+form entry. Rows where `row[2]` (STRENGTHS_AND_FORMS) is absent or empty fall back to the existing per-row shape.

```python
def _parse_strengths(self, data: Any, count: int, dose_hint: str) -> list[CodeResult]:
    rows: list[list[str]] = data[3] or []
    dose_norm = dose_hint.strip().lower()

    matching: list[tuple[str, str, dict]] = []
    others:   list[tuple[str, str, dict]] = []

    for row in rows:
        if len(row) < 3:
            continue
        drug      = row[0]
        cuis      = [c.strip() for c in row[1].split(",") if c.strip()]
        strengths = [s.strip() for s in row[2].split(",") if s.strip()]
        for cui, strength in zip(cuis, strengths):
            if not cui:
                continue
            display = f"{drug} — {strength}"
            raw = {"code": cui, "display": display, "drug": drug, "strength": strength, "row": row}
            bucket = matching if dose_norm in strength.lower() else others
            bucket.append((cui, display, raw))

    ordered  = (matching + others)[:count]
    codes    = [e[0] for e in ordered]
    displays = [e[1] for e in ordered]
    raws     = [e[2] for e in ordered]
    return self._make_results(codes, displays, raws)
```

**Display format:** `"lisinopril (Oral Pill) — 20 MG Tablet"` — drug group name + specific strength. Downstream consumers (evaluator display names, summarizer, Streamlit UI) get a fully qualified string.

**Scoring:** `_make_results` assigns scores by rank (rank 0 → 1.0, descending). Matching strengths fill top ranks; non-matching follow. Matching is a plain `dose_norm in strength.lower()` substring check — sufficient for `"20 mg"` → `"20 MG Tablet"`.

---

## Tests

Four new tests in `tests/tools/test_rxnorm.py`, following the existing real-API pattern:

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

The existing four tests cover the normal path (plain drug-name queries) and are unaffected.

---

## Required imports in `rxnorm.py`

The override adds three new imports to `rxnorm.py`:

```python
import re
import httpx
from clinical_codes.config import settings
from clinical_codes.tools.base import ClinicalTablesClient, _fetch_with_retry
```

(`re` and `httpx` are stdlib/already-installed; `settings` is used for `settings.fetch_results`; `_fetch_with_retry` is the intra-package import.)

---

## What does NOT change

- `ClinicalTablesClient` base class — no new methods, no interface changes
- `executor` node — calls `client.search(term)` uniformly, no awareness of fallback
- `prompts.py` / `SYSTEM_CATALOG` — no prompt changes needed
- All other tool clients (`ICD10CMClient`, `LOINCClient`, etc.)
