# Evaluation Metrics Design

## Goal

Compute system-selection F1, top-3 code recall, must-include hit rate, mean iterations, and mean API calls from a list of `RunResult` objects paired with their `GoldQuery` gold records. Return a typed `MetricsSummary` that reporter.py can format without doing any computation.

---

## File map

| File | Action | Purpose |
|---|---|---|
| `src/clinical_codes/evaluation/metrics.py` | Create | Data models + individual metric functions + `compute_metrics()` aggregator |
| `tests/evaluation/test_metrics.py` | Create | Unit tests for all functions and `compute_metrics()` |

Data models live in `metrics.py` (not `schema.py`) because they are outputs of metrics computation, not runner I/O. `reporter.py` imports `MetricsSummary` and `compute_metrics` from `metrics`.

---

## Data models (`metrics.py`)

```python
from pydantic import BaseModel
from clinical_codes.schemas import SystemName

class QueryMetrics(BaseModel):
    query_id: str
    query: str
    query_type: str
    system_precision: float
    system_recall: float
    system_f1: float
    recall_at_3: float | None          # None for miss queries (expected_codes == {})
    must_include_hit_rate: float | None  # None when must_include == []
    iterations: int
    api_calls: int
    latency_s: float
    error: str | None

class QueryTypeMetrics(BaseModel):
    query_type: str
    n: int
    system_selection_f1: float
    top3_recall: float | None          # None for miss type (no expected codes)
    must_include_hit_rate: float | None  # None if no query in this type has must_include
    mean_iterations: float
    mean_api_calls: float

class MetricsSummary(BaseModel):
    n_total: int
    n_errors: int
    system_selection_f1: float         # macro-averaged over all queries
    top3_recall: float                 # mean over non-miss queries
    must_include_hit_rate: float       # mean over queries with must_include != []
    mean_iterations: float
    mean_api_calls: float
    by_type: dict[str, QueryTypeMetrics]
    per_query: list[QueryMetrics]
```

`None` fields propagate cleanly — reporter.py renders them as `n/a` in tables without special-casing.

**`MetricsSummary` aggregate convention:** `top3_recall` and `must_include_hit_rate` are `float` (non-optional) at the overall level, while the same fields on `QueryTypeMetrics` are `float | None`. `compute_metrics` substitutes `0.0` for `None` using an explicit `if x is not None else 0.0` check. Reporter should skip the overall top-3 recall row (or show `n/a`) when all queries are miss-type — derivable from `by_type` keys.

---

## Individual metric functions (`metrics.py`)

### `_system_f1`

```python
def _system_f1(
    predicted: list[SystemName],
    expected: list[SystemName],
) -> tuple[float, float, float]:  # (precision, recall, f1)
```

Set-intersection math over `SystemName` values.

**Edge cases:**
- `predicted=[], expected=[]` → `(1.0, 1.0, 1.0)` — correct miss; penalizing here would distort the miss-query F1
- `predicted != [], expected=[]` → `(0.0, 1.0, 0.0)` — hallucinated systems on a non-clinical query, F1=0
- `predicted=[], expected != []` → `(1.0, 0.0, 0.0)` — missed all required systems, F1=0

```python
tp = len(set(predicted) & set(expected))
precision = tp / len(predicted) if predicted else 1.0  # no false positives if nothing predicted
recall    = tp / len(expected)  if expected  else 1.0  # nothing required = perfect recall
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
return precision, recall, f1
```

When `predicted=[], expected=[]`: precision=1.0, recall=1.0, F1=1.0 (correct miss). When `predicted≠[], expected=[]`: tp=0, precision=0.0, recall=1.0, F1=0.0 (hallucinated systems).

### `_recall_at_k`

```python
def _recall_at_k(
    predicted_codes: dict[SystemName, list[str]],
    expected_codes: dict[SystemName, list[str]],
    k: int = 3,
) -> float | None:
```

Pools all `(system, code)` pairs from `expected_codes`. For each, checks whether `code in predicted_codes.get(system, [])[:k]`. Returns `None` if `expected_codes` is empty (miss query — no recall to compute).

**Example:** `expected_codes = {ICD10CM: ["E11.9"], RXNORM: ["860975", "6809"]}`, `predicted_codes = {ICD10CM: ["E11.9"], RXNORM: ["6809", "X"]}` → 2/3 = 0.67 (E11.9 hit, 860975 miss, 6809 hit).

### `_must_include_hit_rate`

```python
def _must_include_hit_rate(
    predicted_codes: dict[SystemName, list[str]],
    must_include: list[str],
) -> float | None:
```

Checks each code in `must_include` against **all** predicted codes across all systems (any position, not just top 3 — these are critical codes the system must surface somewhere). Returns `None` if `must_include=[]`.

```python
if not must_include:
    return None
all_predicted = {code for codes in predicted_codes.values() for code in codes}
hits = sum(1 for code in must_include if code in all_predicted)
return hits / len(must_include)
```

### `compute_metrics`

```python
def compute_metrics(
    results: list[RunResult],
    gold: list[GoldQuery],
) -> MetricsSummary:
```

Joins on `result.query_id == gold_query.id` using a dict keyed by id (O(1) per lookup). Calls the three helper functions per query to build `QueryMetrics` objects, then aggregates into `MetricsSummary`.

**Aggregation rules:**
- `system_selection_f1` — macro average: mean of per-query F1 over all queries
- `top3_recall` — mean of per-query `recall_at_3` over queries where it is not `None`
- `must_include_hit_rate` — mean of per-query `must_include_hit_rate` over queries where it is not `None`
- `mean_iterations`, `mean_api_calls` — simple mean over all queries
- `n_errors` — count of results where `result.error is not None`
- `by_type` — group `QueryMetrics` by `query_type`, apply same aggregation rules per group

**Error query handling:** Error results have `predicted_systems=[]` and `predicted_codes={}` (set by runner.py). These naturally score F1=0 for any query with non-empty `expected_systems` — no special-casing needed.

---

## Testing (`tests/evaluation/test_metrics.py`)

All three helper functions are pure — no mocks required. `compute_metrics` needs only lightweight `RunResult` + `GoldQuery` fixture data.

### Tests for `_system_f1`

```python
def test_system_f1_perfect_match():
    p, r, f1 = _system_f1([SystemName.ICD10CM], [SystemName.ICD10CM])
    assert p == 1.0 and r == 1.0 and f1 == 1.0

def test_system_f1_miss_query_both_empty():
    p, r, f1 = _system_f1([], [])
    assert f1 == 1.0

def test_system_f1_hallucinated_systems():
    p, r, f1 = _system_f1([SystemName.ICD10CM], [])
    assert f1 == 0.0

def test_system_f1_partial_overlap():
    p, r, f1 = _system_f1(
        [SystemName.ICD10CM, SystemName.LOINC],
        [SystemName.ICD10CM, SystemName.RXNORM],
    )
    assert p == 0.5 and r == 0.5
    assert abs(f1 - 0.5) < 1e-9
```

### Tests for `_recall_at_k`

```python
def test_recall_at_k_hit_in_top3():
    result = _recall_at_k({SystemName.ICD10CM: ["E11.9", "E10.9"]}, {SystemName.ICD10CM: ["E11.9"]})
    assert result == 1.0

def test_recall_at_k_miss_beyond_k():
    result = _recall_at_k(
        {SystemName.ICD10CM: ["E10.9", "E11.65", "J45.9", "E11.9"]},  # E11.9 at position 4
        {SystemName.ICD10CM: ["E11.9"]},
        k=3,
    )
    assert result == 0.0

def test_recall_at_k_miss_query_returns_none():
    result = _recall_at_k({}, {})
    assert result is None

def test_recall_at_k_multi_system_partial():
    result = _recall_at_k(
        {SystemName.ICD10CM: ["E11.9"], SystemName.RXNORM: ["6809"]},
        {SystemName.ICD10CM: ["E11.9"], SystemName.RXNORM: ["860975", "6809"]},
        k=3,
    )
    assert abs(result - 2/3) < 1e-9
```

### Tests for `_must_include_hit_rate`

```python
def test_must_include_hit():
    result = _must_include_hit_rate({SystemName.ICD10CM: ["E11.9", "E10.9"]}, ["E11.9"])
    assert result == 1.0

def test_must_include_miss():
    result = _must_include_hit_rate({SystemName.ICD10CM: ["E10.9"]}, ["E11.9"])
    assert result == 0.0

def test_must_include_empty_returns_none():
    result = _must_include_hit_rate({SystemName.ICD10CM: ["E11.9"]}, [])
    assert result is None

def test_must_include_partial():
    result = _must_include_hit_rate(
        {SystemName.ICD10CM: ["E11.9"]},
        ["E11.9", "I10"],
    )
    assert result == 0.5
```

### Tests for `compute_metrics`

**Test 1 — happy path with one error:**

Build one `GoldQuery` (simple, ICD10CM, must_include=["E11.9"]) and one errored `RunResult` (error="timeout", predicted_systems=[], predicted_codes={}). Call `compute_metrics`. Assert:
- `summary.n_total == 2`
- `summary.n_errors == 1`
- `summary.system_selection_f1 < 1.0` (error query drags it down)
- `"simple" in summary.by_type`
- `summary.per_query` has 2 entries

**Test 2 — miss query excluded from top3_recall:**

Build one miss `GoldQuery` (expected_systems=[], expected_codes={}) with a correct `RunResult` (predicted_systems=[], predicted_codes={}). Call `compute_metrics`. Assert:
- `summary.system_selection_f1 == 1.0` (correct miss)
- `summary.per_query[0].recall_at_3 is None`
- `summary.by_type["miss"].top3_recall is None`

**Test 3 — by_type keys match input:**

Build results with query types "simple" and "multi_system". Assert `set(summary.by_type.keys()) == {"simple", "multi_system"}`.

---

## Edge case summary

| Situation | F1 | recall_at_3 | must_include_hit_rate |
|---|---|---|---|
| Correct miss (pred=[], exp=[]) | 1.0 | None | None |
| Hallucinated systems (pred≠[], exp=[]) | 0.0 | None | None |
| Error query (runner.py sets pred=[]) | 0.0 (if exp≠[]) | 0.0 (if exp_codes≠{}) | 0.0 (if must_include≠[]) |
| must_include=[] | any | any | None |
| expected_codes={} (miss) | any | None | None |

---

## Out of scope

- Writing results to disk (reporter.py's job)
- Confidence-threshold calibration (already handled in the evaluator node)
- Latency percentiles or histograms (mean only)
