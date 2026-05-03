# Planner Prompt Fixes (B + D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix planner over-selection on single-domain queries (Fix B) and prevent system selection for clearly non-clinical inputs (Fix D) by updating the `_PLANNER_SYSTEM` selection rules block in `prompts.py`.

**Architecture:** Single prompt string replacement in `_PLANNER_SYSTEM`. The graph, nodes, state, tools, and evaluator are untouched. Verification is a manual eval run against the gold set — not automated tests, because the fix changes LLM behavior, not deterministic code.

**Tech Stack:** Python 3.12, LangGraph, `langchain-anthropic`, `uv` (package manager)

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Modify | `src/clinical_codes/graph/prompts.py` | Replace `Selection rules:` block in `_PLANNER_SYSTEM` |

No other files change.

---

### Task 1: Replace the selection rules block in `_PLANNER_SYSTEM`

**Files:**
- Modify: `src/clinical_codes/graph/prompts.py` (lines 22–25, the `Selection rules:` block)

- [ ] **Step 1: Confirm the current text before editing**

  Open `src/clinical_codes/graph/prompts.py` and locate `_PLANNER_SYSTEM`. The `Selection rules:` block currently reads:

  ```
  Selection rules:
  - Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains.
  - Generate exactly one search term per selected system.
  - Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.
  ```

- [ ] **Step 2: Replace the selection rules block**

  In `src/clinical_codes/graph/prompts.py`, replace the entire `Selection rules:` block (4 lines) with the following 10 lines. The surrounding text (`Available systems:` catalog above and `On refinement:` section below) is **not changed**.

  Replace:
  ```python
  Selection rules:
  - Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains.
  - Generate exactly one search term per selected system.
  - Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.
  ```

  With:
  ```python
  Selection rules:
  - Default to 1 system. Add a second only when the query explicitly spans two distinct clinical domains; add a third only for genuinely complex multi-domain queries.
  - Domain anchors for unqualified single-domain queries:
    - Bare disease, condition, or symptom (e.g. "diabetes", "hypertension", "asthma") → ICD-10CM only
    - Drug name or dosage form (e.g. "metformin", "lisinopril 20 mg") → RxNorm only
    - Lab test or clinical measurement (e.g. "glucose test", "hemoglobin a1c") → LOINC only
    - Device or durable medical equipment (e.g. "wheelchair", "CPAP machine") → HCPCS only
    - Unit of measure (e.g. "mg/dL", "mmol/L") → UCUM only
  - If the query is clearly not a clinical term — random characters, keyboard mash, or non-medical questions — return an empty system selection and state this in the rationale.
  - Generate exactly one search term per selected system.
  - Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.
  ```

  After the edit, `_PLANNER_SYSTEM` should look like this in full:

  ```python
  _PLANNER_SYSTEM = f"""You are a clinical coding specialist. Given a natural-language clinical query, select the most relevant medical coding systems and generate a precise search term for each.

  Available systems:
  {_CATALOG_LINES}

  Selection rules:
  - Default to 1 system. Add a second only when the query explicitly spans two distinct clinical domains; add a third only for genuinely complex multi-domain queries.
  - Domain anchors for unqualified single-domain queries:
    - Bare disease, condition, or symptom (e.g. "diabetes", "hypertension", "asthma") → ICD-10CM only
    - Drug name or dosage form (e.g. "metformin", "lisinopril 20 mg") → RxNorm only
    - Lab test or clinical measurement (e.g. "glucose test", "hemoglobin a1c") → LOINC only
    - Device or durable medical equipment (e.g. "wheelchair", "CPAP machine") → HCPCS only
    - Unit of measure (e.g. "mg/dL", "mmol/L") → UCUM only
  - If the query is clearly not a clinical term — random characters, keyboard mash, or non-medical questions — return an empty system selection and state this in the rationale.
  - Generate exactly one search term per selected system.
  - Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.

  On refinement:
  - You will receive the prior attempt's search terms, weak systems, and the evaluator's diagnosis.
  - Based on the diagnosis, you may: retry a weak system with a different search term, drop a weak system that does not cover this query type, or add a system not in the original selection if the diagnosis suggests the query spans a different domain.
  - Systems that returned strong results do not need to be re-queried; omit them from search_terms."""
  ```

- [ ] **Step 3: Run the existing prompt tests to verify no regressions**

  ```bash
  uv run pytest tests/graph/test_prompts.py -v
  ```

  Expected: all 8 tests pass. The tests check structural properties (message count, message types, human message content) — none of them assert on the selection rules text, so the edit does not break any test.

  If any test fails, read the failure carefully. The edit is a string-only change — a failure indicates the prompt function signature or the surrounding `_PLANNER_SYSTEM` text was accidentally altered. Fix the edit (do not change the tests).

- [ ] **Step 4: Run the full test suite to confirm no regressions anywhere**

  ```bash
  uv run pytest -v
  ```

  Expected: all tests pass (102 at last count). Integration tests are excluded by default and do not need to run here.

- [ ] **Step 5: Commit**

  ```bash
  git add src/clinical_codes/graph/prompts.py
  git commit -m "feat: add conservative selection defaults and miss-query catch to planner prompt"
  ```

---

### Task 2: Run the eval and verify improvement

**Files:** None changed — read-only verification step.

- [ ] **Step 1: Run the eval against the gold set**

  ```bash
  uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
  ```

  This takes approximately 5–10 minutes (31 queries × up to 2 iterations × LLM calls). Output is written to `results/` with a timestamped directory.

- [ ] **Step 2: Check the per-query results for the target queries**

  Open the results JSON or markdown in `results/<timestamp>/`. Look up:

  | Query ID | Query text | Before (system_f1) | Expected after |
  |---|---|---|---|
  | q001 | diabetes | 0.50 | 1.0 |
  | q002 | hypertension | 0.50 | 1.0 |
  | q003 | asthma | 0.50 | 1.0 |
  | q009 | CPAP machine | 0.50 | 1.0 |
  | q029 | asdfghjkl | 0.00 | 1.0 |

  If q001–q003 and q009 still show system_f1 < 1.0, the domain anchors are not narrowing selection. Check the raw planner output in the results JSON to see which systems were selected — the planner may have interpreted the query as multi-domain.

  If q029 still shows system_f1 = 0.0, the miss-query instruction is not triggering. Check the planner rationale in the results JSON.

- [ ] **Step 3: Check overall metrics and watch for multi-system regressions**

  Look at the summary table. Expected outcomes:
  - Overall system_selection_f1: was 0.69, expected ≥ 0.75
  - Multi-system queries (q012–q019): system_f1 should not drop below 0.60 (baseline was 0.70). If it drops, the domain anchors are firing incorrectly on ambiguous queries like `"diabetes medication"`.

  If multi-system F1 regresses, compare the planner's system selection for those queries against the gold expected systems. The fix should only affect bare single-domain queries — if it's also clamping multi-domain queries to 1 system, the anchor wording needs adjustment.

- [ ] **Step 4: Update `README.md` with the new eval numbers**

  Open `README.md`. Find the eval results table (under the `## Evaluation` heading). Update the numbers to match the new run:
  - Update the run ID (e.g., `20260502_210940` → new timestamp)
  - Update system-selection F1, top-3 recall, must-include hit rate, mean iterations, mean API calls
  - Update the per-query-type slice table
  - Update the "Remaining gaps" paragraph if multi-system recall or simple query precision changed

  Also update the "What I'd do with more time" section: move "Planner conservative defaults" and "Miss-query catch" from **Planned** to **Completed improvements**.

- [ ] **Step 5: Commit the README update**

  ```bash
  git add README.md
  git commit -m "docs: update eval results after planner prompt fixes B and D"
  ```
