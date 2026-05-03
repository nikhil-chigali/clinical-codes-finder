# Planner Prompt Fixes (B + D) — Design

## Problem

Two failure modes identified from eval run `20260502_210940`:

**Fix B — Over-selection (precision ~0.33 on simple queries)**
Queries like `"diabetes"`, `"hypertension"`, `"asthma"`, and `"CPAP machine"` cause the planner to select 3 systems when only 1 is expected. The current rule — *"Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains"* — is too vague in practice. The LLM defaults to a broad selection rather than committing to 1.

**Fix D — Miss query failure (q029 `"asdfghjkl"` scores system_f1=0)**
The planner selects a coding system for keyboard mash, consuming 2 full iterations before the graph terminates. The model already correctly handles prose non-clinical queries (`"weather forecast"`, `"how do I make pasta"`) but does not recognize random character strings as non-clinical.

## Goal

Reduce planner over-selection on single-domain queries and eliminate system selection for clearly non-clinical inputs. Both fixes are prompt-only — no graph, node, or tool changes.

---

## Architecture

**One file changes. Nothing else.**

| Action | Path |
|---|---|
| Modify | `src/clinical_codes/graph/prompts.py` |

`nodes.py`, `builder.py`, `state.py`, all tools, and all other files are untouched.

---

## The Change

Replace the `Selection rules` block in `_PLANNER_SYSTEM` (currently lines 23–26 of `prompts.py`).

**Before:**
```
Selection rules:
- Select 1-3 systems. Select more only when the query genuinely spans multiple clinical domains.
- Generate exactly one search term per selected system.
- Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.
```

**After:**
```
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

### Design decisions

**HPO excluded from domain anchors.** The HPO vs. ICD-10CM distinction (e.g., `"ataxia"` as a rare-disease phenotype vs. a billable condition) is too subtle for a fixed rule. A domain anchor for HPO would risk under-selection for ambiguous phenotype queries. HPO selection is left to LLM judgment as before.

**No graph changes for empty selection.** When the planner returns `selected_systems=[]`, the executor runs `asyncio.gather(*[])` immediately, raw_results stay empty, and the evaluator/summarizer handle empty results gracefully. This is already proven correct by `q030` and `q031` in the eval. No bypass or new routing edge is needed.

**No evaluator changes.** The evaluator's "sufficient if every selected system returned relevant results" criterion is vacuously true when no systems are selected (zero systems = zero weak systems). This already routes correctly to consolidation.

---

## Verification

No automated tests are added. Verification is a manual eval run after the change:

```bash
uv run python -m scripts.run_eval --gold data/gold/gold_v0.1.1.json
```

**Success signals:**

| Query | Before | Expected after |
|---|---|---|
| `q001` diabetes (system_f1) | 0.50 | 1.0 |
| `q002` hypertension (system_f1) | 0.50 | 1.0 |
| `q003` asthma (system_f1) | 0.50 | 1.0 |
| `q009` CPAP machine (system_f1) | 0.50 | 1.0 |
| `q029` asdfghjkl (system_f1) | 0.00 | 1.0 |
| Overall system_selection_f1 | 0.69 | ≥ 0.75 (estimated) |

**Regression watch:** Multi-system queries (`q012`–`q019`) currently score system_f1=0.67–0.86. The domain anchors apply only to *unqualified* single-domain queries — `"diabetes medication"` explicitly spans two domains and should still route to ICD-10CM + RxNorm. Verify these do not regress after the change.

---

## What Does NOT Change

- `_EVALUATOR_SYSTEM` — no modifications
- `_SUMMARIZER_SYSTEM` — no modifications
- `build_planner_messages`, `build_evaluator_messages`, `build_summarizer_messages` — no modifications
- All graph nodes, edges, and state — untouched
- All tool clients — untouched
- Gold set — not modified or versioned
