# Gold evaluation set

Hand-curated queries used to measure the agent's end-to-end behavior. Versioned files in this directory are the contract that eval results are reproduced against — never overwrite a published version, always bump.

## Verification status

`gold_v0.1.1.json` is the current API-verified file (verified against the NLM Clinical Tables API on **2026-04-29**). It supersedes `gold_v0.1.0.json` for use by the runner. See [`CHANGELOG.md`](CHANGELOG.md) for the full per-code diff.

Two patterns surfaced during verification that are worth knowing before reading eval numbers:

- **19 codes were replaced** because the original (knowledge-derived) code did not appear in the API's top-20 for the gold query. Most were LOINC codes for multi-word queries like "cholesterol panel" or "diabetes management" — the API ranks system-level codes higher than the specific test panels we'd expected.
- **21 (query, system) pairs returned zero API results** for the verbatim gold query (e.g. "metformin 500 mg" → no RxNorm hits because RxTerms doesn't index dose strings; "asthma inhaler" → no ICD-10 hits because the API doesn't tokenize that as a clinical phrase). These codes were retained unchanged — verifying them requires the planner to re-formulate the query first, which is exactly the planner's job. Once the planner is built, these queries will validate the planner's reformulation behavior end-to-end.

The verification harness lives at [`scripts/verify_gold_codes.py`](../../scripts/verify_gold_codes.py) and is re-runnable against any version of the gold file.

## Files

| File | Purpose |
|---|---|
| `gold_v0.1.0.json` | Original gold set, knowledge-derived codes, **not API-verified**. Kept for reproducibility of any prior eval runs. |
| `gold_v0.1.1.json` | Current API-verified gold set (31 queries). This is what the runner should load. |
| `CHANGELOG.md` | Per-version diff of code changes. |
| `README.md` | This file. |

The Pydantic schema lives in `src/clinical_codes/evaluation/schema.py`. Each query has the following fields:

- `id` — stable identifier (`q001`, `q002`, …)
- `query` — the natural-language input the user would type
- `query_type` — one of `simple`, `multi_system`, `ambiguous`, `refinement`, `miss`
- `expected_systems` — coding systems the planner *should* select. Empty list means the agent should select nothing (a "miss")
- `expected_codes` — `{system: [code, …]}` of codes that should appear in top-k for that system
- `must_include` — strict assertion: these codes MUST appear in the final consolidated results. Use sparingly, only for cases where there's exactly one right answer
- `must_not_include` — strict negative assertion: these codes/systems MUST NOT appear. Catches false positives
- `notes` — why this query is in the set; what specific behavior it tests

## Composition (v0.1.0)

| Query type | Count | What it tests |
|---|---|---|
| `simple` | 11 | Per-system happy paths and harder single-system queries. Validates basic plumbing for each of the 6 systems. |
| `multi_system` | 8 | Queries that legitimately span 2-4 systems. Includes one 3-system and one 4-system query to test breadth scaling. |
| `ambiguous` | 8 | Confusion-pair territory where the planner has to make a judgment call. This is where the eval set earns its keep. |
| `refinement` | 1 | Vague query designed to fail on first pass and force the loop to recover. Validates the refinement path actually fires. |
| `miss` | 3 | Out-of-scope and gibberish queries. Agent should select no systems and return empty. |
| **Total** | **31** | |

### Per-system coverage

How many queries have each system in `expected_systems`:

| System | Count | Notes |
|---|---|---|
| ICD10CM | 16 | Heavy because ICD-10 legitimately appears in most multi-system and ambiguous queries (conditions are the most common clinical concept). |
| LOINC | 9 | Lab tests, panels, vital signs. |
| RXNORM | 8 | Medications, with and without explicit strengths. |
| HPO | 6 | Phenotypic traits — six queries because HPO/ICD-10 confusion is the most common ambiguous case. |
| HCPCS | 3 | Wheelchair, CPAP machine, plus inclusion in the 4-system "diabetes management" query. Borderline-thin coverage; bump in v0.2.0 if the planner shows HCPCS-specific failure modes. |
| UCUM | 1 | Closed-vocabulary system. The planner should rarely select UCUM (units almost never appear without a host concept). One query is enough to test the behavior — bumping to 3+ would be redundant. |

## Curation methodology

The set is sized to surface the specific risks the locked architecture introduces, not just to hit a query count. Three constraints in particular drove design choices:

**Soft cap of 1-3 systems per planner call.** Without at least one query that legitimately wants 4 systems, the soft cap can't be evaluated — you can't tell whether the planner is correctly hitting the limit or whether the prompt is just suppressing breadth. `q018` ("diabetes management") is the 4-system test (condition + monitoring + treatment + supplies). If the planner consistently truncates to 3 here, the soft-cap prompt language is too strict and should be loosened.

**One query per selected system.** The architecture trades first-pass recall for simplicity, with the refinement loop as the recovery path. This means the eval set must include queries where the first-pass term is likely to be too vague — otherwise the refinement loop is never exercised and you can't measure whether it works. `q028` ("sugar test") is deliberately vague: the planner's first guess will likely use "sugar" as the LOINC search term, get weak results, and the loop should retry with "glucose". If `q028` resolves in 1 iteration, the evaluator isn't triggering refinement when it should.

**Per-type metrics, not just averages.** The constraint "mean iterations ≥ 1.8 in the ambiguous slice = problem" requires enough ambiguous queries for the average to be meaningful. With 8 ambiguous queries, one outlier won't dominate the metric. With 3, it would.

### Confusion pairs

The 8 ambiguous queries deliberately cluster around the system-pair confusions most likely to surface planner errors:

- **HPO ↔ ICD-10** (`q022` tremor, `q023` fever, `q026` muscle weakness, `q027` shortness of breath) — symptoms and findings overlap conceptually. Both systems are often defensible; the planner's behavior here is more of a calibration question than a correctness one. The eval slice will reveal whether the planner is consistent across these queries.
- **LOINC ↔ ICD-10** (`q020` blood sugar test) — measurement vs. condition. The word "test" should anchor LOINC.
- **HPO ↔ ICD-10 with conventional answer** (`q021` high blood pressure) — even though HPO has a phenotype for it, ICD-10 is the conventional answer. Tests whether the planner over-routes to HPO.
- **RxNorm with adjacent HCPCS** (`q024` insulin) — the substance is RxNorm, but related delivery devices live in HCPCS. The planner shouldn't fan out to HCPCS unless the query is about equipment.

### Pre-emptive false-positive guards

Two simple queries (`q006` metformin 500 mg, `q007` lisinopril 20 mg) include explicit "no UCUM" guidance in their notes. Numeric-with-unit strings can wrongly trigger UCUM if the planner is over-eager. These act as canaries.

## Before you trust the eval results

**Verify the codes against the live API.** The `expected_codes` values are plausible defaults written from prior knowledge, not validated against Clinical Tables. Run a one-off verification script during Phase 1 that queries each system for each `expected_codes` entry and flags any that don't appear in the API's top-10 results. HPO IDs and some RxNorm CUIs are the most likely to be wrong. This is a 30-minute task that prevents misleading eval numbers downstream.

```bash
# Suggested verification harness location
uv run python -m scripts.verify_gold_codes --gold data/gold/gold_v0.1.0.json
```

If a code doesn't appear, either the code is wrong (correct it) or the API ranks it lower than expected (note the actual top-10 and decide whether to relax the expectation or keep it as a known-hard case).

## Calibrate `must_include` after the first eval run

The `must_include` and `must_not_include` fields are deliberately sparse in v0.1.0 — populated only where there's exactly one right answer (e.g., I10 for hypertension, 4548-4 for HbA1c). Strict assertions added pre-emptively become noise: they fail in cases where the agent's behavior is actually fine, just different from your guess.

The pattern: run the full eval, look at the failures, and convert observed failure modes into strict assertions. If the planner wrongly selects UCUM on `q006`, add `must_not_include: ["UCUM"]` (or the equivalent in your metric code). If a specific wrong code keeps appearing in HPO results, add it to `must_not_include`. The eval set should grow teeth based on what actually breaks, not what you imagine might break.

## Versioning policy

- **Patch bumps** (`0.1.0` → `0.1.1`) — fix wrong codes, typo corrections, expand `must_include` based on calibration. Eval results from prior patch versions remain comparable.
- **Minor bumps** (`0.1.0` → `0.2.0`) — add new queries, expand per-system coverage, add new query types. Eval results not directly comparable across minor versions; rerun the previous version's set to compare.
- **Major bumps** (`0.1.0` → `1.0.0`) — schema change (new fields, renamed fields, structural redesign). Old runners may not parse new files.

Never edit a published JSON file in place. Always create a new versioned file and update the runner's default to point at it.

## Future expansion (v0.2.0+)

Track ideas for the next bump here so they're not lost:

- Bump HCPCS to 5-6 queries if planner shows specific HCPCS failure modes
- Add 2-3 queries that test the refinement loop's *failure* mode (queries that don't converge in 2 iterations) — important for honest reporting of the cap's cost
- Add inter-rater agreement on the 8 ambiguous queries (one collaborator, blind labeling) before claiming any specific accuracy number
- Expand to ~50 total once the planner's behavior is well-characterized
- Consider a small "adversarial" slice: queries that look clinical but have no good answer ("treatment for typing speed", "code for the color blue")
