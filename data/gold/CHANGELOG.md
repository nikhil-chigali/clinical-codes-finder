# Gold eval set changelog

## v0.1.1 — 2026-04-29

Verified against NLM Clinical Tables API. Patch-level changes only (no schema or query changes).

### Code replacements
- `q003` ICD10CM: replaced `J45.9` with `J45.998` — original code did not appear in top-20 for query "asthma"
- `q003` ICD10CM: replaced `J45` with `J45.998` — original code did not appear in top-20 for query "asthma"
- `q004` LOINC: replaced `2345-7` with `62856-0` — original code did not appear in top-20 for query "glucose test"
- `q004` LOINC: replaced `2339-0` with `62856-0` — original code did not appear in top-20 for query "glucose test"
- `q008` HCPCS: replaced `E1130` with `E2210` — original code did not appear in top-20 for query "wheelchair"
- `q014` LOINC: replaced `94500-6` with `98732-1` — original code did not appear in top-20 for query "COVID-19 test"
- `q014` LOINC: replaced `94531-1` with `98732-1` — original code did not appear in top-20 for query "COVID-19 test"
- `q017` LOINC: replaced `8480-6` with `34860-7` — original code did not appear in top-20 for query "hypertension management"
- `q017` LOINC: replaced `8462-4` with `34860-7` — original code did not appear in top-20 for query "hypertension management"
- `q018` LOINC: replaced `4548-4` with `38944-5` — original code did not appear in top-20 for query "diabetes management"
- `q018` LOINC: replaced `2345-7` with `38944-5` — original code did not appear in top-20 for query "diabetes management"
- `q018` HCPCS: replaced `A4253` with `G0108` — original code did not appear in top-20 for query "diabetes management"
- `q018` HCPCS: replaced `E0607` with `G0108` — original code did not appear in top-20 for query "diabetes management"
- `q019` LOINC: replaced `57698-3` with `94872-9` — original code did not appear in top-20 for query "cholesterol panel"
- `q019` LOINC: replaced `2093-3` with `94872-9` — original code did not appear in top-20 for query "cholesterol panel"
- `q020` LOINC: replaced `2345-7` with `13534-3` — original code did not appear in top-20 for query "blood sugar test"
- `q024` RXNORM: replaced `6809` with `847230` — original code did not appear in top-20 for query "insulin"
- `q024` RXNORM: replaced `5856` with `847230` — original code did not appear in top-20 for query "insulin"
- `q028` LOINC: replaced `2345-7` with `76133-8` — original code did not appear in top-20 for query "sugar test"

### Lower-than-expected rankings (kept, but flagged)
- `q001` ICD10CM: `E11.9` ranks #11 for "diabetes" (kept, but outside top-10)
- `q025` ICD10CM: `H91.90` ranks #12 for "hearing loss" (kept, but outside top-10)

### Demoted from must_include
- `q001` ICD10CM: `E11.9` moved out of must_include — does not appear in top-10 for "diabetes" (rank 11)
- `q006` RXNORM: `860975` moved out of must_include — does not appear in top-10 for "metformin 500 mg" (not in top-20)
- `q009` HCPCS: `E0601` moved out of must_include — does not appear in top-10 for "CPAP machine" (not in top-20)

### Removed from must_not_include
- _(none)_

### Unverified / unverifiable
- `q006` RXNORM: `860975` retained: API returned 0 results for query "metformin 500 mg" — verification not possible (query string likely needs reformulation by the planner)
- `q007` RXNORM: `314077` retained: API returned 0 results for query "lisinopril 20 mg" — verification not possible (query string likely needs reformulation by the planner)
- `q009` HCPCS: `E0601` retained: API returned 0 results for query "CPAP machine" — verification not possible (query string likely needs reformulation by the planner)
- `q012` ICD10CM: `E11.9` retained: API returned 0 results for query "diabetes medication" — verification not possible (query string likely needs reformulation by the planner)
- `q012` RXNORM: `860975` retained: API returned 0 results for query "diabetes medication" — verification not possible (query string likely needs reformulation by the planner)
- `q012` RXNORM: `6809` retained: API returned 0 results for query "diabetes medication" — verification not possible (query string likely needs reformulation by the planner)
- `q013` ICD10CM: `A15.9` retained: API returned 0 results for query "tuberculosis treatment" — verification not possible (query string likely needs reformulation by the planner)
- `q013` RXNORM: `7407` retained: API returned 0 results for query "tuberculosis treatment" — verification not possible (query string likely needs reformulation by the planner)
- `q014` ICD10CM: `U07.1` retained: API returned 0 results for query "COVID-19 test" — verification not possible (query string likely needs reformulation by the planner)
- `q014` ICD10CM: `Z11.52` retained: API returned 0 results for query "COVID-19 test" — verification not possible (query string likely needs reformulation by the planner)
- `q015` ICD10CM: `J45.9` retained: API returned 0 results for query "asthma inhaler" — verification not possible (query string likely needs reformulation by the planner)
- `q015` RXNORM: `745679` retained: API returned 0 results for query "asthma inhaler" — verification not possible (query string likely needs reformulation by the planner)
- `q016` ICD10CM: `D64.9` retained: API returned 0 results for query "anemia workup" — verification not possible (query string likely needs reformulation by the planner)
- `q016` LOINC: `718-7` retained: API returned 0 results for query "anemia workup" — verification not possible (query string likely needs reformulation by the planner)
- `q016` LOINC: `789-8` retained: API returned 0 results for query "anemia workup" — verification not possible (query string likely needs reformulation by the planner)
- `q017` ICD10CM: `I10` retained: API returned 0 results for query "hypertension management" — verification not possible (query string likely needs reformulation by the planner)
- `q017` RXNORM: `314077` retained: API returned 0 results for query "hypertension management" — verification not possible (query string likely needs reformulation by the planner)
- `q018` ICD10CM: `E11.9` retained: API returned 0 results for query "diabetes management" — verification not possible (query string likely needs reformulation by the planner)
- `q018` RXNORM: `860975` retained: API returned 0 results for query "diabetes management" — verification not possible (query string likely needs reformulation by the planner)
- `q018` RXNORM: `6809` retained: API returned 0 results for query "diabetes management" — verification not possible (query string likely needs reformulation by the planner)
- `q021` ICD10CM: `I10` retained: API returned 0 results for query "high blood pressure" — verification not possible (query string likely needs reformulation by the planner)

### Summary
- 19 codes replaced
- 2 codes flagged as ranked-low (kept)
- 3 must_include demotions
- 0 must_not_include removals
- 21 (query, system) pairs unverifiable (API returned no results)
- Total (query, system) pairs verified: 43
