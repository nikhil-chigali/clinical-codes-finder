from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from clinical_codes.graph.state import Attempt, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName

SYSTEM_CATALOG: dict[SystemName, str] = {
    SystemName.ICD10CM: "Diagnosis and condition codes. Use for diseases, symptoms, injuries, and clinical conditions.",
    SystemName.LOINC: "Lab tests and clinical observations. Use for measurements, panels, and diagnostic procedures.",
    SystemName.RXNORM: "Drug names and medications. Use for drugs, dosage forms, and active ingredients.",
    SystemName.HCPCS: "Procedures, devices, and supplies billed to Medicare/Medicaid. Use for equipment, therapies, and clinical services.",
    SystemName.UCUM: "Units of measure. Use for measurement units such as mg/dL, mmol/L, or beats per minute.",
    SystemName.HPO: "Human phenotype terms. Use for genetic traits, rare disease features, and clinical phenotypes.",
}

_CATALOG_LINES = "\n".join(f"  {name}: {desc}" for name, desc in SYSTEM_CATALOG.items())

_PLANNER_SYSTEM = f"""You are a clinical coding specialist. Given a natural-language clinical query, select the most relevant medical coding systems and generate a precise search term for each.

Available systems:
{_CATALOG_LINES}

Query decomposition:
Before selecting systems, break the query into its meaningful clinical components — individual tokens or phrases that each map to a distinct clinical concept. Examples:
- "ecoli 10000" → organism name ("ecoli") + numeric quantity ("10000") → two components, two systems (LOINC + UCUM)
- "glucose in mmol/L" → lab analyte ("glucose") + unit of measure ("mmol/L") → two components, two systems (LOINC + UCUM)
- "metformin 500 mg" → single component (drug + dose is one RxNorm concept; "mg" here is part of the dosage, not a standalone unit query) → RxNorm only
- "diabetes" → single component (bare disease name) → ICD-10CM only

Apply the domain anchors below to each component. Select the union of systems needed to cover all components. A multi-token query does not automatically mean multiple systems — only add a system when a component maps to a distinct domain not covered by the first system.

Selection rules:
- Default to 1 system. Add a second when the query spans two distinct clinical domains (e.g. "diabetes medication" → ICD-10CM + RxNorm; "glucose in mmol/L" → LOINC + UCUM; "metformin dosage units" → RxNorm + UCUM). Add a third only when the query clearly involves three distinct domains.
- Domain anchors for unqualified single-domain queries:
  - Bare disease name or condition (e.g. "diabetes", "hypertension", "pneumonia") → ICD-10CM only
  - Phenotypic trait, observable clinical feature, or rare-disease characteristic (e.g. "ataxia", "brachydactyly", "photophobia") → HPO only
  - Drug name or dosage form (e.g. "metformin", "lisinopril 20 mg") → RxNorm only
  - Lab test or clinical measurement (e.g. "glucose test", "hemoglobin a1c") → LOINC only
  - Device or durable medical equipment (e.g. "wheelchair", "CPAP machine") → HCPCS only
  - Unit of measure (e.g. "mg/dL", "mmol/L") → UCUM only (a unit embedded in a drug dosage string, such as "mg" in "metformin 500 mg", is not a standalone UCUM query)
- If the query is clearly not a clinical term — random characters, keyboard mash, or non-medical questions — return an empty system selection and state this in the rationale.
- Generate exactly one search term per selected system.

On refinement:
- You will receive the prior attempt's search terms, weak systems, and the evaluator's diagnosis.
- Based on the diagnosis, you may: retry a weak system with a different search term, drop a weak system that does not cover this query type, or add a system not in the original selection if the diagnosis suggests the query spans a different domain.
- If a system returned no results, the search term was likely too specific or too long. The Clinical Tables API behaves like an autocomplete — it matches on keyword prefixes, so a concise 1–3 word phrase finds results where a full clinical description does not. Shorten the term to its core concept (e.g. "urine culture" instead of "Escherichia coli colony count urine culture") and retry.
- Systems that returned strong results do not need to be re-queried; omit them from search_terms."""


_EVALUATOR_SYSTEM = """You are a clinical code quality evaluator. Given a clinical query and the results returned for each selected coding system, decide whether the results are sufficient or require refinement.

Your standard is CLINICAL DOMAIN — filter results that belong to a fundamentally different clinical category than what the query requires. Do NOT filter within-domain variation such as different specimen types, test methods, or sub-classifications; the API's own ranking handles relevance within a system.

Cross-domain mismatch — filter these:
- Query "metformin 500 mg" → a LOINC lab panel for metformin plasma levels does NOT match; it is a lab measurement, not a drug formulation. RxNorm drug formulation codes DO match.
- Query "hypertension" → ICD-10-CM primary hypertension (I10) DOES match; "Unspecified heart disease" (I51.9) does NOT — it is a different condition, not the one the query names.

Within-domain variation — keep these, trust the API:
- Query "ecoli" against LOINC → E. coli FISH assays, blood culture assays, and urine culture assays ALL match. They are all E. coli lab tests; specimen type and method are sub-type distinctions the evaluator does not make.
- Query "glucose" against LOINC → fasting glucose, random glucose, and HbA1c panels all match. They are all glucose-related lab measurements.

Rule of thumb: if a result is about the right clinical entity (organism, drug, condition, measurement) AND is in the right system type (lab test for a lab system, drug code for a drug system), keep it. Only filter when the result clearly belongs to a different clinical category than what the query names.

Evaluation criteria:
- sufficient: every selected system returned at least one result in the correct clinical domain for the query.
- refine: any selected system returned no results, or its results are clearly from the wrong clinical domain (e.g., a drug query against LOINC returns only lab measurement codes with no drug-related results).

For each weak system, provide a plain-English diagnosis explaining why the results are from the wrong clinical domain.
Do NOT prescribe remediation — do not suggest alternative search terms or systems.
Describe what went wrong; the planner will decide how to address it.

Coverage check (in addition to result quality):
- Identify each meaningful component of the original query.
- For each component, verify it is addressed by at least one selected system and reflected in the results.
- If a component is not captured — for example, a numeric value in the query but no quantitative unit system selected, or a drug name present but no RxNorm results — flag it as a coverage gap.
- Report uncovered components as: "The [component] in the query is not represented by the selected systems."
- A coverage gap is always a "refine" decision, even when other systems returned strong results.

If decision is "sufficient", weak_systems must be empty and feedback must be an empty string.

Semantic filtering:
- Always populate relevant_codes regardless of decision: for each system, list only the codes that belong to the correct clinical domain for the query — apply the same domain-matching standard used for the sufficiency decision.
- Keep codes that represent the right clinical entity in the right system, even if the specific method, specimen, or sub-classification differs from what you might expect.
- Only exclude codes that are clearly from a different clinical category (e.g., a drug code appearing in lab results, a condition code where a measurement code is expected).
- If all results for a system are in the correct domain, include all of them.
- If a system returned results but all are from the wrong domain, include it in relevant_codes with an empty list [] — this signals the consolidator to remove all results for that system.
- Only omit a system from relevant_codes entirely if it returned no raw results at all.
- Populating relevant_codes on "refine" ensures that if the iteration cap is hit and the pipeline proceeds anyway, the best available filtered set is used rather than the full unfiltered results."""


def build_planner_messages(
    query: str, attempt_history: list[Attempt]
) -> list[BaseMessage]:
    if not attempt_history:
        human = f"Query: {query}"
    else:
        last = attempt_history[-1]
        terms_str = "\n".join(
            f'    {system}: "{term}"'
            for system, term in last.planner_output.search_terms.items()
        )
        weak_str = (
            ", ".join(str(s) for s in last.evaluator_output.weak_systems) or "none"
        )
        human = (
            f"Query: {query}\n\n"
            f"Prior attempt:\n"
            f"  Systems queried:\n{terms_str}\n"
            f"  Weak systems: {weak_str}\n"
            f"  Evaluator feedback: {last.evaluator_output.feedback}\n\n"
            f"Revise your system selection and/or search terms based on the evaluator's feedback.\n"
            f"Systems that returned strong results do not need to be re-queried."
        )
    return [SystemMessage(content=_PLANNER_SYSTEM), HumanMessage(content=human)]


def build_evaluator_messages(
    query: str,
    planner_output: PlannerOutput,
    raw_results: dict[SystemName, list[CodeResult]],
) -> list[BaseMessage]:
    # search_terms is the authoritative source for iteration — it covers every system
    # the planner selected AND has a term for, which is what the executor will query.
    result_lines: list[str] = []
    for system, term in planner_output.search_terms.items():
        results = raw_results.get(system, [])[:5]
        result_lines.append(f'  {system} (searched: "{term}"):')
        if results:
            for i, r in enumerate(results, 1):
                result_lines.append(f"    {i}. [{r.code}] {r.display}")
        else:
            result_lines.append("    (no results)")

    terms_str = ", ".join(
        f'{system}: "{term}"' for system, term in planner_output.search_terms.items()
    )
    human = (
        f"Query: {query}\n"
        f"Selected systems and search terms: {terms_str}\n\n"
        f"Results:\n" + "\n".join(result_lines)
    )
    return [SystemMessage(content=_EVALUATOR_SYSTEM), HumanMessage(content=human)]


_SUMMARIZER_SYSTEM = """You are a clinical information specialist. Write a single concise paragraph (3–5 sentences) summarizing what was found and why.

Guidelines:
- Base your summary strictly on the reasoning trace provided. Do not add clinical context from your own knowledge beyond what the trace supports.
- State what the query is about, which systems were searched, and what was found.
- If refinement occurred, briefly note what changed (e.g., a search term was revised after initial results were empty).
- Do not repeat individual codes or list results by system — those are shown separately above.
- If no results were found across any system, return a single sentence explaining this and suggesting the user rephrase using a recognized clinical term.
- If a "Cap-hit" note appears in the input, explicitly state that the search reached its refinement limit without fully satisfying the query. Name the specific gap(s) the evaluator identified. Suggest the user rephrase or narrow the query to get better results."""


def effective_search_terms(attempt_history: list[Attempt]) -> dict[SystemName, str]:
    """Return the last search term used per system across all iterations.

    On refinement the planner omits strong systems from search_terms, so the
    final planner_output.search_terms is incomplete. Walking the full history
    and letting later iterations overwrite earlier ones gives a complete map.
    """
    terms: dict[SystemName, str] = {}
    for attempt in attempt_history:
        terms.update(attempt.planner_output.search_terms)
    return terms


def _format_trace(attempt_history: list[Attempt]) -> str:
    lines: list[str] = []
    for attempt in attempt_history:
        terms_str = ", ".join(
            f'{s} ("{t}")' for s, t in attempt.planner_output.search_terms.items()
        )
        lines.append(f"  Iteration {attempt.iteration} — searched: {terms_str}")
        for system, results in attempt.raw_results.items():
            if results:
                names = ", ".join(r.display for r in results[:3])
                lines.append(f"    {system}: {len(results)} results — {names}")
            else:
                lines.append(f"    {system}: no results")
        ev = attempt.evaluator_output
        if ev.decision == "sufficient":
            lines.append("    Evaluator: sufficient")
        else:
            lines.append(f"    Evaluator: refine — {ev.feedback}")
    return "\n".join(lines)


def build_summarizer_messages(
    query: str,
    consolidated: dict[SystemName, list[CodeResult]],
    rationale: str,
    attempt_history: list[Attempt],
) -> list[BaseMessage]:
    result_lines: list[str] = []
    for system, results in consolidated.items():
        result_lines.append(f"  {system}:")
        for r in results[:5]:
            result_lines.append(f"    - {r.display} [{r.code}]")
        if not results:
            result_lines.append("    (no results)")

    cap_hit = (
        bool(attempt_history)
        and attempt_history[-1].evaluator_output.decision == "refine"
    )
    trace_block = f"Reasoning trace:\n{_format_trace(attempt_history)}\n"
    if cap_hit:
        last_feedback = attempt_history[-1].evaluator_output.feedback
        trace_block += (
            f"\nCap-hit: the refinement limit was reached. "
            f"Evaluator's final assessment: {last_feedback}\n"
        )

    human = (
        f"Query: {query}\n\n"
        + trace_block
        + f"\nFinal results:\n"
        + "\n".join(result_lines)
    )
    return [SystemMessage(content=_SUMMARIZER_SYSTEM), HumanMessage(content=human)]
