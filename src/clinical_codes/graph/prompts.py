from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from clinical_codes.graph.state import Attempt, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName

SYSTEM_CATALOG: dict[SystemName, str] = {
    SystemName.ICD10CM: "Diagnosis and condition codes. Use for diseases, symptoms, injuries, and clinical conditions.",
    SystemName.LOINC:   "Lab tests and clinical observations. Use for measurements, panels, and diagnostic procedures.",
    SystemName.RXNORM:  "Drug names and medications. Use for drugs, dosage forms, and active ingredients.",
    SystemName.HCPCS:   "Procedures, devices, and supplies billed to Medicare/Medicaid. Use for equipment, therapies, and clinical services.",
    SystemName.UCUM:    "Units of measure. Use for measurement units such as mg/dL, mmol/L, or beats per minute.",
    SystemName.HPO:     "Human phenotype terms. Use for genetic traits, rare disease features, and clinical phenotypes.",
}

_CATALOG_LINES = "\n".join(
    f"  {name}: {desc}" for name, desc in SYSTEM_CATALOG.items()
)

_PLANNER_SYSTEM = f"""You are a clinical coding specialist. Given a natural-language clinical query, select the most relevant medical coding systems and generate a precise search term for each.

Available systems:
{_CATALOG_LINES}

Selection rules:
- Select 1–3 systems. Select more only when the query genuinely spans multiple clinical domains.
- Generate exactly one search term per selected system.
- Use standardized clinical vocabulary — prefer terms the NLM Clinical Tables API recognizes over colloquial or abbreviated forms.

On refinement:
- You will receive the prior attempt's search terms, weak systems, and the evaluator's diagnosis.
- Based on the diagnosis, you may: retry a weak system with a different search term, drop a weak system that does not cover this query type, or add a system not in the original selection if the diagnosis suggests the query spans a different domain.
- Systems that returned strong results do not need to be re-queried; omit them from search_terms."""


_EVALUATOR_SYSTEM = """You are a clinical code quality evaluator. Given a clinical query and the results returned for each selected coding system, decide whether the results are sufficient or require refinement.

Evaluation criteria:
- sufficient: every selected system returned at least one result that appears semantically relevant to the query.
- refine: any selected system returned no results, or its results do not appear relevant to the query (e.g., a drug query against LOINC returns imaging codes).

For each weak system, provide a plain-English diagnosis explaining why the results are weak.
Do NOT prescribe remediation — do not suggest alternative search terms or systems.
Describe what went wrong; the planner will decide how to address it.

If decision is "sufficient", weak_systems must be empty and feedback must be an empty string."""


def build_planner_messages(query: str, attempt_history: list[Attempt]) -> list[BaseMessage]:
    if not attempt_history:
        human = f"Query: {query}"
    else:
        last = attempt_history[-1]
        terms_str = "\n".join(
            f"    {system}: \"{term}\""
            for system, term in last.planner_output.search_terms.items()
        )
        weak_str = ", ".join(str(s) for s in last.evaluator_output.weak_systems) or "none"
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
        result_lines.append(f"  {system} (searched: \"{term}\"):")
        if results:
            for i, r in enumerate(results, 1):
                result_lines.append(f"    {i}. {r.display}")
        else:
            result_lines.append("    (no results)")

    terms_str = ", ".join(
        f"{system}: \"{term}\"" for system, term in planner_output.search_terms.items()
    )
    human = (
        f"Query: {query}\n"
        f"Selected systems and search terms: {terms_str}\n\n"
        f"Results:\n" + "\n".join(result_lines)
    )
    return [SystemMessage(content=_EVALUATOR_SYSTEM), HumanMessage(content=human)]


def build_summarizer_messages(
    query: str,
    consolidated: dict[SystemName, list[CodeResult]],
    rationale: str,
) -> list[BaseMessage]:
    raise NotImplementedError
