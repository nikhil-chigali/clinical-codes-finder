from clinical_codes.graph.state import Attempt, EvaluatorOutput, PlannerOutput
from clinical_codes.schemas import CodeResult, SystemName


def _make_result(system: SystemName, display: str, code: str = "X00") -> CodeResult:
    return CodeResult(system=system, code=code, display=display, score=1.0, raw={})


def _planner_output(
    systems: list[SystemName] | None = None,
    terms: dict[SystemName, str] | None = None,
    rationale: str = "test rationale",
) -> PlannerOutput:
    systems = systems or [SystemName.ICD10CM]
    terms = terms or {SystemName.ICD10CM: "hypertension"}
    return PlannerOutput(selected_systems=systems, search_terms=terms, rationale=rationale)


def _evaluator_output(
    decision: str = "sufficient",
    weak: list[SystemName] | None = None,
    feedback: str = "",
) -> EvaluatorOutput:
    return EvaluatorOutput(decision=decision, weak_systems=weak or [], feedback=feedback)


def _attempt(iteration: int = 0) -> Attempt:
    return Attempt(
        iteration=iteration,
        planner_output=_planner_output(),
        raw_results={SystemName.ICD10CM: [_make_result(SystemName.ICD10CM, "Essential hypertension")]},
        evaluator_output=_evaluator_output(
            decision="refine",
            weak=[SystemName.LOINC],
            feedback="LOINC returned no results for this drug query",
        ),
    )
