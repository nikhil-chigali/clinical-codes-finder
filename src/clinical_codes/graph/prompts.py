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


def build_planner_messages(query: str, attempt_history: list[Attempt]) -> list[BaseMessage]:
    raise NotImplementedError


def build_evaluator_messages(
    query: str,
    planner_output: PlannerOutput,
    raw_results: dict[SystemName, list[CodeResult]],
) -> list[BaseMessage]:
    raise NotImplementedError


def build_summarizer_messages(
    query: str,
    consolidated: dict[SystemName, list[CodeResult]],
    rationale: str,
) -> list[BaseMessage]:
    raise NotImplementedError
