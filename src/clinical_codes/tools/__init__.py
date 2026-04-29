from clinical_codes.schemas import SystemName
from clinical_codes.tools.base import ClinicalTablesClient
from clinical_codes.tools.hcpcs import HCPCSClient
from clinical_codes.tools.hpo import HPOClient
from clinical_codes.tools.icd10cm import ICD10CMClient
from clinical_codes.tools.loinc import LOINCClient
from clinical_codes.tools.rxnorm import RxNormClient
from clinical_codes.tools.ucum import UCUMClient

CLIENTS: dict[SystemName, type[ClinicalTablesClient]] = {
    SystemName.ICD10CM: ICD10CMClient,
    SystemName.LOINC:   LOINCClient,
    SystemName.RXNORM:  RxNormClient,
    SystemName.HCPCS:   HCPCSClient,
    SystemName.UCUM:    UCUMClient,
    SystemName.HPO:     HPOClient,
}

__all__ = [
    "ClinicalTablesClient",
    "ICD10CMClient",
    "LOINCClient",
    "RxNormClient",
    "HCPCSClient",
    "UCUMClient",
    "HPOClient",
    "CLIENTS",
]
