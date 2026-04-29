from typing import Any

from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.base import ClinicalTablesClient


class ICD10CMClient(ClinicalTablesClient):
    system = SystemName.ICD10CM

    def _endpoint(self) -> str:
        return "icd10cm/v3/search"

    def _build_params(self, query: str, count: int) -> dict[str, str]:
        # sf=code,name required — default sf is "code" only (searches by code string, not name)
        return {"terms": query, "count": str(count), "sf": "code,name"}

    def _parse_response(self, data: Any, count: int) -> list[CodeResult]:
        codes: list[str] = data[1] or []
        rows: list[list[str]] = data[3] or []
        # response[3][i] = [code, name]
        displays = [row[1] if len(row) > 1 else row[0] for row in rows]
        raws = [{"code": c, "display": d, "row": row} for c, d, row in zip(codes, displays, rows)]
        return self._make_results(codes[:count], displays[:count], raws[:count])
