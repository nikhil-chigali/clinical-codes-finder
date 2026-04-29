from typing import Any

from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.base import ClinicalTablesClient


class HPOClient(ClinicalTablesClient):
    system = SystemName.HPO

    def _endpoint(self) -> str:
        return "hpo/v3/search"

    def _build_params(self, query: str, count: int) -> dict[str, str]:
        return {"terms": query, "count": str(count)}

    def _parse_response(self, data: Any, count: int) -> list[CodeResult]:
        codes: list[str] = data[1] or []
        rows: list[list[str]] = data[3] or []
        # response[3][i] = [hp_id, label]
        displays = [row[1] if len(row) > 1 else row[0] for row in rows]
        raws = [{"code": c, "display": d, "row": row} for c, d, row in zip(codes, displays, rows)]
        return self._make_results(codes[:count], displays[:count], raws[:count])
