from typing import Any

from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.base import ClinicalTablesClient


class RxNormClient(ClinicalTablesClient):
    system = SystemName.RXNORM

    def _endpoint(self) -> str:
        return "rxterms/v3/search"

    def _build_params(self, query: str, count: int) -> dict[str, str]:
        return {
            "terms": query,
            "count": str(count),
            "df": "DISPLAY_NAME,RXCUIS,STRENGTHS_AND_FORMS",
        }

    def _parse_response(self, data: Any, count: int) -> list[CodeResult]:
        # response[3][i] = [DISPLAY_NAME, RXCUIS_csv, STRENGTHS_AND_FORMS_csv]
        # One CodeResult per row (drug+form group). Code = first CUI in the row.
        rows: list[list[str]] = data[3] or []
        codes, displays, raws = [], [], []

        for row in rows[:count]:
            if len(row) < 2:
                continue
            drug = row[0]
            cuis = [c.strip() for c in row[1].split(",") if c.strip()]
            code = cuis[0] if cuis else ""
            if not code:
                continue
            codes.append(code)
            displays.append(drug)
            raws.append({"code": code, "display": drug, "row": row})

        return self._make_results(codes, displays, raws)
