import re
from typing import Any

import httpx

from clinical_codes.config import settings
from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.base import ClinicalTablesClient, _fetch_with_retry

_DOSE_RE = re.compile(r'\s*\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b', re.IGNORECASE)


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

    async def search(self, query: str, count: int | None = None) -> list[CodeResult]:
        results = await super().search(query, count)
        if results:
            return results

        dose_match = _DOSE_RE.search(query)
        if not dose_match:
            return []

        n = count if count is not None else settings.fetch_results
        drug_name = _DOSE_RE.sub("", query).strip()
        if not drug_name:
            return []
        try:
            data = await _fetch_with_retry(
                self._client, self._endpoint(), self._build_params(drug_name, n)
            )
        except httpx.HTTPError:
            return []
        return self._parse_strengths(data, n, dose_hint=dose_match.group(0))

    def _parse_strengths(self, data: Any, count: int, dose_hint: str) -> list[CodeResult]:
        try:
            rows: list[list[str]] = data[3] or []
        except (IndexError, TypeError):
            return []
        dose_norm = dose_hint.strip().lower()

        matching: list[tuple[str, str, dict]] = []
        others: list[tuple[str, str, dict]] = []

        for row in rows:
            if len(row) < 3:
                continue
            drug = row[0]
            cuis = [c.strip() for c in row[1].split(",") if c.strip()]
            strengths = [s.strip() for s in row[2].split(",") if s.strip()]
            for cui, strength in zip(cuis, strengths):
                display = f"{drug} — {strength}"
                raw = {"code": cui, "display": display, "drug": drug, "strength": strength, "row": row}
                bucket = matching if dose_norm in strength.lower() else others
                bucket.append((cui, display, raw))

        ordered = (matching + others)[:count]
        codes = [e[0] for e in ordered]
        displays = [e[1] for e in ordered]
        raws = [e[2] for e in ordered]
        return self._make_results(codes, displays, raws)

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
