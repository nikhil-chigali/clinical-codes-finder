import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from clinical_codes.config import settings
from clinical_codes.schemas import CodeResult, SystemName

logger = logging.getLogger(__name__)


def _rank_to_score(rank: int, total: int) -> float:
    """Map 0-indexed rank to a (0, 1] score: rank 0 → 1.0, rank n-1 → 1/n."""
    if total <= 1:
        return 1.0
    return round((total - rank) / total, 4)


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, str],
) -> Any:
    """GET path with params, retrying on httpx.HTTPError with exponential backoff.

    AsyncRetrying is constructed inline so runtime settings values are honoured per-call.
    Raises the last httpx.HTTPError if all attempts are exhausted (reraise=True).
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(settings.api_max_retries + 1),
        wait=wait_exponential(
            multiplier=settings.api_backoff_base,
            min=settings.api_backoff_base,
            max=32,
        ),
        retry=retry_if_exception_type(httpx.HTTPError),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        reraise=True,
    ):
        with attempt:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()


class ClinicalTablesClient(ABC):
    """Base class for per-system NLM Clinical Tables API wrappers.

    Subclasses implement _endpoint, _build_params, and _parse_response.
    The search() method handles HTTP, retry, and error isolation.

    Usage:
        async with ICD10CMClient() as client:
            results = await client.search("hypertension")

    Or without context manager (connection pool closed on GC):
        client = ICD10CMClient()
        results = await client.search("hypertension")
    """

    system: SystemName

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.nlm_api_base,
            timeout=settings.api_timeout,
        )

    async def __aenter__(self) -> "ClinicalTablesClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.__aexit__(*args)

    async def search(self, query: str, count: int | None = None) -> list[CodeResult]:
        """Search this system for query. Returns [] on total API failure (isolated)."""
        n = count if count is not None else settings.fetch_results
        path = self._endpoint()
        params = self._build_params(query, n)

        try:
            data = await _fetch_with_retry(self._client, path, params)
        except httpx.HTTPError:
            logger.error("All retries exhausted for %s query=%r — returning empty", self.system, query)
            return []

        return self._parse_response(data, n)

    @abstractmethod
    def _endpoint(self) -> str:
        """Relative path from nlm_api_base, e.g. 'icd10cm/search'."""

    @abstractmethod
    def _build_params(self, query: str, count: int) -> dict[str, str]:
        """Build Clinical Tables API query parameters."""

    @abstractmethod
    def _parse_response(self, data: Any, count: int) -> list[CodeResult]:
        """Parse raw API response list into normalized CodeResult list."""

    def _make_results(
        self,
        codes: list[str],
        displays: list[str],
        raws: list[dict[str, Any]] | None = None,
    ) -> list[CodeResult]:
        """Zip codes + displays into scored CodeResults. Override raws if you have richer data."""
        total = len(codes)
        results = []
        for rank, (code, display) in enumerate(zip(codes, displays)):
            raw: dict[str, Any] = raws[rank] if raws else {"code": code, "display": display}
            results.append(
                CodeResult(
                    system=self.system,
                    code=code,
                    display=display,
                    score=_rank_to_score(rank, total),
                    raw=raw,
                )
            )
        return results
