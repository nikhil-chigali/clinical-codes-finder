from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.loinc import LOINCClient


async def test_search_result_shape() -> None:
    async with LOINCClient() as client:
        results = await client.search("glucose", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.LOINC
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with LOINCClient() as client:
        results = await client.search("hemoglobin", count=3)

    assert len(results) <= 3


async def test_code_format() -> None:
    # LOINC codes are numeric parts separated by a hyphen, e.g. "4548-4"
    async with LOINCClient() as client:
        results = await client.search("cholesterol", count=5)

    assert results, "expected at least one result for 'cholesterol'"
    for r in results:
        parts = r.code.split("-")
        assert len(parts) == 2 and parts[0].isdigit(), f"unexpected LOINC code format: {r.code!r}"
