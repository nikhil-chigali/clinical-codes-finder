from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.rxnorm import RxNormClient


async def test_search_result_shape() -> None:
    async with RxNormClient() as client:
        results = await client.search("lisinopril", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.RXNORM
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with RxNormClient() as client:
        results = await client.search("metformin", count=3)

    assert len(results) <= 3


async def test_code_is_numeric_cui() -> None:
    # RxNorm CUIs are numeric strings
    async with RxNormClient() as client:
        results = await client.search("aspirin", count=5)

    assert results, "expected at least one result for 'aspirin'"
    assert all(r.code.isdigit() for r in results), "RxNorm codes must be numeric CUIs"


async def test_display_contains_drug_name() -> None:
    # Display is "drug — strength" or just "drug" on fallback
    async with RxNormClient() as client:
        results = await client.search("aspirin", count=3)

    assert results
    for r in results:
        assert "aspirin" in r.display.lower(), f"display should contain drug name: {r.display!r}"
