from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.ucum import UCUMClient


async def test_search_result_shape() -> None:
    async with UCUMClient() as client:
        results = await client.search("kilogram", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.UCUM
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with UCUMClient() as client:
        results = await client.search("meter", count=3)

    assert len(results) <= 3


async def test_display_nonempty() -> None:
    # UCUM unit codes are terse strings (e.g. "kg", "/kg") — test display is human-readable
    async with UCUMClient() as client:
        results = await client.search("liter", count=5)

    assert results, "expected at least one result for 'liter'"
    for r in results:
        assert len(r.display) > len(r.code), (
            f"display should be more descriptive than the unit code: {r.code!r} / {r.display!r}"
        )
