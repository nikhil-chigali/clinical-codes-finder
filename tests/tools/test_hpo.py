from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.hpo import HPOClient


async def test_search_result_shape() -> None:
    async with HPOClient() as client:
        results = await client.search("fever", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.HPO
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with HPOClient() as client:
        results = await client.search("tremor", count=3)

    assert len(results) <= 3


async def test_code_format() -> None:
    # HPO codes follow the pattern HP:XXXXXXX (HP: prefix + 7 digits)
    async with HPOClient() as client:
        results = await client.search("pain", count=5)

    assert results, "expected at least one result for 'pain'"
    for r in results:
        assert r.code.startswith("HP:"), f"HPO code must start with 'HP:': {r.code!r}"
        suffix = r.code[3:]
        assert suffix.isdigit() and len(suffix) == 7, f"unexpected HPO code format: {r.code!r}"
