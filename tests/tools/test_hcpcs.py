from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.hcpcs import HCPCSClient


async def test_search_result_shape() -> None:
    async with HCPCSClient() as client:
        results = await client.search("wheelchair", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.HCPCS
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with HCPCSClient() as client:
        results = await client.search("oxygen", count=3)

    assert len(results) <= 3


async def test_code_format() -> None:
    # HCPCS codes are one uppercase letter followed by 4 digits, e.g. "E2210", "K0001"
    async with HCPCSClient() as client:
        results = await client.search("walker", count=5)

    assert results, "expected at least one result for 'walker'"
    for r in results:
        assert len(r.code) == 5, f"HCPCS code should be 5 chars: {r.code!r}"
        assert r.code[0].isalpha() and r.code[1:].isdigit(), f"unexpected HCPCS format: {r.code!r}"
