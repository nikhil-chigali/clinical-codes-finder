from clinical_codes.schemas import CodeResult, SystemName
from clinical_codes.tools.icd10cm import ICD10CMClient


async def test_search_result_shape() -> None:
    async with ICD10CMClient() as client:
        results = await client.search("hypertension", count=5)

    assert len(results) > 0
    for r in results:
        assert isinstance(r, CodeResult)
        assert r.system == SystemName.ICD10CM
        assert r.code
        assert r.display
        assert 0 < r.score <= 1.0
        assert {"code", "display", "row"} <= r.raw.keys()

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "scores must be descending"


async def test_count_limits_results() -> None:
    async with ICD10CMClient() as client:
        results = await client.search("diabetes", count=3)

    assert len(results) <= 3


async def test_code_format() -> None:
    # ICD-10-CM codes begin with an uppercase letter (category letter)
    async with ICD10CMClient() as client:
        results = await client.search("fracture", count=5)

    assert results, "expected at least one result for 'fracture'"
    assert all(r.code[0].isalpha() for r in results), "ICD-10-CM codes start with a letter"
