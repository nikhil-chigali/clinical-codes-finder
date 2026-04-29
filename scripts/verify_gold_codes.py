"""Verify gold-set expected codes against the live NLM Clinical Tables API.

Usage:
    python -m scripts.verify_gold_codes [--gold path/to/gold.json] [--write-corrected path]

If --write-corrected is omitted, runs in report-only mode (no files written).

The harness deliberately uses stdlib only (urllib + time.sleep) to keep it
re-runnable without project deps.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_BASE = "https://clinicaltables.nlm.nih.gov/api"
DEFAULT_FETCH_COUNT = 20
TOP_N_PRIMARY = 10  # threshold for "in top-10"
TOP_N_STRICT = 3    # threshold for must_include / must_not_include
SLEEP_BETWEEN_CALLS = 0.1
REQUEST_TIMEOUT = 15

# Per-system endpoint config. RxNorm is special: its codes (CUIs) live in
# slot[3] under the RXCUIS df field, not slot[1].
SYSTEM_CONFIG: dict[str, dict[str, Any]] = {
    "ICD10CM": {
        "path": "/icd10cm/v3/search",
        # Default search field is `code`; need sf=code,name to match by name.
        "extra_params": {"sf": "code,name"},
        "code_extractor": "slot1",
    },
    "LOINC": {
        "path": "/loinc_items/v3/search",
        "extra_params": {},
        "code_extractor": "slot1",
    },
    "RXNORM": {
        "path": "/rxterms/v3/search",
        "extra_params": {"df": "DISPLAY_NAME,RXCUIS,STRENGTHS_AND_FORMS"},
        "code_extractor": "rxnorm_cuis",
    },
    "HCPCS": {
        "path": "/hcpcs/v3/search",
        "extra_params": {},
        "code_extractor": "slot1",
    },
    "HPO": {
        "path": "/hpo/v3/search",
        "extra_params": {},
        "code_extractor": "slot1",
    },
    "UCUM": {
        "path": "/ucum/v3/search",
        "extra_params": {},
        "code_extractor": "slot1",
    },
}


def fetch_api(system: str, query: str, count: int = DEFAULT_FETCH_COUNT) -> list[Any]:
    cfg = SYSTEM_CONFIG[system]
    params = {"terms": query, "count": str(count), **cfg["extra_params"]}
    url = f"{API_BASE}{cfg['path']}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read())


def extract_codes(system: str, response: list[Any]) -> list[str]:
    """Return ordered, deduped list of codes from an API response."""
    extractor = SYSTEM_CONFIG[system]["code_extractor"]
    if extractor == "slot1":
        return list(response[1] or [])
    if extractor == "rxnorm_cuis":
        # RxTerms returns one row per drug+form. CUIs (one per strength) are in
        # row[1] as a comma-joined string. Flatten in row order, dedup.
        cuis: list[str] = []
        for row in response[3] or []:
            if len(row) < 2:
                continue
            for cui in row[1].split(","):
                cui = cui.strip()
                if cui and cui not in cuis:
                    cuis.append(cui)
        return cuis
    raise ValueError(f"unknown extractor: {extractor}")


@dataclass
class CodeCheck:
    code: str
    in_top10: bool
    in_top20: bool
    rank: int | None  # 1-indexed; None if not in top-20


@dataclass
class SystemResult:
    system: str
    query: str
    api_top20: list[str]
    api_returned_zero: bool
    expected_checks: list[CodeCheck] = field(default_factory=list)


@dataclass
class QueryResult:
    qid: str
    query: str
    query_type: str
    systems: dict[str, SystemResult] = field(default_factory=dict)


def check_codes(api_top20: list[str], expected: list[str]) -> list[CodeCheck]:
    checks = []
    for code in expected:
        rank = api_top20.index(code) + 1 if code in api_top20 else None
        checks.append(
            CodeCheck(
                code=code,
                in_top10=rank is not None and rank <= TOP_N_PRIMARY,
                in_top20=rank is not None,
                rank=rank,
            )
        )
    return checks


def verify_query(query_obj: dict[str, Any]) -> QueryResult | None:
    """Run verification for one gold query. Returns None for miss-type queries."""
    if query_obj["query_type"] == "miss":
        return None
    qres = QueryResult(
        qid=query_obj["id"],
        query=query_obj["query"],
        query_type=query_obj["query_type"],
    )
    expected_codes = query_obj.get("expected_codes", {})
    for system, expected in expected_codes.items():
        if system not in SYSTEM_CONFIG:
            print(f"  [warn] {qres.qid}: unknown system {system}, skipping")
            continue
        try:
            response = fetch_api(system, qres.query)
            api_codes = extract_codes(system, response)
        except Exception as e:
            print(f"  [error] {qres.qid} {system}: {e}")
            raise
        time.sleep(SLEEP_BETWEEN_CALLS)
        sres = SystemResult(
            system=system,
            query=qres.query,
            api_top20=api_codes[:DEFAULT_FETCH_COUNT],
            api_returned_zero=(len(api_codes) == 0),
            expected_checks=check_codes(api_codes[:DEFAULT_FETCH_COUNT], expected),
        )
        qres.systems[system] = sres
    return qres


@dataclass
class Modification:
    qid: str
    system: str
    kind: str  # "replaced", "ranked_low", "must_include_demoted", "must_not_include_removed", "api_zero_results"
    detail: str


def build_corrected_gold(
    original: dict[str, Any], results: list[QueryResult]
) -> tuple[dict[str, Any], list[Modification]]:
    """Apply correction rules. Returns (new_gold_dict, modifications)."""
    mods: list[Modification] = []
    by_qid = {r.qid: r for r in results}
    new_queries = []

    for q in original["queries"]:
        new_q = json.loads(json.dumps(q))  # deep copy
        qres = by_qid.get(q["id"])

        if qres is None:
            # miss-type or no verification done
            new_queries.append(new_q)
            continue

        for system, sres in qres.systems.items():
            expected_list = new_q["expected_codes"].get(system, [])

            if sres.api_returned_zero:
                # Cannot replace from an empty API response. Flag and keep.
                for check in sres.expected_checks:
                    mods.append(
                        Modification(
                            qid=q["id"],
                            system=system,
                            kind="api_zero_results",
                            detail=(
                                f"`{check.code}` retained: API returned 0 results "
                                f"for query \"{qres.query}\" — verification not possible "
                                f"(query string likely needs reformulation by the planner)"
                            ),
                        )
                    )
                continue

            for check in sres.expected_checks:
                if check.in_top10:
                    continue
                if check.in_top20:
                    # Outside top-10 but inside top-20 → keep, flag.
                    mods.append(
                        Modification(
                            qid=q["id"],
                            system=system,
                            kind="ranked_low",
                            detail=(
                                f"`{check.code}` ranks #{check.rank} for "
                                f"\"{qres.query}\" (kept, but outside top-10)"
                            ),
                        )
                    )
                else:
                    # Not in top-20 → replace with API top result if available.
                    api_top = sres.api_top20[0] if sres.api_top20 else None
                    if api_top is None:
                        continue
                    # Replace in expected_codes
                    if check.code in expected_list:
                        idx = expected_list.index(check.code)
                        expected_list[idx] = api_top
                        # Dedup if api_top was already in the list
                        seen = set()
                        deduped = []
                        for c in expected_list:
                            if c not in seen:
                                seen.add(c)
                                deduped.append(c)
                        new_q["expected_codes"][system] = deduped
                    mods.append(
                        Modification(
                            qid=q["id"],
                            system=system,
                            kind="replaced",
                            detail=(
                                f"replaced `{check.code}` with `{api_top}` — "
                                f"original code did not appear in top-20 for "
                                f"query \"{qres.query}\""
                            ),
                        )
                    )

        # Apply must_include rule: demote codes not in top-10 of any system.
        # Walk a copy so we can mutate.
        for code in list(new_q.get("must_include", [])):
            # Find which system this code belongs to (lookup in expected_codes,
            # falling back to format-based heuristic).
            owner_system = None
            for system, codes in new_q["expected_codes"].items():
                if code in codes:
                    owner_system = system
                    break
            if owner_system is None:
                # Could be a code that was just replaced. Check if it appeared
                # in original q for any system; if so use that system.
                for system, codes in q["expected_codes"].items():
                    if code in codes:
                        owner_system = system
                        break
            if owner_system is None or owner_system not in qres.systems:
                continue
            sres = qres.systems[owner_system]
            in_top10 = code in sres.api_top20[:TOP_N_PRIMARY]
            if not in_top10:
                new_q["must_include"].remove(code)
                # Move to expected_codes if not already there
                exp = new_q["expected_codes"].setdefault(owner_system, [])
                if code not in exp:
                    exp.append(code)
                where = (
                    f"rank {sres.api_top20.index(code) + 1}"
                    if code in sres.api_top20
                    else "not in top-20"
                )
                mods.append(
                    Modification(
                        qid=q["id"],
                        system=owner_system,
                        kind="must_include_demoted",
                        detail=(
                            f"`{code}` moved out of must_include — does not appear "
                            f"in top-{TOP_N_PRIMARY} for \"{qres.query}\" ({where})"
                        ),
                    )
                )

        # Apply must_not_include rule: if a code there appears in top-3, remove.
        for code in list(new_q.get("must_not_include", [])):
            # Check across all queried systems
            for system, sres in qres.systems.items():
                if code in sres.api_top20[:TOP_N_STRICT]:
                    new_q["must_not_include"].remove(code)
                    rank = sres.api_top20.index(code) + 1
                    mods.append(
                        Modification(
                            qid=q["id"],
                            system=system,
                            kind="must_not_include_removed",
                            detail=(
                                f"`{code}` removed from must_not_include — "
                                f"actually appears at rank {rank} in {system} for "
                                f"\"{qres.query}\""
                            ),
                        )
                    )
                    break

        new_queries.append(new_q)

    new_gold = dict(original)
    new_gold["queries"] = new_queries
    return new_gold, mods


def print_summary(results: list[QueryResult], mods: list[Modification]) -> dict[str, int]:
    """Print human-readable summary table. Returns counts dict."""
    counts = {
        "queries_verified": len(results),
        "system_checks": sum(len(r.systems) for r in results),
        "in_top10": 0,
        "ranked_low": 0,
        "replaced": 0,
        "api_zero_results": 0,
        "must_include_demoted": 0,
        "must_not_include_removed": 0,
    }
    for r in results:
        for sres in r.systems.values():
            for check in sres.expected_checks:
                if check.in_top10:
                    counts["in_top10"] += 1
    for m in mods:
        if m.kind in counts:
            counts[m.kind] += 1

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Queries verified                : {counts['queries_verified']}")
    print(f"  Total (query, system) checks    : {counts['system_checks']}")
    print(f"  Codes in API top-10 (clean)     : {counts['in_top10']}")
    print(f"  Codes ranked low (kept, flagged): {counts['ranked_low']}")
    print(f"  Codes replaced (not in top-20)  : {counts['replaced']}")
    print(f"  API returned zero results       : {counts['api_zero_results']}")
    print(f"  must_include demotions          : {counts['must_include_demoted']}")
    print(f"  must_not_include removals       : {counts['must_not_include_removed']}")
    print("=" * 70)
    return counts


def write_changelog(
    path: Path, mods: list[Modification], counts: dict[str, int], date_str: str
) -> None:
    by_kind: dict[str, list[Modification]] = {}
    for m in mods:
        by_kind.setdefault(m.kind, []).append(m)

    lines = [
        "# Gold eval set changelog",
        "",
        f"## v0.1.1 — {date_str}",
        "",
        "Verified against NLM Clinical Tables API. Patch-level changes only "
        "(no schema or query changes).",
        "",
        "### Code replacements",
    ]
    if by_kind.get("replaced"):
        for m in by_kind["replaced"]:
            lines.append(f"- `{m.qid}` {m.system}: {m.detail}")
    else:
        lines.append("- _(none)_")

    lines += ["", "### Lower-than-expected rankings (kept, but flagged)"]
    if by_kind.get("ranked_low"):
        for m in by_kind["ranked_low"]:
            lines.append(f"- `{m.qid}` {m.system}: {m.detail}")
    else:
        lines.append("- _(none)_")

    lines += ["", "### Demoted from must_include"]
    if by_kind.get("must_include_demoted"):
        for m in by_kind["must_include_demoted"]:
            lines.append(f"- `{m.qid}` {m.system}: {m.detail}")
    else:
        lines.append("- _(none)_")

    lines += ["", "### Removed from must_not_include"]
    if by_kind.get("must_not_include_removed"):
        for m in by_kind["must_not_include_removed"]:
            lines.append(f"- `{m.qid}` {m.system}: {m.detail}")
    else:
        lines.append("- _(none)_")

    lines += ["", "### Unverified / unverifiable"]
    if by_kind.get("api_zero_results"):
        for m in by_kind["api_zero_results"]:
            lines.append(f"- `{m.qid}` {m.system}: {m.detail}")
    else:
        lines.append("- _(none)_")

    lines += [
        "",
        "### Summary",
        f"- {counts['replaced']} codes replaced",
        f"- {counts['ranked_low']} codes flagged as ranked-low (kept)",
        f"- {counts['must_include_demoted']} must_include demotions",
        f"- {counts['must_not_include_removed']} must_not_include removals",
        f"- {counts['api_zero_results']} (query, system) pairs unverifiable "
        f"(API returned no results)",
        f"- Total (query, system) pairs verified: {counts['system_checks']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/gold/gold_v0.1.0.json"),
        help="Path to gold JSON to verify",
    )
    parser.add_argument(
        "--write-corrected",
        type=Path,
        default=None,
        help="If set, write a corrected gold JSON to this path",
    )
    parser.add_argument(
        "--write-changelog",
        type=Path,
        default=None,
        help="If set, write a changelog markdown to this path",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date string for the changelog (defaults to today UTC)",
    )
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    print(f"Loaded {args.gold} (version {gold['version']}, "
          f"{len(gold['queries'])} queries)")

    results: list[QueryResult] = []
    for q in gold["queries"]:
        print(f"  verifying {q['id']} ({q['query_type']}): {q['query']!r}")
        res = verify_query(q)
        if res is not None:
            results.append(res)

    new_gold, mods = build_corrected_gold(gold, results)
    counts = print_summary(results, mods)

    if args.write_corrected:
        new_gold["version"] = "0.1.1"
        args.write_corrected.write_text(
            json.dumps(new_gold, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote corrected gold to {args.write_corrected}")

    if args.write_changelog:
        from datetime import datetime, timezone
        date_str = args.date or datetime.now(timezone.utc).date().isoformat()
        write_changelog(args.write_changelog, mods, counts, date_str)
        print(f"Wrote changelog to {args.write_changelog}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
