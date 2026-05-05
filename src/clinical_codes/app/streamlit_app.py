from __future__ import annotations

import asyncio

import streamlit as st

from clinical_codes.graph.builder import build_graph, make_initial_state
from clinical_codes.graph.prompts import SYSTEM_CATALOG, effective_search_terms

st.set_page_config(page_title="Clinical Codes Finder", layout="wide")


@st.cache_resource
def _get_graph():
    return build_graph()


# ── Sidebar — system catalog ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"**SUPPORTED SYSTEMS ({len(SYSTEM_CATALOG)})**")
    for system, description in SYSTEM_CATALOG.items():
        with st.expander(system.value):
            st.caption(description)


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Clinical Codes Finder")
st.caption(
    "Enter a clinical term to search across ICD-10-CM, LOINC, RxNorm, HCPCS, UCUM, and HPO."
)

query = st.text_input("Clinical term", placeholder="e.g. metformin 500 mg")
search = st.button("Search", disabled=not bool(query.strip()))

if search and query.strip():
    try:
        with st.spinner("Running..."):
            state = asyncio.run(_get_graph().ainvoke(make_initial_state(query.strip())))
    except Exception as e:
        st.error(str(e))
        st.stop()

    consolidated = state["consolidated"]
    summary = state["summary"]
    attempt_history = state["attempt_history"]
    search_terms = effective_search_terms(state["attempt_history"])

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Results**")
    if not consolidated:
        st.info("No results found.")
    else:
        rows = []
        for system, results in consolidated.items():
            term = search_terms.get(system, "")
            for r in results:
                row: dict = {
                    "System": system.value,
                    "Code": r.code,
                    "Display": r.display,
                    "Searched as": term,
                }
                if system.value == "RXNORM" and "row" in r.raw and len(r.raw["row"]) > 2:
                    row["Strengths"] = r.raw["row"][2]
                rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary**")
    st.markdown(summary)

    # ── Reasoning trace ───────────────────────────────────────────────────────
    st.divider()
    label = f"Reasoning trace ({len(attempt_history)} iteration{'s' if len(attempt_history) != 1 else ''})"
    with st.expander(label):
        for i, attempt in enumerate(attempt_history):
            po = attempt.planner_output
            ev = attempt.evaluator_output

            st.markdown(f"#### Iteration {attempt.iteration}")

            st.markdown("**Planner**")
            st.caption(po.rationale)
            term_rows = [
                {"System": s.value, "Search term": po.search_terms.get(s, "")}
                for s in po.selected_systems
            ]
            st.dataframe(term_rows, use_container_width=True, hide_index=True)

            st.markdown("**Results**")
            result_rows = []
            for s in po.selected_systems:
                hits = attempt.raw_results.get(s, [])
                result_rows.append({
                    "System": s.value,
                    "Hits": len(hits),
                    "Top results": ", ".join(r.display for r in hits[:3]) or "—",
                })
            st.dataframe(result_rows, use_container_width=True, hide_index=True)

            st.markdown("**Evaluator**")
            if ev.decision == "sufficient":
                st.success("Sufficient — all systems returned relevant results.")
            else:
                weak = ", ".join(s.value for s in ev.weak_systems) if ev.weak_systems else "—"
                st.warning(f"Refine · weak systems: {weak}")
                st.caption(ev.feedback)

            if i < len(attempt_history) - 1:
                st.divider()
