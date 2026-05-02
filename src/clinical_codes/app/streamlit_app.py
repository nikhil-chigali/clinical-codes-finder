from __future__ import annotations

import asyncio

import streamlit as st

from clinical_codes.graph.builder import build_graph, make_initial_state
from clinical_codes.graph.prompts import SYSTEM_CATALOG

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

    # ── Summary ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary**")
    st.markdown(summary)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Results**")
    if not consolidated:
        st.info("No results found.")
    else:
        for system, results in consolidated.items():
            with st.expander(f"{system.value} · {len(results)} results", expanded=True):
                st.dataframe(
                    [
                        {
                            "Code": r.code,
                            "Display": r.display,
                            "Score": f"{r.score:.2f}",
                        }
                        for r in results
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Reasoning trace ───────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Reasoning trace"):
        if len(attempt_history) > 1:
            n = len(attempt_history)
            st.markdown(
                f'<span style="background:#5a3a2a;color:#e8a87c;padding:2px 10px;'
                f'border-radius:12px;font-size:13px">🔁 {n} iterations</span>',
                unsafe_allow_html=True,
            )
            st.write("")
        for i, attempt in enumerate(attempt_history):
            systems = ", ".join(s.value for s in attempt.planner_output.selected_systems)
            st.markdown(f"**Iteration {attempt.iteration}** · {systems}")
            st.caption(attempt.planner_output.rationale)
            if attempt.evaluator_output.decision == "sufficient":
                st.success("✓ Sufficient")
            else:
                st.warning(f"↩ Refine — {attempt.evaluator_output.feedback}")
            if i < len(attempt_history) - 1:
                st.divider()
