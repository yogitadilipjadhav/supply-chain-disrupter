import streamlit as st

from src.agents.langgraph_engine import run_agent_graph
from src.dashboard.data_loader import show_data_loader
from src.utils.db_utils import ensure_schema, fetch_scenario_options
from src.utils.rag_utils import query_chroma_rag


def show_rag_search() -> None:
    st.title("Electronics Knowledge Search")
    query = st.text_input(
        "Search semiconductor events, mitigation guidance, or field definitions",
        "semiconductor factory shutdown risk",
    )
    result_count = st.slider("Results", 1, 10, 5)
    if st.button("Search ChromaDB"):
        hits = query_chroma_rag(query, n_results=result_count)
        if not hits:
            st.warning("No ChromaDB results. Build the databases first.")
            return
        for hit in hits:
            metadata = hit["metadata"]
            with st.expander(
                f"{metadata.get('type', 'document')} · "
                f"distance {hit.get('distance', 0):.3f}"
            ):
                st.write(hit["text"])
                st.json(metadata)


def show_scenario_analyzer() -> None:
    st.title("Scenario Analyzer")
    st.caption(
        "Runs against records from Varun's electronics workbook and live "
        "Open-Meteo weather data."
    )
    ensure_schema()
    try:
        options = fetch_scenario_options()
    except Exception:
        options = []

    if not options:
        st.warning("Build the SQLite database before running a scenario.")
        return

    with st.form(key="scenario_form"):
        disruption_type = st.selectbox(
            "Disruption type",
            [
                "earthquake",
                "port closure",
                "chip shortage",
                "geopolitical",
                "extreme weather",
                "supplier lockdown",
            ],
        )
        selected = st.selectbox(
            "Historical scenario baseline",
            options,
            format_func=lambda row: (
                f"{row['port']} · {row['sku']} · {row['event_date']} "
                f"({row['history_points']} history points)"
            ),
        )
        affected_route = st.text_input("Affected route", "Supplier to destination")
        severity = st.slider("Severity", 0.0, 1.0, 0.6)
        shock_duration_days = st.number_input(
            "Shock duration (days)",
            min_value=0,
            max_value=180,
            value=0,
            help="Set only when modeling a confirmed disruption duration; 0 skips duration escalation.",
        )
        recovery_window_days = st.number_input(
            "Recovery window (days)", min_value=1, max_value=180, value=60
        )
        submit = st.form_submit_button("Run scenario")

    if not submit:
        return

    with st.spinner("Running workflow..."):
        try:
            result = run_agent_graph(
                {
                    "disruption_type": disruption_type,
                    "affected_port": selected["port"],
                    "affected_route": affected_route,
                    "severity": severity,
                    "shock_duration_days": shock_duration_days,
                    "recovery_window_days": recovery_window_days,
                    "synthetic_ratio": 0.0,
                    "sku": selected["sku"],
                    "event_date": selected["event_date"],
                }
            )
        except Exception as exc:
            st.error(f"Scenario failed: {exc}")
            return

    # ── Risk Classifier ───────────────────────────────────────────────────────
    st.subheader("Risk Classifier")
    if result.risk_classification:
        rc = result.risk_classification

        _LABEL_COLOR = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red",
            "CRITICAL": "darkred",
        }
        color = _LABEL_COLOR.get(rc.final_label, "grey")
        escalation_note = (
            f" *(escalated from **{rc.base_label}** — duration {rc.duration_days:.0f}d)*"
            if rc.escalated
            else ""
        )
        st.markdown(
            f"### :{color}[{rc.final_label}]{escalation_note}"
        )

        # Top-line metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite Score", f"{rc.composite_score:.3f}")
        m2.metric("Mode", rc.mode.upper())
        m3.metric("Base Label", rc.base_label)
        m4.metric("Escalated", "Yes" if rc.escalated else "No")

        # Component breakdown
        st.markdown("**Component Breakdown** *(each normalized 0 → 1)*")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Geo Risk", f"{rc.geo_component:.3f}")
        c1.progress(rc.geo_component)
        c2.metric("Supply Disruption", f"{rc.supply_component:.3f}")
        c2.progress(rc.supply_component)
        c3.metric("Freight / News", f"{rc.freight_component:.3f}")
        c3.progress(rc.freight_component)
        c4.metric("Defect Rate", f"{rc.defect_component:.3f}")
        c4.progress(rc.defect_component)

        # Composite formula callout
        st.caption(
            f"Composite = 0.40 × {rc.geo_component:.3f} (geo)"
            f" + 0.30 × {rc.supply_component:.3f} (supply)"
            f" + 0.15 × {rc.freight_component:.3f} (freight)"
            f" + 0.15 × {rc.defect_component:.3f} (defect)"
            f" = **{rc.composite_score:.3f}**"
        )

        if rc.duration_days is not None:
            st.info(f"Disruption duration signal: **{rc.duration_days:.0f} days** (used for escalation matrix)")

        if rc.rationale:
            with st.expander("Rationale / RAG grounding"):
                st.write(rc.rationale)
                if rc.rag_citations:
                    st.markdown("**Citations:**")
                    for cite in rc.rag_citations:
                        st.markdown(f"- `{cite}`")
    else:
        st.warning("Risk classification result not available.")

    st.divider()

    # ── Supporting signals ────────────────────────────────────────────────────
    st.subheader("Supporting Signals")
    s1, s2, s3 = st.columns(3)
    s1.metric("Live Weather Severity", f"{result.live_weather_severity:.3f}" if result.live_weather_severity is not None else "N/A")
    if result.forecast_result:
        s2.metric("Forecast Demand Drop", f"{result.forecast_result.expected_drop_pct:.1f}%")
    if result.simulation_result:
        s3.metric("Stockout Probability", f"{result.simulation_result.stockout_probability_pct:.1f}%")
        st.caption(f"Alternate route: {result.simulation_result.alternate_route}")

    st.divider()

    # ── Mitigation ────────────────────────────────────────────────────────────
    if result.mitigation_action:
        st.subheader("Mitigation Recommendation")
        st.write(result.mitigation_action.summary)
        for rec in result.mitigation_action.recommendations:
            st.markdown(f"- {rec}")
        st.caption(f"Cost delta: {result.mitigation_action.cost_delta}")

    # ── Agent logs ────────────────────────────────────────────────────────────
    with st.expander("Agent Logs"):
        for log in result.agent_logs:
            st.text(log)


def main() -> None:
    st.set_page_config(
        page_title="Supply Chain Disruption Predictor",
        layout="wide",
    )
    page = st.sidebar.radio(
        "Navigate",
        ["Data Ingestion", "RAG Search", "Scenario Analyzer"],
    )
    if page == "Data Ingestion":
        show_data_loader()
    elif page == "RAG Search":
        show_rag_search()
    else:
        show_scenario_analyzer()


if __name__ == "__main__":
    main()
