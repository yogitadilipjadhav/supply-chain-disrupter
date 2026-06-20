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
            "Shock duration (days)", min_value=1, max_value=180, value=30
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

    st.subheader("Workflow Results")
    st.write("Risk label:", result.risk_label)
    st.write("Composite risk:", result.risk_score_composite)
    st.write("Weather severity:", result.live_weather_severity)
    if result.forecast_result:
        st.write("Forecast demand variance (%):", result.forecast_result.expected_drop_pct)
    if result.simulation_result:
        st.write(
            "Stockout probability (%):",
            result.simulation_result.stockout_probability_pct,
        )
        st.write("Alternate route:", result.simulation_result.alternate_route)
    if result.mitigation_action:
        st.subheader("Mitigation Recommendation")
        st.write(result.mitigation_action.summary)
        for recommendation in result.mitigation_action.recommendations:
            st.write(f"- {recommendation}")
        st.write("Cost delta:", result.mitigation_action.cost_delta)

    st.subheader("Agent Logs")
    for log in result.agent_logs:
        st.write(f"- {log}")


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
