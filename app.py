"""Meta Analyst — Streamlit product entrypoint."""

import json

import streamlit as st

from meta_analyst import __product_name__, __version__
from meta_analyst.analysis import (
    get_openai_client,
    llm_chart_insight,
    llm_chat_answer,
    llm_detailed_analysis,
)
from meta_analyst.charts import create_charts
from meta_analyst.data import load_and_clean_data
from meta_analyst.pdf import generate_full_pdf
from meta_analyst.ui import (
    check_password,
    inject_theme,
    render_footer,
    render_home,
    render_kpi_cards,
    sidebar_nav,
)


@st.cache_resource
def cached_client():
    return get_openai_client()


def _insight_box(insight: str, accent: str) -> str:
    return f"""
    <div style="background:#f7f7f7; padding:10px;
                border-left:4px solid {accent};
                border-radius:6px; margin-top:8px;">
        <b>AI Insight:</b><br>
        {insight}
    </div>
    """


def render_dashboard(df, client):
    st.title("Performance Dashboard")
    render_kpi_cards(df)
    st.markdown("<hr>", unsafe_allow_html=True)

    figs = create_charts(df)
    col1, col2 = st.columns(2)

    if "objective_spend_pie" in figs:
        with col1:
            st.subheader("Spend Share by Objective")
            st.pyplot(figs["objective_spend_pie"])
            insight = llm_chart_insight(client, df, "objective_spend_pie")
            st.markdown(_insight_box(insight, "#2563eb"), unsafe_allow_html=True)

    if "objective_roas" in figs:
        with col2:
            st.subheader("ROAS by Objective")
            st.pyplot(figs["objective_roas"])
            insight = llm_chart_insight(client, df, "objective_roas")
            st.markdown(_insight_box(insight, "#16a34a"), unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    if "scatter_spend_roas" in figs:
        with col3:
            st.subheader("Spend vs ROAS")
            st.pyplot(figs["scatter_spend_roas"])
            insight = llm_chart_insight(client, df, "scatter_spend_roas")
            st.markdown(_insight_box(insight, "#dc2626"), unsafe_allow_html=True)

    if "ctr_heatmap" in figs:
        with col4:
            st.subheader("CTR Heatmap (Top Campaigns)")
            st.pyplot(figs["ctr_heatmap"])
            insight = llm_chart_insight(client, df, "ctr_heatmap")
            st.markdown(_insight_box(insight, "#a855f7"), unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Detailed AI Analysis")

    if st.button("Generate Detailed AI Analysis"):
        with st.spinner("Analyzing campaigns and ads..."):
            st.session_state["detailed_ai_data"] = llm_detailed_analysis(client, df)

    if "detailed_ai_data" in st.session_state:
        try:
            parsed = json.loads(st.session_state["detailed_ai_data"])
        except json.JSONDecodeError:
            st.error("Model returned invalid JSON.")
            st.write(st.session_state["detailed_ai_data"])
            return

        st.markdown("### Overall Summary")
        st.write(parsed["overall_summary"])
        st.markdown("### Priority Recommendations")

        color_map = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
        for blk in parsed["priority_blocks"]:
            color = color_map.get(blk["priority"], "#000")
            st.markdown(
                f"""
                <div style="border-left:5px solid {color};
                            padding:10px; background:#fafafa;
                            border-radius:6px; margin-bottom:12px;">
                    <h4 style="margin:0; color:{color}">
                        {blk['priority']} PRIORITY — {blk['title']}
                    </h4>
                    <p>{blk['details']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("Generate Full PDF Report"):
        with st.spinner("Generating PDF..."):
            insights = {key: llm_chart_insight(client, df, key) for key in figs}
            pdf_path = generate_full_pdf(df, figs, insights)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f,
                    file_name="Meta_Report.pdf",
                    mime="application/pdf",
                )


def render_chat(df, client):
    st.title("AI Strategy Chat")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about ROAS, scaling, pauses, or budget shifts...")
    if not user_input:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm_chat_answer(client, df, user_input)
            st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})


def main():
    st.set_page_config(
        page_title=__product_name__,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    check_password()

    try:
        client = cached_client()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    page = sidebar_nav()
    st.caption(f"{__product_name__} v{__version__}")

    if page == "Home":
        render_home()
    elif page == "Upload":
        st.title("Upload Meta Ads Report")
        file = st.file_uploader("Upload CSV or Excel export", type=["csv", "xlsx"])
        if file:
            before_rows = None
            df = load_and_clean_data(file)
            st.session_state["df"] = df
            st.success(f"Processed {len(df)} active ads.")
            st.dataframe(df.head(), use_container_width=True)
    elif page == "Analysis":
        if "df" not in st.session_state:
            st.warning("Upload a report first.")
        else:
            render_dashboard(st.session_state["df"], client)
    elif page == "Chat":
        if "df" not in st.session_state:
            st.warning("Upload a report first to enable chat.")
        else:
            render_chat(st.session_state["df"], client)

    render_footer()


if __name__ == "__main__":
    main()
