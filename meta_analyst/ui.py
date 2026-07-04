import base64

import streamlit as st

from meta_analyst.config import APP_PASSWORD, COMPANY_NAME, LOGO_PATH, PRODUCT_NAME


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif !important;
}

.kpi-card {
    background: white;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 10px;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: white;
    padding: 32px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    height: 100%;
}

.footer-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 6px 20px;
    background: #f9fafb;
    border-top: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    justify-content: center;
}
.footer-text {
    font-size: 12px;
    color: #6b7280;
    display: flex;
    align-items: center;
    gap: 6px;
}
.footer-logo {
    height: 14px;
    opacity: 0.9;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    color: white !important;
}
.sidebar-title {
    font-size: 20px;
    font-weight: bold;
    color: #e5e7eb;
    margin-bottom: 20px;
}
.stSidebar .stButton>button {
    width: 100%;
    background: #111827;
    color: #fff;
    border-radius: 8px;
    padding: 8px 14px;
    border: none;
}
.stSidebar .stButton>button:hover {
    background: #1f2937;
}
</style>
"""


def encode_image_base64(path) -> str:
    try:
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode()
    except OSError:
        return ""


def inject_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def check_password():
    env_pwd = APP_PASSWORD.strip()
    if not env_pwd:
        st.error("APP_PASSWORD is not configured. Add it to your .env file.")
        st.stop()

    if st.session_state.get("authenticated"):
        return True

    def password_entered():
        if st.session_state.get("password_input", "") == env_pwd:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    encoded_logo = encode_image_base64(LOGO_PATH)
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:80px; margin-bottom:24px;">
            <img src="data:image/png;base64,{encoded_logo}"
                 style="height:60px; margin-bottom:12px;" />
            <h2 style="margin-bottom:4px;">{PRODUCT_NAME}</h2>
            <p style="color:#6b7280; font-size:14px; margin:0;">
                AI-powered Meta Ads analytics by {COMPANY_NAME}.
            </p>
            <p style="color:#9ca3af; font-size:12px; margin-top:4px;">
                Enter your access password to continue.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_input("Access password", type="password", key="password_input", on_change=password_entered)

    if "authenticated" in st.session_state and st.session_state["authenticated"] is False:
        st.error("Incorrect password.")

    st.stop()


def render_footer():
    encoded_logo = encode_image_base64(LOGO_PATH)
    st.markdown(
        f"""
        <div class="footer-wrapper">
            <div class="footer-text">
                <span>Powered by</span>
                <img src="data:image/png;base64,{encoded_logo}" class="footer-logo"/>
                <span>· {PRODUCT_NAME}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav() -> str:
    st.sidebar.markdown(f"<div class='sidebar-title'>{PRODUCT_NAME}</div>", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    if st.sidebar.button("Home"):
        st.session_state["page"] = "Home"
    if st.sidebar.button("Upload Report"):
        st.session_state["page"] = "Upload"
    if st.sidebar.button("Dashboard"):
        st.session_state["page"] = "Analysis"
    if st.sidebar.button("AI Chat"):
        st.session_state["page"] = "Chat"

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.get("authenticated"):
        st.sidebar.markdown(
            f"<span style='font-size:12px; color:#9ca3af;'>Signed in · {COMPANY_NAME}</span>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Logout"):
            for key in ["authenticated", "password_input", "chat_history", "detailed_ai_data", "df"]:
                st.session_state.pop(key, None)

    return st.session_state["page"]


def render_home():
    encoded_logo = encode_image_base64(LOGO_PATH)
    st.markdown(
        f"""
        <div class="hero-card">
            <img src="data:image/png;base64,{encoded_logo}" style="height:48px; margin-bottom:12px;" />
            <h1 style="margin:0 0 8px 0;">{PRODUCT_NAME}</h1>
            <p style="margin:0; color:#cbd5e1; font-size:16px;">
                Turn Meta Ads exports into KPI dashboards, AI insights, and client-ready PDF reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    cards = [
        ("Upload", "Import CSV or Excel exports from Meta Ads Manager."),
        ("Analyze", "Auto-clean data, compute ROAS/CPC/CTR, and visualize performance."),
        ("Act", "Chat with an AI strategist and export prioritized recommendations."),
    ]
    for col, (title, body) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h3 style="margin-top:0;">{title}</h3>
                    <p style="color:#4b5563; margin-bottom:0;">{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Get started")
    st.info("Upload a Meta Ads report from the sidebar to unlock the dashboard and AI chat.")


def render_kpi_cards(df):
    from meta_analyst.charts import kpi_metrics

    cols = st.columns(5)
    for i, (name, val) in enumerate(kpi_metrics(df)):
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class='kpi-card'>
                    <h4 style='margin-bottom:4px;'>{name}</h4>
                    <h3 style='margin:0; color:#0f172a;'>{val}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
