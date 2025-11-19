##############################################
#                 META ANALYST APP
##############################################

import os
import json
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import matplotlib.pyplot as plt
import seaborn as sns

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

##############################################
# IMAGE BASE64 ENCODER
##############################################

import base64

def encode_image_base64(path):
    """Reads an image and returns base64 encoded string for HTML embedding."""
    try:
        with open(path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
            return encoded
    except Exception as e:
        print("Logo load error:", e)
        return ""


##############################################
# CONFIG
##############################################

ANALYSIS_MODEL = "gpt-4.1-mini"
CHAT_MODEL = "gpt-4.1-mini"

LOGO_PATH = "RareBoxLogo.png"   # Put file near app.py

##############################################
# GLOBAL CSS (Roboto + Clean SaaS UI)
##############################################

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif !important;
}

/* KPI CARD */
.kpi-card {
    background: white;
    padding: 16px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 10px;
}

/* Insight Box */
.insight-box {
    background: #fafafa;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 6px;
    margin-top: 8px;
    margin-bottom: 25px;
    font-size: 14px;
}

/* Footer */
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

/* Sidebar */
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

##############################################
# OPENAI CLIENT
##############################################

@st.cache_resource
def get_client():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("Missing OPENAI_API_KEY in .env")
        st.stop()
    return OpenAI(api_key=key)

##############################################
# LOGO BASE64 LOADING (fix for Streamlit)
##############################################

def load_logo_base64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

LOGO_BASE64 = load_logo_base64()

##############################################
# UTILITY: Pick matching column
##############################################

def pick_col(names, allcols):
    for n in names:
        if n in allcols: return n
    return None

##############################################
# DATA LOADER + AUTO CLEANER
##############################################

def load_clean_data(file):
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    cols = df.columns

    col_campaign = pick_col(["Campaign name","campaign_name"], cols)
    col_obj = pick_col(["Objective","objective","Campaign objective"], cols)
    col_status = pick_col(["Ad delivery","status"], cols)
    col_spend = pick_col(["Amount spent (INR)","Amount spent","Spend","spend"], cols)
    col_rev = pick_col(["Purchases conversion value","Website purchases conversion value","revenue"], cols)
    col_impr = pick_col(["Impressions","impressions"], cols)
    col_clicks = pick_col(["Clicks (all)","Clicks","clicks"], cols)
    col_lpv = pick_col(["LPview/LC%"], cols)
    col_imp_lpv = pick_col(["Imp to LPV %"], cols)

    rename = {
        col_campaign:"campaign_name",
        col_obj:"objective",
        col_status:"status",
        col_spend:"spend",
        col_rev:"revenue",
        col_impr:"impressions",
        col_clicks:"clicks",
        col_lpv:"lpv_lc",
        col_imp_lpv:"imp_lpv"
    }
    rename = {k:v for k,v in rename.items() if k}

    df = df.rename(columns=rename)

    # Filter active ads
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        before = len(df)
        df = df[df["status"] == "active"]
        st.success(f"Filtered active ads: {before} → {len(df)}")

    # Convert to numeric
    num_cols = ["spend","revenue","impressions","clicks","lpv_lc","imp_lpv"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics
    df["roas"] = df["revenue"] / df["spend"].replace(0, np.nan)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
    df["cpm"] = df["spend"] / df["impressions"].replace(0, np.nan) * 1000
    df["ctr_calc"] = df["clicks"] / df["impressions"].replace(0, np.nan)

    return df
##############################################
# KPI CARDS
##############################################

def render_kpi_cards(df):

    tot_spend = df["spend"].sum()
    tot_rev = df["revenue"].sum()
    tot_impr = df["impressions"].sum()
    tot_clicks = df["clicks"].sum()

    avg_roas = tot_rev / tot_spend if tot_spend else 0
    avg_cpc = df["cpc"].mean()
    avg_cpm = df["cpm"].mean()
    avg_ctr = df["ctr_calc"].mean()

    avg_lpv = df["lpv_lc"].mean() if "lpv_lc" in df.columns else 0
    avg_imp_lpv = df["imp_lpv"].mean() if "imp_lpv" in df.columns else 0

    kpi_list = [
        ("Total Spend", f"₹{tot_spend:,.0f}"),
        ("Total Revenue", f"₹{tot_rev:,.0f}"),
        ("Total Impressions", f"{tot_impr:,.0f}"),
        ("Total Clicks", f"{tot_clicks:,.0f}"),
        ("Overall ROAS", f"{avg_roas:,.2f}"),
        ("Avg CPC", f"₹{avg_cpc:,.2f}"),
        ("Avg CPM", f"₹{avg_cpm:,.2f}"),
        ("Avg CTR", f"{avg_ctr*100:,.2f}%"),
        ("Avg LPview/LC%", f"{avg_lpv:,.2f}%"),
        ("Avg Imp → LPV%", f"{avg_imp_lpv:,.2f}%"),
    ]

    cols = st.columns(5)
    for i, (name, val) in enumerate(kpi_list):
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class='kpi-card'>
                    <h4 style='margin-bottom:4px;'>{name}</h4>
                    <h3 style='margin:0; color:#0f172a;'>{val}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

##############################################
# Chart Colors
##############################################

def pastel_colors(n):
    base = [
        "#A3CEF1", "#F9C8C8", "#C6E5B1",
        "#F7E8A4", "#DCC4FF", "#B5EAEA",
        "#FFB5E8", "#FFCBC1", "#C0FFD4"
    ]
    return base[:n]

##############################################
# CREATE CHARTS (all small & sleek)
##############################################

def create_charts(df):

    figs = {}

    # Spend Share Pie
    if "objective" in df.columns:
        spend_pie = df.groupby("objective")["spend"].sum()
        fig, ax = plt.subplots(figsize=(3.0,2.8))
        ax.pie(
            spend_pie.values,
            labels=spend_pie.index,
            autopct="%1.1f%%",
            colors=pastel_colors(len(spend_pie))
        )
        ax.set_title("Spend Share by Objective", fontsize=11)
        figs["objective_spend_pie"] = fig

    # ROAS by Objective
    if "objective" in df.columns:
        summary = (
            df.groupby("objective")
              .agg({"revenue":"sum","spend":"sum"})
        )
        summary["roas"] = summary["revenue"] / summary["spend"]

        fig, ax = plt.subplots(figsize=(3.5,2.8))
        summary["roas"].sort_values(ascending=False).plot(
            kind="barh",
            ax=ax,
            color="#A3CEF1"
        )
        ax.set_title("ROAS by Objective", fontsize=11)
        ax.set_xlabel("ROAS")
        figs["objective_roas"] = fig

    # Spend vs ROAS scatter
    fig, ax = plt.subplots(figsize=(3.5,2.8))
    ax.scatter(df["spend"], df["roas"], s=14, alpha=0.5)
    ax.set_title("Spend vs ROAS", fontsize=11)
    ax.set_xlabel("Spend")
    ax.set_ylabel("ROAS")
    figs["scatter_spend_roas"] = fig

    # CTR Heatmap (Top 15 Campaigns)
    if "campaign_name" in df.columns:
        heat = (
            df.groupby("campaign_name")["ctr_calc"].mean()
              .sort_values(ascending=False)
              .head(15).to_frame()
        )
        fig, ax = plt.subplots(figsize=(3.8,2.9))
        sns.heatmap(heat, cmap="Blues", ax=ax, annot=True, fmt=".2f")
        ax.set_title("CTR Heatmap (Top Campaigns)", fontsize=11)
        figs["ctr_heatmap"] = fig

    return figs

##############################################
# LLM INSIGHT FOR EACH CHART
##############################################

def llm_chart_insight_single(client, df, chart_key):
    """
    Produce AI insights for each chart individually
    """

    if chart_key == "objective_spend_pie":
        data = df.groupby("objective")["spend"].sum().to_dict()
        title = "Spend Distribution by Objective"

    elif chart_key == "objective_roas":
        data = (
            df.groupby("objective")
              .apply(lambda x: x["revenue"].sum() / x["spend"].sum())
              .replace([np.inf,-np.inf], np.nan)
              .fillna(0).to_dict()
        )
        title = "ROAS by Objective"

    elif chart_key == "scatter_spend_roas":
        data = {
            "spend_min": float(df["spend"].min()),
            "spend_max": float(df["spend"].max()),
            "roas_min": float(df["roas"].min()),
            "roas_max": float(df["roas"].max())
        }
        title = "Spend vs ROAS Relationship"

    elif chart_key == "ctr_heatmap":
        data = (
            df.groupby("campaign_name")["ctr_calc"].mean()
              .sort_values(ascending=False).head(15).to_dict()
        )
        title = "CTR by Campaign"

    else:
        return None

    system_prompt = f"""
You are a senior Meta Ads performance analyst. 
Write a **very sharp insight block** for the chart titled: {title}

Rules:
- Start with **1 crisp summary sentence**
- Then give **4–6 bullet points**
- Keep it extremely actionable (budget shift, creative ideas, audience issues)
- Keep it short and high signal only
"""

    user_msg = json.dumps(data, indent=2)

    res = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_msg}
        ]
    )

    return res.output_text.strip()

##############################################
# LLM — DETAILED ANALYSIS (Button Trigger)
##############################################

def llm_detailed(client, df):

    camp_summary = df.groupby("campaign_name").agg({
        "spend":"sum",
        "revenue":"sum",
        "impressions":"sum",
        "roas":"mean"
    }).reset_index().to_dict(orient="records")

    ads_data = (
        df.sort_values("spend", ascending=False)
          .head(200)
          .to_dict(orient="records")
    )

    system = """
You are a senior Meta Ads strategist.
Return JSON like:

{
 "overall_summary":"",
 "priority_blocks":[
     {"priority":"HIGH","title":"","details":""},
     {"priority":"MEDIUM","title":"","details":""},
     {"priority":"LOW","title":"","details":""}
 ]
}
"""

    user = f"""
Campaign Summary:
{json.dumps(camp_summary, indent=2)}

Top Ads:
{json.dumps(ads_data, indent=2)}
"""

    res = client.responses.create(
        model=ANALYSIS_MODEL,
        input=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )

    return res.output_text
##############################################
# DASHBOARD PAGE
##############################################

def render_dashboard(df, client):

    st.title("Meta Ads Performance Dashboard")

    # =========================================
    # KPI CARDS
    # =========================================
    render_kpi_cards(df)
    st.markdown("<hr>", unsafe_allow_html=True)

    # =========================================
    # CREATE CHART OBJECTS
    # =========================================
    figs = create_charts(df)

    # =========================================
    # RENDER CHARTS IN GRID
    # with LLM INSIGHTS BELOW each chart
    # =========================================

    ######### 1 — Spend Share + ROAS #########
    col1, col2 = st.columns(2)

    # --- Spend Share ---
    if "objective_spend_pie" in figs:
        with col1:
            st.subheader("Spend Share by Objective")
            st.pyplot(figs["objective_spend_pie"])

            # LLM Insight
            insight = llm_chart_insight_single(client, df, "objective_spend_pie")
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    padding:10px 14px;
                    border-left:3px solid #2563eb;
                    border-radius:8px;
                    box-shadow:0px 1px 3px rgba(0,0,0,0.06);
                    font-size:14px;
                    margin-top:6px;
                    line-height:1.4;">
                    <b>AI Insight:</b><br>
                    {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- ROAS by Objective ---
    if "objective_roas" in figs:
        with col2:
            st.subheader("ROAS by Objective")
            st.pyplot(figs["objective_roas"])

            insight = llm_chart_insight_single(client, df, "objective_roas")
            st.markdown(
                f"""
                <div style="background:#f7f7f7; padding:10px; 
                            border-left:4px solid #16a34a; 
                            border-radius:6px; margin-top:8px;">
                    <b>AI Insight:</b><br>
                    {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )


    ######### 2 — Spend vs ROAS + CTR Heatmap #########

    col3, col4 = st.columns(2)

    # --- Spend vs ROAS ---
    if "scatter_spend_roas" in figs:
        with col3:
            st.subheader("Spend vs ROAS")
            st.pyplot(figs["scatter_spend_roas"])

            insight = llm_chart_insight_single(client, df, "scatter_spend_roas")
            st.markdown(
                f"""
                <div style="background:#f7f7f7; padding:10px; 
                            border-left:4px solid #dc2626; 
                            border-radius:6px; margin-top:8px;">
                    <b>AI Insight:</b><br>
                    {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- CTR Heatmap ---
    if "ctr_heatmap" in figs:
        with col4:
            st.subheader("CTR Heatmap (Top Campaigns)")
            st.pyplot(figs["ctr_heatmap"])

            insight = llm_chart_insight_single(client, df, "ctr_heatmap")
            st.markdown(
                f"""
                <div style="background:#f7f7f7; padding:10px; 
                            border-left:4px solid #a855f7; 
                            border-radius:6px; margin-top:8px;">
                    <b>AI Insight:</b><br>
                    {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================
    # DETAILED ANALYSIS BUTTON
    # =========================================

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## 🤖 Ask AI Agent — Detailed Account Analysis")

    if st.button("Generate Detailed AI Analysis"):
        with st.spinner("Analyzing deeper patterns across campaigns & ads..."):
            result = llm_detailed(client, df)
            st.session_state["detailed_ai_data"] = result

    # Display Detailed JSON Response
    if "detailed_ai_data" in st.session_state:
        try:
            parsed = json.loads(st.session_state["detailed_ai_data"])
        except:
            st.error("Model returned invalid JSON. Raw output displayed:")
            st.write(st.session_state["detailed_ai_data"])
            return

        st.markdown("### Overall Summary")
        st.write(parsed["overall_summary"])

        st.markdown("### Priority Recommendations")

        color_map = {
            "HIGH": "#dc2626",
            "MEDIUM": "#d97706",
            "LOW": "#16a34a",
        }

        for blk in parsed["priority_blocks"]:
            st.markdown(
                f"""
                <div style="border-left:5px solid {color_map.get(blk['priority'],'#000')};
                            padding:10px; background:#fafafa; 
                            border-radius:6px; margin-bottom:12px;">
                    <h4 style="margin:0; color:{color_map.get(blk['priority'],'#000')}">
                        {blk['priority']} PRIORITY — {blk['title']}
                    </h4>
                    <p>{blk['details']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    ##############################################
    # PDF GENERATOR (NEW + CORRECT)
    ##############################################

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch

    def generate_full_pdf(df, figs, insights, filename="Meta_Report.pdf"):

        doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
        )

        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<b>Meta Ads Performance Report</b>", styles["Title"]))
        story.append(Spacer(1, 20))

        # KPI Summary
        story.append(Paragraph("<b>KPI Summary</b>", styles["Heading2"]))

        kpis = [
            f"Total Spend: ₹{df['spend'].sum():,.0f}",
            f"Total Revenue: ₹{df['revenue'].sum():,.0f}",
            f"Total Impressions: {df['impressions'].sum():,.0f}",
            f"Total Clicks: {df['clicks'].sum():,.0f}",
            f"Overall ROAS: {df['revenue'].sum() / df['spend'].sum():.2f}",
        ]

        for k in kpis:
            story.append(Paragraph(f"- {k}", styles["Normal"]))
        story.append(Spacer(1, 20))

        # Charts + Insights
        for name, fig in figs.items():

            story.append(Paragraph(f"<b>{name.replace('_',' ').title()}</b>", styles["Heading2"]))

            img_path = f"{name}.png"
            fig.savefig(img_path, dpi=140, bbox_inches="tight")

            story.append(Image(img_path, width=6.2 * inch, height=3.2 * inch))
            story.append(Spacer(1, 10))

            if name in insights:
                story.append(Paragraph(
                    f"<b>AI Insight:</b><br/>{insights[name].replace(chr(10),'<br/>')}",
                    styles["BodyText"]
                ))
                story.append(Spacer(1, 14))

            story.append(PageBreak())

        doc.build(story)
        return filename
    ##############################################
    # DOWNLOAD PDF BUTTON
    ##############################################

    if st.button("Generate Full PDF Report"):
        with st.spinner("Generating PDF..."):
            insights = {
                key: llm_chart_insight_single(client, df, key)
                for key in figs.keys()
            }

            pdf_path = generate_full_pdf(df, figs, insights)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    data=f,
                    file_name="Meta_Report.pdf",
                    mime="application/pdf"
                )
##############################################
# CHAT PAGE
##############################################

def render_chat(df, client):

    st.title("Chat with Your Meta AI Agent")

    # Initialize history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Show previous messages
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about ROAS, campaigns, scaling, pauses…")

    if user_input:
        # Add user message
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):

                # Build a compact dataset for answer
                compact = {
                    "campaign_summary": df.groupby("campaign_name")
                        .agg({"spend":"sum","revenue":"sum","roas":"mean"})
                        .reset_index().to_dict(orient="records"),

                    "top_ads": df.sort_values("spend",ascending=False)
                        .head(100)
                        .to_dict(orient="records")
                }

                system = """
                You are a senior Meta Ads performance strategist.
                Answer clearly, concisely, and with specific recommendations.
                """

                res = client.responses.create(
                    model=CHAT_MODEL,
                    input=[
                        {"role":"system","content":system},
                        {"role":"user","content":f"Context:\n{json.dumps(compact,indent=2)}"},
                        {"role":"user","content":user_input}
                    ]
                )

                answer = res.output_text

                st.markdown(answer)

                # Save assistant reply
                st.session_state["chat_history"].append({"role": "assistant", "content": answer})
##############################################
# Check Password Function
##############################################
def check_password():
    """Password login with RareBox branding using APP_PASSWORD from .env."""

    # Read password from env (strip spaces/newlines just in case)
    env_pwd = os.getenv("APP_PASSWORD", "").strip()

    if not env_pwd:
        st.error("APP_PASSWORD is not set on the server. Please configure it in .env or environment variables.")
        st.stop()

    # ✅ If already authenticated in this session, allow through
    if st.session_state.get("authenticated"):
        return True

    # Nested callback to validate password when user types it
    def password_entered():
        """Callback: runs when user changes the password field."""
        if st.session_state.get("password_input", "") == env_pwd:
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    # Brand header (RareBox logo + text)
    encoded_logo = load_logo_base64()  # you already have this function

    st.markdown(
        f"""
        <div style="text-align:center; margin-top:80px; margin-bottom:24px;">
            <img src="data:image/png;base64,{encoded_logo}" 
                 style="height:60px; margin-bottom:12px;" />
            <h2 style="margin-bottom:4px;">Meta Ads Analyst</h2>
            <p style="color:#6b7280; font-size:14px; margin:0;">
                Private RareBox tool for internal Meta performance analysis.
            </p>
            <p style="color:#9ca3af; font-size:12px; margin-top:4px;">
                Access restricted. Please enter the password shared with you.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 🔑 Password field – no button, validation happens via on_change
    st.text_input(
        "Access password",
        type="password",
        key="password_input",          # stored in st.session_state["password_input"]
        on_change=password_entered     # sets st.session_state["authenticated"]
    )

    # If user tried and it failed, show error
    if "authenticated" in st.session_state and st.session_state["authenticated"] is False:
        st.error("Incorrect password.")

    # ❗ Block the rest of the app until authenticated == True
    st.stop()


##############################################
# FOOTER WITH BASE64 LOGO
##############################################

def render_footer():
    encoded_logo = encode_image_base64(
        "RareBoxLogo.png"
    )

    footer_html = f"""
        <div class="footer-wrapper">
            <div class="footer-text">
                <span>Powered by</span>
                <img src="data:image/png;base64,{encoded_logo}" class="footer-logo"/>
                <span>· Developed by Arka</span>
            </div>
        </div>
    """

    st.markdown(footer_html, unsafe_allow_html=True)



##############################################
# MAIN NAVIGATION
##############################################

def sidebar_nav():
    st.sidebar.markdown("<div class='sidebar-title'>Meta Ads Analyst</div>", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state["page"] = "Upload"

    if st.sidebar.button("Upload File"):
        st.session_state["page"] = "Upload"
    if st.sidebar.button("Analysis Dashboard"):
        st.session_state["page"] = "Analysis"
    if st.sidebar.button("Chat with AI"):
        st.session_state["page"] = "Chat"

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    # 🔐 Logout section
    if st.session_state.get("authenticated"):
        st.sidebar.markdown(
            "<span style='font-size:12px; color:#9ca3af;'>Logged in to RareBox</span>",
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Logout"):
            # Clear auth + password + any heavy session data
            for key in ["authenticated", "password_input", "chat_history", "detailed_ai_data"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Streamlit will rerun automatically after button click
            # On rerun, check_password() will show login screen again

    return st.session_state["page"]



##############################################
# MAIN APP
##############################################

def main():

    st.set_page_config(
        page_title="Meta Ads Analyst",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject global CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # Password protection
    check_password()

    # Load OpenAI client
    client = get_client()

    # Navigation
    page = sidebar_nav()

    # ------------------------ UPLOAD PAGE ------------------------
    if page == "Upload":
        st.title("Upload Meta Ads Report")

        file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

        if file:
            df = load_clean_data(file)
            st.session_state["df"] = df
            st.dataframe(df.head(), use_container_width=True)
            st.success("File processed successfully. Go to Dashboard.")

        render_footer()


    # ------------------------ DASHBOARD PAGE ------------------------
    elif page == "Analysis":

        if "df" not in st.session_state:
            st.warning("Please upload a file first.")
            render_footer()
            return

        df = st.session_state["df"]
        render_dashboard(df, client)
        render_footer()


    # ------------------------ CHAT PAGE ------------------------
    elif page == "Chat":

        if "df" not in st.session_state:
            st.warning("Upload a file first to enable chat.")
            render_footer()
            return

        df = st.session_state["df"]
        render_chat(df, client)
        render_footer()


##############################################
# RUN
##############################################

if __name__ == "__main__":
    main()
