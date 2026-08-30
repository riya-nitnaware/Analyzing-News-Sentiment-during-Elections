"""
Election News Sentiment Analyzer — Main Application
Professional Streamlit UI with full error handling.
No technical errors are ever shown to users.
"""
import os
import logging
import streamlit as st
import pandas as pd
from datetime import datetime

# ── Logging (internal only, never shown in UI) ────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Streamlit page config MUST be first ──────────────────────────────────────
st.set_page_config(
    page_title="Election NLP — Sentiment Analyzer",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Imports (after page config) ───────────────────────────────────────────────
try:
    from auth import render_login, render_signup, check_auth, logout
    from nlp_model import train_model, predict_sentiment
    from history_manager import save_analysis, get_user_history, delete_analysis
    from pdf_generator import generate_pdf_report
    from visualizations import plot_sentiment_distribution, plot_sentiment_trend
except Exception as _import_err:
    logger.exception("Critical import error")
    st.error("The application failed to start. Please contact support.")
    st.stop()

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "news_data.csv")

# ──────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "page": "Landing",
        "authenticated": False,
        "current_user": {},
        "analysis_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _username():
    return st.session_state.get("current_user", {}).get("username", "guest")


def _user_display_name():
    return st.session_state.get("current_user", {}).get("name", "User")


# ──────────────────────────────────────────────────────────────────────────────
# Friendly error helpers — NEVER show tracebacks
# ──────────────────────────────────────────────────────────────────────────────
def _safe_error(user_msg: str, exc: Exception = None):
    if exc:
        logger.exception(user_msg)
    st.error(f"⚠️ {user_msg}")


# ──────────────────────────────────────────────────────────────────────────────
# PDF download helper — generates lazily on click, never on render
# ──────────────────────────────────────────────────────────────────────────────
def _pdf_download_button(label: str, record: dict, filename: str, key: str):
    """Render a download button that lazily generates the PDF only when clicked."""
    col = st.empty()
    if col.button(label, key=key, use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_report(record)
            except Exception:
                logger.exception("PDF generation unexpected error")
                pdf_bytes = None

        if pdf_bytes:
            # Replace the button with a real download button
            col.download_button(
                label="📄 Click to Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"{key}_dl",
                use_container_width=True
            )
        else:
            st.error("⚠️ Unable to generate the PDF report right now. Please try again.")


# ──────────────────────────────────────────────────────────────────────────────
# Main router
# ──────────────────────────────────────────────────────────────────────────────
def main():
    _init_state()

    page = st.session_state["page"]

    # Public pages (no auth required)
    if page == "Landing":
        _render_landing()
        return
    if page == "Login":
        _render_login_page()
        return
    if page == "Signup":
        _render_signup_page()
        return

    # Auth gate
    if not check_auth():
        st.session_state["page"] = "Landing"
        st.rerun()

    # Load model once (cached by Streamlit)
    try:
        with st.spinner("Loading NLP model…"):
            vectorizer, model, df, metrics = train_model(DATA_PATH)
        if vectorizer is None:
            raise RuntimeError(metrics.get("error", "Model could not be loaded."))
    except Exception as e:
        logger.exception("Model loading failed")
        st.error("⚠️ The NLP model could not be loaded. Please make sure the dataset file exists.")
        return

    _render_sidebar()
    _render_page(vectorizer, model, df, metrics)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
_NAV_PAGES = [
    ("🏠", "Dashboard"),
    ("📰", "Analyze News"),
    ("📊", "Analysis Dashboard"),
    ("🕘", "Analysis History"),
    ("📥", "Report Center"),
    ("🔄", "Compare Analyses"),
    ("ℹ️", "About Project"),
]

def _render_sidebar():
    st.sidebar.markdown(
        "<h2 style='text-align:center; color:#1e3a8a;'>🗳️ Election NLP</h2>",
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    for icon, label in _NAV_PAGES:
        if st.sidebar.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state["analysis_result"] = None   # clear result when navigating away
            st.session_state["active_page"] = label
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**👤 {_user_display_name()}**<br>"
        f"<span style='color:#64748b; font-size:0.85rem;'>{_username()}</span>",
        unsafe_allow_html=True
    )
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="nav_logout"):
        try:
            logout()
        except Exception:
            logger.exception("Logout error")
            st.session_state.clear()
            st.rerun()


def _get_active_page() -> str:
    return st.session_state.get("active_page", "Dashboard")


def _render_page(vectorizer, model, df, metrics):
    page = _get_active_page()
    try:
        if page == "Dashboard":
            _render_dashboard()
        elif page == "Analyze News":
            _render_analyze_news(vectorizer, model)
        elif page == "Analysis Dashboard":
            _render_analysis_dashboard()
        elif page == "Analysis History":
            _render_history()
        elif page == "Report Center":
            _render_report_center()
        elif page == "Compare Analyses":
            _render_compare()
        elif page == "About Project":
            _render_about()
        else:
            _render_dashboard()
    except Exception:
        logger.exception(f"Page render error for page: {page}")
        st.error("⚠️ Something went wrong loading this page. Please try again.")


# ──────────────────────────────────────────────────────────────────────────────
# Public pages
# ──────────────────────────────────────────────────────────────────────────────
def _render_landing():
    st.markdown(
        "<h1 style='text-align:center; color:#1e3a8a; font-size:2.8rem;'>"
        "🗳️ Analyzing News Sentiment During Elections</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center; color:#475569; font-weight:400;'>"
        "Understand the tone and sentiment of election-related news using "
        "Natural Language Processing and Machine Learning.</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sign In", type="primary", use_container_width=True):
                st.session_state["page"] = "Login"
                st.rerun()
        with c2:
            if st.button("Create Account", use_container_width=True):
                st.session_state["page"] = "Signup"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "Sentiment predictions are generated by an NLP model and should not be "
            "interpreted as political advice or factual verification."
        )

    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>How It Works</h2>", unsafe_allow_html=True)

    steps = [
        ("01", "Collect News",       "Gather election articles"),
        ("02", "Preprocess Text",    "Clean and normalize text"),
        ("03", "Extract Features",   "Apply TF-IDF vectorization"),
        ("04", "Classify Sentiment", "Logistic Regression model"),
        ("05", "View Results",       "Dashboard & reports"),
    ]
    cols = st.columns(5)
    for col_obj, (num, title, desc) in zip(cols, steps):
        with col_obj:
            st.markdown(
                f"<div style='text-align:center; padding:14px; background:#f8fafc; "
                f"border-radius:8px; border:1px solid #e2e8f0;'>"
                f"<h3 style='color:#1e3a8a; margin:0;'>{num}</h3>"
                f"<strong>{title}</strong><br>"
                f"<span style='font-size:0.8rem; color:#64748b;'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True
            )


def _render_login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        try:
            render_login()
        except Exception:
            logger.exception("Login render error")
            st.error("⚠️ Something went wrong. Please refresh and try again.")


def _render_signup_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        try:
            render_signup()
        except Exception:
            logger.exception("Signup render error")
            st.error("⚠️ Something went wrong. Please refresh and try again.")


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def _render_dashboard():
    st.title("Election News Sentiment Dashboard")
    st.markdown(
        "<p style='color:#64748b;'>Analyze, track and understand the sentiment "
        "of election-related news.</p>",
        unsafe_allow_html=True
    )

    try:
        history = get_user_history(_username())
    except Exception:
        logger.exception("Could not load history for dashboard")
        history = []

    if not history:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info(
            "**Welcome to Election NLP!**\n\n"
            "You haven't analyzed any news yet. Click **📰 Analyze News** in the sidebar to get started."
        )
        return

    total = len(history)
    pos   = sum(1 for h in history if h.get("sentiment") == "Positive")
    neu   = sum(1 for h in history if h.get("sentiment") == "Neutral")
    neg   = sum(1 for h in history if h.get("sentiment") == "Negative")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("TOTAL ANALYSES", total)
    with c2: st.metric("POSITIVE", pos)
    with c3: st.metric("NEUTRAL", neu)
    with c4: st.metric("NEGATIVE", neg)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("**📰 Analyze News**\n\nAnalyze a headline or article for sentiment.")
    with q2:
        st.info("**📊 Analysis Dashboard**\n\nExplore your personal analysis trends.")
    with q3:
        st.info("**📥 Report Center**\n\nDownload professional PDF reports.")


# ──────────────────────────────────────────────────────────────────────────────
# Analyze News + Result
# ──────────────────────────────────────────────────────────────────────────────
def _render_analyze_news(vectorizer, model):
    # If we already have a pending result, show it
    if st.session_state.get("analysis_result"):
        _render_sentiment_result()
        return

    st.title("Analyze Election News")
    st.markdown(
        "<p style='color:#64748b;'>Enter a headline or article to understand its "
        "sentiment using Natural Language Processing.</p>",
        unsafe_allow_html=True
    )

    with st.form("analyze_form", clear_on_submit=False):
        st.markdown("**NEWS HEADLINE**")
        headline = st.text_input(
            "Headline", label_visibility="collapsed",
            placeholder="Enter the news headline…"
        )
        st.markdown("**NEWS ARTICLE**")
        article = st.text_area(
            "Article", label_visibility="collapsed",
            placeholder="Paste the complete news article here…",
            height=200
        )

        fc1, fc2, _ = st.columns([1, 1, 3])
        with fc1:
            submit = st.form_submit_button("Analyze Sentiment →", type="primary")
        with fc2:
            clear = st.form_submit_button("Clear")

    if clear:
        st.session_state["analysis_result"] = None
        st.rerun()

    if submit:
        full_text = f"{headline} {article}".strip()
        if not full_text:
            st.error("Please enter a headline or news article to analyze.")
            return

        with st.spinner("Analyzing sentiment using NLP model…"):
            try:
                prediction, confidence = predict_sentiment(full_text, vectorizer, model)

                probs = {}
                if hasattr(model, "predict_proba"):
                    features  = vectorizer.transform([full_text])
                    prob_arr  = model.predict_proba(features)[0]
                    for cls, p in zip(model.classes_, prob_arr):
                        probs[cls] = float(p)

                st.session_state["analysis_result"] = {
                    "headline":     headline,
                    "article":      article,
                    "sentiment":    prediction,
                    "confidence":   float(confidence),
                    "probabilities": probs,
                    "word_count":   len(full_text.split()),
                    "date":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.rerun()

            except Exception:
                logger.exception("Prediction failed")
                st.error("⚠️ Unable to analyze this news article. Please try again.")


def _render_sentiment_result():
    result = st.session_state["analysis_result"]
    if not result:
        return

    sentiment  = result.get("sentiment", "Unknown")
    confidence = result.get("confidence", 0.0)
    headline   = result.get("headline", "")
    article    = result.get("article", "")
    word_count = result.get("word_count", 0)
    date_str   = result.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    probs      = result.get("probabilities", {})

    # ── Colour scheme ─────────────────────────────────────────────────────────
    if sentiment == "Positive":
        color, icon, bg = "#166534", "🟢", "#f0fdf4"
    elif sentiment == "Negative":
        color, icon, bg = "#991b1b", "🔴", "#fef2f2"
    else:
        color, icon, bg = "#334155", "⚪", "#f8fafc"

    st.title("Sentiment Analysis Result")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main result card ──────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{bg}; border:2px solid {color}20; border-radius:16px; "
        f"padding:40px; text-align:center; margin-bottom:20px;'>"
        f"<div style='font-size:1rem; color:#64748b; margin-bottom:10px;'>SENTIMENT</div>"
        f"<div style='font-size:3.5rem; font-weight:900; color:{color}; line-height:1;'>"
        f"{icon} {sentiment.upper()}</div>"
        f"<div style='font-size:2rem; font-weight:700; color:{color}; margin-top:12px;'>"
        f"{confidence:.1%}</div>"
        f"<div style='color:#64748b; font-size:0.9rem; margin-top:4px;'>Model Confidence</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── Probability bars ───────────────────────────────────────────────────────
    if probs:
        st.markdown("### Sentiment Probabilities")
        prob_colors = {"Positive": "#22c55e", "Neutral": "#94a3b8", "Negative": "#ef4444"}
        for cls in ["Positive", "Neutral", "Negative"]:
            p = probs.get(cls, 0.0)
            bar_html = (
                f"<div style='display:flex; align-items:center; margin-bottom:8px;'>"
                f"<div style='width:90px; font-weight:500;'>{cls}</div>"
                f"<div style='flex:1; background:#f1f5f9; border-radius:6px; height:22px; overflow:hidden;'>"
                f"<div style='width:{p*100:.1f}%; background:{prob_colors.get(cls,'#94a3b8')}; "
                f"height:100%; border-radius:6px;'></div></div>"
                f"<div style='width:55px; text-align:right; font-weight:600; color:#334155;'>{p:.1%}</div>"
                f"</div>"
            )
            st.markdown(bar_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Input text section ─────────────────────────────────────────────────────
    st.markdown("### News Analyzed")
    if headline:
        st.markdown(f"**Headline:** {headline}")
    if article:
        with st.expander("View Article"):
            st.write(article)

    st.markdown("---")

    # ── Details ───────────────────────────────────────────────────────────────
    st.markdown("### Analysis Details")
    d1, d2, d3 = st.columns(3)
    with d1: st.metric("Word Count", word_count)
    with d2: st.metric("Characters", len(f"{headline} {article}".strip()))
    with d3: st.metric("Analyzed On", date_str.split(" ")[0])

    st.markdown("---")

    # ── Action buttons ─────────────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("💾 Save Analysis", use_container_width=True):
            try:
                save_analysis(
                    _username(),
                    headline, article,
                    sentiment, confidence,
                    probs, word_count
                )
                st.success("Analysis saved successfully.")
            except Exception:
                logger.exception("Save analysis failed")
                st.error("⚠️ Unable to save your analysis. Please try again.")

    with b2:
        # Lazy PDF: only generate when this button is explicitly clicked
        if st.button("📥 Download Report", use_container_width=True, key="dl_result"):
            with st.spinner("Generating PDF report…"):
                try:
                    pdf_bytes = generate_pdf_report(result)
                except Exception:
                    logger.exception("PDF generation error from result page")
                    pdf_bytes = None

            if pdf_bytes:
                safe_date = date_str.split(" ")[0]
                st.download_button(
                    label="📄 Click to Download",
                    data=pdf_bytes,
                    file_name=f"Election_NLP_Report_{safe_date}.pdf",
                    mime="application/pdf",
                    key="dl_result_actual",
                    use_container_width=True
                )
            else:
                st.error("⚠️ Unable to generate the PDF report right now. Please try again.")

    with b3:
        if st.button("🔄 Analyze Another", use_container_width=True):
            st.session_state["analysis_result"] = None
            st.rerun()

    with b4:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state["analysis_result"] = None
            st.session_state["active_page"] = "Dashboard"
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Analysis History
# ──────────────────────────────────────────────────────────────────────────────
def _render_history():
    st.title("Analysis History")
    st.markdown(
        "<p style='color:#64748b;'>View and manage your previous sentiment analyses.</p>",
        unsafe_allow_html=True
    )

    try:
        history = get_user_history(_username())
    except Exception:
        logger.exception("History load error")
        st.error("⚠️ Unable to load your analysis history. Please try again.")
        return

    if not history:
        st.info("No saved analyses yet. Click **📰 Analyze News** in the sidebar to get started.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    with f1:
        search = st.text_input("Search history…", placeholder="Type a keyword…")
    with f2:
        filter_sent = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])

    st.markdown(f"**{len(history)} saved analyses**")

    for item in history:
        if filter_sent != "All" and item.get("sentiment") != filter_sent:
            continue
        text_blob = f"{item.get('headline','')} {item.get('article','')}"
        if search and search.lower() not in text_blob.lower():
            continue

        sent      = item.get("sentiment", "Unknown")
        icon      = "🟢" if sent == "Positive" else "🔴" if sent == "Negative" else "⚪"
        conf      = item.get("confidence", 0.0)
        date_disp = item.get("date", "")

        st.markdown(
            f"<div style='background:white; border:1px solid #e2e8f0; border-radius:10px; "
            f"padding:18px; margin-bottom:12px;'>"
            f"<div style='font-size:0.85rem; color:#64748b;'>📅 {date_disp}</div>"
            f"<h4 style='margin:6px 0;'>{item.get('headline') or '(no headline)'}</h4>"
            f"<div style='font-weight:600;'>{icon} {sent.upper()}"
            f"<span style='font-weight:400; color:#64748b; margin-left:12px;'>"
            f"Confidence: {conf:.1%}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        hb1, hb2, _ = st.columns([1, 1, 4])
        with hb1:
            if st.button("🗑️ Delete", key=f"del_{item['id']}", use_container_width=True):
                try:
                    delete_analysis(_username(), item["id"])
                    st.rerun()
                except Exception:
                    logger.exception("Delete error")
                    st.error("⚠️ Unable to delete this analysis. Please try again.")
        with hb2:
            if st.button("📥 Download PDF", key=f"dl_{item['id']}", use_container_width=True):
                with st.spinner("Generating…"):
                    try:
                        pdf_bytes = generate_pdf_report(item)
                    except Exception:
                        logger.exception("History PDF error")
                        pdf_bytes = None

                if pdf_bytes:
                    st.download_button(
                        "📄 Download",
                        data=pdf_bytes,
                        file_name=f"Election_NLP_Report_{item['id'][:8]}.pdf",
                        mime="application/pdf",
                        key=f"dl_actual_{item['id']}"
                    )
                else:
                    st.error("⚠️ Unable to generate the PDF report right now. Please try again.")


# ──────────────────────────────────────────────────────────────────────────────
# Report Center
# ──────────────────────────────────────────────────────────────────────────────
def _render_report_center():
    st.title("Report Center")
    st.markdown(
        "<p style='color:#64748b;'>Generate and download reports from your saved analyses.</p>",
        unsafe_allow_html=True
    )

    try:
        history = get_user_history(_username())
    except Exception:
        logger.exception("Report center history load error")
        st.error("⚠️ Unable to load your reports. Please try again.")
        return

    if not history:
        st.info("No saved analyses yet.")
        return

    st.metric("Total Reports Available", len(history))
    st.markdown("---")

    # Summary table
    try:
        rows = []
        for item in history:
            rows.append({
                "Date":       item.get("date", ""),
                "Headline":   (item.get("headline") or "(no headline)")[:60],
                "Sentiment":  item.get("sentiment", ""),
                "Confidence": f"{item.get('confidence', 0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception:
        logger.exception("Report table render error")

    st.markdown("---")
    st.markdown("### Download Individual Reports")

    for item in history:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(
                f"**{item.get('date','')}** — "
                f"{item.get('headline') or '(no headline)'} "
                f"({item.get('sentiment','')})"
            )
        with c2:
            if st.button("📥 Download PDF", key=f"rep_dl_{item['id']}", use_container_width=True):
                with st.spinner("Generating…"):
                    try:
                        pdf_bytes = generate_pdf_report(item)
                    except Exception:
                        logger.exception("Report center PDF error")
                        pdf_bytes = None

                if pdf_bytes:
                    st.download_button(
                        "📄 Download",
                        data=pdf_bytes,
                        file_name=f"Election_NLP_Report_{item['id'][:8]}.pdf",
                        mime="application/pdf",
                        key=f"rep_actual_{item['id']}"
                    )
                else:
                    st.error("⚠️ Unable to generate the PDF report right now. Please try again.")

    # CSV download
    st.markdown("---")
    st.markdown("### Download Full History as CSV")
    try:
        csv_df = pd.DataFrame(history)
        csv_cols = [c for c in ["date", "headline", "sentiment", "confidence", "word_count"] if c in csv_df.columns]
        csv_data = csv_df[csv_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "📊 Download CSV",
            data=csv_data,
            file_name="Election_NLP_History.csv",
            mime="text/csv"
        )
    except Exception:
        logger.exception("CSV generation error")
        st.error("⚠️ Unable to generate the CSV file right now.")


# ──────────────────────────────────────────────────────────────────────────────
# Compare Analyses
# ──────────────────────────────────────────────────────────────────────────────
def _render_compare():
    st.title("Compare Analyses")
    st.markdown(
        "<p style='color:#64748b;'>Compare two saved analyses side by side.</p>",
        unsafe_allow_html=True
    )

    try:
        history = get_user_history(_username())
    except Exception:
        logger.exception("Compare history load error")
        st.error("⚠️ Unable to load your analyses. Please try again.")
        return

    if len(history) < 2:
        st.warning("You need at least **2 saved analyses** to use this feature.")
        return

    options = {
        f"{item['date']} — {(item.get('headline') or item.get('article','')[:30])[:50]}": item
        for item in history
    }
    keys = list(options.keys())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Analysis A**")
        sel_a = st.selectbox("Select Analysis A", keys, key="cmp_a", label_visibility="collapsed")
    with c2:
        st.markdown("**Analysis B**")
        sel_b = st.selectbox("Select Analysis B", keys, index=min(1, len(keys)-1), key="cmp_b", label_visibility="collapsed")

    item_a = options[sel_a]
    item_b = options[sel_b]

    st.markdown("---")
    st.markdown("### Sentiment Comparison")

    hdr, col_a, col_b = st.columns([1.5, 1, 1])
    with hdr:
        st.markdown("**Metric**")
    with col_a:
        st.markdown("**Analysis A**")
    with col_b:
        st.markdown("**Analysis B**")

    rows = [
        ("Sentiment",   item_a.get("sentiment",""),                   item_b.get("sentiment","")),
        ("Confidence",  f"{item_a.get('confidence',0):.1%}",          f"{item_b.get('confidence',0):.1%}"),
        ("Word Count",  str(item_a.get("word_count",0)),              str(item_b.get("word_count",0))),
        ("Date",        item_a.get("date","").split(" ")[0],          item_b.get("date","").split(" ")[0]),
    ]
    for label, va, vb in rows:
        hdr, col_a, col_b = st.columns([1.5, 1, 1])
        with hdr: st.write(f"**{label}**")
        with col_a: st.write(va)
        with col_b: st.write(vb)

    st.markdown("---")
    st.markdown(
        "> Comparison helps you understand how the sentiment of different election-related "
        "news articles differs over time."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Analysis Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def _render_analysis_dashboard():
    st.title("Analysis Dashboard")
    st.markdown(
        "<p style='color:#64748b;'>Deeper insights into your personal analysis history.</p>",
        unsafe_allow_html=True
    )

    try:
        history = get_user_history(_username())
    except Exception:
        logger.exception("Analysis dashboard load error")
        st.error("⚠️ Unable to load your analysis data. Please try again.")
        return

    if not history:
        st.info("No analyses yet. Start by clicking **📰 Analyze News** in the sidebar.")
        return

    df = pd.DataFrame(history)
    total    = len(df)
    pos      = len(df[df["sentiment"] == "Positive"])
    neu      = len(df[df["sentiment"] == "Neutral"])
    neg      = len(df[df["sentiment"] == "Negative"])
    avg_conf = df["confidence"].mean() if "confidence" in df else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Analyses", total)
    with c2: st.metric("Positive %",     f"{pos/total:.1%}")
    with c3: st.metric("Neutral %",      f"{neu/total:.1%}")
    with c4: st.metric("Negative %",     f"{neg/total:.1%}")
    with c5: st.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.markdown("---")

    chart_a, chart_b = st.columns(2)
    with chart_a:
        try:
            st.markdown(
                "<div style='background:white; padding:15px; border-radius:10px; border:1px solid #e2e8f0;'>",
                unsafe_allow_html=True
            )
            fig = plot_sentiment_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            logger.exception("Distribution chart error")
            st.info("Chart unavailable.")

    with chart_b:
        try:
            st.markdown(
                "<div style='background:white; padding:15px; border-radius:10px; border:1px solid #e2e8f0;'>",
                unsafe_allow_html=True
            )
            fig2 = plot_sentiment_trend(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough date information for trend chart.")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            logger.exception("Trend chart error")
            st.info("Chart unavailable.")


# ──────────────────────────────────────────────────────────────────────────────
# About Project
# ──────────────────────────────────────────────────────────────────────────────
def _render_about():
    st.title("About Election NLP")
    st.markdown("---")

    st.markdown("### Project Overview")
    st.write(
        "**Analyzing News Sentiment During Elections** is an NLP-based web application "
        "designed to classify the sentiment expressed in election-related news articles "
        "as Positive, Neutral, or Negative."
    )

    st.markdown("### Objective")
    st.write(
        "To understand the tone and sentiment of election-related news using Natural "
        "Language Processing and Machine Learning — without predicting election outcomes "
        "or voter behavior."
    )

    st.markdown("### NLP Pipeline")
    st.markdown(
        "```\n"
        "News Headline / Article\n"
        "        ↓\n"
        "Text Preprocessing\n"
        "(lowercase → URL removal → punctuation → tokenize → stopwords → lemmatize)\n"
        "        ↓\n"
        "TF-IDF Feature Extraction\n"
        "(ngram_range=(1,2), min_df=2, sublinear_tf=True)\n"
        "        ↓\n"
        "Machine Learning Model\n"
        "(Logistic Regression, max_iter=1000)\n"
        "        ↓\n"
        "Sentiment Classification\n"
        "(Positive / Neutral / Negative)\n"
        "        ↓\n"
        "Confidence Score + Analysis History + PDF Report\n"
        "```"
    )

    st.markdown("### Technology Used")
    st.markdown(
        "| Component | Technology |\n"
        "|---|---|\n"
        "| Language | Python 3 |\n"
        "| Frontend | Streamlit |\n"
        "| NLP | NLTK (preprocessing) |\n"
        "| ML | Scikit-learn — Logistic Regression |\n"
        "| Features | TF-IDF Vectorizer |\n"
        "| Visualizations | Plotly, WordCloud, Matplotlib |\n"
        "| Reports | fpdf2 |\n"
        "| Data | Pandas |"
    )

    st.markdown("### Training Dataset Overview")
    st.info(
        "The model was trained on a curated dataset of **110 election-related news articles**.\n\n"
        "- Positive: 44 articles\n"
        "- Neutral: 48 articles\n"
        "- Negative: 18 articles\n\n"
        "These statistics represent the training dataset, **not** the user's personal analysis history."
    )

    st.markdown("### Limitations")
    st.warning(
        "This system analyzes textual sentiment only. It does not determine factual accuracy, "
        "political fairness, bias, or truthfulness of any news article. "
        "Predictions are model-generated estimates and should not be used as political advice."
    )

    st.markdown("### Future Scope")
    st.markdown(
        "- Real-time news API integration\n"
        "- Multi-language support\n"
        "- Deep learning models (BERT/RoBERTa)\n"
        "- Named entity recognition for political figures\n"
        "- Topic modeling for election themes"
    )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
