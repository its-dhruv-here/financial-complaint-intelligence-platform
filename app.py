"""
Financial Complaint Intelligence Platform — Streamlit Application.

Enterprise-grade complaint routing and analytics dashboard backed by a
serialised scikit-learn pipeline.

Usage:
    streamlit run app.py
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.preprocessor import TextPreprocessor  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Complaint Intelligence Platform",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG         = "#0E1117"
SURFACE    = "#161B22"
CARD       = "#1C2333"
BORDER     = "#30363D"
ACCENT     = "#58A6FF"
ACCENT2    = "#3FB950"
WARN       = "#D29922"
DANGER     = "#F85149"
TEXT       = "#E6EDF3"
TEXT_DIM   = "#8B949E"
TEXT_FAINT = "#484F58"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

/* ---- main area ---- */
.stApp {{
    background: {BG};
}}
.main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}}

/* ---- sidebar compact ---- */
section[data-testid="stSidebar"] {{
    background: {SURFACE};
    width: 220px !important;
    min-width: 220px !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
    padding: 1rem 0.75rem;
}}

/* ---- tabs ---- */
button[data-baseweb="tab"] {{
    color: {TEXT_DIM} !important;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    padding: 8px 16px !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom-color: {ACCENT} !important;
}}

/* ---- KPI card ---- */
.kpi {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 14px 12px 12px;
    text-align: center;
}}
.kpi-val {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1.15;
}}
.kpi-lbl {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}}

/* ---- result card ---- */
.res-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 3px solid {ACCENT};
    border-radius: 6px;
    padding: 18px 16px;
}}

/* ---- chip / tag ---- */
.chip {{
    display: inline-block;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 10px;
    margin: 3px 4px 3px 0;
    font-size: 0.78rem;
    font-weight: 500;
    color: {TEXT};
}}
.chip-pos {{
    border-color: {ACCENT2};
    color: {ACCENT2};
}}
.chip-neg {{
    border-color: {DANGER};
    color: {DANGER};
}}

/* ---- misc ---- */
.section-hdr {{
    font-size: 0.75rem;
    font-weight: 700;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
hr {{
    border-color: {BORDER} !important;
    margin: 12px 0 !important;
}}

/* hide default header/footer */
header[data-testid="stHeader"] {{
    background: {BG};
}}

/* button overrides */
.stButton > button {{
    background: {ACCENT};
    color: #0D1117;
    font-weight: 600;
    border: none;
    border-radius: 4px;
}}
.stButton > button:hover {{
    background: #79C0FF;
}}

/* dataframe */
.stDataFrame {{
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* text area */
textarea {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 4px !important;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(os.path.join("models", "best_model_pipeline.pkl"))
    except Exception as e:
        logger.error("Pipeline load failed: %s", e)
        return None


@st.cache_data
def load_metrics() -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join("reports", "metrics.json")) as f:
            return json.load(f)
    except Exception:
        return None


pipeline = load_pipeline()
metrics = load_metrics()

# ---------------------------------------------------------------------------
# Compact sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<p style='font-size:0.9rem;font-weight:700;color:{TEXT};margin:0;'>📋 CIP</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.65rem;color:{TEXT_DIM};margin-top:2px;'>Complaint Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    if metrics:
        bm = metrics.get("best_model", {})
        ds = metrics.get("dataset", {})
        st.markdown(f"<p class='section-hdr'>Production Model</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.78rem;color:{TEXT};margin:0;'>{bm.get('Model','—')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.72rem;color:{TEXT_DIM};margin:2px 0 12px;'>{bm.get('Accuracy',0)*100:.1f}% acc · {bm.get('Weighted F1',0)*100:.1f}% F1</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='section-hdr'>Dataset</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.72rem;color:{TEXT_DIM};margin:0;'>{ds.get('Total Rows',0):,} records · {ds.get('Classes',0)} classes</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<p style='font-size:0.62rem;color:{TEXT_FAINT};'>v1.0 · scikit-learn · Streamlit</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:10px; margin-bottom:4px;">
    <span style="font-size:1.25rem; font-weight:800; color:{TEXT};">Financial Complaint Intelligence Platform</span>
    <span style="font-size:0.72rem; color:{TEXT_DIM}; font-weight:500;">NLP-Powered Routing Engine</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Executive KPI row
# ---------------------------------------------------------------------------
if metrics:
    bm = metrics["best_model"]
    ds = metrics["dataset"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, f"{bm.get('Accuracy',0)*100:.1f}%", "Accuracy"),
        (k2, f"{bm.get('Weighted F1',0)*100:.1f}%", "Weighted F1"),
        (k3, f"{bm.get('Macro F1',0)*100:.1f}%", "Macro F1"),
        (k4, f"{ds.get('Classes',0)}", "Classes"),
        (k5, f"{ds.get('Total Rows',0):,}", "Training Records"),
        (k6, f"{ds.get('Avg Complaint Length',0):,}", "Avg Length (chars)"),
    ]
    for col, val, lbl in kpis:
        col.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_route, tab_analytics, tab_arch = st.tabs(["ROUTING ENGINE", "ANALYTICS DASHBOARD", "ARCHITECTURE & DATA"])


# =====================================================================
# TAB 1 — ROUTING ENGINE
# =====================================================================
with tab_route:
    # ---- Input row ----
    inp_col, act_col = st.columns([4, 1])

    with inp_col:
        examples = {
            "— select example —": "",
            "Credit Card Dispute": "I applied for a credit card with Macy's and was approved, but I was charged an annual fee that was never disclosed to me. I want a refund.",
            "Mortgage Issue": "My mortgage was transferred to Ocwen and they immediately claimed I missed a payment. I sent them proof of my escrow account statements but they are still threatening foreclosure.",
            "Bank Overdraft": "I checked my checking account balance and saw three overdraft fees of $35 each. I never authorised this and my debit card should have been declined.",
            "Debt Collection": "I keep getting calls from a recovery agency for a medical debt that I already paid off two years ago. They are threatening to ruin my credit.",
            "Student Loan": "My student loan servicer increased my monthly payment without notification. When I called they said it was due to a recalculation but refused to provide documentation.",
        }
        sel = st.selectbox("Test scenario", list(examples.keys()), label_visibility="collapsed")
        default = examples[sel] if sel != "— select example —" else ""
        user_text = st.text_area("complaint_input", value=default, height=110, label_visibility="collapsed", placeholder="Paste a consumer complaint here …")

    with act_col:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("▶ Classify", use_container_width=True)
        clear_btn = st.button("✕ Clear", use_container_width=True)
        if clear_btn:
            st.rerun()

    # ---- Inference ----
    if predict_btn:
        if not user_text.strip():
            st.error("Enter a complaint to classify.")
        elif pipeline is None:
            st.error("Model not loaded.")
        else:
            t0 = time.time()
            if hasattr(pipeline, "predict_proba"):
                probas = pipeline.predict_proba([user_text])[0]
            else:
                decisions = pipeline.decision_function([user_text])[0]
                # Convert to pseudo-probabilities using softmax
                e_x = np.exp(decisions - np.max(decisions))
                probas = e_x / e_x.sum()
            classes = pipeline.classes_
            latency_ms = (time.time() - t0) * 1000

            top_idx = np.argsort(probas)[::-1]
            top5_cls = classes[top_idx][:5]
            top5_prob = probas[top_idx][:5]
            primary = top5_cls[0]
            conf = top5_prob[0] * 100

            # confidence tier
            if conf >= 70:
                tier_label = "AUTO-ROUTE"
                tier_color = ACCENT2
                risk = "Low"
            elif conf >= 40:
                tier_label = "REVIEW"
                tier_color = WARN
                risk = "Medium"
            else:
                tier_label = "ESCALATE"
                tier_color = DANGER
                risk = "High"

            # ---- Result row ----
            r1, r2 = st.columns([2, 3])

            with r1:
                st.markdown(f"""
                <div class="res-card">
                    <div class="section-hdr">Routing Decision</div>
                    <div style="font-size:1.15rem; font-weight:700; color:{TEXT}; margin:6px 0 4px;">{primary}</div>
                    <div style="display:flex; gap:16px; margin-top:10px;">
                        <div>
                            <div class="kpi-lbl">Model Score</div>
                            <div style="font-size:1.1rem; font-weight:700; color:{ACCENT};">{conf:.1f}%</div>
                        </div>
                        <div>
                            <div class="kpi-lbl">Action</div>
                            <div style="font-size:0.85rem; font-weight:700; color:{tier_color};">{tier_label}</div>
                        </div>
                        <div>
                            <div class="kpi-lbl">Risk</div>
                            <div style="font-size:0.85rem; font-weight:600; color:{tier_color};">{risk}</div>
                        </div>
                        <div>
                            <div class="kpi-lbl">Latency</div>
                            <div style="font-size:0.85rem; font-weight:600; color:{TEXT_DIM};">{latency_ms:.0f}ms</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                df_bar = pd.DataFrame({"Department": top5_cls, "Probability": top5_prob * 100})
                fig = go.Figure(go.Bar(
                    y=df_bar["Department"],
                    x=df_bar["Probability"],
                    orientation="h",
                    marker_color=[ACCENT if i == 0 else TEXT_FAINT for i in range(len(df_bar))],
                    text=[f"{p:.1f}%" for p in df_bar["Probability"]],
                    textposition="outside",
                    textfont=dict(size=11, color=TEXT_DIM),
                ))
                fig.update_layout(
                    height=200,
                    margin=dict(l=0, r=40, t=4, b=4),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False, range=[0, max(df_bar["Probability"]) * 1.3]),
                    yaxis=dict(
                        categoryorder="total ascending",
                        tickfont=dict(size=11, color=TEXT_DIM),
                        automargin=True,
                    ),
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ---- Explainability ----
            st.markdown("---")

            prep = pipeline.named_steps["preprocessor"]
            tfidf = pipeline.named_steps["tfidf"]
            clf = pipeline.named_steps["clf"]

            clean = prep.transform(pd.Series([user_text]))[0]
            vec = tfidf.transform([clean])
            feat_names = np.array(tfidf.get_feature_names_out())
            nz = vec.nonzero()[1]

            if len(nz) > 0:
                cls_idx = int(np.where(clf.classes_ == primary)[0][0])
                coefs_cls = clf.coef_[cls_idx]
                doc_words = feat_names[nz]
                doc_coefs = coefs_cls[nz]
                order = np.argsort(doc_coefs)

                pos = [(doc_words[i], doc_coefs[i]) for i in order[::-1] if doc_coefs[i] > 0][:8]
                neg = [(doc_words[i], doc_coefs[i]) for i in order if doc_coefs[i] < 0][:5]

                e1, e2 = st.columns(2)

                with e1:
                    st.markdown(f'<div class="section-hdr">Supporting Evidence</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip chip-pos">{w}</span>' for w, _ in pos)
                    st.markdown(chips, unsafe_allow_html=True)
                    with st.expander("Coefficient details"):
                        for w, v in pos:
                            st.markdown(f"- `{w}` → **+{v:.3f}**")

                with e2:
                    st.markdown(f'<div class="section-hdr">Contradicting Evidence</div>', unsafe_allow_html=True)
                    if neg:
                        chips = "".join(f'<span class="chip chip-neg">{w}</span>' for w, _ in neg)
                        st.markdown(chips, unsafe_allow_html=True)
                        with st.expander("Coefficient details"):
                            for w, v in neg:
                                st.markdown(f"- `{w}` → **{v:.3f}**")
                    else:
                        st.markdown(f"<span style='font-size:0.8rem;color:{TEXT_DIM};'>No contradicting signals detected.</span>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:{SURFACE}; border:1px solid {BORDER}; border-radius:4px; padding:10px 14px; margin-top:12px; font-size:0.78rem; color:{TEXT_DIM};">
                    <strong style="color:{TEXT};">Why this routing?</strong>
                    The model identified vocabulary in the complaint that correlates strongly with <strong style="color:{ACCENT};">{primary}</strong>
                    in historical CFPB data.  Green keywords pushed the prediction toward this class; red keywords opposed it.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("No recognised vocabulary found.")


# =====================================================================
# TAB 2 — ANALYTICS DASHBOARD
# =====================================================================
with tab_analytics:
    if not metrics:
        st.info("Run `python run_training.py` to generate analytics data.")
    else:
        bm = metrics["best_model"]
        ds = metrics["dataset"]

        # ---- Metrics cards ----
        st.markdown(f'<div class="section-hdr">Production Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        metric_items = [
            (m1, "Accuracy",    bm.get("Accuracy", 0)),
            (m2, "Precision",   bm.get("Precision", 0)),
            (m3, "Recall",      bm.get("Recall", 0)),
            (m4, "Weighted F1", bm.get("Weighted F1", 0)),
            (m5, "Macro F1",    bm.get("Macro F1", 0)),
            (m6, "Classes",     None),
        ]
        for col, label, val in metric_items:
            if val is not None:
                display = f"{val*100:.1f}%"
            else:
                display = str(ds.get("Classes", 0))
            col.markdown(f'<div class="kpi"><div class="kpi-val">{display}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("")

        # ---- Dataset stats row ----
        st.markdown(f'<div class="section-hdr">Dataset Provenance & Statistics</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(f'<div class="kpi"><div class="kpi-val">~1.28M</div><div class="kpi-lbl">Raw CFPB Dataset</div></div>', unsafe_allow_html=True)
        p2.markdown(f'<div class="kpi"><div class="kpi-val">{ds.get("Total Rows", 0):,}</div><div class="kpi-lbl">Cleaned Records Used</div></div>', unsafe_allow_html=True)
        p3.markdown(f'<div class="kpi"><div class="kpi-val">{ds.get("Train Samples", 0):,}</div><div class="kpi-lbl">Train Split</div></div>', unsafe_allow_html=True)
        p4.markdown(f'<div class="kpi"><div class="kpi-val">{ds.get("Test Samples", 0):,}</div><div class="kpi-lbl">Test Split</div></div>', unsafe_allow_html=True)
        
        st.markdown("")
        s1, s2, s3 = st.columns([1, 1, 2])
        s1.markdown(f'<div class="kpi"><div class="kpi-val">{ds.get("Vocabulary Size", 0):,}</div><div class="kpi-lbl">TF-IDF Features</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="kpi"><div class="kpi-val">{ds.get("Avg Complaint Length", 0):,}</div><div class="kpi-lbl">Avg Complaint Length</div></div>', unsafe_allow_html=True)
        
        top_cats = ds.get("Top Categories", [])
        if top_cats:
            cat_list = ", ".join(top_cats)
            s3.markdown(f'<div class="kpi"><div class="kpi-val" style="font-size:1.1rem; padding-top:6px;">{cat_list}</div><div class="kpi-lbl">Top Categories</div></div>', unsafe_allow_html=True)


        st.markdown("---")

        # ---- Charts ----
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown(f'<div class="section-hdr">Class Distribution</div>', unsafe_allow_html=True)
            p = os.path.join("reports", "figures", "class_distribution.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
        with ch2:
            st.markdown(f'<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
            p = os.path.join("reports", "figures", "cm_linear_svm.png")
            # Also check for tuned variant
            p2 = os.path.join("reports", "figures", "cm_linear_svm_tuned.png")
            if os.path.exists(p2):
                st.image(p2, use_container_width=True)
            elif os.path.exists(p):
                st.image(p, use_container_width=True)

        st.markdown("---")

        # ---- Benchmark table ----
        st.markdown(f'<div class="section-hdr">Model Benchmark Comparison</div>', unsafe_allow_html=True)
        if "benchmarks" in metrics:
            df_b = pd.DataFrame(metrics["benchmarks"])
            st.dataframe(
                df_b.style
                    .highlight_max(subset=["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"], color="#1a3a2a")
                    .format({
                        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
                        "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}",
                        "Train Time (s)": "{:.2f}", "Inference Time (s)": "{:.3f}",
                    }),
                use_container_width=True, hide_index=True, height=210,
            )

        # ---- Benchmark charts ----
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown(f'<div class="section-hdr">Accuracy vs Weighted F1</div>', unsafe_allow_html=True)
            p = os.path.join("reports", "figures", "benchmark_comparison.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)
        with bc2:
            st.markdown(f'<div class="section-hdr">All Metrics Comparison</div>', unsafe_allow_html=True)
            p = os.path.join("reports", "figures", "benchmark_all_metrics.png")
            if os.path.exists(p):
                st.image(p, use_container_width=True)

        # ---- Tuning info ----
        if "tuning" in metrics:
            st.markdown("---")
            st.markdown(f'<div class="section-hdr">Hyperparameter Tuning</div>', unsafe_allow_html=True)
            t = metrics["tuning"]
            t1, t2, t3 = st.columns(3)
            t1.markdown(f'<div class="kpi"><div class="kpi-val">C = {t.get("best_C", "—")}</div><div class="kpi-lbl">Best Regularisation</div></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="kpi"><div class="kpi-val">{t.get("cv_weighted_f1",0)*100:.1f}%</div><div class="kpi-lbl">CV Weighted F1</div></div>', unsafe_allow_html=True)
            t3.markdown(f'<div class="kpi"><div class="kpi-val">{t.get("test_weighted_f1_after",0)*100:.1f}%</div><div class="kpi-lbl">Test Weighted F1</div></div>', unsafe_allow_html=True)

# =====================================================================
# TAB 3 — ARCHITECTURE & DATA
# =====================================================================
with tab_arch:
    st.markdown(f'<div class="section-hdr">Machine Learning Pipeline</div>', unsafe_allow_html=True)
    pipe_img = os.path.join("reports", "pipeline_diagram.png")
    if os.path.exists(pipe_img):
        st.image(pipe_img, use_container_width=False, width=800)
    else:
        st.info("Pipeline diagram not found. Run diagram generation script.")

    st.markdown("---")
    st.markdown(f'<div class="section-hdr">System Architecture</div>', unsafe_allow_html=True)
    arch_img = os.path.join("reports", "architecture_diagram.png")
    if os.path.exists(arch_img):
        st.image(arch_img, use_container_width=True)
    else:
        st.info("Architecture diagram not found.")
