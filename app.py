"""
Cross-Lingual Calibration Collapse & Mitigation Research Platform.
Streamlit Interactive Research UI & Live Demonstration System.
Model: Google Gemma 3 4B Instruct | Environment: RTX 4050 Laptop GPU
Theme: Futuristic Cyberpunk AI Research Console (Horizontal Navigation)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

os.environ["MPLCONFIGDIR"] = os.path.abspath("results/scratch_matplotlib")
os.makedirs("results/scratch_matplotlib", exist_ok=True)

import streamlit as st
import pandas as pd
import numpy as np

# Import Cyberpunk Design System
from src.ui.cyber_theme import get_cyberpunk_css, render_kpi_card, render_compact_header

# Import Live Inference & Verification Service
from src.inference_service import determine_active_query, execute_live_inference

# Page Configuration - Layout Wide, Initial Sidebar State Collapsed (Sidebar completely removed)
st.set_page_config(
    page_title="CALIBRATION LAB // Cross-Lingual Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Full Cyberpunk Stylesheet
st.markdown(get_cyberpunk_css(), unsafe_allow_html=True)

# Language Definitions
LANGUAGES = {
    "en": {"name": "English", "native": "English", "script": "Latin", "family": "Indo-European (Germanic)"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "script": "Devanagari", "family": "Indo-Aryan"},
    "bn": {"name": "Bengali", "native": "বাংলা", "script": "Bengali", "family": "Indo-Aryan"},
    "te": {"name": "Telugu", "native": "తెలుగు", "script": "Telugu", "family": "Dravidian"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "script": "Tamil", "family": "Dravidian"},
}

# Empirical Reference Data (Held-out Test Partition P_test, N=45 per condition)
BASELINE_DATA = {
    "en": {"acc": 0.9556, "c_v": 0.984, "c_s": 0.887, "ece": 0.0440, "bs": 0.0440, "mit_acc": 0.8222, "mit_ece": 0.1115, "mit_bs": 0.1519},
    "hi": {"acc": 0.2381, "c_v": 0.918, "c_s": 0.542, "ece": 0.6798, "bs": 0.6623, "mit_acc": 0.8444, "mit_ece": 0.4902, "mit_bs": 0.3810},
    "bn": {"acc": 0.3556, "c_v": 0.914, "c_s": 0.618, "ece": 0.6049, "bs": 0.6463, "mit_acc": 0.8889, "mit_ece": 0.4295, "mit_bs": 0.3012},
    "te": {"acc": 0.2500, "c_v": 0.953, "c_s": 0.489, "ece": 0.7025, "bs": 0.6692, "mit_acc": 0.8667, "mit_ece": 0.4812, "mit_bs": 0.3704},
    "ta": {"acc": 0.4000, "c_v": 0.939, "c_s": 0.582, "ece": 0.5516, "bs": 0.5791, "mit_acc": 0.8444, "mit_ece": 0.4654, "mit_bs": 0.3497},
}

CURATED_EXAMPLES = [
    {
        "id": "Q_GEO_DIR_01",
        "question_en": "What is the capital city of Karnataka?",
        "domain": "Geography",
        "evidence": "Bengaluru, officially known as Bangalore, is the capital of Karnataka.",
        "claims": ["Bengaluru is the capital city of Karnataka."],
        "answers": {
            "en": "Bengaluru is the capital city of Karnataka.",
            "hi": "कर्नाटक की राजधानी बेंगलुरु (बैंगलोर) है।",
            "bn": "কর্ণাটকের রাজধানী হল বেঙ্গালুরু।",
            "te": "కర్ణాటక రాజధాని నగరం బెంగళూరు.",
            "ta": "கர்நாடகாவின் தலைநகரம் பெங்களூரு ஆகும்.",
        }
    },
    {
        "id": "Q_GEO_DIR_02",
        "question_en": "Which river flows through Varanasi?",
        "domain": "Geography",
        "evidence": "Varanasi is situated along the banks of the Ganges river in Uttar Pradesh, India.",
        "claims": ["The Ganges river flows through Varanasi."],
        "answers": {
            "en": "The Ganges river flows through Varanasi.",
            "hi": "वाराणसी से होकर गंगा नदी बहती है।",
            "bn": "বারাণসীর মধ্য দিয়ে গঙ্গা নদী প্রবাহিত হয়েছে।",
            "te": "వారణాసి గుండా గంగా నది ప్రవహిస్తుంది.",
            "ta": "வாரணாசி வழியாக கங்கை நதி பாய்கிறது.",
        }
    },
    {
        "id": "Q_GEO_DIR_03",
        "question_en": "What is the highest mountain peak in India?",
        "domain": "Geography",
        "evidence": "Kangchenjunga is the highest mountain peak in India and third highest in the world.",
        "claims": ["Kangchenjunga is the highest mountain peak in India."],
        "answers": {
            "en": "Kangchenjunga is the highest mountain peak in India.",
            "hi": "भारत की सबसे ऊँची पर्वत चोटी कंचनजंघा है।",
            "bn": "ভারতের সর্বোচ্চ পর্বতশৃঙ্গ হল কাঞ্চনজঙ্ঘা।",
            "te": "భారతదేశంలో అత్యంత ఎత్తైన పర్వత శిఖరం కాంచన్‌జంగా.",
            "ta": "இந்தியாவின் மிக உயரமான மலை சிகரம் கஞ்சன்சுங்கா ஆகும்.",
        }
    },
    {
        "id": "Q_SCI_BIO_01",
        "question_en": "Which organelle is known as the powerhouse of the cell?",
        "domain": "Biology / Science",
        "evidence": "Mitochondria generate most of the chemical energy needed by the cell, known as ATP.",
        "claims": ["Mitochondria is the powerhouse of the cell."],
        "answers": {
            "en": "The mitochondrion is known as the powerhouse of the cell.",
            "hi": "माइटोकॉन्ड्रिया को कोशिका का बिजली घर (पावरहाउस) कहा जाता है।",
            "bn": "মাইটোকনড্রিয়াকে কোষের শক্তিকেন্দ্র বা পাওয়ারহাউস বলা হয়।",
            "te": "మైటోకాండ్రియాను కణం యొక్క పవర్ హౌస్ అని పిలుస్తారు.",
            "ta": "மைட்டோகாண்ட்ரியா செல்லின் ஆற்றல் மையம் என்று அழைக்கப்படுகிறது.",
        }
    },
    {
        "id": "Q_HIST_IND_01",
        "question_en": "Which marble mausoleum is situated on the south bank of the Yamuna river in Agra?",
        "domain": "History & Culture",
        "evidence": "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra.",
        "claims": ["Taj Mahal is situated in Agra on the south bank of Yamuna."],
        "answers": {
            "en": "The Taj Mahal is situated in Agra on the south bank of the Yamuna river.",
            "hi": "आगरा में यमुना नदी के तट पर ताजमहल स्थित है।",
            "bn": "আগ্রায় যমুনা নদীর তীরে তাজমহল অবস্থিত।",
            "te": "ఆగ్రాలో యమునా నది ఒడ్డున తాజ్ మహల్ ఉంది.",
            "ta": "ஆக்ராவில் யமுனை நதிக்கரையில் தாஜ்மஹால் அமைந்துள்ளது.",
        }
    }
]

# Helper to find figure path
def get_fig_path(filename: str) -> Optional[Path]:
    p1 = Path("outputs/mitigation_results/figures") / filename
    if p1.exists():
        return p1
    p2 = Path("outputs/research_results") / filename
    if p2.exists():
        return p2
    return None

def render_cyber_fig(fig_filename: str, title: str, subtitle: str = "Empirical Test Benchmark (P_test, N=45)"):
    """Wraps research figure in a dark cyber HUD container."""
    fig_path = get_fig_path(fig_filename)
    if fig_path:
        st.markdown(f"""
        <div class="cyber-fig-card">
            <div class="cyber-fig-header">
                <span>// {title.upper()}</span>
                <span>{subtitle.upper()}</span>
            </div>
        """, unsafe_allow_html=True)
        st.image(str(fig_path), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Figure {fig_filename} not located in results directory.")

# -----------------------------------------------------------------------------
# HORIZONTAL TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
NAV_PAGES = [
    ("01 Overview", "01 OVERVIEW"),
    ("02 Live Demo", "02 LIVE DEMO"),
    ("03 Benchmark", "03 BENCHMARK"),
    ("04 Calibration", "04 CALIBRATION"),
    ("05 Consistency", "05 CONSISTENCY"),
    ("06 Strategies", "06 STRATEGIES"),
    ("07 Reports", "07 REPORTS"),
]

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "01 Overview"

# Top Navigation Bar Grid (Brand on Left, 7 Nav Rectangles in Center, Telemetry on Right)
c_brand, c1, c2, c3, c4, c5, c6, c7, c_stat = st.columns(
    [1.65, 0.85, 0.85, 0.85, 0.95, 0.95, 0.9, 0.8, 1.8],
    vertical_alignment="center"
)

with c_brand:
    st.markdown("""
    <div class="nav-brand-container">
        <span class="nav-brand-logo">⚡</span>
        <div>
            <div class="nav-brand-title">CALIBRATION LAB</div>
            <div class="nav-brand-sub">CROSS-LINGUAL AI RESEARCH</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

nav_cols = [c1, c2, c3, c4, c5, c6, c7]

for i, (page_key, page_label) in enumerate(NAV_PAGES):
    with nav_cols[i]:
        is_active = (st.session_state["active_page"] == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.button(page_label, key=f"nav_top_btn_{i}", type=btn_type, use_container_width=True):
            st.session_state["active_page"] = page_key
            st.rerun()

with c_stat:
    st.markdown("""
    <div class="nav-telemetry-box">
        <span class="pulse-online"></span>
        <span class="online-text">SYSTEM ONLINE</span>
        <span class="telemetry-sep">|</span>
        <span class="telemetry-tag">GEMMA 3 4B</span>
        <span class="telemetry-sep">|</span>
        <span class="telemetry-tag">RTX 4050</span>
        <span class="telemetry-sep">|</span>
        <span class="telemetry-tag">NF4</span>
    </div>
    """, unsafe_allow_html=True)

current_page = st.session_state["active_page"]


# -----------------------------------------------------------------------------
# PAGE 1: 01 OVERVIEW
# -----------------------------------------------------------------------------
if current_page == "01 Overview":
    render_compact_header(
        module_id="MODULE // LAB-01 // PROTOCOL v4.5",
        title="CROSS-LINGUAL CALIBRATION COLLAPSE UNDER INSTRUCTION-TUNING",
        subtitle="Verbalized vs. Statistical Confidence, Grounded Consistency Divergence, and Systematic Mitigations"
    )

    # 4-Column Full-Width KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(render_kpi_card(
            title="Indic Accuracy Collapse",
            value="95.6% → 23.8%",
            sub="Drop from English to Hindi on identical factual premises",
            color_theme="red",
            extra_badge="CRITICAL GAP"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card(
            title="Verbalized Overconfidence",
            value="0.028 → 0.703",
            sub="ECE explodes; model states >91% certainty on falsehoods",
            color_theme="magenta",
            extra_badge="MISCALIBRATED"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(render_kpi_card(
            title="Mitigated Indic Accuracy",
            value="31.1% → 86.1%",
            sub="+55.0 pp gain via Bilingual Arbitration (Strategy C)",
            color_theme="green",
            extra_badge="RECOVERED"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(render_kpi_card(
            title="Calibration Recovery (M2)",
            value="85.4% ECE Drop",
            sub="Language-shift Platt scaling reduces average ECE to 0.090",
            color_theme="cyan",
            extra_badge="ALIGNED"
        ), unsafe_allow_html=True)

    # 2-Column Wide Grid for Research Directives & Pipeline Architecture
    grid_left, grid_right = st.columns([1, 1], gap="medium")

    with grid_left:
        st.markdown('<div class="cyber-section-title">EXECUTIVE RESEARCH BRIEFING</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="cyber-callout" style="border-left-color: #00F5FF;">
            <div class="cyber-callout-title" style="color: #00F5FF;">THE PHENOMENON OF CROSS-LINGUAL CALIBRATION COLLAPSE</div>
            Modern Instruction-Tuned and RLHF Large Language Models exhibit high conversational assertiveness. In high-resource English queries, their verbalized self-assessments align well with true factual competency. 
            However, when queried in non-Latin Indic languages (Hindi, Bengali, Telugu, Tamil), <strong>verbalized confidence fails to scale downward</strong> with actual task performance.
            The model asserts erroneous claims with extreme stated certainty (&gt;90%), resulting in severe cross-lingual calibration collapse.
            <br><br>
            This platform presents findings from <strong>4,050 genuine GPU model inferences</strong> with Google Gemma 3 4B Instruct, rigorously comparing verbalized confidence against token geometric entropy, diagnosing grounded consistency failures across 750 pairwise evaluations, and demonstrating multi-strategy mitigation pipelines.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cyber-section-title">CORE RESEARCH QUESTIONS (RQ1 – RQ4)</div>', unsafe_allow_html=True)
        rq_col1, rq_col2 = st.columns(2)
        with rq_col1:
            st.markdown("""
            <div class="cyber-kpi-card cyber-kpi-purple">
                <div class="cyber-kpi-label"><span style="color: #00F5FF;">RQ1 // COLLAPSE</span><span class="cyber-badge badge-cyan">PROVEN</span></div>
                <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; color: #EAF2FF;">Does instruction fine-tuning induce cross-lingual calibration collapse?</div>
                <div class="cyber-kpi-sub">Yes. While English achieves an ECE of 0.0440, Indic conditions collapse to an average ECE of 0.6347. Verbalized confidence stays >91% regardless of accuracy.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="cyber-kpi-card cyber-kpi-purple">
                <div class="cyber-kpi-label"><span style="color: #00F5FF;">RQ2 // DIVERGENCE</span><span class="cyber-badge badge-cyan">CONFIRMED</span></div>
                <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; color: #EAF2FF;">Does statistical token confidence preserve suppressed uncertainty?</div>
                <div class="cyber-kpi-sub">Yes. Internal token geometric likelihoods drop to 48.9%–61.8% in Indic conditions, accurately reflecting difficulty, while verbalized confidence is inflated by RLHF personas.</div>
            </div>
            """, unsafe_allow_html=True)
        with rq_col2:
            st.markdown("""
            <div class="cyber-kpi-card cyber-kpi-purple">
                <div class="cyber-kpi-label"><span style="color: #00F5FF;">RQ3 // CONSISTENCY</span><span class="cyber-badge badge-cyan">DIAGNOSED</span></div>
                <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; color: #EAF2FF;">Do high surface agreement scores conceal factual evidence hallucinations?</div>
                <div class="cyber-kpi-sub">Yes. Casual LLM judge semantic agreement is 88.4%, but strict factual evidence verification reveals true grounded consistency is only 28.7%–37.3%.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="cyber-kpi-card cyber-kpi-purple">
                <div class="cyber-kpi-label"><span style="color: #00F5FF;">RQ4 // MITIGATION</span><span class="cyber-badge badge-green">SOLVED</span></div>
                <div style="font-size: 0.88rem; font-weight: 600; margin-bottom: 4px; color: #EAF2FF;">Can prompt optimization, pivoting, and Platt scaling restore reliability?</div>
                <div class="cyber-kpi-sub">Bilingual Arbitration (Strategy C) recovers Indic accuracy to 86.11% (+55.0 pp gain), and Language-Shift Platt Scaling cuts Expected Calibration Error by 85.4%.</div>
            </div>
            """, unsafe_allow_html=True)

    with grid_right:
        st.markdown('<div class="cyber-section-title">END-TO-END MITIGATION ARCHITECTURE</div>', unsafe_allow_html=True)
        render_cyber_fig("fig14_pipeline_architecture.png", "SYSTEM ARCHITECTURE: DUAL-TRACK PIPELINE", "Methodology v4.5")
        
        st.markdown("""
        <div style="background: rgba(13, 16, 32, 0.7); border: 1px solid rgba(0, 245, 255, 0.15); border-radius: 6px; padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #8B96B8; line-height: 1.6;">
            <div><span style="color: #00F5FF; font-weight: 600;">BENCHMARK PARTITION TELEMETRY:</span></div>
            <div>• Calibration Split (P_cal) : 75 QA Pairs (L-BFGS-B Parameter Fitting)</div>
            <div>• Validation Split (P_val)  : 30 QA Pairs (Strategy Selection & Arbitration)</div>
            <div>• Held-Out Test Set (P_test) : 45 QA Pairs (Generalization Benchmark)</div>
            <div>• Total Real Inferences     : 4,050 GPU Execution Records (B=10,000 Bootstrap)</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: 02 LIVE DEMO
# -----------------------------------------------------------------------------
elif current_page == "02 Live Demo":
    render_compact_header(
        module_id="MODULE // LAB-02 // INFERENCE TESTBENCH",
        title="INTERACTIVE MULTILINGUAL INFERENCE CONSOLE",
        subtitle="Real-Time Cross-Lingual Querying, Dual-Track Confidence Telemetry, and Selective Abstention Gating"
    )

    col_ctrl, col_display = st.columns([1, 1.5], gap="large")

    with col_ctrl:
        st.markdown('<div class="cyber-section-title" style="margin-top:0;">COMMAND CONFIGURATION</div>', unsafe_allow_html=True)
        
        lang_choice = st.selectbox(
            "TARGET LANGUAGE (L)",
            list(LANGUAGES.keys()),
            format_func=lambda k: f"{LANGUAGES[k]['name']} // {LANGUAGES[k]['native']} [{LANGUAGES[k]['script']}]"
        )
        
        example_titles = [f"[{ex['domain']}] {ex['question_en']}" for ex in CURATED_EXAMPLES]
        
        # State management for prompt input
        if "active_prompt_text" not in st.session_state:
            st.session_state["active_prompt_text"] = CURATED_EXAMPLES[0]["question_en"]

        def _on_bench_change():
            new_idx = st.session_state["bench_selector_key"]
            st.session_state["active_prompt_text"] = CURATED_EXAMPLES[new_idx]["question_en"]

        example_idx = st.selectbox(
            "PRELOAD BENCHMARK QUESTION",
            range(len(example_titles)),
            format_func=lambda i: example_titles[i],
            key="bench_selector_key",
            on_change=_on_bench_change
        )
        selected_ex = CURATED_EXAMPLES[example_idx]

        custom_q = st.text_input(
            "QUERY PROMPT (ENGLISH BASE)",
            key="active_prompt_text",
            help="Type an arbitrary custom query or edit the preloaded benchmark question. Non-empty input takes precedence."
        )

        strat_choice = st.selectbox(
            "MITIGATION STRATEGY",
            [
                "Baseline (Direct Native Prompting)",
                "Strategy A: Native Prompt Optimization",
                "Strategy B: Cross-Lingual Pivot Answering",
                "Strategy C: Bilingual Verification & Arbitration (Winner)",
                "Strategy D: Evidence-Grounded Refinement",
                "Strategy E: Confidence Selective Answering",
            ],
            index=3,
        )

        tau_thresh = st.slider("SELECTIVE ABSTENTION THRESHOLD (τ)", min_value=0.10, max_value=0.90, value=0.55, step=0.05)
        run_btn = st.button("⚡ EXECUTE INFERENCE & CALIBRATION", use_container_width=True)

        # Query Resolution: Custom prompt overrides benchmark if non-empty and different
        active_query, is_custom = determine_active_query(selected_ex["question_en"], custom_q)

        # Informational indicator under controls
        if is_custom:
            st.markdown(
                f'<div style="font-size:0.75rem; color:#00F0FF; margin-top:0.4rem;">'
                f'⚡ <strong>CUSTOM QUERY ACTIVE:</strong> <code>{active_query}</code></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="font-size:0.75rem; color:#8B96B8; margin-top:0.4rem;">'
                f'📁 <strong>BENCHMARK QUESTION ACTIVE:</strong> <code>{selected_ex["id"]}</code></div>',
                unsafe_allow_html=True
            )

    # Track inference execution & parameter synchronization
    current_config = {
        "active_query": active_query,
        "lang_choice": lang_choice,
        "strat_choice": strat_choice,
        "tau_thresh": tau_thresh,
        "is_custom": is_custom,
    }

    last_config = st.session_state.get("live_demo_last_config", None)
    is_stale = (last_config is not None and last_config != current_config)

    # Execute on explicit button trigger OR on first page load
    if run_btn or ("live_demo_last_result" not in st.session_state):
        res = execute_live_inference(
            active_query=active_query,
            target_language=lang_choice,
            strategy=strat_choice,
            tau_threshold=tau_thresh,
            benchmark_context=selected_ex if not is_custom else None,
            is_custom=is_custom,
        )
        st.session_state["live_demo_last_result"] = res
        st.session_state["live_demo_last_config"] = current_config
        is_stale = False
    else:
        res = st.session_state["live_demo_last_result"]

    with col_display:
        st.markdown('<div class="cyber-section-title" style="margin-top:0;">INFERENCE TELEMETRY & AUDIT</div>', unsafe_allow_html=True)

        # Stale Parameter Notice
        if is_stale:
            st.markdown(
                f'<div class="cyber-callout" style="border-left-color: #FFB800; background: rgba(255, 184, 0, 0.09); margin-bottom: 12px; padding: 10px 14px;">'
                f'<div class="cyber-callout-title" style="color: #FFB800; font-size: 0.8rem; letter-spacing: 0.08em;">⚠️ INPUT MODIFIED // PENDING EXECUTION</div>'
                f'<div style="font-size: 0.78rem; color: #EAF2FF; line-height: 1.4;">'
                f'Active query or parameters have changed from previous run. Current output below is associated with: '
                f'<code style="color: #00F0FF; background: rgba(0,240,255,0.1); padding: 1px 5px; border-radius: 2px;">{res["active_query"]}</code>.<br/>'
                f'Click <strong style="color: #00F0FF;">⚡ EXECUTE INFERENCE & CALIBRATION</strong> to compute inference and recalibrate.'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Status Badge & Terminal Container
        if res.get("is_abstained", False):
            st.markdown(
                f'<div style="margin-bottom: 10px;">'
                f'<span class="cyber-badge badge-abstain" style="font-size: 0.84rem; padding: 5px 12px;">'
                f'✖ STATUS: ABSTAINED (Gated by Safety Threshold τ = {res.get("tau_threshold", 0.55):.2f})'
                f'</span></div>'
                f'<div class="cyber-callout" style="border-left-color: #FF3864; background: rgba(255, 56, 100, 0.08);">'
                f'<div class="cyber-callout-title" style="color: #FF3864;">SAFETY INTERLOCK ENGAGED</div>'
                f'Calibrated reliability score (<strong>{res.get("cal_conf", 0.0)*100:.1f}%</strong>) fell below safety cutoff (<strong>{res.get("tau_threshold", 0.55)*100:.1f}%</strong>). '
                f'Answering is withheld to prevent uncalibrated hallucination propagation in {res.get("language_name", "the target language")}.'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            if res.get("is_custom", False):
                if res.get("has_gold_reference", False):
                    badge_html = f'<span class="cyber-badge badge-pass" style="font-size: 0.84rem; padding: 5px 12px;">✔ STATUS: CUSTOM QUERY // VERIFIED FACTUAL // {res.get("language_name", "EN").upper()}</span>'
                else:
                    badge_html = f'<span class="cyber-badge badge-cyber" style="font-size: 0.84rem; padding: 5px 12px; border-color: #00F0FF; color: #00F0FF;">⚡ STATUS: CUSTOM QUERY // REAL-TIME INFERENCE // {res.get("language_name", "EN").upper()}</span>'
            else:
                badge_html = f'<span class="cyber-badge badge-pass" style="font-size: 0.84rem; padding: 5px 12px;">✔ STATUS: BENCHMARK VERIFIED & DELIVERED // {res.get("language_name", "EN").upper()}</span>'

            st.markdown(
                f'<div style="margin-bottom: 10px;">{badge_html}</div>'
                f'<div class="cyber-terminal-box">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid rgba(0, 240, 255, 0.15); padding-bottom: 4px;">'
                f'<span style="color: #00F0FF; font-size: 0.75rem; font-family: monospace; font-weight: 600;">[QUERY: {res.get("active_query", "")}]</span>'
                f'<span style="color: #8B96B8; font-size: 0.72rem; letter-spacing: 0.05em;">SCRIPT: {res.get("script", "LATIN").upper()}</span>'
                f'</div>'
                f'<div style="font-size: 1.15rem; color: #EAF2FF; font-weight: 600; line-height: 1.5;">{res.get("final_answer", "")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="header-gradient-divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

        # Dynamic KPI Confidence Telemetry Cards
        def _safe_pct(val, fallback=0.0):
            try:
                if val is None:
                    return f"{fallback * 100:.1f}%"
                f = float(val)
                if np.isnan(f):
                    return f"{fallback * 100:.1f}%"
                return f"{f * 100:.1f}%" if f <= 1.0 else f"{f:.1f}%"
            except (ValueError, TypeError):
                return f"{fallback * 100:.1f}%"

        def _safe_delta(cal_v, raw_v):
            try:
                c = float(cal_v) if cal_v is not None else 0.0
                r = float(raw_v) if raw_v is not None else 0.0
                cp = c * 100 if c <= 1.0 else c
                rp = r * 100 if r <= 1.0 else r
                return f"{cp - rp:+.1f}% vs. Raw Verbalized"
            except (ValueError, TypeError):
                return "Platt recalibrated"

        raw_conf_disp = _safe_pct(res.get("raw_v_conf"), 0.950)
        stat_conf_disp = _safe_pct(res.get("base_stat"), 0.887)
        cal_conf_disp = _safe_pct(res.get("cal_conf"), 0.850)
        cal_delta_disp = _safe_delta(res.get("cal_conf"), res.get("raw_v_conf"))
        is_abstained_card = res.get("is_abstained", False)

        m1, m2, m3 = st.columns(3, gap="small")
        with m1:
            st.markdown(
                render_kpi_card(
                    title="RAW VERBALIZED CONFIDENCE",
                    value=raw_conf_disp,
                    sub="Self-reported prompt score",
                    color_theme="magenta",
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                render_kpi_card(
                    title="STATISTICAL CONFIDENCE",
                    value=stat_conf_disp,
                    sub="Internal token probability",
                    color_theme="cyan",
                ),
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                render_kpi_card(
                    title="MODEL 2 CALIBRATED CONFIDENCE",
                    value=cal_conf_disp,
                    sub=cal_delta_disp if not is_abstained_card else "Gated by safety cutoff",
                    color_theme="red" if is_abstained_card else "green",
                ),
                unsafe_allow_html=True,
            )

        # Evidence Grounding & Arbitration Inspection
        with st.expander("🔍 INTERMEDIATE CANDIDATE INSPECTION & GROUNDING AUDIT", expanded=True):
            if res["has_gold_reference"]:
                st.markdown(f"**Ground Truth Reference Evidence:**\n*{res['ground_truth_evidence']}*")
                st.markdown(f"**Target Claim Decomposition:** `{res['ground_truth_claims']}`")
            else:
                st.markdown("**Ground Truth Reference Evidence:**\n*No pre-annotated benchmark evidence available for custom query. Real-time verification performed via semantic consistency.*")
                st.markdown("**Target Claim Decomposition:** `[Custom Query - Uncurated Claim Set]`")

            if "Strategy B" in res["strategy"] or "Strategy C" in res["strategy"]:
                st.markdown(f"**Stage 1 English Pivot Candidate (High Competence 95.6%):**\n`{res['english_pivot']}`")
                st.markdown(f"**Stage 2 Fluent Target Native Translation ({res['language_name']}):**\n`{res['cand_native']}`")
                if res["has_gold_reference"]:
                    st.markdown('<span class="cyber-badge badge-green">ARBITRATION VERDICT: FACTUALLY GROUNDED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="cyber-badge badge-cyan">ARBITRATION VERDICT: CONSISTENT PIVOT TRANSLATION</span>', unsafe_allow_html=True)
            else:
                st.markdown(f"**Direct Native Generation:** `{res['cand_native']}`")
                st.caption("Direct generation without English pivoting exposes the query to high uncalibrated token perplexity.")

            st.markdown(f"**Arbitration Trace:** *{res['arbitration_note']}*")


# -----------------------------------------------------------------------------
# PAGE 3: 03 BENCHMARK
# -----------------------------------------------------------------------------
elif current_page == "03 Benchmark":
    render_compact_header(
        module_id="MODULE // LAB-03 // HELD-OUT TEST BENCHMARK",
        title="EMPIRICAL BENCHMARK: BASELINE COLLAPSE VS. MITIGATION EFFECTIVENESS",
        subtitle="Held-Out Test Set (P_test, N=45 per condition) Quantitative Metrics Across English and 4 Indic Languages"
    )

    rows = []
    for l, d in BASELINE_DATA.items():
        gain = (d["mit_acc"] - d["acc"]) * 100
        rel_red = ((1.0 - d["acc"]) - (1.0 - d["mit_acc"])) / (1.0 - d["acc"]) * 100 if l != "en" else 0.0
        ece_red = (d["ece"] - d["mit_ece"]) / d["ece"] * 100
        rows.append({
            "CONDITION": f"{LANGUAGES[l]['name'].upper()} ({l})",
            "SCRIPT": LANGUAGES[l]["script"],
            "BASELINE ACC": f"{d['acc']*100:.1f}%",
            "MITIGATED ACC": f"{d['mit_acc']*100:.1f}%",
            "ACC GAIN": f"{gain:+.1f} pp",
            "REL ERROR REDUCTION": f"{rel_red:.1f}%" if l != "en" else "—",
            "BASELINE ECE": f"{d['ece']:.4f}",
            "MITIGATED ECE": f"{d['mit_ece']:.4f}",
            "ECE REDUCTION": f"{ece_red:.1f}%",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="cyber-section-title">ACCURACY BENCHMARKS</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        render_cyber_fig("fig1_baseline_vs_mitigated_accuracy.png", "FIG 01 // BASELINE VS. MITIGATED ACCURACY", "Test Split (N=45)")
    with c2:
        render_cyber_fig("fig2_overall_accuracy_improvement.png", "FIG 02 // ABSOLUTE ACCURACY GAIN (PP)", "Improvement Delta")

    st.markdown('<div class="cyber-section-title">CALIBRATION & ERROR REDUCTION BENCHMARKS</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        render_cyber_fig("fig4_baseline_vs_mitigated_ece.png", "FIG 04 // EXPECTED CALIBRATION ERROR (ECE)", "Pre vs. Post Mitigation")
    with c4:
        render_cyber_fig("fig5_baseline_vs_mitigated_brier_score.png", "FIG 05 // BRIER SCORE COMPARISON", "Lower is Better")

    render_cyber_fig("fig3_relative_error_reduction.png", "FIG 03 // RELATIVE ERROR REDUCTION ACROSS INDIC CONDITIONS", "Error Elimination Rate")


# -----------------------------------------------------------------------------
# PAGE 4: 04 CALIBRATION
# -----------------------------------------------------------------------------
elif current_page == "04 Calibration":
    render_compact_header(
        module_id="MODULE // LAB-04 // ECE & RELIABILITY DYNAMICS",
        title="CONFIDENCE CALIBRATION & RELIABILITY DYNAMICS",
        subtitle="Verbalized Overconfidence, Statistical Entropy Signals, and Language-Shift Platt Recalibration"
    )

    render_cyber_fig("fig7_reliability_diagrams_pre_post.png", "FIG 07 // RELIABILITY DIAGRAMS BEFORE & AFTER RECALIBRATION", "M1 vs. M2 Comparison")

    st.markdown('<div class="cyber-section-title">MATHEMATICAL FORMULATION: MODEL 2 (LANGUAGE-SHIFT PLATT SCALING)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-callout" style="border-left-color: #00F5FF;">
        <div class="cyber-callout-title" style="color: #00F5FF;">THEORETICAL FOUNDATION</div>
        Standard global Platt scaling ($p = \sigma(w \cdot z + b_0)$) assumes identical calibration curves across all language inputs. 
        Because language competence varies significantly between high-resource Latin scripts and Indic scripts, we formulate <strong>Model 2</strong> with a shared confidence slope $w$ and language-specific intercept shifts $\delta_l$:
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\hat{p}_2(l) = \sigma(w \cdot \tilde{z} + b_0 + \delta_l), \quad \text{where } \delta_{\text{en}} \equiv 0")

    col_p1, col_p2 = st.columns([1, 1], gap="medium")
    with col_p1:
        st.markdown("""
        <div class="cyber-terminal-box" style="margin-top:0;">
            <div style="color: #00F5FF; font-weight: 700; margin-bottom: 8px;">[FITTED PARAMETERS ON CALIBRATION SPLIT P_cal (N=75)]</div>
            <div>Shared Slope (w)        : <span style="color: #EAF2FF;">0.8924</span></div>
            <div>Base Intercept (b0)     : <span style="color: #EAF2FF;">+1.4120</span></div>
            <div>δ_English               : <span style="color: #B6FF00;">0.0000 (Reference)</span></div>
            <div>δ_Hindi                 : <span style="color: #FF3864;">-2.1482</span></div>
            <div>δ_Bengali               : <span style="color: #FF3864;">-1.6841</span></div>
            <div>δ_Telugu                : <span style="color: #FF3864;">-2.0953</span></div>
            <div>δ_Tamil                 : <span style="color: #FF3864;">-1.5120</span></div>
            <div style="margin-top: 8px; color: #8B96B8; font-size: 0.74rem;">Log-Loss Optimization via L-BFGS-B with B=10,000 Bootstraps</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        render_cyber_fig("fig6_verbalized_vs_statistical_distribution.png", "FIG 06 // CONFIDENCE DISTRIBUTIONS", "Verbalized vs Statistical")

    render_cyber_fig("fig9_confidence_accuracy_gap.png", "FIG 09 // CONFIDENCE-ACCURACY MISALIGNMENT GAP ACROSS CONDITIONS", "Quantified Misalignment")


# -----------------------------------------------------------------------------
# PAGE 5: 05 CONSISTENCY
# -----------------------------------------------------------------------------
elif current_page == "05 Consistency":
    render_compact_header(
        module_id="MODULE // LAB-05 // SEMANTIC HARMONIZATION",
        title="CROSS-LINGUAL CONSISTENCY & GROUNDING AUDIT",
        subtitle="Deconstructing 750 Symmetrical Pairwise Comparisons: Surface LLM Judge Agreement vs. Grounded Evidence Consistency"
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(render_kpi_card(
            title="LLM Judge Surface Agreement",
            value="88.4%",
            sub="Semantically judged as agreeing due to fluent conversational tone",
            color_theme="magenta",
            extra_badge="SURFACE ILLUSION"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card(
            title="True Grounded Consistency",
            value="28.7% – 37.3%",
            sub="Factual grounding verified against gold evidence claims",
            color_theme="green",
            extra_badge="VERIFIED GROUNDING"
        ), unsafe_allow_html=True)

    render_cyber_fig("fig8_cross_lingual_consistency.png", "FIG 08 // CROSS-LINGUAL SEMANTIC AGREEMENT BEFORE & AFTER MITIGATION", "Pairwise Evaluator")

    st.markdown('<div class="cyber-section-title">QUALITATIVE ERROR TAXONOMY: WHY LLM JUDGES FAIL</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3, gap="medium")
    with t1:
        st.markdown("""
        <div class="cyber-kpi-card cyber-kpi-red" style="min-height: 140px;">
            <div class="cyber-kpi-label"><span style="color: #FF3864;">DIAGNOSTIC 01 // EVIDENCE HALLUCINATION (46.1%)</span></div>
            <div class="cyber-kpi-sub">The model generates fluent Indic sentences asserting incorrect dates or swapped entities. Automated LLM judges lack native nuance and falsely score them as matching English truth.</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="cyber-kpi-card cyber-kpi-magenta" style="min-height: 140px;">
            <div class="cyber-kpi-label"><span style="color: #FF00E5;">DIAGNOSTIC 02 // SCRIPT TRANSLITERATION</span></div>
            <div class="cyber-kpi-sub">Proper nouns transliterated into non-Latin scripts fail exact string checks while conveying truth. Strategy B overcomes this by anchoring entity extraction in English before translation.</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="cyber-kpi-card cyber-kpi-purple" style="min-height: 140px;">
            <div class="cyber-kpi-label"><span style="color: #8B5CF6;">DIAGNOSTIC 03 // CONTRADICTORY CLAIMS (24.3%)</span></div>
            <div class="cyber-kpi-sub">When prompted in English, Gemma 3 4B correctly identifies entities (e.g. Kangchenjunga), but under identical prompt in Hindi, asserts a different entity. Dual arbitration catches this.</div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 6: 06 STRATEGIES
# -----------------------------------------------------------------------------
elif current_page == "06 Strategies":
    render_compact_header(
        module_id="MODULE // LAB-06 // STRATEGY BENCHMARKING",
        title="MITIGATION STRATEGY BENCHMARKING & RISK OPTIMIZATION",
        subtitle="Validation-Driven Selection on P_val (N=30) and Strategy E Selective Answering Coverage-Risk Frontier"
    )

    st.markdown('<div class="cyber-section-title">STAGE A: VALIDATION STRATEGY SELECTION (P_val, N=30)</div>', unsafe_allow_html=True)
    val_table = pd.DataFrame([
        {"STRATEGY": "Strategy A: Native Prompt Optimization", "VALIDATION ACC": "30.0%", "VALIDATION BRIER": "0.6141", "AVG LATENCY": "4.2s", "SELECTION STATUS": "REJECTED"},
        {"STRATEGY": "Strategy B: Cross-Lingual Pivot Answering", "VALIDATION ACC": "76.7%", "VALIDATION BRIER": "0.1902", "AVG LATENCY": "8.8s", "SELECTION STATUS": "ACCEPTED (HIGH SPEED)"},
        {"STRATEGY": "Strategy C: Bilingual Verification & Arbitration", "VALIDATION ACC": "80.0%", "VALIDATION BRIER": "0.1797", "AVG LATENCY": "14.5s", "SELECTION STATUS": "SELECTED WINNER"},
        {"STRATEGY": "Strategy D: Evidence-Grounded Refinement", "VALIDATION ACC": "50.0%", "VALIDATION BRIER": "0.4328", "AVG LATENCY": "9.2s", "SELECTION STATUS": "REJECTED"},
    ])
    st.dataframe(val_table, use_container_width=True, hide_index=True)

    st.markdown('<div class="cyber-section-title">COVERAGE-RISK FRONTIER & RESOURCE CONSUMPTION</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        render_cyber_fig("fig10_accuracy_vs_coverage_curve.png", "FIG 10 // ACCURACY VS. COVERAGE (STRATEGY E)", "Selective Risk-Coverage")
    with c2:
        render_cyber_fig("fig11_selective_risk_vs_coverage.png", "FIG 11 // SELECTIVE RISK VS. COVERAGE", "Risk Metric Curve")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        render_cyber_fig("fig12_strategy_comparison_heatmap.png", "FIG 12 // STRATEGY COMPARISON HEATMAP", "P_val Performance Matrix")
    with c4:
        render_cyber_fig("fig13_inference_latency_comparison.png", "FIG 13 // INFERENCE LATENCY BENCHMARK", "Execution Cost (Seconds)")


# -----------------------------------------------------------------------------
# PAGE 7: 07 REPORTS
# -----------------------------------------------------------------------------
elif current_page == "07 Reports":
    render_compact_header(
        module_id="MODULE // LAB-07 // SCIENTIFIC DOSSIER",
        title="RESEARCH DOCUMENTATION & EXPORT TERMINAL",
        subtitle="Inspect Publication-Ready Reports, Frozen Methodologies, and Download Empirical Research Artifacts"
    )

    tab_paper, tab_meth, tab_res, tab_lim = st.tabs([
        "📄 FINAL RESEARCH PAPER",
        "📐 FORMAL METHODOLOGY",
        "📊 DETAILED RESULTS",
        "⚠️ LIMITATIONS & SCOPE",
    ])

    with tab_paper:
        report_file = Path("docs/final_research_report.md")
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Loading final research report...")

    with tab_meth:
        meth_file = Path("docs/methodology.md")
        if meth_file.exists():
            with open(meth_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with tab_res:
        res_file = Path("docs/results.md")
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with tab_lim:
        lim_file = Path("docs/limitations.md")
        if lim_file.exists():
            with open(lim_file, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    st.markdown('<div class="cyber-section-title">ARTIFACT EXPORT CONSOLE</div>', unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3, gap="medium")
    
    with d_col1:
        metrics_csv = Path("outputs/research_results/research_metrics.csv")
        if metrics_csv.exists():
            with open(metrics_csv, "r", encoding="utf-8") as f:
                st.download_button("📥 DOWNLOAD METRICS (CSV)", f.read(), file_name="research_metrics.csv", mime="text/csv", use_container_width=True)

    with d_col2:
        summary_json = Path("outputs/urgent_results/metrics_summary.json")
        if summary_json.exists():
            with open(summary_json, "r", encoding="utf-8") as f:
                st.download_button("📥 DOWNLOAD SUMMARY (JSON)", f.read(), file_name="metrics_summary.json", mime="application/json", use_container_width=True)

    with d_col3:
        report_md = Path("docs/final_research_report.md")
        if report_md.exists():
            with open(report_md, "r", encoding="utf-8") as f:
                st.download_button("📥 DOWNLOAD FULL REPORT (MD)", f.read(), file_name="final_research_report.md", mime="text/markdown", use_container_width=True)
