"""
Cyberpunk UI Theme & Design System for Cross-Lingual Calibration Platform.
Includes custom CSS rules, Google Fonts injection, Horizontal Navbar, HUD headers, KPI cards, and styling helpers.
"""

import streamlit as st

def get_cyberpunk_css() -> str:
    """Returns the full cyberpunk stylesheet for Streamlit with horizontal navigation and no sidebar."""
    return """
<style>
/* -------------------------------------------------------------------------
   FONTS INJECTION
   ------------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* -------------------------------------------------------------------------
   ROOT DESIGN TOKENS
   ------------------------------------------------------------------------- */
:root {
    --bg-dark: #070812;
    --bg-secondary: #0D1020;
    --bg-card: #121629;
    --bg-card-hover: #171D36;
    --neon-cyan: #00F5FF;
    --neon-cyan-subtle: rgba(0, 245, 255, 0.12);
    --neon-magenta: #FF00E5;
    --neon-purple: #8B5CF6;
    --acid-green: #B6FF00;
    --error-red: #FF3864;
    --text-primary: #EAF2FF;
    --text-muted: #8B96B8;
    --border-subtle: rgba(0, 245, 255, 0.16);
    --border-active: rgba(0, 245, 255, 0.45);
}

/* -------------------------------------------------------------------------
   COMPLETE SIDEBAR & HEADER REMOVAL FOR FULL-WIDTH VIEWPORT
   ------------------------------------------------------------------------- */
section[data-testid="stSidebar"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"],
.stSidebarCollapsedControl,
header[data-testid="stHeader"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    visibility: hidden !important;
}

/* Eliminate reserved sidebar margins in Streamlit layout */
.main, .stMain, [data-testid="stMain"] {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    padding-left: 0 !important;
}

.main .block-container, [data-testid="stMainBlockContainer"] {
    max-width: 96% !important;
    width: 96% !important;
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    margin: 0 auto !important;
}

/* -------------------------------------------------------------------------
   APP CONTAINER & CYBERPUNK BACKGROUND GRID
   ------------------------------------------------------------------------- */
.stApp {
    background-color: #070812 !important;
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.10) 0%, transparent 65%),
        linear-gradient(rgba(0, 245, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 245, 255, 0.02) 1px, transparent 1px) !important;
    background-size: 100% 100%, 36px 36px, 36px 36px !important;
    color: #EAF2FF !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Global Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', 'Space Grotesk', sans-serif !important;
    color: #EAF2FF !important;
    letter-spacing: 0.02em !important;
}
p, span, label, li {
    font-family: 'Inter', sans-serif;
    color: #EAF2FF;
}
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* -------------------------------------------------------------------------
   TOP HORIZONTAL NAVBAR CONTAINER & RECTANGULAR BUTTONS
   ------------------------------------------------------------------------- */
.nav-brand-container {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 2px 0;
}
.nav-brand-logo {
    font-size: 1.3rem;
    color: #00F5FF;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.6);
}
.nav-brand-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: #00F5FF;
    line-height: 1.1;
}
.nav-brand-sub {
    font-family: 'Space Grotesk', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #8B96B8;
    text-transform: uppercase;
}

.nav-telemetry-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8B96B8;
    background: rgba(13, 16, 32, 0.75);
    border: 1px solid rgba(0, 245, 255, 0.18);
    border-radius: 6px;
    padding: 7px 12px;
    text-align: right;
    white-space: nowrap;
}
.nav-telemetry-box span.online-text {
    color: #B6FF00;
    font-weight: 700;
    margin-right: 6px;
}
.nav-telemetry-box span.telemetry-sep {
    color: rgba(0, 245, 255, 0.25);
    margin: 0 4px;
}
.nav-telemetry-box span.telemetry-tag {
    color: #EAF2FF;
    font-weight: 500;
}

/* Pulsing Status Beacon */
.pulse-online {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #B6FF00;
    box-shadow: 0 0 8px #B6FF00;
    margin-right: 6px;
    vertical-align: middle;
    animation: neonPulse 1.8s infinite;
}
@keyframes neonPulse {
    0% { transform: scale(0.9); box-shadow: 0 0 3px #B6FF00; }
    50% { transform: scale(1.2); box-shadow: 0 0 10px #B6FF00; }
    100% { transform: scale(0.9); box-shadow: 0 0 3px #B6FF00; }
}

/* Custom Rectangular Top Navigation Buttons */
div[data-testid="stHorizontalBlock"]:first-of-type button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 6px 4px !important;
    border-radius: 4px !important;
    height: 38px !important;
    min-height: 38px !important;
    white-space: nowrap !important;
    text-transform: uppercase !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Inactive Nav Button */
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-bottom: 2px solid transparent !important;
    color: #8B96B8 !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"]:hover {
    color: #00F5FF !important;
    background: rgba(0, 245, 255, 0.08) !important;
    border-color: rgba(0, 245, 255, 0.4) !important;
    border-bottom: 2px solid #00F5FF !important;
}

/* Active Nav Button */
div[data-testid="stHorizontalBlock"]:first-of-type button[kind="primary"] {
    background: linear-gradient(180deg, rgba(0, 245, 255, 0.05) 0%, rgba(0, 245, 255, 0.16) 100%) !important;
    border: 1px solid rgba(0, 245, 255, 0.35) !important;
    border-bottom: 2px solid #00F5FF !important;
    color: #00F5FF !important;
    box-shadow: 0 4px 14px rgba(0, 245, 255, 0.22) !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
}

/* -------------------------------------------------------------------------
   COMPACT BALANCED APPLICATION HEADER
   ------------------------------------------------------------------------- */
.app-header-container {
    margin-bottom: 1.4rem;
    position: relative;
}
.header-module-pill {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #00F5FF;
    background: rgba(0, 245, 255, 0.08);
    border: 1px solid rgba(0, 245, 255, 0.25);
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}
.header-page-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.48rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    background: linear-gradient(90deg, #EAF2FF 0%, #00F5FF 55%, #FF00E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
    line-height: 1.25;
}
.header-page-subtitle {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.92rem;
    color: #8B96B8;
    margin-bottom: 0.8rem;
    line-height: 1.45;
}
.header-gradient-divider {
    width: 100%;
    height: 1.5px;
    background: linear-gradient(90deg, #00F5FF 0%, #8B5CF6 45%, #FF00E5 80%, transparent 100%);
    opacity: 0.75;
    margin-bottom: 1.2rem;
}

/* -------------------------------------------------------------------------
   REFINED GLASSMORPHIC KPI CARDS
   ------------------------------------------------------------------------- */
.cyber-kpi-card {
    background: rgba(18, 22, 41, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 245, 255, 0.16);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
    width: 100%;
    transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
    min-height: 122px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.cyber-kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 245, 255, 0.4);
    box-shadow: 0 6px 20px rgba(0, 245, 255, 0.12);
}
.cyber-kpi-cyan {
    border-left: 3px solid #00F5FF;
}
.cyber-kpi-cyan .cyber-kpi-val {
    color: #00F5FF;
    text-shadow: 0 0 12px rgba(0, 245, 255, 0.45);
}
.cyber-kpi-magenta {
    border-left: 3px solid #FF00E5;
}
.cyber-kpi-magenta .cyber-kpi-val {
    color: #FF00E5;
    text-shadow: 0 0 12px rgba(255, 0, 229, 0.45);
}
.cyber-kpi-green {
    border-left: 3px solid #B6FF00;
}
.cyber-kpi-green .cyber-kpi-val {
    color: #B6FF00;
    text-shadow: 0 0 12px rgba(182, 255, 0, 0.45);
}
.cyber-kpi-red {
    border-left: 3px solid #FF3864;
}
.cyber-kpi-red .cyber-kpi-val {
    color: #FF3864;
    text-shadow: 0 0 12px rgba(255, 56, 100, 0.45);
}
.cyber-kpi-purple {
    border-left: 3px solid #8B5CF6;
}
.cyber-kpi-purple .cyber-kpi-val {
    color: #8B5CF6;
    text-shadow: 0 0 12px rgba(139, 92, 246, 0.45);
}

.cyber-kpi-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8B96B8;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.cyber-kpi-val {
    font-family: 'Orbitron', 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    color: #EAF2FF;
    margin-bottom: 4px;
    line-height: 1.1;
}
.cyber-kpi-sub {
    font-size: 0.74rem;
    color: #8B96B8;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* -------------------------------------------------------------------------
   SECTION HEADERS
   ------------------------------------------------------------------------- */
.cyber-section-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #00F5FF !important;
    display: flex;
    align-items: center;
    margin-top: 1.4rem;
    margin-bottom: 0.85rem;
}
.cyber-section-title::before {
    content: "■";
    font-size: 0.75rem;
    color: #FF00E5;
    margin-right: 8px;
    text-shadow: 0 0 6px #FF00E5;
}

/* -------------------------------------------------------------------------
   TACTICAL STATUS BADGES
   ------------------------------------------------------------------------- */
.cyber-badge {
    display: inline-flex;
    align-items: center;
    font-family: 'Space Grotesk', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    margin-right: 6px;
}
.badge-pass {
    background: rgba(182, 255, 0, 0.10);
    border: 1px solid #B6FF00;
    color: #B6FF00;
    box-shadow: 0 0 8px rgba(182, 255, 0, 0.18);
}
.badge-abstain {
    background: rgba(255, 56, 100, 0.10);
    border: 1px solid #FF3864;
    color: #FF3864;
    box-shadow: 0 0 8px rgba(255, 56, 100, 0.18);
}
.badge-cyan {
    background: rgba(0, 245, 255, 0.10);
    border: 1px solid #00F5FF;
    color: #00F5FF;
}
.badge-magenta {
    background: rgba(255, 0, 229, 0.10);
    border: 1px solid #FF00E5;
    color: #FF00E5;
}

/* -------------------------------------------------------------------------
   NATIVE CONTROLS (BUTTONS, INPUTS, SELECT, SLIDERS, TABS, EXPANDERS)
   ------------------------------------------------------------------------- */
/* Content Buttons (excluding top navbar) */
div[data-testid="stHorizontalBlock"]:not(:first-of-type) button {
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.14) 0%, rgba(139, 92, 246, 0.25) 100%) !important;
    border: 1px solid #00F5FF !important;
    color: #00F5FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 0 10px rgba(0, 245, 255, 0.15) !important;
}
div[data-testid="stHorizontalBlock"]:not(:first-of-type) button:hover {
    background: linear-gradient(135deg, #00F5FF 0%, #8B5CF6 100%) !important;
    color: #070812 !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Download Button */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(255, 0, 229, 0.14) 0%, rgba(139, 92, 246, 0.22) 100%) !important;
    border: 1px solid #FF00E5 !important;
    color: #FF00E5 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 0 10px rgba(255, 0, 229, 0.15) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #FF00E5 0%, #8B5CF6 100%) !important;
    color: #070812 !important;
    box-shadow: 0 0 20px rgba(255, 0, 229, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Selectboxes, Inputs & Text Areas */
div[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea {
    background-color: #0D1020 !important;
    border: 1px solid rgba(0, 245, 255, 0.22) !important;
    color: #EAF2FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.90rem !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] > div:hover,
.stTextInput input:hover,
.stTextArea textarea:hover {
    border-color: #00F5FF !important;
}
div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #00F5FF !important;
    box-shadow: 0 0 10px rgba(0, 245, 255, 0.35) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    margin-top: 8px;
}
div[role="slider"] {
    background-color: #00F5FF !important;
    border: 2px solid #EAF2FF !important;
    box-shadow: 0 0 10px #00F5FF !important;
}

/* Cyber Tabs */
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #8B96B8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.90rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #00F5FF !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00F5FF !important;
    border-bottom: 2px solid #00F5FF !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(18, 22, 41, 0.8) !important;
    border: 1px solid rgba(0, 245, 255, 0.20) !important;
    border-radius: 8px !important;
    margin-top: 12px !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: #00F5FF !important;
    letter-spacing: 0.03em !important;
}

/* Tables & Dataframes */
div[data-testid="stDataFrame"], table {
    border: 1px solid rgba(0, 245, 255, 0.18) !important;
    border-radius: 6px !important;
}

/* -------------------------------------------------------------------------
   FIGURE FRAMES & CONTAINERS
   ------------------------------------------------------------------------- */
.cyber-fig-card {
    background: rgba(13, 16, 32, 0.85);
    border: 1px solid rgba(0, 245, 255, 0.20);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.4);
}
.cyber-fig-card:hover {
    border-color: rgba(0, 245, 255, 0.45);
    box-shadow: 0 6px 20px rgba(0, 245, 255, 0.12);
}
.cyber-fig-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 6px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(0, 245, 255, 0.12);
    font-family: 'Space Grotesk', monospace;
    font-size: 0.74rem;
    color: #00F5FF;
    letter-spacing: 0.08em;
}

/* Terminal Console Box */
.cyber-terminal-box {
    background-color: #060812;
    border: 1px solid rgba(0, 245, 255, 0.22);
    border-radius: 6px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    color: #B6FF00;
    box-shadow: inset 0 0 14px rgba(0, 0, 0, 0.7);
    margin: 10px 0;
    line-height: 1.5;
}

/* Tactical Alert / Info Callouts */
.cyber-callout {
    background: rgba(18, 22, 41, 0.75);
    border-left: 3px solid #8B5CF6;
    border-top: 1px solid rgba(139, 92, 246, 0.18);
    border-right: 1px solid rgba(139, 92, 246, 0.18);
    border-bottom: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 6px;
    padding: 14px 16px;
    margin: 10px 0;
    color: #EAF2FF;
}
.cyber-callout-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #8B5CF6;
    letter-spacing: 0.05em;
    margin-bottom: 5px;
}
</style>
"""

def render_kpi_card(title: str, value: str, sub: str, color_theme: str = "cyan", extra_badge: str = "") -> str:
    """
    Generates a clean HTML string for a futuristic cyberpunk glass KPI card.
    Strictly uses single-line zero-indentation HTML to avoid CommonMark 4-space code block conversion.
    """
    badge_html = f'<span class="cyber-badge badge-{color_theme}">{extra_badge}</span>' if extra_badge else ""
    return (
        f'<div class="cyber-kpi-card cyber-kpi-{color_theme}">'
        f'<div class="cyber-kpi-label"><span title="{title}">{title}</span>{badge_html}</div>'
        f'<div class="cyber-kpi-val">{value}</div>'
        f'<div class="cyber-kpi-sub" title="{sub}">{sub}</div>'
        f'</div>'
    )


def render_compact_header(module_id: str, title: str, subtitle: str) -> None:
    """
    Renders the balanced, compact page header with module identifier, title, subtitle, and gradient divider.
    Zero-indentation ensures direct DOM rendering without code block escaping.
    """
    header_html = (
        f'<div class="app-header-container">'
        f'<div class="header-module-pill">{module_id}</div>'
        f'<div class="header-page-title">{title}</div>'
        f'<div class="header-page-subtitle">{subtitle}</div>'
        f'<div class="header-gradient-divider"></div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)
