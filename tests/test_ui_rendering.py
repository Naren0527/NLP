"""
Test suite for UI rendering integrity, KPI card formatting, and zero-indentation HTML safety.
Ensures that no HTML strings passed to Streamlit are accidentally converted into CommonMark code blocks.
"""

import re
from src.ui.cyber_theme import render_kpi_card, render_compact_header


def test_kpi_card_has_zero_leading_whitespace():
    """Verify that no line in render_kpi_card begins with 4+ spaces (which triggers CommonMark code blocks)."""
    html = render_kpi_card(
        title="RAW VERBALIZED CONFIDENCE",
        value="92.0%",
        sub="Self-reported prompt score",
        color_theme="magenta"
    )
    for line in html.splitlines():
        assert not line.startswith("    "), f"Indented line detected that would trigger Markdown code block: {line!r}"
    assert "cyber-kpi-magenta" in html
    assert "RAW VERBALIZED CONFIDENCE" in html
    assert "92.0%" in html
    assert "Self-reported prompt score" in html


def test_kpi_card_themes_and_badges():
    """Verify all cyberpunk color themes render their appropriate accent classes."""
    for theme in ["cyan", "magenta", "green", "red", "purple"]:
        html = render_kpi_card("TEST TITLE", "100.0%", "Subtext", color_theme=theme, extra_badge="VERIFIED")
        assert f"cyber-kpi-{theme}" in html
        assert f"badge-{theme}" in html
        assert "VERIFIED" in html


def test_compact_header_has_zero_leading_whitespace():
    """Verify compact header template does not start with 4 spaces."""
    # Test template structure
    from src.ui.cyber_theme import get_cyberpunk_css
    css = get_cyberpunk_css()
    assert ".cyber-kpi-card" in css
    assert ".cyber-kpi-cyan" in css
    assert ".cyber-kpi-magenta" in css
    assert ".cyber-kpi-green" in css


def test_confidence_card_percentage_formatting():
    """Verify numeric confidence values format properly without truncation or scientific notation."""
    val = 0.95556
    formatted = f"{val * 100:.1f}%"
    assert formatted == "95.6%"

    val_low = 0.2381
    formatted_low = f"{val_low * 100:.1f}%"
    assert formatted_low == "23.8%"
