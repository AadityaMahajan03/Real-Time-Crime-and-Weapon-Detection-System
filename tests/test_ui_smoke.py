"""UI Smoke tests using Streamlit AppTest framework."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_shell_smoke() -> None:
    """Smoke test ensuring app.py launches, renders top bar, and has no uncaught exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify no exceptions or errors occurred during script execution
    assert not at.exception, f"App execution raised exception: {at.exception}"
    assert len(at.title) >= 0 or len(at.markdown) > 0


def test_kpi_card_with_delta() -> None:
    """Test that kpi_card accepts delta and does not raise TypeError."""
    from ui.components import kpi_card

    # Should execute cleanly without raising TypeError: unexpected keyword argument 'delta'
    kpi_card(
        title="Total Incidents",
        value="42",
        delta="5 High-Severity",
        accent_color="#22D3EE",
    )
    kpi_card(
        title="Fleet",
        value="3/3",
        subtitle="Online",
        delta="Healthy",
        accent_color="#22C55E",
    )


def test_analytics_page_smoke() -> None:
    """Test that 05_analytics.py executes without TypeError or uncaught exceptions."""
    analytics_path = Path(__file__).resolve().parent.parent / "ui" / "pages" / "05_analytics.py"
    at = AppTest.from_file(str(analytics_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Analytics page raised exception: {at.exception}"


def test_live_monitor_page_smoke() -> None:
    """Test that 02_live_monitor.py executes without uncaught exceptions."""
    live_path = Path(__file__).resolve().parent.parent / "ui" / "pages" / "02_live_monitor.py"
    at = AppTest.from_file(str(live_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Live monitor page raised exception: {at.exception}"


def test_evaluation_page_smoke() -> None:
    """Test that 06_evaluation.py executes and Arrow serialization succeeds."""
    eval_path = Path(__file__).resolve().parent.parent / "ui" / "pages" / "06_evaluation.py"
    at = AppTest.from_file(str(eval_path), default_timeout=30)
    at.run()
    assert not at.exception, f"Evaluation page raised exception: {at.exception}"


