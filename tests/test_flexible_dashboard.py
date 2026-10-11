"""Tests verifying the ultra-flexible dashboard and visualization capabilities of generate_html_viz."""

import json
from pathlib import Path
import pytest

from datavault.tools.html_viz import generate_html_viz, _ensure_defaults


def test_standalone_deep_agent_hooking_without_manual_config():
    """Verify tool can be invoked standalone (as by any Deep Agent or LangGraph agent) with zero setup."""
    # Reset any explicit global state if needed; _ensure_defaults handles it
    result = generate_html_viz.invoke({
        "data_ref": "vehicle_sales",
        "chart_type": "bar",
        "x_col": "vehicle_type",
        "y_col": "price_usd",
        "aggregation": "AVG",
        "title": "Standalone Deep Agent Test"
    })
    assert "Visualization saved to" in result
    path_str = result.split("saved to: ")[1].strip()
    assert Path(path_str).exists()


def test_custom_charts_and_kpis_dashboard_format():
    """Verify custom chart specs, KPIs, comparison layout, and midnight theme."""
    charts_spec = [
        {
            "type": "bar",
            "title": "Top 5 States by Volume",
            "sql": "SELECT customer_state, COUNT(*) as volume FROM vehicle_sales GROUP BY customer_state ORDER BY volume DESC LIMIT 5",
            "x": "customer_state",
            "y": "volume"
        },
        {
            "type": "pie",
            "title": "Electric vs Gas Brand Distribution",
            "sql": "SELECT CASE WHEN brand IN ('Tesla') THEN 'EV' ELSE 'Conventional' END as category, COUNT(*) as cnt FROM vehicle_sales GROUP BY category",
            "x": "category",
            "y": "cnt"
        }
    ]

    kpi_spec = [
        {
            "label": "Total EV Volume",
            "sql": "SELECT COUNT(*) FROM vehicle_sales WHERE brand = 'Tesla'",
            "suffix": " Units"
        },
        {
            "label": "Average SUV Price",
            "sql": "SELECT AVG(price_usd) FROM vehicle_sales WHERE vehicle_type = 'SUV'",
            "prefix": "$"
        }
    ]

    result = generate_html_viz.invoke({
        "data_ref": "vehicle_sales",
        "title": "Executive EV & SUV Market Intelligence",
        "charts": json.dumps(charts_spec),
        "kpis": json.dumps(kpi_spec),
        "layout": "comparison",
        "theme": "midnight"
    })

    assert "Dashboard saved to" in result
    path_str = result.split("saved to: ")[1].strip()
    content = Path(path_str).read_text(encoding="utf-8")
    assert "Top 5 States by Volume" in content
    assert "Total EV Volume" in content
    assert "Average SUV Price" in content
    assert "layout-comparison" in content
    assert "Midnight" in content


def test_custom_raw_html_dashboard():
    """Verify raw custom HTML is supported for 100% unrestricted dashboard formats."""
    custom_markup = """
    <div style="background:#111; color:#00ffcc; padding:30px; border-radius:12px;">
        <h2>Tailored Executive Performance Matrix</h2>
        <p>Custom D3 / Plotly / SVG component rendered directly</p>
        <div id="custom-kpi" style="font-size:32px; font-weight:bold;">$94.2M Revenue</div>
    </div>
    """

    result = generate_html_viz.invoke({
        "data_ref": "vehicle_sales",
        "title": "Raw Custom Dashboard",
        "custom_html": custom_markup
    })

    assert "Visualization saved to" in result
    path_str = result.split("saved to: ")[1].strip()
    content = Path(path_str).read_text(encoding="utf-8")
    assert "Tailored Executive Performance Matrix" in content
    assert "$94.2M Revenue" in content


def test_dynamic_focal_dashboard():
    """Verify passing specific focus columns adapts the dashboard layout to that dimension."""
    result = generate_html_viz.invoke({
        "data_ref": "vehicle_sales",
        "chart_type": "dashboard",
        "x_col": "customer_state",
        "y_col": "price_usd",
        "aggregation": "AVG",
        "title": "State-Focused Price Analytics",
        "theme": "dark"
    })

    assert "Dashboard saved to" in result
    path_str = result.split("saved to: ")[1].strip()
    content = Path(path_str).read_text(encoding="utf-8")
    assert "State-Focused Price Analytics" in content
    assert "customer_state" in content
    assert "stat-card" in content
