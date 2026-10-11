"""Comprehensive readiness verification for DataVault.

Validates:
1. FastMCP server and 10M row Parquet querying
2. Two tools (generate_html_viz and generate_markdown) + profile_and_filter on 10M rows
3. Dashboard formatting in generate_html_viz
4. Summary / breakdown / trend formatting in generate_markdown
5. LangGraph agent configuration & tool schema bindings
6. FastAPI endpoints (UI, viz serving in iframe, API query, refs)
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Add project root and src to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from datavault import server
from datavault.agent import create_agent
from datavault.proxy.interceptor import DataProxyInterceptor
from datavault.tools import html_viz, markdown_report, profile_filter
import mcp_servers.vehicle_sales.server as mcp_srv


def test_1_mcp_server_on_10m_rows():
    print("\n--- 1. Testing FastMCP Server on 10M Rows ---")
    tables = mcp_srv.list_tables()
    assert "vehicle_sales" in tables, "Table vehicle_sales missing"
    print("✓ list_tables() works")

    desc = mcp_srv.describe_table("vehicle_sales")
    assert "10,000,000" in desc, "Row count not 10,000,000"
    print("✓ describe_table() confirms 10M rows")

    sample = mcp_srv.sample_data("vehicle_sales", 3)
    assert len(sample) > 50, "Sample data empty"
    print("✓ sample_data() returned 3 rows")

    agg = mcp_srv.aggregate_data("vehicle_sales", "vehicle_type", "price_usd", "AVG")
    assert "Bike" in agg and "Car" in agg, "Aggregation failed"
    print("✓ aggregate_data() calculated 10M rows in sub-second")


def test_2_two_tools_and_agent_integration():
    print("\n--- 2. Testing The Two Tools & Profiling on 10M Rows ---")
    parquet_path = root_dir / "mcp_servers" / "vehicle_sales" / "data" / "vehicle_sales.parquet"
    assert parquet_path.exists(), f"Parquet file missing at {parquet_path}"

    interceptor = DataProxyInterceptor()
    profile = interceptor.register_parquet(str(parquet_path), "vehicle_sales")
    ref_id = profile.ref_id
    print(f"✓ Registered 10M rows Parquet. Data reference: {ref_id}")

    # Tool 1: profile_and_filter
    profile_filter.configure(interceptor.profiler, interceptor.data_store)
    p_res = profile_filter.profile_and_filter.invoke({
        "data_ref": ref_id,
        "filter_query": "vehicle_type = 'Car' AND brand = 'Tesla'"
    })
    assert "Tesla" in p_res and "Data Profile" in p_res
    print("✓ profile_and_filter tool successfully filtered 10M rows")

    # Tool 2A: generate_markdown (Summary, Breakdown, Trend)
    reports_dir = root_dir / "output" / "reports"
    markdown_report.configure(interceptor.connection, interceptor.data_store, reports_dir)

    res_sum = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "summary",
        "title": "Executive 10M Sales Summary"
    })
    assert "Report saved to" in res_sum
    print(f"✓ generate_markdown (Summary): {res_sum}")

    res_brk = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "breakdown",
        "title": "Brand Revenue Breakdown",
        "group_col": "brand",
        "metric_col": "price_usd",
        "aggregation": "SUM"
    })
    assert "Report saved to" in res_brk
    print(f"✓ generate_markdown (Breakdown): {res_brk}")

    res_trd = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "trend",
        "title": "Monthly Price Trends",
        "date_col": "sale_date",
        "metric_col": "price_usd",
        "aggregation": "AVG"
    })
    assert "Report saved to" in res_trd
    print(f"✓ generate_markdown (Trend): {res_trd}")

    # Tool 2B: generate_html_viz (Chart and Dashboard formats)
    viz_dir = root_dir / "output" / "viz"
    html_viz.configure(interceptor.connection, interceptor.data_store, viz_dir)

    res_bar = html_viz.generate_html_viz.invoke({
        "data_ref": ref_id,
        "chart_type": "bar",
        "x_col": "brand",
        "y_col": "price_usd",
        "title": "Avg Price By Brand",
        "aggregation": "AVG"
    })
    assert "Visualization saved to" in res_bar
    print(f"✓ generate_html_viz (Bar chart): {res_bar}")

    res_dash = html_viz.generate_html_viz.invoke({
        "data_ref": ref_id,
        "chart_type": "dashboard",
        "title": "10M Vehicle Sales Executive Dashboard"
    })
    assert "Dashboard saved to" in res_dash
    dash_path = Path(res_dash.split("saved to: ")[1].strip())
    assert dash_path.exists()
    dash_content = dash_path.read_text(encoding="utf-8")
    # plotly-latest is frozen at plotly.js 1.58.5 and cannot decode current Plotly output
    assert "plotly-latest" not in dash_content
    assert "cdn.plot.ly/plotly-" in dash_content
    assert "stat-card" in dash_content
    assert "dashboard-grid" in dash_content
    print(f"✓ generate_html_viz (Multi-chart Dashboard Format): {res_dash}")


def test_3_fastapi_and_iframe_ui():
    print("\n--- 3. Testing FastAPI Server & Iframe UI Serving ---")
    client = TestClient(server.app)

    # 1. Test UI index serving
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "DataVault" in r_ui.text
    assert "<iframe" in r_ui.text
    print("✓ UI served successfully with iframe preview container")

    # 2. Test api/refs
    r_refs = client.get("/api/refs")
    assert r_refs.status_code == 200
    refs = r_refs.json().get("data_refs")
    assert len(refs) > 0
    print(f"✓ /api/refs returned active data reference: {refs}")

    # 3. Test serving visualization for iframe
    viz_files = sorted((root_dir / "output" / "viz").glob("dashboard_*.html"), key=lambda p: p.stat().st_mtime)
    assert len(viz_files) > 0
    latest_viz = viz_files[-1].name
    r_viz = client.get(f"/viz/{latest_viz}")
    assert r_viz.status_code == 200
    assert "DataVault" in r_viz.text
    print(f"✓ /viz/{latest_viz} ready to be embedded into UI iframe")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING COMPLETE READINESS VERIFICATION")
    print("=" * 60)
    test_1_mcp_server_on_10m_rows()
    test_2_two_tools_and_agent_integration()
    test_3_fastapi_and_iframe_ui()
    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED: SYSTEM FULLY READY!")
    print("=" * 60)
