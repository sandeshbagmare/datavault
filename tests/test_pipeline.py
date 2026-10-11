"""End-to-end test script for the DataVault pipeline.

Tests the full flow WITHOUT requiring an LLM API key:
1. Data generation (if needed)
2. Profiler + DataStore
3. Interceptor
4. All 3 tools (profile_filter, html_viz, markdown_report)

Run with:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb

from datavault.proxy.profiler import DataProfiler, DataStore, DataProfile
from datavault.proxy.interceptor import DataProxyInterceptor
from datavault.tools import profile_filter, html_viz, markdown_report


def _check_profiler():
    """Test the DataProfiler with inline test data."""
    print("\n=== Test 1: DataProfiler ===")
    conn = duckdb.connect(":memory:")
    store = DataStore()
    profiler = DataProfiler(conn, store)

    # Create test data
    conn.execute("""
        CREATE TABLE test_sales AS
        SELECT
            i AS sale_id,
            CASE WHEN i % 3 = 0 THEN 'Car' WHEN i % 3 = 1 THEN 'Bike' ELSE 'Truck' END AS vehicle_type,
            CASE WHEN i % 4 = 0 THEN 'Toyota' WHEN i % 4 = 1 THEN 'Honda'
                 WHEN i % 4 = 2 THEN 'Ford' ELSE 'Tesla' END AS brand,
            (10000 + i * 7.3)::DECIMAL(10,2) AS price_usd,
            DATE '2020-01-01' + INTERVAL (i % 1800) DAY AS sale_date
        FROM range(10000) t(i)
    """)

    profile = profiler.profile("test_sales")
    print(f"  ref_id: {profile.ref_id}")
    print(f"  rows: {profile.row_count}")
    print(f"  columns: {len(profile.columns)}")
    print(f"  stats: {len(profile.statistics)}")
    print(f"  sample: {len(profile.sample_rows)}")

    # Check profile text
    text = profile.to_llm_text()
    token_est = len(text.split())
    print(f"  profile tokens (approx): {token_est}")
    assert profile.row_count == 10000, f"Expected 10000 rows, got {profile.row_count}"
    assert len(profile.columns) == 5
    assert len(profile.sample_rows) == 5
    assert token_est < 500, f"Profile too large: {token_est} tokens"
    print("  PASSED")
    return conn, store, profiler, profile


def _check_profiler_with_filter(conn, store, profiler, initial_ref_id):
    """Test profiling with a SQL filter."""
    print("\n=== Test 2: Profile with Filter ===")

    ref = store.get(initial_ref_id)
    filtered_profile = profiler.profile(ref.view_name, filter_sql="brand = 'Tesla'")
    print(f"  ref_id: {filtered_profile.ref_id}")
    print(f"  filtered rows: {filtered_profile.row_count}")
    print(f"  filter: {filtered_profile.filter_applied}")
    assert filtered_profile.row_count == 2500, f"Expected 2500 Tesla rows, got {filtered_profile.row_count}"
    assert filtered_profile.filter_applied == "brand = 'Tesla'"
    print("  PASSED")
    return filtered_profile


def _check_interceptor():
    """Test the DataProxyInterceptor."""
    print("\n=== Test 3: DataProxyInterceptor ===")
    interceptor = DataProxyInterceptor()

    # Test passthrough
    result = interceptor.intercept("list_tables", "Tables: vehicle_sales")
    assert result == "Tables: vehicle_sales", "Passthrough failed"
    print("  Passthrough: PASSED")

    # Test markdown table interception
    md_table = """| brand | count |
|-------|-------|
| Toyota | 1500 |
| Honda | 1200 |
| Ford | 1100 |"""
    result = interceptor.intercept("query_data", md_table)
    assert "Data Profile" in result, f"Interception failed, got: {result[:100]}"
    assert "ref_" in result, "No reference ID in profile"
    print("  Markdown interception: PASSED")

    # Test JSON interception
    json_data = '[{"brand": "Tesla", "count": 500}, {"brand": "BMW", "count": 300}]'
    result = interceptor.intercept("query_data", json_data)
    assert "Data Profile" in result, f"JSON interception failed, got: {result[:100]}"
    print("  JSON interception: PASSED")

    print("  PASSED")
    return interceptor


def _check_profile_filter_tool(conn, store, profiler, ref_id):
    """Test the profile_and_filter tool."""
    print("\n=== Test 4: profile_and_filter Tool ===")

    profile_filter.configure(profiler, store)

    # Test basic profile
    result = profile_filter.profile_and_filter.invoke({
        "data_ref": ref_id,
    })
    assert "Data Profile" in result, f"Tool failed: {result[:100]}"
    print("  Basic profile: PASSED")

    # Test with filter
    result = profile_filter.profile_and_filter.invoke({
        "data_ref": ref_id,
        "filter_query": "brand = 'Honda'",
    })
    assert "Data Profile" in result, f"Filter tool failed: {result[:100]}"
    assert "Honda" in result
    print("  Filtered profile: PASSED")

    # Test invalid ref
    result = profile_filter.profile_and_filter.invoke({
        "data_ref": "ref_nonexistent",
    })
    assert "Error" in result
    print("  Error handling: PASSED")

    print("  PASSED")


def _check_html_viz_tool(conn, store):
    """Test the generate_html_viz tool."""
    print("\n=== Test 5: generate_html_viz Tool ===")

    output_dir = Path(__file__).parent.parent / "output" / "viz"
    html_viz.configure(conn, store, output_dir)

    refs = store.list_refs()
    ref_id = refs[0]

    # Test bar chart
    result = html_viz.generate_html_viz.invoke({
        "data_ref": ref_id,
        "chart_type": "bar",
        "x_col": "brand",
        "y_col": "price_usd",
        "title": "Average Price by Brand",
        "aggregation": "AVG",
    })
    assert "saved to" in result.lower(), f"Bar chart failed: {result}"
    print(f"  Bar chart: {result}")

    # Test dashboard
    result = html_viz.generate_html_viz.invoke({
        "data_ref": ref_id,
        "chart_type": "dashboard",
        "x_col": "",
        "y_col": "",
        "title": "Vehicle Sales Dashboard",
    })
    assert "saved to" in result.lower() or "dashboard" in result.lower(), f"Dashboard failed: {result}"
    print(f"  Dashboard: {result}")

    # Test pie chart
    result = html_viz.generate_html_viz.invoke({
        "data_ref": ref_id,
        "chart_type": "pie",
        "x_col": "vehicle_type",
        "title": "Sales by Vehicle Type",
    })
    assert "saved to" in result.lower(), f"Pie chart failed: {result}"
    print(f"  Pie chart: {result}")

    print("  PASSED")


def _check_markdown_report_tool(conn, store):
    """Test the generate_markdown tool."""
    print("\n=== Test 6: generate_markdown Tool ===")

    output_dir = Path(__file__).parent.parent / "output" / "reports"
    markdown_report.configure(conn, store, output_dir)

    refs = store.list_refs()
    ref_id = refs[0]

    # Test summary report
    result = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "summary",
        "title": "Vehicle Sales Summary",
    })
    assert "saved to" in result.lower(), f"Summary report failed: {result}"
    print(f"  Summary: {result}")

    # Test breakdown report
    result = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "breakdown",
        "title": "Sales by Brand",
        "group_col": "brand",
        "metric_col": "price_usd",
    })
    assert "saved to" in result.lower(), f"Breakdown failed: {result}"
    print(f"  Breakdown: {result}")

    # Test trend report
    result = markdown_report.generate_markdown.invoke({
        "data_ref": ref_id,
        "report_type": "trend",
        "title": "Price Trend",
        "date_col": "sale_date",
        "metric_col": "price_usd",
        "aggregation": "AVG",
    })
    assert "saved to" in result.lower(), f"Trend failed: {result}"
    print(f"  Trend: {result}")

    print("  PASSED")


def main():
    """Run all tests."""
    print("=" * 60)
    print("DataVault End-to-End Pipeline Test")
    print("=" * 60)

    start = time.time()
    passed = 0
    failed = 0

    try:
        conn, store, profiler, profile = _check_profiler()
        passed += 1

        _check_profiler_with_filter(conn, store, profiler, profile.ref_id)
        passed += 1

        _check_interceptor()
        passed += 1

        _check_profile_filter_tool(conn, store, profiler, profile.ref_id)
        passed += 1

        _check_html_viz_tool(conn, store)
        passed += 1

        _check_markdown_report_tool(conn, store)
        passed += 1

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed in {elapsed:.2f}s")
    print(f"{'=' * 60}")

    return 0 if failed == 0 else 1


def test_pipeline() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
