"""The SQL guard and the structured query builder."""

import duckdb
import pytest

from datavault.query import QueryError, build_select, validate_select

ALLOWED = {"sales", "ref_abc"}


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE sales AS SELECT i AS a, i % 3 AS g FROM range(10) t(i)")
    return c


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT g, SUM(a) FROM sales GROUP BY g",
        "WITH x AS (SELECT * FROM sales) SELECT a, ROW_NUMBER() OVER (PARTITION BY g ORDER BY a) FROM x",
        "SELECT a FROM sales USING SAMPLE 3 ROWS",
        "SELECT s1.a FROM sales s1 JOIN sales s2 ON s1.a = s2.a",
        "SELECT * FROM range(5)",
        "SELECT a FROM sales WHERE a IN (SELECT a FROM ref_abc)",
        "SELECT a FROM sales;",
        "WITH a AS (SELECT * FROM sales), b AS (SELECT * FROM a) SELECT * FROM b",
        "SELECT * FROM (WITH x AS (SELECT 1 AS n) SELECT n FROM x) t",
    ],
)
def test_allows_read_only_selects(conn, sql):
    conn.execute("CREATE OR REPLACE VIEW ref_abc AS SELECT * FROM sales")
    assert validate_select(conn, sql, ALLOWED)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE z AS SELECT 1",
        "DROP TABLE sales",
        "COPY sales TO 'out.csv'",
        "SELECT 1; DROP TABLE sales",
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM read_parquet('x.parquet')",
        "SELECT * FROM 'secret.csv'",
        "SELECT * FROM sales, read_text('x')",
        "SELECT getenv('PATH')",
        "SELECT * FROM glob('*')",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM duckdb_tables()",
        "SELECT * FROM other_table",
        "SELECT a FROM sales WHERE a IN (SELECT 1 FROM read_csv('x'))",
        "SELECT * FROM query_table('sales')",
        "",
        # a CTE must not make a real, unregistered table look allowed (regression)
        "WITH secret AS (SELECT * FROM secret) SELECT * FROM secret",
        "SELECT * FROM (WITH secret AS (SELECT * FROM secret) SELECT * FROM secret) t",
        "WITH a AS (SELECT * FROM b), b AS (SELECT 1) SELECT * FROM a",
        "SELECT * FROM x WHERE 1 IN (WITH x AS (SELECT 1) SELECT * FROM x)",
    ],
)
def test_rejects_everything_else(conn, sql):
    conn.execute("CREATE TABLE secret AS SELECT 'TOP-SECRET' AS pw")
    with pytest.raises(QueryError):
        validate_select(conn, sql, ALLOWED)


COLUMNS = [("brand", "VARCHAR"), ("price", "DOUBLE"), ("sale_date", "DATE")]


def test_group_by_metrics_and_default_order():
    q = build_select("v", COLUMNS, {"group_by": ["brand"], "metrics": [{"agg": "avg", "col": "price", "as": "avg_price"}]})
    assert 'AVG("price") AS "avg_price"' in q.sql
    assert 'GROUP BY 1' in q.sql and 'ORDER BY "avg_price" DESC' in q.sql
    assert q.aggregated and q.ordered


def test_time_grain_orders_chronologically():
    q = build_select("v", COLUMNS, {"group_by": [{"col": "sale_date", "grain": "month", "as": "m"}], "metrics": ["sum(price)"]})
    assert "DATE_TRUNC('month'" in q.sql and 'ORDER BY "m" ASC' in q.sql


def test_metric_strings_and_percentiles():
    q = build_select("v", COLUMNS, {"metrics": ["COUNT(*)", "p90(price)", "count_distinct(brand) as brands"]})
    assert "COUNT(*)" in q.sql and "quantile_cont" in q.sql and 'COUNT(DISTINCT "brand") AS "brands"' in q.sql


def test_unknown_column_error_lists_available_columns():
    with pytest.raises(QueryError, match="Available columns.*brand"):
        build_select("v", COLUMNS, {"group_by": ["colour"]})


def test_unknown_aggregation_and_grain():
    with pytest.raises(QueryError, match="Unknown aggregation"):
        build_select("v", COLUMNS, {"metrics": [{"agg": "geomean", "col": "price"}]})
    with pytest.raises(QueryError, match="Unknown grain"):
        build_select("v", COLUMNS, {"group_by": [{"col": "sale_date", "grain": "fortnight"}], "metrics": ["count(*)"]})


def test_identifiers_are_quoted_not_interpolated():
    cols = [('we"ird; DROP TABLE x', "VARCHAR")]
    q = build_select("v", cols, {"select": ['we"ird; DROP TABLE x']})
    assert '"we""ird; DROP TABLE x"' in q.sql
