"""Shared fixtures: a small deterministic dataset in an isolated DataContext."""

from __future__ import annotations

import json
import re
from typing import Any

import duckdb
import pytest

from datavault.context import DataContext, Policy

BRANDS = ["Toyota", "Honda", "Ford", "Tesla"]
N_ROWS = 1000


def sales_rows() -> list[dict[str, Any]]:
    """The same rows the SQL below produces, for computing expected values in Python."""
    from datetime import date, timedelta

    return [
        {
            "id": i,
            "brand": BRANDS[i % 4],
            "vehicle_type": ("Car", "SUV", "Truck")[i % 3],
            "price": float(1000 + (i % 50) * 10),
            "sale_date": date(2022, 1, 1) + timedelta(days=i),
            "buyer_email": f"user{i}@example.com",
        }
        for i in range(N_ROWS)
    ]


def make_context(tmp_path, *, name: str = "sales", **policy: Any) -> DataContext:
    conn = duckdb.connect(":memory:")
    conn.execute(
        f"""
        CREATE TABLE {name} AS SELECT
            i AS id,
            ['Toyota','Honda','Ford','Tesla'][1 + i % 4] AS brand,
            CASE WHEN i % 3 = 0 THEN 'Car' WHEN i % 3 = 1 THEN 'SUV' ELSE 'Truck' END AS vehicle_type,
            (1000 + (i % 50) * 10)::DOUBLE AS price,
            DATE '2022-01-01' + INTERVAL (i) DAY AS sale_date,
            'user' || i || '@example.com' AS buyer_email,
            repeat('x', 300) AS notes
        FROM range({N_ROWS}) t(i)
        """
    )
    ctx = DataContext(connection=conn, policy=Policy(**policy), output_dir=tmp_path)
    ctx.register_table(name)
    return ctx


@pytest.fixture
def ctx(tmp_path) -> DataContext:
    return make_context(tmp_path)


def tools_by_name(ctx: DataContext) -> dict[str, Any]:
    from datavault.tools import create_tools

    return {t.name: t for t in create_tools(ctx)}


def bridge(html: str) -> dict[str, Any]:
    """The window.DATAVAULT object embedded in a generated page."""
    match = re.search(r"window\.DATAVAULT = (.*?);</script>", html, re.S)
    assert match, "page has no DATAVAULT bridge"
    return json.loads(match.group(1))


def saved_path(result: str):
    from pathlib import Path

    return Path(result.strip().splitlines()[-1].split("saved to: ")[1].strip())
