"""Generate 10 million rows of realistic vehicle sales data using DuckDB.

Uses DuckDB's internal SQL functions for maximum speed — no Python loops,
no pandas. The generated data is saved as a Parquet file.

Run:
    python generate_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def generate_vehicle_sales(
    num_rows: int = 10_000_000,
    output_dir: Path | None = None,
) -> Path:
    """Generate vehicle sales data and save as Parquet.

    Args:
        num_rows: Number of rows to generate (default 10M).
        output_dir: Directory to save the Parquet file.

    Returns:
        Path to the generated Parquet file.
    """
    import duckdb

    if output_dir is None:
        output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vehicle_sales.parquet"

    print(f"Generating {num_rows:,} vehicle sales rows...")
    start = time.time()

    con = duckdb.connect(":memory:")

    con.execute(f"""
        CREATE TABLE vehicle_sales AS
        SELECT
            i AS sale_id,
            DATE '2019-01-01' + INTERVAL (CAST(floor(random() * 2190) AS INTEGER)) DAY AS sale_date,

            -- Vehicle type
            CASE CAST(floor(random() * 5) AS INTEGER)
                WHEN 0 THEN 'Car'
                WHEN 1 THEN 'Bike'
                WHEN 2 THEN 'Truck'
                WHEN 3 THEN 'SUV'
                ELSE 'Van'
            END AS vehicle_type,

            -- Brand
            CASE CAST(floor(random() * 15) AS INTEGER)
                WHEN 0 THEN 'Toyota' WHEN 1 THEN 'Honda' WHEN 2 THEN 'Ford'
                WHEN 3 THEN 'BMW' WHEN 4 THEN 'Tesla' WHEN 5 THEN 'Hyundai'
                WHEN 6 THEN 'Chevrolet' WHEN 7 THEN 'Nissan' WHEN 8 THEN 'Mercedes'
                WHEN 9 THEN 'Audi' WHEN 10 THEN 'Yamaha' WHEN 11 THEN 'Harley-Davidson'
                WHEN 12 THEN 'Ducati' WHEN 13 THEN 'Kawasaki' ELSE 'Suzuki'
            END AS brand,

            -- Model (simplified)
            CASE CAST(floor(random() * 10) AS INTEGER)
                WHEN 0 THEN 'Sedan LX' WHEN 1 THEN 'Sport GT' WHEN 2 THEN 'Touring XL'
                WHEN 3 THEN 'Classic 500' WHEN 4 THEN 'EV Pro' WHEN 5 THEN 'Hybrid SE'
                WHEN 6 THEN 'Adventure R' WHEN 7 THEN 'Cruiser V2' WHEN 8 THEN 'Compact S'
                ELSE 'Premium Max'
            END AS model,

            -- Color
            CASE CAST(floor(random() * 7) AS INTEGER)
                WHEN 0 THEN 'White' WHEN 1 THEN 'Black' WHEN 2 THEN 'Silver'
                WHEN 3 THEN 'Red' WHEN 4 THEN 'Blue' WHEN 5 THEN 'Gray'
                ELSE 'Green'
            END AS color,

            -- Price based on vehicle type (realistic ranges)
            ROUND(CASE
                WHEN floor(random() * 5) = 1 THEN 5000 + random() * 25000   -- Bike
                WHEN floor(random() * 5) = 2 THEN 30000 + random() * 60000  -- Truck
                WHEN floor(random() * 5) = 3 THEN 25000 + random() * 45000  -- SUV
                WHEN floor(random() * 5) = 4 THEN 25000 + random() * 25000  -- Van
                ELSE 20000 + random() * 60000                                -- Car
            END, 2)::DECIMAL(10,2) AS price_usd,

            -- Quantity
            CAST(1 + floor(random() * 5) AS INTEGER) AS quantity,

            -- US State
            CASE CAST(floor(random() * 50) AS INTEGER)
                WHEN 0 THEN 'AL' WHEN 1 THEN 'AK' WHEN 2 THEN 'AZ' WHEN 3 THEN 'AR'
                WHEN 4 THEN 'CA' WHEN 5 THEN 'CO' WHEN 6 THEN 'CT' WHEN 7 THEN 'DE'
                WHEN 8 THEN 'FL' WHEN 9 THEN 'GA' WHEN 10 THEN 'HI' WHEN 11 THEN 'ID'
                WHEN 12 THEN 'IL' WHEN 13 THEN 'IN' WHEN 14 THEN 'IA' WHEN 15 THEN 'KS'
                WHEN 16 THEN 'KY' WHEN 17 THEN 'LA' WHEN 18 THEN 'ME' WHEN 19 THEN 'MD'
                WHEN 20 THEN 'MA' WHEN 21 THEN 'MI' WHEN 22 THEN 'MN' WHEN 23 THEN 'MS'
                WHEN 24 THEN 'MO' WHEN 25 THEN 'MT' WHEN 26 THEN 'NE' WHEN 27 THEN 'NV'
                WHEN 28 THEN 'NH' WHEN 29 THEN 'NJ' WHEN 30 THEN 'NM' WHEN 31 THEN 'NY'
                WHEN 32 THEN 'NC' WHEN 33 THEN 'ND' WHEN 34 THEN 'OH' WHEN 35 THEN 'OK'
                WHEN 36 THEN 'OR' WHEN 37 THEN 'PA' WHEN 38 THEN 'RI' WHEN 39 THEN 'SC'
                WHEN 40 THEN 'SD' WHEN 41 THEN 'TN' WHEN 42 THEN 'TX' WHEN 43 THEN 'UT'
                WHEN 44 THEN 'VT' WHEN 45 THEN 'VA' WHEN 46 THEN 'WA' WHEN 47 THEN 'WV'
                WHEN 48 THEN 'WI' ELSE 'WY'
            END AS customer_state,

            -- Dealer ID
            CAST(1 + floor(random() * 500) AS INTEGER) AS dealer_id

        FROM range({num_rows}) t(i)
    """)

    gen_time = time.time() - start
    print(f"  Generated in {gen_time:.2f}s")

    # Export to Parquet
    print("Exporting to Parquet...")
    export_start = time.time()
    con.execute(f"COPY vehicle_sales TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    export_time = time.time() - export_start

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - start

    print(f"  Export time: {export_time:.2f}s")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Total: {total_time:.2f}s")

    # Quick verification
    row_count = con.execute("SELECT COUNT(*) FROM vehicle_sales").fetchone()[0]
    print(f"  Verified rows: {row_count:,}")

    con.close()
    return output_path


if __name__ == "__main__":
    generate_vehicle_sales()
