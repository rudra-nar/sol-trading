"""Tests for the download module."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from sol_ew.data.download import (
    _daily_url,
    _monthly_url,
    _parquet_path,
    extract_csv_from_zip,
    generate_month_ranges,
    parse_kline_csv,
    save_parquet,
)


class TestURLGeneration:
    """Test URL pattern generation."""

    def test_monthly_url(self) -> None:
        url = _monthly_url("SOLUSDT", "1m", 2023, 6)
        assert url == (
            "https://data.binance.vision/data/futures/um/monthly/klines/"
            "SOLUSDT/1m/SOLUSDT-1m-2023-06.zip"
        )

    def test_daily_url(self) -> None:
        url = _daily_url("SOLUSDT", "1m", date(2023, 6, 15))
        assert url == (
            "https://data.binance.vision/data/futures/um/daily/klines/"
            "SOLUSDT/1m/SOLUSDT-1m-2023-06-15.zip"
        )


class TestParquetPath:
    """Test Parquet path generation."""

    def test_path_structure(self) -> None:
        path = _parquet_path(Path("data"), "SOLUSDT", "1m", 2023, 6)
        assert "symbol=SOLUSDT" in str(path)
        assert "interval=1m" in str(path)
        assert "year=2023" in str(path)
        assert "month=06" in str(path)
        assert str(path).endswith("part.parquet")


class TestGenerateMonthRanges:
    """Test month range generation."""

    def test_single_month(self) -> None:
        months = generate_month_ranges(date(2023, 6, 1), date(2023, 6, 30))
        assert months == [(2023, 6)]

    def test_cross_year(self) -> None:
        months = generate_month_ranges(date(2022, 11, 1), date(2023, 2, 28))
        assert months == [(2022, 11), (2022, 12), (2023, 1), (2023, 2)]

    def test_same_day(self) -> None:
        months = generate_month_ranges(date(2023, 6, 15), date(2023, 6, 15))
        assert months == [(2023, 6)]

    def test_multi_year(self) -> None:
        months = generate_month_ranges(date(2020, 1, 1), date(2020, 3, 31))
        assert len(months) == 3


class TestParseKlineCsv:
    """Test CSV parsing with header/headerless handling."""

    def test_headerless_csv(self) -> None:
        csv = (
            "1700000000000,100.0,105.0,98.0,103.0,1000.0,"
            "1700000059999,50000.0,500,600.0,30000.0,0\n"
            "1700000060000,103.0,106.0,101.0,104.0,1200.0,"
            "1700000119999,60000.0,600,700.0,35000.0,0\n"
        )
        df = parse_kline_csv(csv.encode())
        assert len(df) == 2
        assert "open_time" in df.columns
        assert "ignore" not in df.columns
        assert df["open"].iloc[0] == 100.0

    def test_with_header(self) -> None:
        csv = (
            "open_time,open,high,low,close,volume,"
            "close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
            "1700000000000,100.0,105.0,98.0,103.0,1000.0,"
            "1700000059999,50000.0,500,600.0,30000.0,0\n"
        )
        df = parse_kline_csv(csv.encode())
        assert len(df) == 1
        assert df["open_time"].iloc[0] == 1700000000000

    def test_empty_csv(self) -> None:
        df = parse_kline_csv(b"")
        assert len(df) == 0

    def test_microsecond_timestamps_normalized(self) -> None:
        # Use microsecond timestamps (> 1e15)
        us_ts = 1_700_000_000_000_000
        csv = (
            f"{us_ts},100.0,105.0,98.0,103.0,1000.0,"
            f"{us_ts + 59_999_000},50000.0,500,600.0,30000.0,0\n"
        )
        df = parse_kline_csv(csv.encode())
        assert df["open_time"].iloc[0] == 1_700_000_000_000


class TestExtractCsvFromZip:
    """Test ZIP extraction."""

    def test_extract(self) -> None:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.csv", "1,2,3\n4,5,6\n")
        zip_bytes = buf.getvalue()

        csv_bytes = extract_csv_from_zip(zip_bytes)
        assert b"1,2,3" in csv_bytes

    def test_no_csv_raises(self) -> None:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.txt", "not a csv")
        zip_bytes = buf.getvalue()

        with pytest.raises(ValueError, match="No CSV"):
            extract_csv_from_zip(zip_bytes)


class TestSaveParquet:
    """Test Parquet save/load roundtrip."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        df = pd.DataFrame({
            "open_time": [1000, 2000],
            "open": [100.0, 101.0],
            "close": [103.0, 102.0],
        })

        path = tmp_path / "sub" / "dir" / "test.parquet"
        save_parquet(df, path)

        assert path.exists()
        loaded = pq.read_table(path).to_pandas()
        assert len(loaded) == 2
        assert loaded["open"].iloc[0] == 100.0


import pytest  # noqa: E402
