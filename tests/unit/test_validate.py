"""Tests for data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sol_ew.data.validate import (
    check_monotonic,
    detect_gaps,
    detect_header,
    interval_to_ms,
    normalize_timestamps,
    remove_duplicates,
    validate_klines,
    validate_ohlc,
    validate_volume,
)


def _make_klines(n: int, start_ts: int = 1_000_000_000_000, interval_ms: int = 60_000) -> pd.DataFrame:
    """Create synthetic 1m kline data for testing."""
    ts = [start_ts + i * interval_ms for i in range(n)]
    close_ts = [t + interval_ms - 1 for t in ts]
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    highs = prices + rng.uniform(0.1, 1.0, n)
    lows = prices - rng.uniform(0.1, 1.0, n)
    opens = lows + rng.uniform(0, 1, n) * (highs - lows)
    closes = lows + rng.uniform(0, 1, n) * (highs - lows)

    return pd.DataFrame({
        "open_time": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": rng.uniform(100, 10000, n),
        "close_time": close_ts,
        "quote_volume": rng.uniform(1000, 100000, n),
        "count": rng.integers(10, 1000, n),
        "taker_buy_volume": rng.uniform(50, 5000, n),
        "taker_buy_quote_volume": rng.uniform(500, 50000, n),
    })


class TestNormalizeTimestamps:
    """Test timestamp normalization (ms vs us detection)."""

    def test_already_milliseconds(self) -> None:
        df = pd.DataFrame({"open_time": [1_700_000_000_000], "close_time": [1_700_000_059_999]})
        result = normalize_timestamps(df)
        assert result["open_time"].iloc[0] == 1_700_000_000_000

    def test_microseconds_converted(self) -> None:
        # Microseconds: > 1e15
        us_ts = 1_700_000_000_000_000
        df = pd.DataFrame({"open_time": [us_ts], "close_time": [us_ts + 59_999_000]})
        result = normalize_timestamps(df)
        assert result["open_time"].iloc[0] == 1_700_000_000_000

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"open_time": pd.Series(dtype="int64"), "close_time": pd.Series(dtype="int64")})
        result = normalize_timestamps(df)
        assert len(result) == 0


class TestDetectHeader:
    """Test CSV header detection."""

    def test_numeric_first_row(self) -> None:
        assert detect_header(["1700000000000", "100.5", "101.0"]) is False

    def test_string_first_row(self) -> None:
        assert detect_header(["open_time", "open", "high"]) is True

    def test_empty_row(self) -> None:
        assert detect_header([]) is False


class TestValidateOHLC:
    """Test OHLC sanity validation."""

    def test_valid_ohlc(self) -> None:
        df = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [105.0, 103.0],
            "low": [98.0, 99.0],
            "close": [103.0, 100.5],
        })
        result = validate_ohlc(df)
        assert result.all()

    def test_low_above_high(self) -> None:
        df = pd.DataFrame({
            "open": [100.0],
            "high": [95.0],  # high < low = invalid
            "low": [98.0],
            "close": [97.0],
        })
        result = validate_ohlc(df)
        assert not result.iloc[0]

    def test_open_above_high(self) -> None:
        df = pd.DataFrame({
            "open": [110.0],  # open > high = invalid
            "high": [105.0],
            "low": [98.0],
            "close": [103.0],
        })
        result = validate_ohlc(df)
        assert not result.iloc[0]

    def test_close_below_low(self) -> None:
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [98.0],
            "close": [95.0],  # close < low = invalid
        })
        result = validate_ohlc(df)
        assert not result.iloc[0]


class TestValidateVolume:
    """Test volume validation."""

    def test_valid_volume(self) -> None:
        df = pd.DataFrame({"volume": [100.0, 200.0], "taker_buy_volume": [50.0, 100.0]})
        assert validate_volume(df).all()

    def test_negative_volume(self) -> None:
        df = pd.DataFrame({"volume": [-1.0], "taker_buy_volume": [50.0]})
        assert not validate_volume(df).iloc[0]

    def test_negative_taker_volume(self) -> None:
        df = pd.DataFrame({"volume": [100.0], "taker_buy_volume": [-1.0]})
        assert not validate_volume(df).iloc[0]


class TestRemoveDuplicates:
    """Test duplicate removal."""

    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 2000, 3000]})
        result, n = remove_duplicates(df)
        assert n == 0
        assert len(result) == 3

    def test_with_duplicates(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 2000, 2000, 3000]})
        result, n = remove_duplicates(df)
        assert n == 1
        assert len(result) == 3

    def test_all_duplicates(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 1000, 1000]})
        result, n = remove_duplicates(df)
        assert n == 2
        assert len(result) == 1


class TestDetectGaps:
    """Test gap detection."""

    def test_no_gaps(self) -> None:
        df = pd.DataFrame({"open_time": [0, 60000, 120000, 180000]})
        gaps = detect_gaps(df, interval_ms=60000)
        assert len(gaps) == 0

    def test_single_gap(self) -> None:
        # Missing bar at 120000
        df = pd.DataFrame({"open_time": [0, 60000, 180000, 240000]})
        gaps = detect_gaps(df, interval_ms=60000)
        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_bars"] == 1
        assert gaps.iloc[0]["gap_start"] == 60000
        assert gaps.iloc[0]["gap_end"] == 180000

    def test_large_gap(self) -> None:
        # Missing 5 bars
        df = pd.DataFrame({"open_time": [0, 60000, 420000]})
        gaps = detect_gaps(df, interval_ms=60000)
        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_bars"] == 5

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"open_time": pd.Series(dtype="int64")})
        gaps = detect_gaps(df, interval_ms=60000)
        assert len(gaps) == 0

    def test_single_row(self) -> None:
        df = pd.DataFrame({"open_time": [1000]})
        gaps = detect_gaps(df, interval_ms=60000)
        assert len(gaps) == 0


class TestCheckMonotonic:
    """Test monotonic timestamp check."""

    def test_monotonic(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 2000, 3000]})
        assert check_monotonic(df) is True

    def test_not_monotonic(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 3000, 2000]})
        assert check_monotonic(df) is False

    def test_duplicates_not_monotonic(self) -> None:
        df = pd.DataFrame({"open_time": [1000, 2000, 2000]})
        assert check_monotonic(df) is False


class TestIntervalToMs:
    """Test interval string to milliseconds conversion."""

    def test_minutes(self) -> None:
        assert interval_to_ms("1m") == 60_000
        assert interval_to_ms("5m") == 300_000
        assert interval_to_ms("15m") == 900_000

    def test_hours(self) -> None:
        assert interval_to_ms("1h") == 3_600_000
        assert interval_to_ms("4h") == 14_400_000

    def test_days(self) -> None:
        assert interval_to_ms("1D") == 86_400_000

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            interval_to_ms("1x")


class TestValidateKlines:
    """Test full validation pipeline."""

    def test_valid_data(self) -> None:
        df = _make_klines(100)
        cleaned, report = validate_klines(df, "SOLUSDT", "1m")
        assert report.is_valid
        assert report.duplicates_removed == 0
        assert report.ohlc_violations == 0
        assert report.volume_violations == 0
        assert len(cleaned) == 100

    def test_with_duplicates(self) -> None:
        df = _make_klines(10)
        df = pd.concat([df, df.iloc[[3, 5]]], ignore_index=True)
        cleaned, report = validate_klines(df, "SOLUSDT", "1m")
        assert report.duplicates_removed == 2
        assert len(cleaned) == 10

    def test_with_gaps(self) -> None:
        df = _make_klines(20)
        # Remove rows 5-8 to create a gap
        df = df.drop(index=[5, 6, 7, 8]).reset_index(drop=True)
        _, report = validate_klines(df, "SOLUSDT", "1m")
        assert report.gaps_found > 0

    def test_microsecond_normalization(self) -> None:
        df = _make_klines(5)
        # Convert to microseconds
        df["open_time"] = df["open_time"] * 1000
        df["close_time"] = df["close_time"] * 1000
        cleaned, report = validate_klines(df, "SOLUSDT", "1m")
        assert report.timestamp_unit_detected == "us"
        # Timestamps should now be in ms range
        assert cleaned["open_time"].iloc[0] < 1e15
