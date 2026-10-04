"""Data validation utilities.

Handles timestamp normalization, OHLC sanity checks, gap detection,
duplicate removal, and gap report generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Threshold to distinguish milliseconds from microseconds.
# Timestamps > 1e15 are microseconds (year ~2001+ in us vs ~33658 in ms).
_US_THRESHOLD = 1_000_000_000_000_000  # 1e15


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a kline dataset."""

    symbol: str
    interval: str
    total_rows: int
    duplicates_removed: int
    gaps_found: int
    ohlc_violations: int
    volume_violations: int
    timestamp_unit_detected: str  # "ms" or "us"
    is_valid: bool
    messages: tuple[str, ...]


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps to UTC integer milliseconds.

    Detects whether timestamps are in milliseconds or microseconds by magnitude,
    and converts to milliseconds. All timestamp columns (open_time, close_time)
    are normalized.

    Args:
        df: DataFrame with open_time and close_time columns.

    Returns:
        DataFrame with timestamps normalized to UTC milliseconds.
    """
    df = df.copy()
    for col in ("open_time", "close_time"):
        if col not in df.columns:
            continue
        values = df[col].values
        if len(values) == 0:
            continue
        # Check the first non-null value
        sample = values[values > 0][0] if np.any(values > 0) else 0
        if sample >= _US_THRESHOLD:
            logger.info("Detected microsecond timestamps in %s, converting to ms", col)
            df[col] = df[col] // 1000
    return df


def detect_header(first_row: list[str]) -> bool:
    """Detect if the first row of a CSV is a header.

    Args:
        first_row: List of string values from the first CSV row.

    Returns:
        True if the row appears to be a header (non-numeric first field).
    """
    if not first_row:
        return False
    try:
        float(first_row[0])
        return False
    except (ValueError, IndexError):
        return True


def validate_ohlc(df: pd.DataFrame) -> pd.Series:
    """Check OHLC sanity: low <= open,close <= high.

    Args:
        df: DataFrame with o/h/l/c or open/high/low/close columns.

    Returns:
        Boolean Series where True means the row is valid.
    """
    o = df["open"] if "open" in df.columns else df["o"]
    h = df["high"] if "high" in df.columns else df["h"]
    low = df["low"] if "low" in df.columns else df["l"]
    c = df["close"] if "close" in df.columns else df["c"]

    valid = (low <= o) & (o <= h) & (low <= c) & (c <= h) & (low <= h)
    return valid


def validate_volume(df: pd.DataFrame) -> pd.Series:
    """Check that volume fields are non-negative.

    Args:
        df: DataFrame with volume and taker_buy_volume columns.

    Returns:
        Boolean Series where True means volumes are valid.
    """
    valid = df["volume"] >= 0
    if "taker_buy_volume" in df.columns:
        valid = valid & (df["taker_buy_volume"] >= 0)
    return valid


def remove_duplicates(df: pd.DataFrame, ts_col: str = "open_time") -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows by timestamp, keeping the first occurrence.

    Args:
        df: DataFrame with a timestamp column.
        ts_col: Name of the timestamp column.

    Returns:
        Tuple of (deduplicated DataFrame, number of duplicates removed).
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=[ts_col], keep="first")
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning("Removed %d duplicate rows by %s", n_removed, ts_col)
    return df, n_removed


def detect_gaps(
    df: pd.DataFrame,
    interval_ms: int,
    ts_col: str = "open_time",
) -> pd.DataFrame:
    """Detect gaps (missing bars) in a time series.

    Args:
        df: DataFrame sorted by timestamp.
        interval_ms: Expected interval between bars in milliseconds.
        ts_col: Name of the timestamp column.

    Returns:
        DataFrame with columns [gap_start, gap_end, missing_bars] for each gap.
    """
    if len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_bars"])

    ts = df[ts_col].values
    diffs = np.diff(ts)
    # A gap exists where the difference is more than 1.5x the expected interval
    gap_mask = diffs > interval_ms * 1.5

    if not np.any(gap_mask):
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_bars"])

    gap_indices = np.where(gap_mask)[0]
    gaps = []
    for idx in gap_indices:
        gap_start = int(ts[idx])
        gap_end = int(ts[idx + 1])
        missing = int((gap_end - gap_start) / interval_ms) - 1
        gaps.append({
            "gap_start": gap_start,
            "gap_end": gap_end,
            "missing_bars": missing,
        })

    return pd.DataFrame(gaps)


def check_monotonic(df: pd.DataFrame, ts_col: str = "open_time") -> bool:
    """Check that timestamps are strictly monotonically increasing.

    Args:
        df: DataFrame with a timestamp column.
        ts_col: Name of the timestamp column.

    Returns:
        True if timestamps are strictly monotonically increasing.
    """
    if len(df) < 2:
        return True
    return bool(np.all(np.diff(df[ts_col].values) > 0))


def interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds.

    Args:
        interval: Interval string like "1m", "5m", "1h", "4h", "1D".

    Returns:
        Duration in milliseconds.
    """
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "D": 86_400_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "W": 604_800_000,
    }
    unit = interval[-1]
    value = int(interval[:-1])
    if unit not in multipliers:
        raise ValueError(f"Unknown interval unit: {unit!r} in {interval!r}")
    return value * multipliers[unit]


def validate_klines(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Run full validation pipeline on kline data.

    Normalizes timestamps, removes duplicates, checks OHLC/volume sanity,
    detects gaps, and ensures monotonic ordering.

    Args:
        df: Raw kline DataFrame.
        symbol: Trading pair symbol.
        interval: Bar interval string.

    Returns:
        Tuple of (cleaned DataFrame, ValidationReport).
    """
    messages: list[str] = []
    total_rows = len(df)

    # 1. Normalize timestamps
    ts_unit = "ms"
    if len(df) > 0:
        sample = df["open_time"].iloc[0]
        if sample >= _US_THRESHOLD:
            ts_unit = "us"
    df = normalize_timestamps(df)
    messages.append(f"Timestamp unit detected: {ts_unit}")

    # 2. Sort by timestamp
    df = df.sort_values("open_time").reset_index(drop=True)

    # 3. Remove duplicates
    df, n_dupes = remove_duplicates(df, "open_time")
    if n_dupes > 0:
        messages.append(f"Removed {n_dupes} duplicate rows")

    # 4. Check monotonic
    if not check_monotonic(df, "open_time"):
        messages.append("WARNING: timestamps not monotonic after dedup+sort")

    # 5. OHLC sanity
    ohlc_valid = validate_ohlc(df)
    n_ohlc_bad = int((~ohlc_valid).sum())
    if n_ohlc_bad > 0:
        messages.append(f"OHLC violations: {n_ohlc_bad} rows")

    # 6. Volume sanity
    vol_valid = validate_volume(df)
    n_vol_bad = int((~vol_valid).sum())
    if n_vol_bad > 0:
        messages.append(f"Volume violations: {n_vol_bad} rows")

    # 7. Gap detection
    interval_ms = interval_to_ms(interval)
    gaps_df = detect_gaps(df, interval_ms, "open_time")
    n_gaps = len(gaps_df)
    if n_gaps > 0:
        total_missing = int(gaps_df["missing_bars"].sum())
        messages.append(f"Gaps found: {n_gaps} gaps, {total_missing} total missing bars")

    is_valid = n_ohlc_bad == 0 and n_vol_bad == 0
    if is_valid:
        messages.append("Validation PASSED")
    else:
        messages.append("Validation FAILED")

    report = ValidationReport(
        symbol=symbol,
        interval=interval,
        total_rows=total_rows,
        duplicates_removed=n_dupes,
        gaps_found=n_gaps,
        ohlc_violations=n_ohlc_bad,
        volume_violations=n_vol_bad,
        timestamp_unit_detected=ts_unit,
        is_valid=is_valid,
        messages=tuple(messages),
    )

    return df, report


def save_gap_report(
    gaps_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save gap report as Parquet file.

    Args:
        gaps_df: DataFrame with gap_start, gap_end, missing_bars columns.
        output_path: Path to write the Parquet file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(gaps_df)
    pq.write_table(table, output_path)
    logger.info("Gap report saved to %s (%d gaps)", output_path, len(gaps_df))
