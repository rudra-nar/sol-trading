"""Historical kline downloader from Binance public data dump.

Downloads kline CSVs from data.binance.vision, validates them, and stores
as partitioned Parquet files. Resumable and idempotent.

URL patterns (verify at implementation time; layout may change):
  Monthly: https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-YYYY-MM.zip
  Daily:   https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-YYYY-MM-DD.zip
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sol_ew.data.validate import (
    detect_gaps,
    detect_header,
    interval_to_ms,
    normalize_timestamps,
    save_gap_report,
    validate_klines,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

KLINE_DTYPES = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_volume": "float64",
    "count": "int64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "ignore": "str",
}


def _parquet_path(data_dir: Path, symbol: str, interval: str, year: int, month: int) -> Path:
    """Build the Hive-partitioned Parquet path."""
    return (
        data_dir / "raw" / "klines"
        / f"symbol={symbol}" / f"interval={interval}"
        / f"year={year}" / f"month={month:02d}"
        / "part.parquet"
    )


def _monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    """Build the monthly kline download URL."""
    return (
        f"{BASE_URL}/monthly/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{year}-{month:02d}.zip"
    )


def _daily_url(symbol: str, interval: str, d: date) -> str:
    """Build the daily kline download URL."""
    return (
        f"{BASE_URL}/daily/klines/{symbol}/{interval}/"
        f"{symbol}-{interval}-{d.isoformat()}.zip"
    )


def generate_month_ranges(
    start: date,
    end: date,
) -> list[tuple[int, int]]:
    """Generate (year, month) tuples for the given date range.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        List of (year, month) tuples.
    """
    months = []
    current = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    while current <= end_month:
        months.append((current.year, current.month))
        # Advance to next month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def parse_kline_csv(raw_bytes: bytes, source_name: str = "") -> pd.DataFrame:
    """Parse a kline CSV from raw bytes.

    Handles both header and headerless CSVs. Detects and normalizes
    timestamp units (ms vs us).

    Args:
        raw_bytes: Raw CSV file content.
        source_name: Name for logging purposes.

    Returns:
        Parsed and normalized DataFrame.
    """
    text = raw_bytes.decode("utf-8").strip()
    if not text:
        logger.warning("Empty CSV: %s", source_name)
        return pd.DataFrame(columns=KLINE_COLUMNS)

    # Check first line for header
    first_line = text.split("\n")[0].strip()
    first_fields = first_line.split(",")
    has_header = detect_header(first_fields)

    df = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=None if has_header else KLINE_COLUMNS,
        dtype=KLINE_DTYPES,
    )

    # If header was present, ensure column names match expected
    if has_header and list(df.columns) != KLINE_COLUMNS:
        # Try to map by position if column count matches
        if len(df.columns) == len(KLINE_COLUMNS):
            df.columns = KLINE_COLUMNS
        else:
            logger.warning(
                "Unexpected column count in %s: got %d, expected %d",
                source_name, len(df.columns), len(KLINE_COLUMNS),
            )

    # Drop the 'ignore' column if present
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    # Normalize timestamps
    df = normalize_timestamps(df)

    return df


async def download_zip(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
) -> bytes | None:
    """Download a ZIP file from URL with retries.

    Args:
        session: aiohttp session.
        url: URL to download.
        max_retries: Maximum retry attempts.

    Returns:
        Raw bytes of the ZIP file, or None if download failed.
    """
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 404:
                    logger.debug("Not found (404): %s", url)
                    return None
                if resp.status == 200:
                    data = await resp.read()
                    logger.info("Downloaded %s (%d bytes)", url, len(data))
                    return data
                logger.warning(
                    "HTTP %d for %s (attempt %d/%d)",
                    resp.status, url, attempt + 1, max_retries,
                )
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.warning(
                "Download error for %s (attempt %d/%d): %s",
                url, attempt + 1, max_retries, e,
            )
        if attempt < max_retries - 1:
            import asyncio
            await asyncio.sleep(2 ** attempt)

    logger.error("Failed to download after %d attempts: %s", max_retries, url)
    return None


def extract_csv_from_zip(zip_bytes: bytes) -> bytes:
    """Extract the first CSV file from a ZIP archive.

    Args:
        zip_bytes: Raw bytes of the ZIP file.

    Returns:
        Raw bytes of the CSV file.

    Raises:
        ValueError: If no CSV found in the ZIP.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            raise ValueError("No CSV file found in ZIP archive")
        return zf.read(csv_files[0])


def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save DataFrame as a Parquet file.

    Args:
        df: DataFrame to save.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    logger.info("Saved %d rows to %s", len(df), path)


async def download_symbol_klines(
    symbol: str,
    interval: str,
    start: date,
    end: date,
    data_dir: Path,
    force: bool = False,
) -> tuple[int, int, list[str]]:
    """Download kline data for a symbol over a date range.

    Downloads monthly ZIP files from Binance data dump, validates,
    and stores as partitioned Parquet. Resumable: skips months that
    already have a Parquet file unless force=True.

    Args:
        symbol: Trading pair (e.g. "SOLUSDT").
        interval: Kline interval (e.g. "1m").
        start: Start date.
        end: End date.
        data_dir: Root data directory.
        force: If True, re-download even if file exists.

    Returns:
        Tuple of (months_downloaded, months_skipped, error_messages).
    """
    months = generate_month_ranges(start, end)
    downloaded = 0
    skipped = 0
    errors: list[str] = []

    all_gaps = pd.DataFrame(columns=["gap_start", "gap_end", "missing_bars"])

    async with aiohttp.ClientSession() as session:
        for year, month in months:
            pq_path = _parquet_path(data_dir, symbol, interval, year, month)

            # Skip if already exists (resumable)
            if pq_path.exists() and not force:
                logger.info("Skipping %s %s %d-%02d (already exists)", symbol, interval, year, month)
                skipped += 1
                continue

            url = _monthly_url(symbol, interval, year, month)
            zip_bytes = await download_zip(session, url)

            if zip_bytes is None:
                # Try daily files for this month (might be current/recent month)
                logger.info("Monthly file not found, trying daily for %d-%02d", year, month)
                dfs = []
                d = date(year, month, 1)
                while d.month == month and d <= end:
                    daily_url = _daily_url(symbol, interval, d)
                    daily_zip = await download_zip(session, daily_url)
                    if daily_zip is not None:
                        try:
                            csv_bytes = extract_csv_from_zip(daily_zip)
                            day_df = parse_kline_csv(csv_bytes, f"{symbol}-{interval}-{d}")
                            dfs.append(day_df)
                        except (ValueError, zipfile.BadZipFile) as e:
                            errors.append(f"Error parsing {daily_url}: {e}")
                    d += timedelta(days=1)

                if not dfs:
                    errors.append(f"No data found for {symbol} {interval} {year}-{month:02d}")
                    continue
                df = pd.concat(dfs, ignore_index=True)
            else:
                try:
                    csv_bytes = extract_csv_from_zip(zip_bytes)
                    df = parse_kline_csv(csv_bytes, f"{symbol}-{interval}-{year}-{month:02d}")
                except (ValueError, zipfile.BadZipFile) as e:
                    errors.append(f"Error parsing {url}: {e}")
                    continue

            if df.empty:
                errors.append(f"Empty data for {symbol} {interval} {year}-{month:02d}")
                continue

            # Validate
            df, report = validate_klines(df, symbol, interval)
            for msg in report.messages:
                logger.info("[%s %d-%02d] %s", symbol, year, month, msg)

            if not report.is_valid:
                errors.append(
                    f"Validation failed for {symbol} {interval} {year}-{month:02d}: "
                    + "; ".join(report.messages)
                )
                # Still save the data but log the issues

            # Detect gaps for this month
            interval_ms = interval_to_ms(interval)
            month_gaps = detect_gaps(df, interval_ms)
            if not month_gaps.empty:
                all_gaps = pd.concat([all_gaps, month_gaps], ignore_index=True)

            # Save Parquet
            save_parquet(df, pq_path)
            downloaded += 1

    # Save combined gap report
    if not all_gaps.empty:
        gap_path = data_dir / "raw" / "klines" / f"symbol={symbol}" / f"interval={interval}" / "gaps.parquet"
        save_gap_report(all_gaps, gap_path)
        logger.info("Total gaps across all months: %d", len(all_gaps))

    return downloaded, skipped, errors


async def download_all(
    symbols: list[str],
    interval: str,
    start: date,
    end: date,
    data_dir: Path,
    force: bool = False,
) -> dict[str, tuple[int, int, list[str]]]:
    """Download kline data for multiple symbols.

    Args:
        symbols: List of trading pair symbols.
        interval: Kline interval.
        start: Start date.
        end: End date.
        data_dir: Root data directory.
        force: If True, re-download even if files exist.

    Returns:
        Dict mapping symbol to (downloaded, skipped, errors) tuple.
    """
    results = {}
    for symbol in symbols:
        logger.info("=== Downloading %s %s from %s to %s ===", symbol, interval, start, end)
        result = await download_symbol_klines(symbol, interval, start, end, data_dir, force)
        results[symbol] = result
        dl, sk, errs = result
        logger.info(
            "%s complete: %d downloaded, %d skipped, %d errors",
            symbol, dl, sk, len(errs),
        )
    return results
