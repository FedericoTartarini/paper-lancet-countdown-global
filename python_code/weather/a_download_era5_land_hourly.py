"""
Download missing ERA5-Land hourly monthly files for a given year.

The script checks for monthly files in DirsLocal.e5l_h / <year> using the naming
pattern: 2t_era5-land_oper_sfc_YYYYMM01-YYYYMMDD.nc

If a month is missing (or the file is empty), it is downloaded from CDS.
No argument parser is used; call main(year) directly.
"""

import sys
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import cdsapi

from my_config import DirsLocal, ensure_directories
from python_code.log_config import setup_logging
from python_code.secrets import copernicus_api_key

logger = setup_logging(project_root)

DEFAULT_DATASET = "reanalysis-era5-land"
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"


def build_month_filename(year: int, month: int) -> str:
    """Return the expected monthly file name for ERA5-Land hourly data."""
    last_day = monthrange(year, month)[1]
    return (
        f"2t_era5-land_oper_sfc_{year}{month:02d}01-{year}{month:02d}{last_day:02d}.nc"
    )


def build_request_payload(year: int, month: int) -> dict:
    """Create a CDS API request payload for a single month."""
    last_day = monthrange(year, month)[1]
    return {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{day:02d}" for day in range(1, last_day + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "format": "netcdf",
    }


def download_month(
    year: int,
    month: int,
    output_path: Path,
    dataset: str = DEFAULT_DATASET,
    url: str = DEFAULT_URL,
) -> Path:
    """Download a single month of ERA5-Land hourly data to the output path."""
    if not copernicus_api_key or copernicus_api_key.startswith("your-"):
        raise ValueError("Missing Copernicus API key in python_code/secrets.py")

    client = cdsapi.Client(url=url, key=copernicus_api_key)
    payload = build_request_payload(year, month)

    logger.info(f"⬇️ Downloading {output_path.name}")
    client.retrieve(dataset, payload, str(output_path))
    return output_path


def iter_missing_months(year_dir: Path, year: int) -> Iterable[tuple[int, Path]]:
    """Yield (month, output_path) for missing or empty monthly files."""
    for month in range(1, 13):
        filename = build_month_filename(year, month)
        output_path = year_dir / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✅ Exists: {output_path.name}")
            continue

        if output_path.exists():
            logger.warning(f"⚠️ Empty or corrupted file found: {output_path.name}")
            output_path.unlink()

        yield month, output_path


def main(
    year: int,
    dataset: str = DEFAULT_DATASET,
    url: str = DEFAULT_URL,
    max_workers: Optional[int] = 3,
) -> List[Path]:
    """Check for missing monthly files and download them for the given year."""
    if year < 1950 or year > 2100:
        raise ValueError("Year out of expected range: 1950-2100")

    year_dir = DirsLocal.e5l_h / str(year)
    ensure_directories([year_dir])

    missing = list(iter_missing_months(year_dir, year))
    if not missing:
        logger.info("🎉 All monthly files are present. Nothing to download.")
        return []

    if max_workers is None or max_workers < 1:
        max_workers = 1

    logger.info(f"🚀 Starting downloads with up to {max_workers} parallel worker(s)...")

    downloaded: List[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_month,
                year,
                month,
                output_path,
                dataset,
                url,
            ): output_path
            for month, output_path in missing
        }
        for future in as_completed(futures):
            output_path = futures[future]
            try:
                result = future.result()
                downloaded.append(result)
                logger.info(f"✅ Downloaded: {output_path.name}")
            except Exception as exc:
                logger.error(f"❌ Failed: {output_path.name} ({exc})")

    logger.info(f"🎉 Completed. Downloaded {len(downloaded)} file(s).")
    return downloaded


if __name__ == "__main__":
    main(year=2025)
