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
from zipfile import ZipFile, BadZipFile

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


def is_netcdf_file(path: Path) -> bool:
    """Quick check for NetCDF/HDF5 file signatures."""
    if not path.exists() or path.stat().st_size < 4:
        return False

    with path.open("rb") as handle:
        header = handle.read(8)

    # NetCDF classic: 'CDF\001' or 'CDF\002'
    if header[:3] == b"CDF":
        return True

    # HDF5 (NetCDF4): '\x89HDF\r\n\x1a\n'
    if header == b"\x89HDF\r\n\x1a\n":
        return True

    return False


def is_zip_file(path: Path) -> bool:
    """Return True if the file starts with a ZIP signature."""
    if not path.exists() or path.stat().st_size < 4:
        return False

    with path.open("rb") as handle:
        header = handle.read(4)

    return header.startswith(b"PK")


def extract_zip_to_nc(zip_path: Path, target_nc: Path) -> bool:
    """Extract the first .nc file from a ZIP into target_nc, preserving the ZIP."""
    try:
        with ZipFile(zip_path, "r") as archive:
            members = [name for name in archive.namelist() if name.endswith(".nc")]
            if not members:
                return False

            member = members[0]
            logger.info(f"📦 Extracting {member} from {zip_path.name}")
            with archive.open(member) as src, target_nc.open("wb") as dst:
                dst.write(src.read())
        return True
    except BadZipFile:
        return False


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
        "download_format": "unarchived",
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

    if is_zip_file(output_path):
        zip_path = output_path.with_suffix(output_path.suffix + ".zip")
        if not zip_path.exists():
            output_path.replace(zip_path)
        extracted = extract_zip_to_nc(zip_path, output_path)
        if extracted and is_netcdf_file(output_path):
            logger.info(f"✅ Extracted NetCDF from ZIP: {output_path.name}")
            return output_path

    if not is_netcdf_file(output_path):
        preview = "<empty>"
        try:
            preview = output_path.read_bytes()[:500].decode("utf-8", errors="replace")
        except OSError:
            preview = "<unreadable>"
        logger.error("Downloaded file is not NetCDF; first 500 bytes:\n%s", preview)
        raise ValueError(f"Downloaded file is not NetCDF: {output_path.name}")

    return output_path


def iter_missing_months(year_dir: Path, year: int) -> Iterable[tuple[int, Path]]:
    """Yield (month, output_path) for missing or empty monthly files."""
    for month in range(1, 13):
        filename = build_month_filename(year, month)
        output_path = year_dir / filename

        if output_path.exists() and is_netcdf_file(output_path):
            logger.info(f"✅ Exists: {output_path.name}")
            continue

        if output_path.exists():
            logger.warning(f"⚠️ Invalid file found: {output_path.name}")
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
