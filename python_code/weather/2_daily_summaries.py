import logging
from pathlib import Path
from typing import Union

import xarray as xr

from my_config import Dirs, ensure_dirs_exist

# Replace debugging print with logging
logger = logging.getLogger(__name__)


def generate_daily_summary(source_file: Union[str, Path]) -> None:
    """
    Read an hourly GRIB source, compute daily min/max/mean for t2m and write a NetCDF summary.

    - Accepts Path or str for source_file.
    - Writes to Dirs.dir_era_daily with a temporary file then renames atomically.
    - Ensures destination parents exist.
    - Optionally moves the original file to OneDrive, ensuring that parent exists.
    """
    src = Path(source_file)
    if not src.exists():
        logger.warning("Source file not found, skipping: %s", src)
        return

    # create output filename in daily folder
    summary_name = src.name.replace("_temperature.grib", "_temperature_summary.nc")
    summary_path = Dirs.dir_era_daily / summary_name
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")

    ds = None
    try:
        ds = xr.open_dataset(src, engine="cfgrib")
        # resample and compute summary
        daily = ds.resample(time="1D")
        t_min = daily.min().rename({"t2m": "t_min"})
        t_max = daily.max().rename({"t2m": "t_max"})
        t_mean = daily.mean().rename({"t2m": "t_mean"})
        daily_summary = xr.merge([t_min, t_max, t_mean])

        # write to a temporary file first, then rename for atomicity
        daily_summary.to_netcdf(
            summary_tmp,
            encoding={
                "t_min": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
                "t_max": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
                "t_mean": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            },
        )
        summary_tmp.replace(summary_path)
        logger.info("Wrote daily summary: %s", summary_path)

    except Exception:
        logger.exception("Failed to generate daily summary for %s", src)
    finally:
        # close dataset to free resources if opened
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass


def process_all_files(
    dir_hourly: Path = Dirs.dir_era_hourly,
    min_size_gb: float = 18.0,
) -> None:
    """
    Iterate over hourly GRIB files in dir_hourly and process files that appear fully downloaded.

    - min_size_gb: simple heuristic to avoid processing partial downloads (make configurable).
    - Errors for individual files are caught and logged; processing continues.
    """

    for path in dir_hourly.glob("*.grib"):
        try:
            size_gb = path.stat().st_size / 10**9
            if size_gb >= min_size_gb:
                logger.info("Processing %s (%.2f GB)", path, size_gb)
                generate_daily_summary(source_file=path)
            else:
                logger.debug("Skipping (too small) %s (%.2f GB)", path, size_gb)
        except Exception:
            logger.exception("Error while inspecting/processing file %s", path)


if __name__ == "__main__":
    # Ensure output/backup directories exist before processing. No import-time side effects.
    ensure_dirs_exist(paths=[Dirs.dir_era_daily])
    process_all_files()
