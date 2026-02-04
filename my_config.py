from pathlib import Path

# constants
project_name = "lancet_global_data"
gadi_project_code = "ua88"


class Dirs:
    """Local paths for data processing on personal computer."""

    dir_one_drive = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global"
    )
    # 1. Input Data (Read-Only from NCI)
    # Note: '2t' is 2-metre temperature
    dir_era_land = Path("/g/data/zz93/era5-land/reanalysis/2t")
    dir_era_land_hourly_local = Path(dir_one_drive / "era5-land" / "hourly" / "2t")
    dir_era_land_daily_local = Path(dir_one_drive / "era5-land" / "daily" / "2t")

    # 2. Results (Write to Scratch - it is faster)
    dir_results_heatwaves = Path(
        f"/scratch/{gadi_project_code}/{project_name}/heatwaves"
    )

    # 3. Intermediate Data (Daily Summaries)
    dir_era_daily = dir_results_heatwaves / "daily_summaries"


class DirsGadi:
    """Gadi-specific paths for HPC data processing."""

    # Gadi project codes
    gadi_project_compute = "mn51"  # For compute resources
    gadi_project_storage = "ua88"  # For storage access (zz93)

    # Input: ERA5-Land hourly data on Gadi (read-only)
    dir_era_land_hourly = Path("/g/data/zz93/era5-land/reanalysis/2t")

    # Output: Daily summaries on scratch (fast write access)
    dir_era_daily = Path("/scratch/mn51/ft8695/era5-land/daily/2t")

    # Results: Heatwave calculations
    dir_results_heatwaves = Path("/scratch/mn51/ft8695/heatwaves")


def ensure_directories(path_dirs: list[Path]):
    for path_dir in path_dirs:
        path_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(Dirs.dir_era_daily)
