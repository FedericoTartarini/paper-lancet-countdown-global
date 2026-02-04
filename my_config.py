from pathlib import Path

# constants
project_name = "lancet_global_data"
gadi_usr = "ft8695"
gadi_prj_my = "ua88"
gadi_prj_era = "zz93"
gadi_prj_compute = "mn51"
e5l = "era5-land"
e5l_t = "2t"
e5l_h = "hourly"
e5l_d = "daily"
heatwaves = "heatwaves"
reanalysis = "reanalysis"


class Dirs:
    """Local paths for data processing on personal computer."""

    dir_one_drive = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global"
    )
    # remote directory for ERA5-Land data on Gadi
    dir_era_land = Path(f"/g/data/{gadi_prj_era}/{e5l}/{reanalysis}/{e5l_t}")
    # local directory for ERA5-Land data
    dir_era_land_hourly_local = dir_one_drive / e5l / e5l_h / e5l_t
    dir_era_land_daily_local = dir_one_drive / e5l / e5l_d / e5l_t


class DirsGadi:
    """Gadi-specific paths for HPC data processing."""

    # Input: ERA5-Land hourly data on Gadi (read-only)
    dir_era_land_hourly = Path(f"/g/data/{gadi_prj_era}/{e5l}/{reanalysis}/{e5l_t}")

    # Output: Daily summaries on scratch (fast write access)
    dir_era_daily = Path(
        f"/scratch/{gadi_prj_compute}/{gadi_usr}/{e5l}/{e5l_d}/{e5l_t}"
    )

    # Results: Heatwave calculations
    dir_results_heatwaves = Path(f"/scratch/{gadi_prj_compute}/{gadi_usr}/{heatwaves}")


def ensure_directories(path_dirs: list[Path]):
    for path_dir in path_dirs:
        path_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(Dirs.dir_era_land_daily_local)
