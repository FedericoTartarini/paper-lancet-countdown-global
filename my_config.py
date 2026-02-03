from pathlib import Path

# constants
project_name = "lancet_global_data"
gadi_project_code = "ua88"


class Dirs:
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


def ensure_directories(path_dirs: list[Path]):
    for path_dir in path_dirs:
        path_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(Dirs.dir_era_daily)
