from datetime import datetime
from pathlib import Path

year_report: int = datetime.now().year
year_max_analysis: int = year_report - 1
year_min_analysis: int = 1980
year_reference_start: int = 1986
year_reference_end: int = 2005

weather_data: str = "era5"
weather_resolution: str = "0.25deg"

# Paths to local folders, SSD and HD
dir_local: Path = (
    Path.home() / "Documents" / "lancet_countdown"
)  # used to store data for analysis
dir_ssd = Path("/Volumes/T7/lancet_countdown")  # used to store large datasets

dir_figures = Path("python_code/figures")

# ======== no need to change below this line ========

# Paths to local data folders
dir_weather = dir_local / "weather"
dir_results = dir_local / "results"
dir_pop_era_grid = dir_results / f"worldpop_{weather_data}_grid"
dir_pop_era_grid.mkdir(parents=True, exist_ok=True)
dir_results_pop_exposure = (
    dir_results / f"results_{year_report}" / "pop_exposure" / "worldpop_hw_exposure"
)
dir_results_pop_exposure.mkdir(parents=True, exist_ok=True)
dir_pop_hybrid = dir_results / "hybrid_pop"

dir_sub_daily_era_folder = (
    dir_local / weather_data / weather_resolution / "hourly_temperature_2m"
)
dir_sub_daily_era_folder.mkdir(parents=True, exist_ok=True)

dir_era_quantiles = dir_weather / weather_data / weather_resolution / "quantiles"
dir_era_quantiles.mkdir(parents=True, exist_ok=True)

dir_results_heatwaves = dir_results / "heatwaves"
dir_results_heatwaves.mkdir(parents=True, exist_ok=True)
dir_results_heatwaves_tmp = dir_results_heatwaves / f"results_{year_report}"
dir_results_heatwaves_tmp.mkdir(parents=True, exist_ok=True)

# Paths to SSD data folders
dir_era_daily = dir_ssd / "daily_temperature_summary"
dir_pop_raw = dir_ssd / "population"

try:
    dir_pop_raw.mkdir(parents=True, exist_ok=True)
    dir_era_daily.mkdir(parents=True, exist_ok=True)
except PermissionError:  # just in case the SSD is not mounted
    pass
