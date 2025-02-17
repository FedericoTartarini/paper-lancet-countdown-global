from pathlib import Path

max_year = 2024
min_year = 1980
reference_year_start = 1986
reference_year_end = 2005
report_year = 2025

# Paths to local folders, SSD and HD
path_local = Path.home() / "Documents" / "lancet_countdown"
path_hd = Path("/Volumes/Fede 2Tb/lancet")
path_ssd = Path("/Volumes/T7/lancet_countdown")

# Paths to local data folders
weather_src = path_local / "weather"
dir_results = path_local / "results"
dir_pop_data_era_grid = dir_results / "worldpop_era5_grid"
dir_pop_data_era_grid.mkdir(parents=True, exist_ok=True)
dir_results_pop_exposure = (
    dir_results / f"results_{report_year}" / "pop_exposure" / "worldpop_hw_exposure"
)
dir_results_pop_exposure.mkdir(parents=True, exist_ok=True)

subdaily_temperatures_folder = (
    path_local / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)
subdaily_temperatures_folder.mkdir(parents=True, exist_ok=True)

hd_path_population = path_hd / "population data"

climatology_quantiles_folder = weather_src / "era5" / "era5_0.25deg" / "quantiles"
climatology_quantiles_folder.mkdir(parents=True, exist_ok=True)

dir_results_heatwaves = dir_results / "heatwaves"
dir_results_heatwaves.mkdir(parents=True, exist_ok=True)
dir_results_heatwaves_tmp = dir_results_heatwaves / f"results_{max_year + 1}"
dir_results_heatwaves_tmp.mkdir(parents=True, exist_ok=True)

# Paths to SSD data folders
temperature_summary_folder = path_ssd / "daily_temperature_summary"
temperature_summary_folder.mkdir(parents=True, exist_ok=True)
pop_data_src = path_ssd / "population"
