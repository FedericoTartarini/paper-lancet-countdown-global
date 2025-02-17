from pathlib import Path

DATA_SRC = Path.home() / "Documents" / "lancet_countdown"
WEATHER_SRC = DATA_SRC / "weather"
POP_DATA_SRC = DATA_SRC / "population"
dir_results = DATA_SRC / "results"
dir_pop_data_era_grid = dir_results / "worldpop_era5_grid"
dir_pop_data_era_grid.mkdir(parents=True, exist_ok=True)

SUBDAILY_TEMPERATURES_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)
SUBDAILY_TEMPERATURES_FOLDER.mkdir(parents=True, exist_ok=True)

TEMPERATURE_SUMMARY_FOLDER = (
    DATA_SRC / "weather" / "era5" / "era5_0.25deg" / "daily_temperature_summary"
)
TEMPERATURE_SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)

hd_path = Path("/Volumes/Fede 2Tb/lancet")
hd_path_population = hd_path / "population data"
hd_path_daily_temperature_summary = hd_path / "daily temperature summary"

ssd_path = Path("")

climatology_quantiles_folder = WEATHER_SRC / "era5" / "era5_0.25deg" / "quantiles"
climatology_quantiles_folder.mkdir(parents=True, exist_ok=True)

max_year = 2024
reference_year_start = 1986
reference_year_end = 2005

dir_results_heatwaves = dir_results / "heatwaves"
dir_results_heatwaves.mkdir(parents=True, exist_ok=True)
dir_results_heatwaves_tmp = dir_results_heatwaves / f"results_{max_year + 1}"
dir_results_heatwaves_tmp.mkdir(parents=True, exist_ok=True)
