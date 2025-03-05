import os
from datetime import datetime
from pathlib import Path
from cartopy import crs as ccrs

year_report: int = datetime.now().year
year_max_analysis: int = year_report - 1
year_min_analysis: int = 1980
year_reference_start: int = 1986
year_reference_end: int = 2005
year_worldpop_start: int = 2000
year_worldpop_end: int = 2020

worldpop_sex = ["f", "m"]
worldpop_ages = [0, 65, 70, 75, 80]

quantiles = [0.95]

map_projection = ccrs.EckertIII()

weather_data: str = "era5"
weather_resolution: str = "0.25deg"

# Paths to local folders, SSD and HD
dir_local: Path = (
    Path.home() / "Documents" / "lancet_countdown"
)  # used to store data for analysis
dir_ssd = Path("/Volumes/T7/lancet_countdown")  # used to store large datasets
dir_one_drive = Path(
    "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)"
)
dir_one_drive_era_hourly = dir_one_drive / "Temporary" / "lancet"

dir_figures = Path("python_code/figures")
dir_figures_interim = dir_figures / "interim"

# Paths to local data folders
dir_weather = dir_local / "weather"
dir_results = dir_local / "results"
dir_population = dir_local / "population"
dir_population_hybrid = dir_results / "hybrid_pop"
dir_file_population_before_2000 = (
    dir_population_hybrid / "Hybrid Demographics 1950-2020.nc"
)

# ======== no need to change below this line ========

dir_population.mkdir(parents=True, exist_ok=True)
dir_population_tmp = dir_population / "tmp"
dir_population_tmp.mkdir(parents=True, exist_ok=True)
dir_pop_era_grid = dir_results / f"worldpop_{weather_data}_grid"
dir_pop_era_grid.mkdir(parents=True, exist_ok=True)
dir_results_pop_exposure = (
    dir_results / f"results_{year_report}" / "pop_exposure" / "worldpop_hw_exposure"
)
dir_results_pop_exposure.mkdir(parents=True, exist_ok=True)
dir_pop_hybrid = dir_results / "hybrid_pop"

dir_era_hourly = dir_local / weather_data / weather_resolution / "hourly_temperature_2m"
dir_era_hourly.mkdir(parents=True, exist_ok=True)
dir_era_quantiles = (
    dir_weather / weather_data / f"{weather_data}_{weather_resolution}" / "quantiles"
)
dir_era_quantiles.mkdir(parents=True, exist_ok=True)

dir_results_heatwaves = dir_results / "heatwaves"
dir_results_heatwaves.mkdir(parents=True, exist_ok=True)
dir_results_heatwaves_tmp = dir_results_heatwaves / f"results_{year_report}"
dir_results_heatwaves_tmp.mkdir(parents=True, exist_ok=True)
dir_results_heatwaves_monthly = dir_results_heatwaves_tmp / "heatwaves_monthly_era5"
dir_results_heatwaves_monthly.mkdir(exist_ok=True)
dir_results_heatwaves_days = dir_results_heatwaves_tmp / "heatwaves_days_era5"
dir_results_heatwaves_days.mkdir(exist_ok=True)
dir_results_heatwaves_count = dir_results_heatwaves_tmp / "heatwaves_count_era5"
dir_results_heatwaves_count.mkdir(exist_ok=True)
dir_worldpop_exposure_by_region = (
    dir_results_pop_exposure / "exposure_by_region_or_grouping"
)
dir_worldpop_exposure_by_region.mkdir(parents=True, exist_ok=True)

# Paths to SSD data folders
dir_era_daily = dir_ssd / "daily_temperature_summary"
dir_pop_raw = dir_ssd / "population"

try:
    dir_pop_raw.mkdir(parents=True, exist_ok=True)
    dir_era_daily.mkdir(parents=True, exist_ok=True)
except PermissionError:  # just in case the SSD is not mounted
    pass

# remove hidden files from the population folder
for f in os.listdir(dir_pop_raw):
    if f.startswith(".") and f.endswith(".tif"):
        os.remove(dir_pop_raw / f)

# paths to important files
dir_pop_infants_file = (
    dir_population_hybrid
    / f"worldpop_infants_1950_{year_max_analysis}_era5_compatible.nc"
)
dir_pop_elderly_file = (
    dir_population_hybrid
    / f"worldpop_elderly_1950_{year_max_analysis}_era5_compatible.nc"
)
dir_pop_above_75_file = (
    dir_population_hybrid
    / f"worldpop_75_80_1950_{year_max_analysis}_era5_compatible.nc"
)
dir_file_elderly_exposure_abs = (
    dir_results_pop_exposure
    / f"heatwave_exposure_over65_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
)
dir_file_elderly_exposure_change = (
    dir_results_pop_exposure
    / f"heatwave_exposure_change_over65_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
)
dir_file_infants_exposure_abs = (
    dir_results_pop_exposure
    / f"heatwave_exposure_infants_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
)
dir_file_infants_exposure_change = (
    dir_results_pop_exposure
    / f"heatwave_exposure_change_infants_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
)
dir_file_all_exposure_abs = (
    dir_results_pop_exposure
    / f"heatwave_exposure_multi_threshold_{year_min_analysis}-{year_max_analysis}_worldpop.nc"
)

# boundaries and rasters
dir_admin_boundaries = dir_local / "admin_boundaries"
dir_file_detailed_boundaries = dir_admin_boundaries / "Detailed_Boundary_ADM0"
dir_file_country_polygons = dir_file_detailed_boundaries / "GLOBAL_ADM0.shp"
dir_file_admin1_polygons = (
    dir_admin_boundaries / "Detailed_Boundary_ADM1" / "Detailed_Boundary_ADM1.shp"
)
dir_file_country_raster_report = dir_admin_boundaries / "admin0_raster_report_2024.nc"
dir_file_who_raster_report = dir_admin_boundaries / "WHO_regions_raster_report_2024.nc"
dir_file_hdi_raster_report = dir_admin_boundaries / "HDI_group_raster_report_2024.nc"
dir_file_lancet_raster_report = dir_admin_boundaries / "LC_group_raster_report_2024.nc"
dir_file_admin1_raster_report = dir_admin_boundaries / "admin1_raster_report_2024.nc"
dir_file_lancet_country_info = (
    dir_admin_boundaries / "2025 Global Report Country Names and Groupings.xlsx"
)
