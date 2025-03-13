import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Iterator

from cartopy import crs as ccrs
from matplotlib import pyplot as plt


plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["savefig.dpi"] = 300


class AutoEnum(Enum):
    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return str(self.value)

    # Allow direct comparison with the value
    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self is other
        return self.value == other

    def __iter__(self) -> Iterator:
        """Make list values iterable directly through the enum member."""
        if isinstance(self.value, (list, tuple, set)):
            return iter(self.value)
        raise TypeError(f"{self.__class__.__name__}.{self.name} is not iterable")

    def __len__(self) -> int:
        """Support len() for iterable values."""
        if isinstance(self.value, (list, tuple, set)):
            return len(self.value)
        raise TypeError(
            f"object of type '{self.__class__.__name__}.{self.name}' has no len()"
        )


class Vars(AutoEnum):
    year_report: int = datetime.now().year
    year_max_analysis: int = year_report - 1
    year_min_analysis: int = 1980
    year_reference_start: int = 1986
    year_reference_end: int = 2005
    quantiles: List[float] = [0.95]
    map_projection = ccrs.EckertIII()

    @classmethod
    def get_reference_years(cls) -> List[int]:
        """Return all years in the reference period as a list."""
        return list(
            range(cls.year_reference_start.value, cls.year_reference_end.value + 1)
        )

    @classmethod
    def get_analysis_years(cls) -> List[int]:
        """Return all years in the analysis period as a list."""
        return list(range(cls.year_min_analysis.value, cls.year_max_analysis.value + 1))


class VarsWorldPop(AutoEnum):
    year_worldpop_start: int = 2000
    year_worldpop_end: int = 2020
    worldpop_sex: List[str] = ["f", "m"]
    worldpop_ages: List[int] = [0, 65, 70, 75, 80]
    url_base_data: str = (
        "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020/"
    )

    @classmethod
    def get_years_range(cls) -> List[int]:
        """Return all years in the reference period as a list."""
        return list(
            range(cls.year_worldpop_start.value, cls.year_worldpop_end.value + 1)
        )

    @classmethod
    def get_url_download(cls, year, sex, age) -> str:
        return f"{cls.url_base_data}{year}/0_Mosaicked/global_mosaic_1km/global_{sex}_{age}_{year}_1km.tif"

    @classmethod
    def get_slice_years(cls, period) -> slice:
        if period == "before":
            return slice(Vars.year_min_analysis, cls.year_worldpop_start.value - 1)
        elif period == "after":
            return slice(cls.year_worldpop_end.value + 1, Vars.year_report)
        else:
            return slice(cls.year_worldpop_start.value, cls.year_worldpop_end.value)


weather_data: str = "era5"
weather_resolution: str = "0.25deg"


class Dirs(AutoEnum):
    # Paths to local folders, SSD and HD
    dir_local: Path = (
        Path.home() / "Documents" / "lancet_countdown"
    )  # used to store data for analysis
    dir_ssd: Path = Path("/Volumes/T7/lancet_countdown")  # used to store large datasets
    dir_one_drive: Path = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)"
    )
    dir_one_drive_era_hourly: Path = dir_one_drive / "Temporary" / "lancet"

    dir_figures: Path = Path("python_code/figures")
    dir_figures_interim: Path = dir_figures / "interim"

    # Paths to local data folders
    dir_weather: Path = dir_local / "weather"
    dir_results: Path = dir_local / "results"
    dir_population: Path = dir_local / "population"
    dir_population_hybrid: Path = dir_results / "hybrid_pop"
    dir_file_population_before_2000: Path = (
        dir_population_hybrid / "Hybrid Demographics 1950-2020.nc"
    )
    # ======== no need to change below this line ========
    dir_population_tmp = dir_population / "tmp"

    dir_pop_era_grid = dir_results / f"worldpop_{weather_data}_grid"
    dir_results_pop_exposure = (
        dir_results
        / f"results_{Vars.year_report}"
        / "pop_exposure"
        / "worldpop_hw_exposure"
    )
    dir_pop_hybrid = dir_results / "hybrid_pop"

    dir_era_hourly = (
        dir_local / weather_data / weather_resolution / "hourly_temperature_2m"
    )
    dir_era_quantiles = (
        dir_weather
        / weather_data
        / f"{weather_data}_{weather_resolution}"
        / "quantiles"
    )

    dir_results_heatwaves = dir_results / "heatwaves"
    dir_results_heatwaves_tmp = dir_results_heatwaves / f"results_{Vars.year_report}"
    dir_results_heatwaves_monthly = dir_results_heatwaves_tmp / "heatwaves_monthly_era5"
    dir_results_heatwaves_days = dir_results_heatwaves_tmp / "heatwaves_days_era5"
    dir_results_heatwaves_count = dir_results_heatwaves_tmp / "heatwaves_count_era5"
    dir_worldpop_exposure_by_region = (
        dir_results_pop_exposure / "exposure_by_region_or_grouping"
    )

    # Paths to SSD data folders
    dir_era_daily = dir_ssd / "daily_temperature_summary"
    dir_pop_raw = dir_ssd / "population"  # paths to important files
    dir_pop_infants_file = (
        dir_population_hybrid
        / f"worldpop_infants_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_pop_elderly_file = (
        dir_population_hybrid
        / f"worldpop_elderly_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_pop_above_75_file = (
        dir_population_hybrid
        / f"worldpop_75_80_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_file_elderly_exposure_abs = (
        dir_results_pop_exposure
        / f"heatwave_exposure_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_elderly_exposure_change = (
        dir_results_pop_exposure
        / f"heatwave_exposure_change_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_infants_exposure_abs = (
        dir_results_pop_exposure
        / f"heatwave_exposure_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_infants_exposure_change = (
        dir_results_pop_exposure
        / f"heatwave_exposure_change_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_all_exposure_abs = (
        dir_results_pop_exposure
        / f"heatwave_exposure_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_countries_heatwave_exposure = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_weighted_change_1980-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_exposures_abs_by_lc_group_worldpop = (
        dir_worldpop_exposure_by_region / f"exposures_abs_by_lc_group_worldpop.nc"
    )
    dir_file_countries_heatwaves_exposure_change = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_countries_heatwaves_exposure = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_who_regions_heatwaves_exposure = (
        dir_worldpop_exposure_by_region
        / f"who_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_who_regions_heatwaves_exposure_change = (
        dir_worldpop_exposure_by_region
        / f"who_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_hdi_regions_heatwaves_exposure = (
        dir_worldpop_exposure_by_region
        / f"hdi_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_hdi_regions_heatwaves_exposure_change = (
        dir_worldpop_exposure_by_region
        / f"hdi_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_manuscript_submission = Path("manuscript") / f"{Vars.year_report}"
    dir_file_excel_submission = (
        dir_manuscript_submission
        / f"1.1.1 - {Vars.year_report} Global Report - Data Submission - Tartarini.xlsx"
    )
    # boundaries and rasters
    dir_admin_boundaries = dir_local / "admin_boundaries"
    dir_file_detailed_boundaries = dir_admin_boundaries / "Detailed_Boundary_ADM0"
    dir_file_country_polygons = dir_file_detailed_boundaries / "GLOBAL_ADM0.shp"
    dir_file_admin1_polygons = (
        dir_admin_boundaries / "Detailed_Boundary_ADM1" / "Detailed_Boundary_ADM1.shp"
    )
    dir_file_country_raster_report = (
        dir_admin_boundaries / "admin0_raster_report_2024.nc"
    )
    dir_file_who_raster_report = (
        dir_admin_boundaries / "WHO_regions_raster_report_2024.nc"
    )
    dir_file_hdi_raster_report = (
        dir_admin_boundaries / "HDI_group_raster_report_2024.nc"
    )
    dir_file_lancet_raster_report = (
        dir_admin_boundaries / "LC_group_raster_report_2024.nc"
    )
    dir_file_admin1_raster_report = (
        dir_admin_boundaries / "admin1_raster_report_2024.nc"
    )
    dir_file_lancet_country_info = (
        dir_admin_boundaries / "2025 Global Report Country Names and Groupings.xlsx"
    )


try:
    Dirs.dir_worldpop_exposure_by_region.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_pop_raw.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_population.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_population_tmp.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_pop_era_grid.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_results_pop_exposure.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_era_hourly.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_era_quantiles.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_results_heatwaves.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_results_heatwaves_tmp.value.mkdir(parents=True, exist_ok=True)
    Dirs.dir_results_heatwaves_monthly.value.mkdir(exist_ok=True)
    Dirs.dir_results_heatwaves_days.value.mkdir(exist_ok=True)
    Dirs.dir_results_heatwaves_count.value.mkdir(exist_ok=True)
    Dirs.dir_era_daily.value.mkdir(parents=True, exist_ok=True)
except PermissionError:  # just in case the SSD is not mounted
    pass

try:
    # remove hidden files from the population folder
    for f in os.listdir(Dirs.dir_pop_raw.value):
        if f.startswith(".") and f.endswith(".tif"):
            os.remove(Dirs.dir_pop_raw.value / f)
except FileNotFoundError:
    pass


class SheetsFinalSubmission(Enum):
    global_total = "Global total"
    global_average = "Global average"
    country_infants = "Country infants"
    country_over65 = "Country 65+"
    country_over75 = "Country 75+"
    hdi_group = "HDI Group"
    who_region = "WHO Region"
    lc_region = "LC Region"
