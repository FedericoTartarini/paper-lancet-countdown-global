import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from cartopy import crs as ccrs
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["savefig.dpi"] = 300

logger = logging.getLogger(__name__)


class Vars:
    """Project-wide variables (use attributes directly, e.g. Vars.year_report)."""

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
        return list(range(cls.year_reference_start, cls.year_reference_end + 1))

    @classmethod
    def get_analysis_years(cls) -> List[int]:
        """Return all years in the analysis period as a list."""
        return list(range(cls.year_min_analysis, cls.year_max_analysis + 1))


class VarsWorldPop:
    """WorldPop-specific config."""

    year_worldpop_start: int = 2000
    year_worldpop_end: int = 2020
    worldpop_sex: List[str] = ["f", "m"]
    worldpop_ages: List[int] = [0, 65, 70, 75, 80]
    # todo I will need to change this base url
    url_base_data: str = f"https://data.worldpop.org/GIS/AgeSex_structures/Global_{year_worldpop_start}_{year_worldpop_end}/"
    @classmethod
    def get_years_range(cls) -> List[int]:
        """Return all years in the WorldPop range."""
        return list(range(cls.year_worldpop_start, cls.year_worldpop_end + 1))

    @classmethod
    def get_url_download(cls, year: int, sex: str, age: int) -> str:
        return f"{cls.url_base_data}{year}/0_Mosaicked/global_mosaic_1km/global_{sex}_{age}_{year}_1km.tif"

    @classmethod
    def get_slice_years(cls, period: str) -> slice:
        if period == "before":
            return slice(Vars.year_min_analysis, cls.year_worldpop_start - 1)
        elif period == "after":
            return slice(cls.year_worldpop_end + 1, Vars.year_report)
        else:
            return slice(cls.year_worldpop_start, cls.year_worldpop_end)


weather_data: str = "era5"
weather_resolution: str = "0.25deg"


class Dirs:
    """
    Directory and file path configuration.
    Call Dirs.ensure_dirs_exist() at runtime to create directories instead of
    performing side effects at import time.
    """

    # Paths to local folders, SSD and HD
    dir_local: Path = (
        Path.home()
        / "Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/Academia/Datasets/lancet_countdown_global"
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
    dir_population_tmp: Path = dir_population / "tmp"

    dir_pop_era_grid: Path = dir_results / f"worldpop_{weather_data}_grid"
    dir_results_pop_exposure: Path = (
        dir_results
        / f"results_{Vars.year_report}"
        / "pop_exposure"
        / "worldpop_hw_exposure"
    )
    dir_pop_hybrid: Path = dir_results / "hybrid_pop"

    dir_era_hourly: Path = (
        dir_local / weather_data / weather_resolution / "hourly_temperature_2m"
    )
    dir_era_quantiles: Path = (
        dir_weather
        / weather_data
        / f"{weather_data}_{weather_resolution}"
        / "quantiles"
    )

    dir_results_heatwaves: Path = dir_results / "heatwaves"
    dir_results_heatwaves_tmp: Path = (
        dir_results_heatwaves / f"results_{Vars.year_report}"
    )
    dir_results_heatwaves_monthly: Path = (
        dir_results_heatwaves_tmp / "heatwaves_monthly_era5"
    )
    dir_results_heatwaves_days: Path = dir_results_heatwaves_tmp / "heatwaves_days_era5"
    dir_results_heatwaves_count: Path = (
        dir_results_heatwaves_tmp / "heatwaves_count_era5"
    )
    dir_worldpop_exposure_by_region: Path = (
        dir_results_pop_exposure / "exposure_by_region_or_grouping"
    )

    # Paths to SSD data folders
    dir_era_daily: Path = dir_ssd / "daily_temperature_summary"
    dir_pop_raw: Path = dir_ssd / "population"  # paths to important files
    dir_pop_infants_file: Path = (
        dir_population_hybrid
        / f"worldpop_infants_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_pop_elderly_file: Path = (
        dir_population_hybrid
        / f"worldpop_elderly_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_pop_above_75_file: Path = (
        dir_population_hybrid
        / f"worldpop_75_80_1950_{Vars.year_max_analysis}_era5_compatible.nc"
    )
    dir_file_elderly_exposure_abs: Path = (
        dir_results_pop_exposure
        / f"heatwave_exposure_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_elderly_exposure_change: Path = (
        dir_results_pop_exposure
        / f"heatwave_exposure_change_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_infants_exposure_abs: Path = (
        dir_results_pop_exposure
        / f"heatwave_exposure_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_infants_exposure_change: Path = (
        dir_results_pop_exposure
        / f"heatwave_exposure_change_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_all_exposure_abs: Path = (
        dir_results_pop_exposure
        / f"heatwave_exposure_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_countries_heatwave_exposure: Path = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_weighted_change_1980-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_exposures_abs_by_lc_group_worldpop: Path = (
        dir_worldpop_exposure_by_region / f"exposures_abs_by_lc_group_worldpop.nc"
    )
    dir_file_countries_heatwaves_exposure_change: Path = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_countries_heatwaves_exposure: Path = (
        dir_worldpop_exposure_by_region
        / f"countries_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_who_regions_heatwaves_exposure: Path = (
        dir_worldpop_exposure_by_region
        / f"who_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_who_regions_heatwaves_exposure_change: Path = (
        dir_worldpop_exposure_by_region
        / f"who_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_hdi_regions_heatwaves_exposure: Path = (
        dir_worldpop_exposure_by_region
        / f"hdi_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_file_hdi_regions_heatwaves_exposure_change: Path = (
        dir_worldpop_exposure_by_region
        / f"hdi_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    )
    dir_manuscript_submission: Path = Path("manuscript") / f"{Vars.year_report}"
    dir_file_excel_submission: Path = (
        dir_manuscript_submission
        / f"1.1.1 - {Vars.year_report} Global Report - Data Submission - Tartarini.xlsx"
    )
    # boundaries and rasters
    dir_admin_boundaries: Path = dir_local / "admin_boundaries"
    dir_file_detailed_boundaries: Path = dir_admin_boundaries / "Detailed_Boundary_ADM0"
    dir_file_country_polygons: Path = dir_file_detailed_boundaries / "GLOBAL_ADM0.shp"
    dir_file_admin1_polygons: Path = (
        dir_admin_boundaries / "Detailed_Boundary_ADM1" / "Detailed_Boundary_ADM1.shp"
    )
    dir_file_country_raster_report: Path = (
        dir_admin_boundaries / "admin0_raster_report_2024.nc"
    )
    dir_file_who_raster_report: Path = (
        dir_admin_boundaries / "WHO_regions_raster_report_2024.nc"
    )
    dir_file_hdi_raster_report: Path = (
        dir_admin_boundaries / "HDI_group_raster_report_2024.nc"
    )
    dir_file_lancet_raster_report: Path = (
        dir_admin_boundaries / "LC_group_raster_report_2024.nc"
    )
    dir_file_admin1_raster_report: Path = (
        dir_admin_boundaries / "admin1_raster_report_2024.nc"
    )
    dir_file_lancet_country_info: Path = (
        dir_admin_boundaries / "2025 Global Report Country Names and Groupings.xlsx"
    )

    @classmethod
    def ensure_dirs_exist(cls) -> None:
        """
        Create required directories. No-op on import; call explicitly at runtime.

        Example:
            Dirs.ensure_dirs_exist()
        """

        # explicit list of directories we want to create when requested
        dirs_to_create = [
            # keep this list minimal and explicit to avoid creating directories for file paths
            cls.dir_worldpop_exposure_by_region,
            cls.dir_pop_raw,
            cls.dir_population,
            cls.dir_population_tmp,
            cls.dir_pop_era_grid,
            cls.dir_era_hourly,
            cls.dir_era_quantiles,
            cls.dir_results_heatwaves,
            cls.dir_results_heatwaves_tmp,
            cls.dir_results_heatwaves_monthly,
            cls.dir_results_heatwaves_days,
            cls.dir_results_heatwaves_count,
            cls.dir_era_daily,
        ]

        for p in dirs_to_create:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.warning("Permission denied creating directory %s (skipping)", p)
            except Exception:
                logger.exception("Unexpected error creating directory %s", p)

    @classmethod
    def clean_pop_raw(cls) -> None:
        """
        Remove hidden .tif files from the population raw directory (if mounted).
        Call explicitly; does not run on import.
        """
        try:
            for fname in os.listdir(cls.dir_pop_raw):
                if fname.startswith(".") and fname.endswith(".tif"):
                    try:
                        (cls.dir_pop_raw / fname).unlink()
                    except PermissionError:
                        logger.warning("Permission denied removing %s", fname)
        except FileNotFoundError:
            logger.info(
                "Population raw directory %s not found, skipping cleanup",
                cls.dir_pop_raw,
            )
        except Exception:
            logger.exception("Unexpected error cleaning %s", cls.dir_pop_raw)


class SheetsFinalSubmission:
    global_total = "Global total"
    global_average = "Global average"
    country_infants = "Country infants"
    country_over65 = "Country 65+"
    country_over75 = "Country 75+"
    hdi_group = "HDI Group"
    who_region = "WHO Region"
    lc_region = "LC Region"


# Convenience initializer: call from your main script (not at import time)
if __name__ == "__main__":
    Dirs.ensure_dirs_exist()
    Dirs.clean_pop_raw()
