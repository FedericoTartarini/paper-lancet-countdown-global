import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from cartopy import crs as ccrs
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7, 3]
plt.rcParams["savefig.dpi"] = 300

weather_data: str = "era5"
weather_resolution: str = "0.25deg"

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

    # todo I need to fix the following
    year_worldpop_start: int = 2015
    year_worldpop_end: int = 2030
    worldpop_sex: List[str] = ["f", "m"]
    worldpop_ages: List[int] = [0, 65, 70, 75, 80, 85, 90]
    url_base_data: str = (
        # todo I will need to change this base url
        "https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030"
    )

    @classmethod
    def get_years_range(cls) -> List[int]:
        """Return all years in the WorldPop range."""
        return list(range(cls.year_worldpop_start, cls.year_worldpop_end + 1))

    @classmethod
    def get_url_download(cls, year: int, sex: str, age: int) -> str:
        if age < 10:
            age_str = f"0{age}"
        else:
            age_str = str(age)

        # I am downloading the UN adjusted constrained data
        return f"{cls.url_base_data}/R2025A/{year}/0_Mosaicked/v1/1km_ua/constrained/global_{sex}_{age_str}_{year}_CN_1km_R2025A_UA_v1.tif"

    @classmethod
    def get_slice_years(cls, period: str) -> slice:
        if period == "before":
            return slice(Vars.year_min_analysis, cls.year_worldpop_start - 1)
        elif period == "after":
            return slice(cls.year_worldpop_end + 1, Vars.year_report)
        else:
            return slice(cls.year_worldpop_start, cls.year_worldpop_end)


class Dirs:
    """
    Directory and file path configuration.
    Call Dirs.ensure_dirs_exist() at runtime to create directories instead of
    performing side effects at import time.
    """

    dir_one_drive: Path = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/data/lancet/countdown-global"
    )

    # Paths to local folders, SSD and HD
    dir_local: Path = (
        Path.home() / "/Documents/github-projects/paper-lancet-countdown-global"
    )

    dir_results: Path = dir_one_drive / "results"
    dir_figures: Path = dir_results / f"results_{Vars.year_report}" / "figures"
    dir_figures_interim: Path = dir_figures / "interim"
    dir_population_hybrid: Path = dir_results / "hybrid_pop"
    dir_file_population_before_2000: Path = (
        dir_population_hybrid / "Hybrid Demographics 1950-2020.nc"
    )
    dir_ssd: Path = Path("/Volumes/T7/lancet_countdown")  # used to store large datasets

    # ======== no need to change below this line ========

    # WEATHER DATA
    dir_weather: Path = dir_one_drive / weather_data
    dir_era_hourly: Path = (
        dir_weather / "hourly" / weather_resolution / "hourly_temperature_2m"
    )
    dir_era_daily: Path = dir_weather / "daily"

    # POPULATION DATA
    dir_population: Path = dir_one_drive / "population"
    dir_population_tmp: Path = dir_population / "tmp"
    dir_pop_raw: Path = (
        dir_ssd / "population"
    )  # todo I need to copy these files in OneDrive as well
    #
    dir_pop_era_grid: Path = dir_results / f"worldpop_{weather_data}_grid"
    # dir_results_pop_exposure: Path = (
    #     dir_results
    #     / f"results_{Vars.year_report}"
    #     / "pop_exposure"
    #     / "worldpop_hw_exposure"
    # )
    # dir_pop_hybrid: Path = dir_results / "hybrid_pop"
    dir_era_quantiles: Path = dir_weather / "quantiles"
    #
    dir_results_heatwaves: Path = dir_results / "heatwaves"
    # dir_worldpop_exposure_by_region: Path = (
    #     dir_results_pop_exposure / "exposure_by_region_or_grouping"
    # )
    #
    # # Paths to SSD data folders
    # dir_pop_raw: Path = dir_ssd / "population"  # paths to important files
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
    # dir_file_elderly_exposure_abs: Path = (
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_elderly_exposure_change: Path = (
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_change_over65_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_infants_exposure_abs: Path = (
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_infants_exposure_change: Path = (
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_change_infants_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_all_exposure_abs: Path = (
    #     dir_results_pop_exposure
    #     / f"heatwave_exposure_multi_threshold_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_countries_heatwave_exposure: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"countries_heatwaves_exposure_weighted_change_1980-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_exposures_abs_by_lc_group_worldpop: Path = (
    #     dir_worldpop_exposure_by_region / f"exposures_abs_by_lc_group_worldpop.nc"
    # )
    # dir_file_countries_heatwaves_exposure_change: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"countries_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_countries_heatwaves_exposure: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"countries_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_who_regions_heatwaves_exposure: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"who_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_who_regions_heatwaves_exposure_change: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"who_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_hdi_regions_heatwaves_exposure: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"hdi_regions_heatwaves_exposure_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_file_hdi_regions_heatwaves_exposure_change: Path = (
    #     dir_worldpop_exposure_by_region
    #     / f"hdi_regions_heatwaves_exposure_change_{Vars.year_min_analysis}-{Vars.year_max_analysis}_worldpop.nc"
    # )
    # dir_manuscript_submission: Path = Path("manuscript") / f"{Vars.year_report}"
    # dir_file_excel_submission: Path = (
    #     dir_manuscript_submission
    #     / f"1.1.1 - {Vars.year_report} Global Report - Data Submission - Tartarini.xlsx"
    # )
    # # boundaries and rasters
    # dir_admin_boundaries: Path = dir_local / "admin_boundaries"
    # dir_file_detailed_boundaries: Path = dir_admin_boundaries / "Detailed_Boundary_ADM0"
    # dir_file_country_polygons: Path = dir_file_detailed_boundaries / "GLOBAL_ADM0.shp"
    # dir_file_admin1_polygons: Path = (
    #     dir_admin_boundaries / "Detailed_Boundary_ADM1" / "Detailed_Boundary_ADM1.shp"
    # )
    # dir_file_country_raster_report: Path = (
    #     dir_admin_boundaries / "admin0_raster_report_2024.nc"
    # )
    # dir_file_who_raster_report: Path = (
    #     dir_admin_boundaries / "WHO_regions_raster_report_2024.nc"
    # )
    # dir_file_hdi_raster_report: Path = (
    #     dir_admin_boundaries / "HDI_group_raster_report_2024.nc"
    # )
    # dir_file_lancet_raster_report: Path = (
    #     dir_admin_boundaries / "LC_group_raster_report_2024.nc"
    # )
    # dir_file_admin1_raster_report: Path = (
    #     dir_admin_boundaries / "admin1_raster_report_2024.nc"
    # )
    # dir_file_lancet_country_info: Path = (
    #     dir_admin_boundaries / "2025 Global Report Country Names and Groupings.xlsx"
    # )


def clean_pop_raw(path=Dirs.dir_pop_raw) -> None:
    """
    Remove hidden .tif files from the population raw directory (if mounted).
    Call explicitly; does not run on import.
    """
    try:
        for fname in os.listdir(path):
            if fname.startswith(".") and fname.endswith(".tif"):
                try:
                    (path / fname).unlink()
                except PermissionError:
                    logger.warning("Permission denied removing %s", fname)
    except FileNotFoundError:
        logger.info(
            "Population raw directory %s not found, skipping cleanup",
            path,
        )
    except Exception:
        logger.exception("Unexpected error cleaning %s", path)


def ensure_dirs_exist(paths: list) -> None:
    """
    Create required directories. No-op on import; call explicitly at runtime.

    Example:
        ensure_dirs_exist()
    """

    for p in paths:
        try:
            if not p.exists():
                # ask the user whether to create the missing directory
                try:
                    resp = (
                        input(f"Directory `{p}` does not exist. Create it? [y/N]: ")
                        .strip()
                        .lower()
                    )
                except EOFError:
                    # non-interactive environment: skip creation
                    logger.info(
                        "Non-interactive environment, skipping creation of %s", p
                    )
                    continue

                if resp in ("y", "yes"):
                    logger.info("Creating directory: %s", p)
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info("Skipping creation of directory: %s", p)
                    continue
            else:
                # ensure it exists (no-op if already present)
                p.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning("Permission denied creating directory %s (skipping)", p)
        except Exception:
            logger.exception("Unexpected error creating directory %s", p)


class SheetsFinalSubmission:
    global_total = "Global total"
    global_average = "Global average"
    country_infants = "Country infants"
    country_over65 = "Country 65+"
    country_over75 = "Country 75+"
    hdi_group = "HDI Group"
    who_region = "WHO Region"
    lc_region = "LC Region"
