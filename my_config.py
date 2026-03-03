import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from collections.abc import Mapping

from cartopy import crs as ccrs

from matplotlib import pyplot as plt

# plt.rcParams["figure.figsize"] = [8, 4]
# plt.rcParams["savefig.dpi"] = 300
# plt.rcParams["figure.autolayout"] = True  # Enable tight_layout by default
# plt.rcParams["legend.frameon"] = False  # Disable legend frame by default
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


class Vars:
    """Project-wide variables (use attributes directly, e.g. Vars.year_report)."""

    year_report: int = datetime.now().year
    year_max_analysis: int = year_report - 1
    year_min_analysis: int = 1980
    year_reference_start: int = 1986
    year_reference_end: int = 2005
    baseline_periods: List[tuple[int, int]] = [(1986, 2005), (2007, 2016)]
    baseline_period = "baseline_period"
    quantiles: List[float] = 0.95
    t_vars: List[str] = ["t_max", "t_min", "t_mean"]
    infants = "under_1"
    over_65 = "over_65"
    hw_count = "heatwave_counts"
    hw_days = "heatwave_days"
    hw_q_min_max = "q_min_max"
    age_band = "age_band"
    map_projection = ccrs.EckertIV()

    @classmethod
    def get_reference_years(cls) -> List[int]:
        """Return all years in the reference period as a list."""
        return list(range(cls.year_reference_start, cls.year_reference_end + 1))

    @classmethod
    def get_baseline_periods(cls) -> List[tuple[int, int]]:
        """Return all baseline periods used for change calculations."""
        return cls.baseline_periods

    @classmethod
    def format_baseline_period(cls, period: tuple[int, int]) -> str:
        """Format a baseline period tuple as a string label."""
        return f"{period[0]}-{period[1]}"

    @classmethod
    def get_baseline_labels(cls) -> List[str]:
        """Return string labels for configured baseline periods."""
        return [cls.format_baseline_period(period) for period in cls.baseline_periods]

    @classmethod
    def get_analysis_years(cls) -> List[int]:
        """Return all years in the analysis period as a list."""
        return list(range(cls.year_min_analysis, cls.year_max_analysis + 1))


class Dirs:
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
    quantiles = "quantiles"
    results = "results"
    pop = "population"


class DirsLocal:
    """Local paths for data processing on personal computer."""

    one_drive = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)"
    )
    data = one_drive / "data" / "lancet" / "countdown-global"
    project = Path(os.getcwd())

    # local directory for ERA5-Land data
    e5l_h = data / Dirs.e5l / Dirs.e5l_h / Dirs.e5l_t
    e5l_d = data / Dirs.e5l / Dirs.e5l_d / Dirs.e5l_t
    e5l_q: Path = data / Dirs.e5l / Dirs.quantiles

    results = data / Dirs.results
    results_report = results / f"results_{Vars.year_report}"
    hw = results / Dirs.heatwaves
    hw_q_min_max = hw / Vars.hw_q_min_max
    aggregates = results_report / "aggregates"
    aggregates_figures = aggregates / "figures"

    # population data
    # pop_raw_ssd = Path("/Volumes/T7/lancet_countdown/population")
    pop = data / Dirs.pop
    pop_raw_ssd = pop / "raw"
    pop_e5l_grid = pop / (Dirs.e5l + "-grid")
    pop_e5l_grid_combined = data / Dirs.pop / (Dirs.e5l + "-grid-combined")

    # boundaries and rasters
    boundaries: Path = data / "boundaries-2026"
    world_bank_boundaries: Path = boundaries / "world-bank-shapefile-admin-0"
    region_rasters: Path = boundaries / "region-rasters"

    manuscript = project / "manuscript" / f"{Vars.year_report}"
    figures = manuscript / "figures"


class FilesLocal:
    pop_before_2000 = (
        DirsLocal.data / Dirs.pop / "lancet-2023" / "Hybrid Demographics 1950-2020.nc"
    )
    pop_inf = (
        DirsLocal.pop_e5l_grid_combined / f"t-{Vars.infants}-regridded-combined.nc"
    )
    pop_over_65 = (
        DirsLocal.pop_e5l_grid_combined / f"t-{Vars.over_65}-regridded-combined.nc"
    )
    hw_combined_q = (
        DirsLocal.results_report / f"hw-{Vars.hw_q_min_max}-combined-all-ages.nc"
    )
    hw_change_combined = (
        DirsLocal.results_report / f"hw-{Vars.hw_q_min_max}-change-combined-all-ages.nc"
    )

    world_bank_shapefile: Path = (
        DirsLocal.world_bank_boundaries / "WB_GAD_ADM0_complete.shp"
    )
    raster_country: Path = DirsLocal.region_rasters / "country.nc"
    raster_who: Path = DirsLocal.region_rasters / "who.nc"
    raster_hdi: Path = DirsLocal.region_rasters / "hdi.nc"
    raster_lancet: Path = DirsLocal.region_rasters / "lc.nc"
    country_names_groupings: Path = (
        DirsLocal.boundaries / "2026-country-names-groupings.xlsx"
    )
    aggregate_country = DirsLocal.aggregates / "country.nc"
    aggregate_who = DirsLocal.aggregates / "who.nc"
    aggregate_hdi = DirsLocal.aggregates / "hdi.nc"
    aggregate_lancet = DirsLocal.aggregates / "lancet.nc"
    aggregate_submission = (
        DirsLocal.manuscript
        / "1.1.1 - 2026 Global Report - Data Submission - Tartarini.xlsx"
    )
    typst_variables = DirsLocal.manuscript / "typst_variables.json"


class DirsGadi:
    """Gadi-specific paths for HPC data processing."""

    # Input: ERA5-Land hourly data on Gadi (read-only)
    e5l_h = Path(
        f"/g/data/{Dirs.gadi_prj_era}/{Dirs.e5l}/{Dirs.reanalysis}/{Dirs.e5l_t}"
    )

    scratch = Path(
        f"/scratch/{Dirs.gadi_prj_compute}/{Dirs.gadi_usr}/{Dirs.project_name}"
    )
    data = scratch / "data" / "lancet" / "countdown-global"

    # Output: Daily summaries on scratch (fast write access)
    e5l_d = data / Dirs.e5l / Dirs.e5l_d / Dirs.e5l_t
    e5l_q = data / Dirs.e5l / Dirs.quantiles

    results = data / Dirs.results
    hw = results / Dirs.heatwaves
    hw_min_max = hw / "q_min_max"

    # population data
    pop_raw = data / Dirs.pop / "raw"
    pop_e5l_grid = data / Dirs.pop / (Dirs.e5l + "_grid")


class VarsWorldPop:
    """Variables for WorldPop data processing."""

    year_worldpop_start: int = 2000
    year_worldpop_end: int = 2030
    age_groups: List[List[int]] = [[0], [65, 70, 75, 80, 85, 90]]
    year_min: int = 2000
    year_max: int = 2025
    under_1_label: str = "under_1"
    over_65_label: str = "65_over"

    @classmethod
    def get_years_range(cls) -> List[int]:
        """Return all years for WorldPop processing as a list."""
        return list(range(cls.year_min, cls.year_max + 1))


def ensure_directories(path_dirs: list[Path]):
    for path_dir in path_dirs:
        path_dir.mkdir(parents=True, exist_ok=True)


class Labels:
    """Unified, human-friendly labels for age bands and variables."""

    # Main age bands
    infants = "Infants"
    older_adults = "Older adults"
    # Exposure metrics
    person_days = "Person-days"
    person_events = "Person-events"
    avg_days_per_person = "Average heatwave days per person"
    exposure = "Heatwave exposure"
    # Add more as needed
    age_band_labels = {
        Vars.infants: infants,
        Vars.over_65: older_adults,
        "under_1": infants,
        "over_65": older_adults,
        "65_over": older_adults,
        "infants": infants,
        "older_adults": older_adults,
    }
    metric_labels = {
        "person_days": person_days,
        "person-events": person_events,
        "person-events": person_events,
        "person-days": person_days,
        "avg_days_per_person": avg_days_per_person,
        "exposure": exposure,
        Vars.hw_days: person_days,
        Vars.hw_count: person_events,
    }

    @classmethod
    def get_label(cls, key: str) -> str:
        """Return a human-friendly label for an age band or variable name."""
        if key in cls.age_band_labels:
            return cls.age_band_labels[key]
        if key in cls.metric_labels:
            return cls.metric_labels[key]
        return str(key)


def deep_update(base_dict, update_with):
    """
    Recursively updates a dictionary.
    If a key exists in both and both values are dictionaries, it merges them.
    Otherwise, it updates/adds the value from update_with.
    """
    for key, value in update_with.items():
        # Round floats to 1 decimal place as requested
        if isinstance(value, float):
            value = round(value, 1)

        if (
            isinstance(value, Mapping)
            and key in base_dict
            and isinstance(base_dict[key], Mapping)
        ):
            # Recursively merge nested dictionaries
            deep_update(base_dict[key], value)
        else:
            # Otherwise, just set/overwrite the value
            # If value is a dict but key wasn't in base_dict,
            # we should still round floats inside that new dict.
            if isinstance(value, dict):
                _round_nested_floats(value)
            base_dict[key] = value
    return base_dict


def _round_nested_floats(d):
    """Helper to round floats in newly added nested dictionaries."""
    for k, v in d.items():
        if isinstance(v, dict):
            _round_nested_floats(v)
        elif isinstance(v, float):
            d[k] = round(v, 1)


def update_typst_json(new_data, file_path=FilesLocal.typst_variables):
    """Update the typst_variables.json file with new data, performing a deep merge and rounding
    floats to 1 decimal place."""
    data = {}

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {file_path} was corrupted. Starting fresh.")

    # 1. Perform the deep merge (this also handles the rounding)
    updated_data = deep_update(data, new_data)

    # 2. Save back to JSON
    with open(file_path, "w") as f:
        json.dump(updated_data, f, indent=2)


if __name__ == "__main__":
    # print(DirsGadi.e5l_h)

    update_typst_json(
        {
            "methods": {
                "year_report": Vars.year_report,
                "year_max_analysis": Vars.year_max_analysis,
                "year_reference_period": f"{Vars.year_reference_start}-{Vars.year_reference_end}",
                "year_reference_periods": Vars.get_baseline_labels(),
                "quantiles": Vars.quantiles,
            }
        }
    )
