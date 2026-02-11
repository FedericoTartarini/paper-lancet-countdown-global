from datetime import datetime
from pathlib import Path
from typing import List


class Vars:
    """Project-wide variables (use attributes directly, e.g. Vars.year_report)."""

    year_report: int = datetime.now().year
    year_max_analysis: int = year_report - 1
    year_min_analysis: int = 1980
    year_reference_start: int = 1986
    year_reference_end: int = 2005
    quantiles: List[float] = 0.95
    t_vars: List[str] = ["t_max", "t_min", "t_mean"]
    infants = "under_1"
    over_65 = "over_65"

    @classmethod
    def get_reference_years(cls) -> List[int]:
        """Return all years in the reference period as a list."""
        return list(range(cls.year_reference_start, cls.year_reference_end + 1))

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

    # local directory for ERA5-Land data
    e5l_h = data / Dirs.e5l / Dirs.e5l_h / Dirs.e5l_t
    e5l_d = data / Dirs.e5l / Dirs.e5l_d / Dirs.e5l_t
    e5l_q: Path = data / Dirs.e5l / Dirs.quantiles

    results = data / Dirs.results
    hw = results / Dirs.heatwaves
    hw_min_max = hw / "q_min_max"

    # population data
    # dir_pop_raw_ssd = Path("/Volumes/T7/lancet_countdown/population")
    pop_raw_ssd = data / Dirs.pop / "raw"
    pop_e5l_grid = data / Dirs.pop / (Dirs.e5l + "-grid")
    pop_e5l_grid_combined = data / Dirs.pop / (Dirs.e5l + "-grid-combined")


class FilesLocal:
    pop_before_2000 = (
        DirsLocal.data / Dirs.pop / "lancet-2023" / "Hybrid Demographics 1950-2020.nc"
    )
    pop_infant = (
        DirsLocal.pop_e5l_grid_combined / f"t-{Vars.infants}-regridded-combined.nc"
    )
    pop_over_65 = (
        DirsLocal.pop_e5l_grid_combined / f"t-{Vars.over_65}-regridded-combined.nc"
    )


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


if __name__ == "__main__":
    print(DirsGadi.e5l_h)
