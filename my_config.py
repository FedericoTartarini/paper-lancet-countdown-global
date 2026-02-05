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


def ensure_directories(path_dirs: list[Path]):
    for path_dir in path_dirs:
        path_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(DirsLocal.e5l_d)
