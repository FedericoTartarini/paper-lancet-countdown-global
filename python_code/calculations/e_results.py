"""Plotting helpers for report figures."""

from __future__ import annotations

import logging
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs

from my_config import DirsLocal, FilesLocal, Vars
from python_code.calculations.a_heatwave_exposure_pop_abs import standardize_grid


def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _load_heatwave_metrics() -> xr.Dataset:
    """Load heatwave metrics from the ERA5-Land output folder."""
    hw_files = sorted(DirsLocal.hw_q_min_max.glob("*.nc"))
    if not hw_files:
        raise FileNotFoundError(f"No heatwave files found in {DirsLocal.hw_q_min_max}")

    ds = xr.open_mfdataset(hw_files, combine="by_coords", decode_timedelta=True)
    return standardize_grid(ds)


def _load_land_mask() -> xr.DataArray:
    """Load a land mask from the country raster (Admin0)."""
    if not FilesLocal.raster_country.exists():
        raise FileNotFoundError(
            f"Country raster not found: {FilesLocal.raster_country}. "
            "Run c_regions_rasters.py first."
        )
    ds = xr.open_dataset(FilesLocal.raster_country)
    var = next(iter(ds.data_vars))
    mask = ds[var].notnull()
    return standardize_grid(mask)


def _shift_longitudes(da: xr.DataArray) -> xr.DataArray:
    """Ensure longitude is in -180..180 range for plotting."""
    if "longitude" in da.coords and da.longitude.max() > 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby(
            "longitude"
        )
    return da


def plot_change_in_heatwave_days(
    report_year: int,
    baseline_years: tuple[int, int],
    output_path: Path | None = None,
) -> Path:
    """
    Plot change in heatwave days for a report year relative to a baseline period.

    Parameters
    ----------
    report_year
        Year to compare against the baseline.
    baseline_years
        Tuple of (start_year, end_year) for the baseline period.
    output_path
        Optional output path for the figure. Defaults to DirsLocal.figures.

    Returns
    -------
    Path
        The path to the saved figure.
    """
    setup_logging()

    hw_ds = _load_heatwave_metrics()
    land_mask = _load_land_mask()

    if Vars.hw_days not in hw_ds.data_vars:
        raise ValueError(
            f"Missing '{Vars.hw_days}' in heatwave dataset: {list(hw_ds.data_vars)}"
        )

    baseline = (
        hw_ds[Vars.hw_days]
        .sel(year=slice(baseline_years[0], baseline_years[1]))
        .mean(dim="year")
    )
    target = hw_ds[Vars.hw_days].sel(year=report_year)

    change = target - baseline

    if np.issubdtype(change.dtype, np.timedelta64):
        change = change / np.timedelta64(1, "D")

    change = _shift_longitudes(change)
    land_mask = _shift_longitudes(land_mask)
    change = change.where(land_mask)

    f, ax = plt.subplots(
        1,
        1,
        figsize=(7, 5),
        subplot_kw=dict(projection=Vars.map_projection),
        constrained_layout=True,
    )

    ax.coastlines(linewidths=0.25, resolution="110m")
    ax.add_feature(cfeature.OCEAN, facecolor="lightgray")

    cbar_ax = f.add_axes([0.1, 0.1, 0.8, 0.03])  # Position of the colorbar
    plot = change.plot(
        transform=ccrs.PlateCarree(),
        ax=ax,
        vmin=-50,
        vmax=50,
        cmap="RdBu_r",
        cbar_kwargs={
            "label": "Change in number of heatwave days",
            "orientation": "horizontal",
            "cax": cbar_ax,
        },
    )

    # cbar = f.colorbar(
    #     plot,
    #     ax=ax,
    #     orientation="horizontal",
    #     pad=0.08,
    #     fraction=0.05,
    #     aspect=40,
    # )
    # cbar.set_label("Change in number of heatwave days")

    for spine in ax.spines.values():
        spine.set_edgecolor("lightgray")
        spine.set_linewidth(0.5)

    ax.set_title(
        f"Change in {report_year} relative to {baseline_years[0]}–{baseline_years[1]}"
    )

    if output_path is None:
        output_path = DirsLocal.figures / f"map_hw_change_{report_year}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(output_path, dpi=300)
    plt.close(f)

    logging.info("Saved heatwave change plot to %s", output_path)
    return output_path


if __name__ == "__main__":
    plot_change_in_heatwave_days(
        report_year=2025,
        baseline_years=(Vars.year_reference_start, Vars.year_reference_end),
    )
