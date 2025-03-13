"""
# Exposure to change in heatwave occurrence

> Using upscaled population data for pre-2000, this is an approximation! Needs to be shown as such on the graphs
> Using VERY ROUGH ESTIMATE of yearly newborn pop, this is EVEN MORE ROUGH for the pre-2000 data
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from my_config import (
    Vars,
    Dirs,
)
from python_code.shared_functions import (
    read_pop_data_processed,
    calculate_exposure_population,
)


def main():
    # Load population and demographic data
    population_infants_worldpop, population_elderly_worldpop, population_worldpop = (
        read_pop_data_processed()
    )

    heatwave_metrics_files = sorted(Dirs.dir_results_heatwaves_days.value.glob("*.nc"))
    heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

    """
    ### Calculate the difference from the mean number of days and number of events
    
    > NOTE: we do a sort of double-difference in this case, since we first define heatwaves based on a historical 
    distribution (99th percentile) then we again calculate the change. Question is open as to whether it's also 
    critical to use the same period for calculating the percentile and calculating the reference period. Also a 
    question whether we should compare the change in exposures rather than the change in heatwaves. The problem (or 
    maybe it's a benefit??) is that you also mix in changes in total population and demographics"""

    heatwaves_metrics_reference = heatwave_metrics.sel(
        year=slice(Vars.year_reference_start, Vars.year_reference_end)
    ).mean(dim="year")
    heatwave_metrics_delta = heatwave_metrics - heatwaves_metrics_reference

    # Get the total population for normalising
    total_pop_over_65 = population_elderly_worldpop.sum(
        dim=["latitude", "longitude"], skipna=True
    )

    exposures_over65 = calculate_exposure_population(
        data=population_elderly_worldpop, heatwave_metrics=heatwave_metrics_delta
    )

    exposures_infants = calculate_exposure_population(
        data=population_infants_worldpop, heatwave_metrics=heatwave_metrics_delta
    )

    exposures_over65.to_netcdf(Dirs.dir_file_elderly_exposure_change.value)

    exposures_infants.to_netcdf(Dirs.dir_file_infants_exposure_change.value)

    weighted_mean_over65 = (
        (
            exposures_over65
            / population_elderly_worldpop.sum(dim=["latitude", "longitude"])
        )
        .where(~np.isnan(population_elderly_worldpop))
        .sum(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
    ).drop("age_band_lower_bound", axis=1)

    weighted_mean_infants = (
        (
            exposures_infants
            / population_infants_worldpop.sum(dim=["latitude", "longitude"])
        )
        .where(~np.isnan(population_infants_worldpop))
        .sum(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
        .drop("age_band_lower_bound", axis=1)
    )

    # Get the grid weighting factor from the latitude
    cos_lat = np.cos(np.radians(heatwave_metrics.latitude))

    f, ax = plt.subplots(1, 1, constrained_layout=True)

    # heatwave_metrics_delta_mean.heatwaves_days.plot(ax=ax, label="Global mean (land)")

    weighted_mean_infants.plot(ax=ax, label="Weighted mean (infants)")
    weighted_mean_over65.plot(ax=ax, label="Weighted mean (over-65)")
    ax.legend()
    ax.set(
        ylabel="Heatwave days",
        title="Change in heatwave days relative to 1986-2005 baseline",
    )
    plt.show()

    heatwave_metrics_delta_mean = (
        (heatwave_metrics_delta * cos_lat)
        .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
        .mean(dim=["latitude", "longitude"], skipna=True)
        .to_dataframe()
        .drop("age_band_lower_bound", axis=1)
    )


if __name__ == "__main__":
    main()
    pass
