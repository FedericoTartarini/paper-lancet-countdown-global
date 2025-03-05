"""
# Exposure to change in heatwave occurrence

> Using upscaled population data for pre-2000, this is an approximation! Needs to be shown as such on the graphs
> Using VERY ROUGH ESTIMATE of yearly newborn pop, this is EVEN MORE ROUGH for the pre-2000 data
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from my_config import (
    dir_results_pop_exposure,
    year_reference_end,
    year_reference_start,
    dir_results_heatwaves_days,
    dir_file_elderly_exposure_change,
    dir_file_infants_exposure_change,
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

    heatwave_metrics_files = sorted(dir_results_heatwaves_days.glob("*.nc"))
    heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

    """
    ### Calculate the difference from the mean number of days and number of events
    
    > NOTE: we do a sort of double-difference in this case, since we first define heatwaves based on a historical 
    distribution (99th percentile) then we again calculate the change. Question is open as to whether it's also 
    critical to use the same period for calculating the percentile and calculating the reference period. Also a 
    question whether we should compare the change in exposures rather than the change in heatwaves. The problem (or 
    maybe it's a benefit??) is that you also mix in changes in total population and demographics"""

    heatwaves_metrics_reference = heatwave_metrics.sel(
        year=slice(year_reference_start, year_reference_end)
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

    exposures_over65.to_netcdf(dir_file_elderly_exposure_change)

    exposures_infants.to_netcdf(dir_file_infants_exposure_change)

    total_exposures_over65 = exposures_over65.sum(
        dim=["latitude", "longitude"]
    ).to_dataframe()
    total_exposures_infants = exposures_infants.sum(
        dim=["latitude", "longitude"]
    ).to_dataframe()

    total_exposures_over65.to_csv(
        dir_results_pop_exposure / "heatwave_exposure_change_totals_over65.csv"
    )
    total_exposures_infants.to_csv(
        dir_results_pop_exposure / "heatwave_exposure_change_totals_infants.csv"
    )

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

    weighted_mean_over65.to_csv(
        dir_results_pop_exposure / "heatwave_days_change_weighted_over65.csv"
    )
    weighted_mean_infants.to_csv(
        dir_results_pop_exposure / "heatwave_days_change_weighted_infants.csv"
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

    # n = 10
    # rolling_stats = pd.DataFrame(
    #     {
    #         "heatwave_days_change_land": heatwave_metrics_delta_mean.rolling(n)
    #         .mean()
    #         .heatwaves_days,
    #         "heatwave_days_change_infants": weighted_mean_infants.rolling(n)
    #         .mean()
    #         .heatwaves_days,
    #         "heatwave_days_change_over65": weighted_mean_over65.rolling(n)
    #         .mean()
    #         .heatwaves_days,
    #     }
    # ).dropna()
    #
    # rolling_stats.to_csv(
    #     dir_results_pop_exposure / "heatwave_days_change_10_year_rolling_mean.csv"
    # )
    #
    # (weighted_mean_infants - heatwave_metrics_delta_mean).rolling(
    #     n
    # ).mean().heatwaves_days.plot()
    #
    # (weighted_mean_over65 - heatwave_metrics_delta_mean).rolling(
    #     n
    # ).mean().heatwaves_days.plot()


if __name__ == "__main__":
    main()
    pass
