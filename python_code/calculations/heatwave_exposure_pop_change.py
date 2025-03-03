"""
# Exposure to change in heatwave occurrance

- INFANTS and over 65


> Now using the new heatwave definitions that depend on both the tmin and tmax percentiles

---

> Using upscaled population data for pre-2000, this is an approximation! Needs to be shown as such on the graphs

---

> Using VERY ROUGH ESTIMATE of yearly newborn pop, this is EVEN MORE ROUGH for the pre-2000 data
"""

import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt

from cartopy import crs as ccrs
from scipy import stats
import os
import sys

from my_config import (
    dir_results_pop_exposure,
    dir_results,
    year_max_analysis,
    year_min_analysis,
    year_report,
    year_reference_end,
    year_reference_start,
)

# Figure settings
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (5, 2.5)
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["savefig.bbox"] = "tight"

MAP_PROJECTION = ccrs.EckertIII()

# max_year = 2023

# REFERENCE_YEAR_START = 1986
# REFERENCE_YEAR_END = 2005

# MIN_YEAR = 1980

# Load population and demographic data
population_infants_worldpop = xr.open_dataset(
    dir_results
    / "hybrid_pop"
    / f"worldpop_infants_1950_{year_max_analysis}_era5_compatible.nc"
)
population_elderly_worldpop = xr.open_dataset(
    dir_results
    / "hybrid_pop"
    / f"worldpop_elderly_1950_{year_max_analysis}_era5_compatible.nc"
)

population_worldpop = xr.concat(
    [
        population_infants_worldpop.rename({"infants": "pop"}),
        population_elderly_worldpop.rename({"elderly": "pop"}),
    ],
    dim=pd.Index([0, 65], name="age_band_lower_bound"),
)

heatwave_metrics_files = sorted(
    (dir_results / "heatwaves" / f"results_{year_report}" / "heatwaves_days_era5").glob(
        "*.nc"
    )
)
heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

"""
### Calculate the difference from the mean number of days and number of events

> NOTE: we do a sort of double-difference in this case, since we first define heatwaves based on a historical distribution (99th percentile) then we again calculate the change. Question is open as to whether it's also critical to use the same period for calculating the percentile and calculating the reference period. Also a question whether we should compare the change in exposures rather than the change in heatwaves. The problem (or maybe it's a benefit??) is that you also mix in changes in total population and demographics
"""

heatwaves_metrics_reference = heatwave_metrics.sel(
    year=slice(year_reference_start, year_reference_end)
).mean(dim="year")
heatwave_metrics_delta = heatwave_metrics - heatwaves_metrics_reference

# Get the grid weighting factor from the latitude
cos_lat = np.cos(np.radians(heatwave_metrics.latitude))

# Get the total population for normalising
total_pop_over_65 = population_elderly_worldpop.sum(
    dim=["latitude", "longitude"], skipna=True
)

exposures_over65 = heatwave_metrics_delta["heatwaves_days"].transpose(
    "year", "latitude", "longitude"
) * population_elderly_worldpop.transpose("year", "latitude", "longitude")

exposures_over65 = exposures_over65.drop_vars("age_band_lower_bound")

exposures_infants = heatwave_metrics_delta["heatwaves_days"].transpose(
    "year", "latitude", "longitude"
) * population_infants_worldpop.transpose("year", "latitude", "longitude")

exposures_over65.to_netcdf(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_over65_multi_threshold_{year_min_analysis}-{year_max_analysis}.nc"
)

exposures_infants.to_netcdf(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_infants_multi_threshold_{year_min_analysis}-{year_max_analysis}.nc"
)

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
    (exposures_over65 / population_elderly_worldpop.sum(dim=["latitude", "longitude"]))
    .where(~np.isnan(population_elderly_worldpop))
    .sum(dim=["latitude", "longitude"], skipna=True)
    .to_dataframe()
).drop("age_band_lower_bound", axis=1)

weighted_mean_infants = (
    (exposures_infants / population_infants_worldpop.sum(dim=["latitude", "longitude"]))
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

heatwave_metrics_delta_mean = (
    (heatwave_metrics_delta * cos_lat)
    .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
    .mean(dim=["latitude", "longitude"], skipna=True)
    .to_dataframe()
    .drop("age_band_lower_bound", axis=1)
)

f, ax = plt.subplots(1, 1)

# heatwave_metrics_delta_mean.heatwaves_days.plot(ax=ax, label="Global mean (land)")

weighted_mean_infants.plot(ax=ax, label="Weighted mean (infants)")
weighted_mean_over65.plot(ax=ax, label="Weighted mean (over-65)")
ax.legend()
ax.set(
    ylabel="Heatwave days",
    title="Change in heatwave days relative to 1986-2005 baseline",
)
plt.show()

n = 10
rolling_stats = pd.DataFrame(
    {
        "heatwave_days_change_land": heatwave_metrics_delta_mean.rolling(n)
        .mean()
        .heatwaves_days,
        "heatwave_days_change_infants": weighted_mean_infants.rolling(n)
        .mean()
        .heatwaves_days,
        "heatwave_days_change_over65": weighted_mean_over65.rolling(n)
        .mean()
        .heatwaves_days,
    }
).dropna()

rolling_stats.to_csv(
    dir_results_pop_exposure / "heatwave_days_change_10_year_rolling_mean.csv"
)

(weighted_mean_infants - heatwave_metrics_delta_mean).rolling(
    n
).mean().heatwaves_days.plot()

(weighted_mean_over65 - heatwave_metrics_delta_mean).rolling(
    n
).mean().heatwaves_days.plot()
