"""
# Exposure to change in heatwave occurrance

- INFANTS and over 65


> Now using the new heatwave definitions that depend on both the tmin and tmax percentiles

---

> Using upscaled population data for pre-2000, this is an approximation! Needs to be shown as such on the graphs

---

> Using VERY ROUGH ESTIMATE of yearly newborn pop, this is EVEN MORE ROUGH for the pre-2000 data
"""

from pathlib import Path
import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt

from cartopy import crs as ccrs
from scipy import stats
import os
import sys

from my_config import (
    max_year,
    dir_results,
    report_year,
    reference_year_start,
    reference_year_end,
    min_year,
    dir_results_pop_exposure,
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (5, 2.5)
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["savefig.bbox"] = "tight"

population_infants_worldpop = xr.open_dataset(
    dir_results / "hybrid_pop" / f"worldpop_infants_1950_{max_year}_era5_compatible.nc"
)
population_elderly_worldpop = xr.open_dataset(
    dir_results / "hybrid_pop" / f"worldpop_elderly_1950_{max_year}_era5_compatible.nc"
)

population_worldpop = xr.concat(
    [
        population_infants_worldpop.rename({"infants": "pop"}),
        population_elderly_worldpop.rename({"elderly": "pop"}),
    ],
    dim=pd.Index([0, 65], name="age_band_lower_bound"),
)

heatwave_metrics_files = sorted(
    (dir_results / "heatwaves" / f"results_{report_year}" / "heatwaves_days_era5").glob(
        "*.nc"
    )
)
heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

"""
### Calculate the difference from the mean number of days and number of events

> NOTE: we do a sort of double-difference in this case, since we first define heatwaves based on a historical distribution (99th percentile) then we again calculate the change. Question is open as to whether it's also critical to use the same period for calculating the percentile and calculating the reference period. Also a question whether we should compare the change in exposures rather than the change in heatwaves. The problem (or maybe it's a benefit??) is that you also mix in changes in total population and demographics
"""

heatwaves_metrics_reference = heatwave_metrics.sel(
    year=slice(reference_year_start, reference_year_end)
).mean(dim="year")
heatwave_metrics_delta = heatwave_metrics - heatwaves_metrics_reference

print(heatwave_metrics_delta)

# Get the grid weighting factor from the latitude
cos_lat = np.cos(np.radians(heatwave_metrics.latitude))
# Get the total population for normalising
total_pop_over_65 = population_elderly_worldpop.sum(
    dim=["latitude", "longitude"], skipna=True
)

population_elderly_worldpop = population_elderly_worldpop.sortby(
    "latitude", ascending=False
)

exposures_over65 = (
    heatwave_metrics_delta["heatwaves_days"].transpose("year", "latitude", "longitude")
    * population_elderly_worldpop.transpose("year", "latitude", "longitude")["elderly"]
)

exposures_over65 = exposures_over65.drop_vars("age_band_lower_bound")

exposures_over65 = exposures_over65.rename("heatwaves_days")

print(exposures_over65)

exposures_over65.to_netcdf(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_over65_multi_threshold_{min_year}-{max_year}_worldpop.nc"
)

exposures_infants = (
    heatwave_metrics_delta["heatwaves_days"].transpose("year", "latitude", "longitude")
    * population_infants_worldpop.transpose("year", "latitude", "longitude")["infants"]
)

exposures_infants = exposures_infants.drop_vars("age_band_lower_bound")
exposures_infants = exposures_infants.rename("heatwaves_days")

exposures_infants.to_netcdf(
    dir_results_pop_exposure
    / f"heatwave_exposure_change_infants_multi_threshold_{min_year}-{max_year}_worldpop.nc"
)

total_exposures_over65 = exposures_over65.sum(
    dim=["latitude", "longitude"]
).to_dataframe()
total_exposures_infants = exposures_infants.sum(
    dim=["latitude", "longitude"]
).to_dataframe()

total_exposures_over65.to_csv(
    dir_results_pop_exposure / "heatwave_exposure_change_totals_over65_worldpop.csv"
)
total_exposures_infants.to_csv(
    dir_results_pop_exposure / "heatwave_exposure_change_totals_infants_worldpop.csv"
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
    dir_results_pop_exposure / "heatwave_days_change_weighted_over65_worldpop.csv"
)
weighted_mean_infants.to_csv(
    dir_results_pop_exposure / "heatwave_days_change_weighted_infants_worldpop.csv"
)

heatwave_metrics_delta_mean = (
    (heatwave_metrics_delta * cos_lat)
    .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
    .mean(dim=["latitude", "longitude"], skipna=True)
    .to_dataframe()
    .drop("age_band_lower_bound", axis=1)
)

heatwave_metrics_delta_mean = (
    (heatwave_metrics_delta["heatwaves_days"] * cos_lat)
    .where(~np.isnan(population_elderly_worldpop.max(dim="year")))
    .mean(dim=["latitude", "longitude"], skipna=True)
    .to_dataframe()
    .drop("age_band_lower_bound", axis=1)
)

population_elderly_worldpop.max(dim="year")

f, ax = plt.subplots(1, 1)

heatwave_metrics_delta_mean.plot(ax=ax, label="Global mean (land)")

weighted_mean_infants.infants.plot(ax=ax, label="Weighted mean (infants)")
weighted_mean_over65.elderly.plot(ax=ax, label="Weighted mean (over-65)")
ax.legend()
ax.set(
    ylabel="Heatwave days",
    title="Change in heatwave days relative to 1986-2005 baseline",
)
plt.show()
# f.savefig(RESULTS_FOLDER / 'weighted heatwave change comparison worldpop.png', dpi=300)
# f.savefig(RESULTS_FOLDER / 'weighted heatwave change comparison worldpop.pdf')

f, ax = plt.subplots(1, 1)
n = 10
heatwave_metrics_delta_mean.rolling(n).mean().plot(ax=ax, label="Global mean (land)")

weighted_mean_infants.rolling(n).mean().plot(ax=ax, label="Weighted mean (infants)")
weighted_mean_over65.rolling(n).mean().plot(ax=ax, label="Weighted mean (over-65)")
ax.legend()
ax.set(
    ylabel="Heatwave days",
    title=f"Change in heatwave days relative to 1986-2005 baseline,\n {n}-year rolling mean",
)
# f.savefig(RESULTS_FOLDER / 'weighted rolling heatwave change comparison worldpop.png', dpi=300)
# f.savefig(RESULTS_FOLDER / 'weighted rolling heatwave change comparison worldpop.pdf')

print(heatwave_metrics_delta_mean.rolling(n).mean())

n = 10
rolling_stats = pd.DataFrame(
    {
        "heatwave_days_change_land": heatwave_metrics_delta_mean.rolling(n).mean()[
            "elderly"
        ],
        "heatwave_days_change_infants": weighted_mean_infants.rolling(n).mean()[
            "infants"
        ],
        "heatwave_days_change_over65": weighted_mean_over65.rolling(n).mean()[
            "elderly"
        ],
    }
).dropna()
rolling_stats.to_csv(
    dir_results_pop_exposure / "heatwave_days_change_10_year_rolling_mean_worldpop.csv"
)
(weighted_mean_infants - heatwave_metrics_delta_mean).rolling(n).mean()[
    "infants"
].plot()

(weighted_mean_over65 - heatwave_metrics_delta_mean).rolling(n).mean()["elderly"].plot()
