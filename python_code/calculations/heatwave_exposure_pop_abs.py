"""
# Total heatwave exposures

From the health system perspective, we'd like to know not just exposures to change (which in the climate perspective
is useful to demonstrate that  HWs are really forming a trend relative to the null hypothesis of being normally
distributed around 0) - but the absolute values with the idea to know a) how big is this change from 'normal' and b)
how it compares to what we already cope with. More generally the idea is that if you measure millions more exposure
days but on a total value of billions, then even if you pick out a statistically significant trend you might not (
from the policy POV) care that much. On the other hand if you are talking 2x historical it's an issue.

The ideal is to show 'percentage change' rel. to a baseline. the problem is the population data doesn't exist and
even if it does, it doesn't make sense to average over 20years like we do for climatology.

The first step is to just calculate absolute values - these aren't too problematic since anyway the 'HW delta' is
kinda double-normalising since we 1x used 20y period for climatology then again for the baseline of the delta. Just
plotting then the time series gives a pretty good idea of where you stand relative to 'normal'

The next idea is to copy how GDP is presented as a percentage year-to-year. Since it doesn't make sense with pop to
normalise to a baseline period, and it's very arbitrary to pick one year of period, instead plot the percentage
change from previous year (e.g. https://fred.stlouisfed.org/graph/?g=eUmi)"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd

from cartopy import crs as ccrs
from scipy import stats
import os
import sys

# Figure settings
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (5, 2.5)
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["savefig.bbox"] = "tight"
MAP_PROJECTION = ccrs.EckertIII()

MIN_YEAR = 1980
MAX_YEAR = 2023

POP_FOLDER = POP_DATA_SRC / "lancet" / "populations" / "hybrid_2024"
population_infants_worldpop = xr.open_dataset(
    POP_DATA_SRC / "hybrid_2024" / f"worldpop_infants_1950_2023_era5_compatible.nc"
).sel(year=slice(1980, 2023))
population_elderly_worldpop = xr.open_dataset(
    POP_DATA_SRC / "hybrid_2024" / f"worldpop_elderly_1950_2023_era5_compatible.nc"
).sel(year=slice(1980, 2023))

population_worldpop = xr.concat(
    [
        population_infants_worldpop.rename({"infants": "pop"}),
        population_elderly_worldpop.rename({"elderly": "pop"}),
    ],
    dim=pd.Index([0, 65], name="age_band_lower_bound"),
)
HEATWAVE_FOLDER = DATA_SRC / "lancet/results/results_2024"
heatwave_metrics_files = sorted((HEATWAVE_FOLDER / "heatwave_days_era5").glob("*.nc"))
heatwave_metrics = xr.open_mfdataset(heatwave_metrics_files, combine="by_coords")

# Get the grid weighting factor from the latitude
cos_lat = np.cos(np.radians(heatwave_metrics.latitude))

exposures_over65 = heatwave_metrics["heatwaves_days"].transpose(
    "year", "latitude", "longitude"
) * population_elderly_worldpop.transpose("year", "latitude", "longitude")

exposures_over65 = exposures_over65.to_array()
exposures_over65 = exposures_over65.squeeze().drop("variable")

exposures_infants = heatwave_metrics["heatwaves_days"].transpose(
    "year", "latitude", "longitude"
) * population_infants_worldpop.transpose("year", "latitude", "longitude")

exposures_infants = exposures_infants.to_array()
exposures_infants = exposures_infants.squeeze().drop("variable")

exposures = xr.concat(
    [exposures_infants, exposures_over65],
    dim=pd.Index([0, 65], name="age_band_lower_bound"),
)

RESULTS_FOLDER = DATA_SRC / "lancet/results/results_2024/worldpop_hw_exposure"

exposures_over65 = exposures_over65.rename("heatwaves_days")

exposures_over65.to_netcdf(
    RESULTS_FOLDER
    / f"heatwave_exposure_over65_multi_threshold_{MIN_YEAR}-{MAX_YEAR}_worldpop.nc"
)
exposures_infants = exposures_infants.rename("heatwaves_days")

exposures_infants.to_netcdf(
    RESULTS_FOLDER
    / f"heatwave_exposure_infants_multi_threshold_{MIN_YEAR}-{MAX_YEAR}_worldpop.nc"
)
exposures_abs = xr.open_dataset(
    RESULTS_FOLDER
    / f"heatwave_exposure_multi_threshold_{MIN_YEAR}-{MAX_YEAR}_worldpop.nc"
)

exposures = exposures.rename("heatwaves_days")
exposures.to_netcdf(
    RESULTS_FOLDER
    / f"heatwave_exposure_multi_threshold_{MIN_YEAR}-{MAX_YEAR}_worldpop.nc"
)

total_exposures_over65 = exposures_over65.sum(
    dim=["latitude", "longitude"]
).to_dataframe()
total_exposures_infants = exposures_infants.sum(
    dim=["latitude", "longitude"]
).to_dataframe()

# total_exposures_infants.to_excel(RESULTS_FOLDER / 'heatwave_exposure_indicator_totals_infants.xlsx')
# total_exposures_infants.to_csv(RESULTS_FOLDER / 'heatwave_exposure_indicator_totals_infants.csv')

weighted_mean_infants = exposures_infants / population_infants_worldpop.sum(
    dim=["latitude", "longitude"]
)

divnorm = colors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=400)

# baseline = weighted_mean_infants.sel(year=slice(2001,2010)).mean(dim='year')
# decadal = 100 * (weighted_mean_infants.sel(year=slice(2011,2020)).mean(dim='year') - baseline) / baseline
# decadal = decadal.compute()

# f, ax = plt.subplots(figsize=(6,3), subplot_kw=dict(projection=MAP_PROJECTION),dpi=300)

# decadal.heatwaves_days.plot.pcolormesh(
#     norm=divnorm,
#     cbar_kwargs=dict(label='%'),
#     transform=ccrs.PlateCarree(),
#     ax=ax)

# ax.coastlines(linewidth=0.5)
# ax.set_title(f'Exposure change of infants between decades\n 2001-2010 and 2011-2020')
# f.savefig(RESULTS_FOLDER / 'decade change lt 1.png')
# f.savefig(RESULTS_FOLDER / 'decade change lt 1.pdf')

# weighted_mean_over65 = (exposures_over65 / population_over_65.sum(dim=['latitude', 'longitude']))

# baseline = weighted_mean_over65.sel(year=slice(2001,2010)).mean(dim='year')
# decadal = 100 * (weighted_mean_over65.sel(year=slice(2011,2020)).mean(dim='year') - baseline) / baseline
# decadal = decadal.compute()

# f, ax = plt.subplots(figsize=(6,3), subplot_kw=dict(projection=MAP_PROJECTION),dpi=300)

# decadal.heatwaves_days.plot(norm=divnorm,
# #                             robust=True,
# #                             vmin=-100, vmax=400, cmap='plasma',
#                             cbar_kwargs=dict(label='%'),
#                             transform=ccrs.PlateCarree(),
#                             ax=ax)

# ax.coastlines(linewidth=0.5)
# ax.set_title(f'Exposure change of over-65s between decades\n 2001-2010 and 2011-2020')
# f.savefig(RESULTS_FOLDER / 'decade change over 65.png')
