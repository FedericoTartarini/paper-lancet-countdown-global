"""
# Calculate heatwave occurrences

## Heatwave definition

Heatwaves are now defined as:

Tmin > 95percentile AND Tmax > 95percentile

For more than 2 consecutive days (i.e. total of 3 or more days).

This replaces the definition of only Tmin > 99percentile for more than 3 consecutive days (total of 4 or more days).

This is what is requested from the Lancet. To be honest it's not clear whether this produces a substantially 'better' indicator since all heatwave indicators are arbitrary in absence of covariate data (i.e. impact data). Furthermore we know that the health impacts are mediated by many other things, so in any case we are truely interested just in the trends i.e. demonstrating that there is a) more heatwaves and b) more exposure to heatwaves - this can be followed by local studies but (as always) the point is to present a general risk factor trend.

> NOTE: considered just adding the newest year each time instead of re-calculating the whole thing. HOWEVER in reality, the input data is still changing year to year, so far have needed to re-calculate anyway (e.g. change in resolution, change from ERAI to ERA5, in the future probably use ERA5-Land, etc). Although it seems like a cool idea to have a reproducible method where each year you just add one thing, in practice its better to have one 'frozen' output corresponding to each publication, so that it's easy to go back later to find data corresponding to specific results. Additionally, generating one file per year means you have a folder full of files that are harder to share, and the outputs are in the end pretty small (<50MB in Float32)}.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from my_config import (
    temperature_summary_folder,
    max_year,
    dir_results_heatwaves_tmp,
    climatology_quantiles_folder,
)
from python_code.weather import heatwave_indices

xr.set_options(keep_attrs=True)


def ds_for_year(year):
    ds = xr.open_dataset(temperature_summary_folder / f"{year}_temperature_summary.nc")
    ds = ds.transpose("time", "latitude", "longitude")
    return ds


def apply_func_for_file(func, year, t_thresholds, t_var_names, days_threshold=2):
    ds = ds_for_year(year)

    datasets_year = [ds[name] for name in t_var_names]
    result = func(datasets_year, t_thresholds, days_threshold)

    # Add a year dimension matching the input file
    result = result.expand_dims(dim={"year": [year]})
    return year, result


def apply_func_and_save(
    func,
    year,
    output_folder,
    t_thresholds,
    t_var_names=["tmin", "tmax"],
    days_threshold=2,
    overwrite=False,
    filename_pattern="indicator_{year}.nc",
):
    output_file = output_folder / filename_pattern.format(year=year)
    if output_file.exists() is False and overwrite is False:
        year, result = apply_func_for_file(
            func,
            year,
            t_thresholds,
            t_var_names=t_var_names,
            days_threshold=days_threshold,
        )
        result.to_netcdf(output_file)
        return f"Created {output_file}"
    else:
        return f"Skipped {output_file}, already exists"


def apply_func_for_month(func, ds, month, t_thresholds, t_var_names, days_threshold=2):
    monthly_ds = ds.sel(time=ds["time"].dt.month == month)
    datasets_month = [monthly_ds[name] for name in t_var_names]

    result = func(datasets_month, t_thresholds, days_threshold)
    return result  # Returning only the monthly result


def apply_func_and_save_yearly(
    func,
    year,
    output_folder,
    t_thresholds,
    t_var_names=["tmin", "tmax"],
    days_threshold=2,
    overwrite=False,
    filename_pattern="indicator_{year}.nc",
):
    ds = ds_for_year(year)
    yearly_results = []

    for month in range(1, 13):
        monthly_result = apply_func_for_month(
            func, ds, month, t_thresholds, t_var_names, days_threshold
        )
        yearly_results.append(monthly_result)

    # Combine all monthly results into one dataset
    combined_result = xr.concat(yearly_results, pd.Index(range(1, 13), name="month"))
    combined_result = combined_result.assign_coords({"year": year})

    # Save the combined yearly file
    output_file = output_folder / filename_pattern.format(year=year)
    if not output_file.exists() or overwrite:
        combined_result.to_netcdf(output_file)
        return f"Created {output_file}"
    else:
        return f"Skipped {output_file}, already exists"


if __name__ == "__main__":
    temperature_files = [
        (year, temperature_summary_folder / f"{year}_temperature_summary.nc")
        for year in range(2022, max_year + 1)
    ]

    quantiles = [0.95]
    quantile = quantiles[0]
    t_var = "tmin"

    CLIMATOLOGY_QUANTILES = (
        climatology_quantiles_folder
        / f'daily_{t_var}_quantiles_{"_".join([str(int(100*q)) for q in quantiles])}_1986-2005.nc'
    )
    t_min_quantiles = xr.open_dataset(CLIMATOLOGY_QUANTILES)
    t_min_threshold = t_min_quantiles.sel(
        quantile=quantile, drop=True, tolerance=0.001, method="nearest"
    )

    t_var = "tmax"
    CLIMATOLOGY_QUANTILES = (
        climatology_quantiles_folder
        / f'daily_{t_var}_quantiles_{"_".join([str(int(100*q)) for q in quantiles])}_1986-2005.nc'
    )
    t_max_quantiles = xr.open_dataset(CLIMATOLOGY_QUANTILES)
    t_max_threshold = t_max_quantiles.sel(
        quantile=quantile, drop=True, tolerance=0.001, method="nearest"
    )

    t_var = "tmean"
    CLIMATOLOGY_QUANTILES = (
        climatology_quantiles_folder
        / f'daily_{t_var}_quantiles_{"_".join([str(int(100*q)) for q in quantiles])}_1986-2005.nc'
    )

    t_thresholds = [
        t_min_threshold.to_array().squeeze(),
        t_max_threshold.to_array().squeeze(),
    ]

    cos_lat = np.cos(np.radians(t_min_threshold.latitude))

    out_folder = dir_results_heatwaves_tmp / "heatwaves_monthly_era5"
    out_folder.mkdir(exist_ok=True)

    # Loop over years only
    res = Parallel(n_jobs=6, verbose=3)(
        delayed(apply_func_and_save_yearly)(
            heatwave_indices.heatwaves_days_multi_threshold,
            year,
            out_folder,
            t_thresholds,
            ["t_min", "t_max"],
        )
        for year, _ in temperature_files
    )

    out_folder = dir_results_heatwaves_tmp / "heatwaves_days_era5"

    out_folder.mkdir(exist_ok=True)

    res = Parallel(n_jobs=6, verbose=3)(
        delayed(apply_func_and_save)(
            heatwave_indices.heatwaves_days_multi_threshold,
            year,
            out_folder,
            t_thresholds,
            ["t_min", "t_max"],
        )
        for year, file in temperature_files
    )

    out_folder = dir_results_heatwaves_tmp / "heatwaves_counts_era5"

    out_folder.mkdir(exist_ok=True)

    # apply_func_and_save(heatwave_indices.heatwaves_counts_multi_threshold, 2000, out_folder, t_thresholds, t_var_names=['tmin', 'tmax'])

    res = Parallel(n_jobs=6, verbose=2)(
        delayed(apply_func_and_save)(
            heatwave_indices.heatwaves_counts_multi_threshold,
            year,
            out_folder,
            t_thresholds,
            ["t_min", "t_max"],
        )
        for year, file in temperature_files
    )
