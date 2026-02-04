from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from my_config import (
    Vars,
    DirsLocal,
)

if __name__ == "__main__":
    # Collect file list once
    file_list = []
    for file in DirsLocal.e5l_d.rglob("*.nc"):
        file = Path(file)
        year = int(file.name.split("_")[0])
        if Vars.year_reference_start <= year <= Vars.year_reference_end:
            file_list.append(file)
    file_list = sorted(file_list)

    # Open dataset once with larger chunks for better performance on local machine
    daily_temperatures_ds = xr.open_mfdataset(file_list, combine="by_coords")
    daily_temperatures_ds.chunk(chunks={"latitude": 200, "longitude": 200})

    for t_var in Vars.t_vars:
        # Select variable
        daily_temperatures = daily_temperatures_ds[t_var].chunk({"time": -1})

        DirsLocal.e5l_q.mkdir(parents=True, exist_ok=True)

        climatology_quantiles = (
            DirsLocal.e5l_q
            / f"daily_{t_var}_quantiles_{Vars.quantiles}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
        )

        daily_quantiles = daily_temperatures.quantile(Vars.quantiles, dim="time")

        with dask.config.set(scheduler="processes"), ProgressBar():
            daily_quantiles = daily_quantiles.compute()
            daily_quantiles.to_netcdf(climatology_quantiles)
