from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

from my_config import (
    Vars,
    Dirs,
)

if __name__ == "__main__":
    for t_var in ["t_max", "t_min", "t_mean"]:
        file_list = []
        for file in Dirs.dir_era_daily.rglob("*.nc"):
            file = Path(file)

            year = int(file.name.split("_")[0])

            if Vars.year_reference_start <= year <= Vars.year_reference_end:
                file_list.append(file)

        file_list = sorted(file_list)

        daily_temperatures = xr.open_mfdataset(
            file_list, combine="by_coords", chunks={"latitude": 100, "longitude": 100}
        )[t_var]

        daily_temperatures = daily_temperatures.chunk({"time": -1})

        climatology_quantiles = (
            Dirs.dir_era_quantiles
            / f"daily_{t_var}_quantiles_{'_'.join([str(int(100 * q)) for q in Vars.quantiles])}_{Vars.year_reference_start}-{Vars.year_reference_end}.nc"
        )

        daily_quantiles = daily_temperatures.quantile(Vars.quantiles, dim="time")

        with dask.config.set(scheduler="processes"), ProgressBar():
            daily_quantiles = daily_quantiles.compute()
            daily_quantiles.to_netcdf(climatology_quantiles)
