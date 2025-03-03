import glob
import os
import shutil

import xarray as xr
from icecream import ic

from my_config import dir_era_daily, dir_era_hourly, dir_one_drive_era_hourly


def generate_daily_summary(source_file, move_source: bool = True) -> None:

    summary_file = source_file.replace("_temperature.grib", "_temperature_summary.nc")
    summary_file = summary_file.replace(str(dir_era_hourly), str(dir_era_daily))

    daily = xr.open_dataset(source_file, engine="cfgrib").load()
    daily = daily.resample(time="1D")
    t_min = daily.min()
    t_max = daily.max()
    t_mean = daily.mean()

    t_min = t_min.rename({"t2m": "t_min"})
    t_max = t_max.rename({"t2m": "t_max"})
    t_mean = t_mean.rename({"t2m": "t_mean"})
    daily_summary = xr.merge([t_min, t_max, t_mean])

    daily_summary.to_netcdf(
        summary_file,
        encoding={
            "t_min": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            "t_max": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            "t_mean": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
        },
    )

    # I am moving the file to the OneDrive folder to have a backup and save space on the computer
    if move_source:
        shutil.move(
            source_file,
            source_file.replace(str(dir_era_hourly), dir_one_drive_era_hourly),
        )


if __name__ == "__main__":
    for file in glob.glob(str(dir_era_hourly) + "/*.grib"):
        # checking that the file is fully downloaded before processing it
        size_gb = os.path.getsize(file) / 10**9  # Convert bytes to GB
        if size_gb > 18:
            ic(f"File: {file}, Size: {size_gb:.2f} GB")
            generate_daily_summary(source_file=file)
