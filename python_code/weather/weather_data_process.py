import glob
import os

import xarray as xr
import shutil

from my_config import DATA_SRC, TEMPERATURE_SUMMARY_FOLDER

SUBDAILY_TEMPERATURES_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)

one_drive_folder = "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/Temporary/lancet"


def generate_daily_summary(source_file) -> None:

    summary_file = source_file.replace("_temperature.grib", "_temperature_summary.nc")
    summary_file = summary_file.replace(
        str(SUBDAILY_TEMPERATURES_FOLDER), str(TEMPERATURE_SUMMARY_FOLDER)
    )

    daily = xr.open_dataset(source_file, engine="cfgrib").load()
    daily = daily.resample(time="1D")
    tmin = daily.min()
    tmax = daily.max()
    tmean = daily.mean()

    tmin = tmin.rename({"t2m": "t_min"})
    tmax = tmax.rename({"t2m": "t_max"})
    tmean = tmean.rename({"t2m": "t_mean"})
    daily_summary = xr.merge([tmin, tmax, tmean])

    daily_summary.to_netcdf(
        summary_file,
        encoding={
            "t_min": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            "t_max": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            "t_mean": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
        },
    )

    shutil.move(
        source_file,
        source_file.replace(str(SUBDAILY_TEMPERATURES_FOLDER), one_drive_folder),
    )


if __name__ == "__main__":
    for file in glob.glob(str(SUBDAILY_TEMPERATURES_FOLDER) + "/*.grib"):
        # do something with the file
        size_gb = os.path.getsize(file) / 10**9  # Convert bytes to GB
        if size_gb > 18:
            print(f"File: {file}, Size: {size_gb:.2f} GB")
            generate_daily_summary(source_file=file)
