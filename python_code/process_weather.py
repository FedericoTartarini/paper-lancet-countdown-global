import glob
import xarray as xr

from my_config import DATA_SRC

SUBDAILY_TEMPERATURES_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)
TEMPERATURE_SUMMARY_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "daily_temperature_summary"
)
TEMPERATURE_SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)


def generate_daily_summary(source_file) -> None:

    summary_file = source_file.replace("_temperature.grib", "_temperature_summary.nc")
    summary_file = summary_file.replace("hourly_temperature_2m", "daily_temperature_summary")

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


def process_weather():
    # loop through all the files in the folder
    for file in glob.glob(str(SUBDAILY_TEMPERATURES_FOLDER) + "/*.grib"):
        # do something with the file
        print(file)
        generate_daily_summary(source_file=file)


if __name__ == "__main__":
    process_weather()
