import cdsapi
from icecream import ic

from my_config import DATA_SRC
from python_code.secrets import copernicus_api_key

SUBDAILY_TEMPERATURES_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)
SUBDAILY_TEMPERATURES_FOLDER.mkdir(parents=True, exist_ok=True)

TEMPERATURE_SUMMARY_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "daily_temperature_summary"
)
TEMPERATURE_SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)


def download_year_era5(year: int = 2022):
    """
    Download the ERA5 data for a given year.

    Parameters
    ----------
    year : int
        The year to download.
    """

    out_file = SUBDAILY_TEMPERATURES_FOLDER / f"{year}_temperature.grib"

    if not out_file.exists():
        c = cdsapi.Client(
            key=copernicus_api_key,
            url="https://cds.climate.copernicus.eu/api",
        )
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": str(year),
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "day": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                    "20",
                    "21",
                    "22",
                    "23",
                    "24",
                    "25",
                    "26",
                    "27",
                    "28",
                    "29",
                    "30",
                    "31",
                ],
                "time": [
                    "00:00",
                    "01:00",
                    "02:00",
                    "03:00",
                    "04:00",
                    "05:00",
                    "06:00",
                    "07:00",
                    "08:00",
                    "09:00",
                    "10:00",
                    "11:00",
                    "12:00",
                    "13:00",
                    "14:00",
                    "15:00",
                    "16:00",
                    "17:00",
                    "18:00",
                    "19:00",
                    "20:00",
                    "21:00",
                    "22:00",
                    "23:00",
                ],
                "format": "grib",
            },
            str(out_file),
        )
    else:
        ic(f"Era5 file for {year} already exists.")


# def download_summarise_year_data(year: int, overwrite: bool = False):
#     out_file = SUBDAILY_TEMPERATURES_FOLDER / f"{year}_temperature.grib"
#     summary_file = TEMPERATURE_SUMMARY_FOLDER / f"{year}_temperature_summary.nc"
#
#     if overwrite is True or not summary_file.exists():
#         daily_summary = generate_daily_summary(out_file)
#
#         # Encode using rounding to nearest 0.01K to save space, looses a small amount of precision
#         # but should be ok since it not used for detailed simulations...
#         daily_summary.to_netcdf(
#             summary_file,
#             encoding={
#                 "t_min": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
#                 "t_max": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
#                 "t_mean": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
#             },
#         )
#
#     else:
#         print(f"Skip {summary_file}, already exists.")  # # Run each year one at a time


if __name__ == "__main__":
    year_max = 2024
    year_min = 2022
    for y in range(year_min, year_max + 1):
        download_year_era5(y)
