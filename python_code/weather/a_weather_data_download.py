import cdsapi
from icecream import ic

from my_config import (
    Vars,
    Dirs,
)
from python_code.secrets import copernicus_api_key


def download_year_era5(year: int = 2022):
    """
    Download the ERA5 data for a given year.

    Parameters
    ----------
    year : int
        The year to download.
    """

    c = cdsapi.Client(
        key=copernicus_api_key,
        url="https://cds.climate.copernicus.eu/api",
    )
    # todo we should be downloading the t-max and t-min ad the daily level not hourly
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


if __name__ == "__main__":
    for y in Vars.get_analysis_years():
        out_file = Dirs.dir_era_hourly / f"{y}_temperature.grib"
        summary_file = Dirs.dir_era_daily / f"{y}_temperature_summary.nc"

        if not out_file.exists() and not summary_file.exists():
            ic(f"Downloading ERA5 data for year: {y}")
            download_year_era5(y)
        else:
            ic(f"{out_file.stem} file already exists, skipping")
