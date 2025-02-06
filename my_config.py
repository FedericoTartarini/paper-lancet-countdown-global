from pathlib import Path

DATA_SRC = Path.home() / "Documents" / "lancet_countdown"
WEATHER_SRC = DATA_SRC / "weather"
POP_DATA_SRC = DATA_SRC / "population"

SUBDAILY_TEMPERATURES_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "hourly_temperature_2m"
)
SUBDAILY_TEMPERATURES_FOLDER.mkdir(parents=True, exist_ok=True)

TEMPERATURE_SUMMARY_FOLDER = (
    DATA_SRC / "era5" / "era5_0.25deg" / "daily_temperature_summary"
)
TEMPERATURE_SUMMARY_FOLDER.mkdir(parents=True, exist_ok=True)
