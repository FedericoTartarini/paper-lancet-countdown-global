# Lancet Countdown 2025 Heatwave Indicator

This repository contains the code needed to generate the 2025 heatwave indicator from the Lancet Countdown, as well as the figures in the report and appendix (1.1.2: Exposure of Vulnerable Populations to Heatwaves).

## How to run the code

1. Clone the repository
2. Install the required packages using conda
3. Get your Personal Access Token from your profile on the CDS portal at the address: https://cds.climate.copernicus.eu/profile
4. Update `my_config.py` with your personal paths

## Weather data

The weather data used in this analysis is the ERA5 reanalysis data from the Copernicus Climate Data Store (CDS). The data is available at https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

1. Register on their portal
2. Download the data using the code in `weather/weather_data_download.py`
3. Calculate the quantiles using the code in `weather/calculate_quantiles.py`
4. Calculate the heatwaves occurrences using the code in `weather/calculate_heatwaves.py`

## Population data

1. Download the WorldPop data using the code in `population/pop_data_download.py`
2. Regrid the data to the ERA5 grid using the code in `population/pop_data_process.py`
3. Combine the age groups using the code in `population/pop_data_combine.py`
3. Calculate the absolute exposure to heatwaves using the code in `population/heatwave_exposure_pop_abs.py`
4. Calculate the change in exposure to heatwaves using the code in `population/heatwave_exposure_pop_change.py`


# TODO

## Weather data
- [x] Download ERA5 data
- [x] Preprocess the data
- [x] Calculate the quantiles
- [x] Calculate the heatwaves occurrences

## Population data
- [x] Download WorldPop data
- [x] Process the data (regrid and combine ages)
- [x] Combine the population data
- [ ] Calculate the absolute exposure to heatwaves

## Other
- [x] Rasterize the data

## Issues