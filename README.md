# Lancet Countdown - 1.1.1 Indicator - Global

This repository contains the code needed to generate the heatwave indicator for the Lancet Countdown Global report.
This code allows you to:

1. download the raw weather and population data
2. process, clean, and combine the data
3. generate the figures for the report and appendix.

## How to run the code

1. Clone the repository
2. Install the required packages using conda, the dependencies are listed
   in [requirements_conda.txt](requirements_conda.txt). To create a conda environment use
   `conda create --name <env> --file requirements.txt`
3. Get your Personal Access Token from your profile on the CDS portal at the
   address: https://cds.climate.copernicus.eu/profile
4. Update [my_config.py](my_config.py), change the paths and the dates if needed.

## Weather data

The weather data used in this analysis is the ERA5-Land reanalysis data from the Copernicus Climate Data Store (CDS).
The data is available at https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
To download the data you need to:

1. Register on their portal and save the Personal Access Token in [secrets.py](python_code/secrets.py). Create the file
   if the file does not exist and save the token as `copernicus_api_key = "XX"`.
2. Preprocess the data into daily summaries
   using [a_daily_summaries_gadi.py](python_code/weather/a_daily_summaries_gadi.py)
3. Calculate the quantiles if the reference period has changed
   using [b_calculate_quantiles.py](python_code/weather/b_calculate_quantiles.py). Otherwise, you can skip it.
4. Calculate the heatwave occurrences
   using [c_calculate_heatwaves_gadi.py](python_code/weather/c_calculate_heatwaves_gadi.py)

For the moment I am keeping the old heatwave data used in the previous report, but I should remove the
`results/heatwave/results_2025` folder
once the new report is finalised.

## Population data

1. Download the WorldPop using [a_pop_data_download.py](python_code/population/a_pop_data_download.py)
2. Re-grid the data to the ERA5-Land grid
   using [b_pop_merge_and_coarsen.py](python_code/population/b_pop_merge_and_coarsen.py)
3. Combine the age groups using [c_pop_data_combine.py](python_code/population/c_pop_data_combine.py)

## Heatwaves exposure

1. Calculate the absolute exposure to heatwaves
   using [a_heatwave_exposure_pop_abs.py](python_code/calculations/a_heatwave_exposure_pop_abs.py)
2. Calculate the change in exposure to heatwaves relative to the reference period
   using [b_heatwave_exposure_pop_change.py](python_code/calculations/b_heatwave_exposure_pop_change.py)
3. Rasterize the World Bank Admin0 polygons onto the ERA5-Land grid
   using [c_regions_rasters.py](python_code/calculations/c_regions_rasters.py)
4. Aggregate the exposure data by country, WHO, HDI, and Lancet groupings
   using [d_aggregate_results.py](python_code/calculations/d_aggregate_results.py)

## Planned updates

- Heatwave severity (EHF) calculation.
- Pregnant women exposure estimation (pending a robust demographic proxy).

# Other info

To update the list of dependencies use:

```bash
conda list -e > requirements_conda.txt
pip list --format=freeze > requirements.txt
```