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

The weather data used in this analysis is the ERA5 reanalysis data from the Copernicus Climate Data Store (CDS).
The data is available at https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
To download the data you need to:

1. Register on their portal and save the Personal Access Token in [secrets.py](python_code/secrets.py). Create the file
   if the file does not exist and save the token as `copernicus_api_key = "XX".
2. Download the data using [1_weather_data_download.py](python_code/weather/a_weather_data_download.py)
3. Preprocess the data using [2_daily_summaries.py](python_code/weather/b_daily_summaries.py)
4. Calculate the quantiles if the reference period has changed
   using [calculate_quantiles.py](python_code/weather/c_calculate_quantiles.py). Otherwise, you can skip it.
5. Calculate the heatwaves occurrences using [calculate_heatwaves.py](python_code/weather/d_calculate_heatwaves_gadi.py)

For the moment I am keeping the old heatwave data used in the previous report, but I should remove the
`results/heatwave/results_2025` folder
once the new report is finalised.

## Population data

1. Download the WorldPop using [1_pop_data_download.py](python_code/population/a_pop_data_download.py)
    1. [2.1_pop_create_t_file.py](python_code/population/ba_pop_create_t_file.py) creates total files for each age
       group and year for the old WorldPop data.
    2. I am using the new WorldPop data from 2015 since they published a new
       dataset in 2020 with data from 2015 to 2030.
    3. [b_compare_worldpop_gpw.py](python_code/population/cb_compare_worldpop_gpw.py) compares the WorldPop and GPW data
       for the infant
       and elderly population. Not essential for the new report.
2. Re-grid the data to the ERA5 grid using [2_pop_data_process.py](python_code/population/b_pop_data_process.py)
3. Combine the age groups using [pop_data_combine.py](python_code/population/c_pop_data_combine.py)

## Other files to analyse

1. Generate the rasterized data using [region_raster.py](python_code/calculations/region_raster.py)

## Heatwaves exposure

1. Calculate the absolute exposure to heatwaves
   using [heatwave_exposure_pop_abs.py](python_code/calculations/1_heatwave_exposure_pop_abs.py)
2. Run the file [heatwaves_aggregates_worldpop.py](python_code/calculations/3_heatwaves_aggregates_worldpop.py)
3. Generate most of the results
   using [results_heatwaves_worldpop.py](python_code/calculations/results_heatwaves_worldpop.py)
4. Calculate the exposure due to climate change and pop growth
   using [results_climate_change_vs_pop_growth.py](python_code/calculations/5_results_climate_change_vs_pop_growth.py)

## Other calculations

1. Calculate the change in exposure to heatwaves
   using [heatwave_exposure_pop_change.py](python_code/calculations/2_heatwave_exposure_pop_change.py). These data are
   not used in the report.
2. Calculate the worldpop exposure to heatwaves
   using [heatwave_exposure_worldpop_change.py](python_code/calculations/6_heatwave_exposure_worldpop_change.py). These
   data are not used in the report.

# Other info

To update the list of dependencies use:

```bash
conda list -e > requirements_conda.txt
pip list --format=freeze > requirements.txt
```