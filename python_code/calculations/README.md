# Heatwave Exposure (Population)

This folder contains scripts to compute heatwave exposure for vulnerable groups.

## a_heatwave_exposure_pop_abs.py

Calculates exposure as:

- `heatwave_days * population`
- `heatwave_counts * population`

Outputs a combined NetCDF file with dimensions:
`age_band`, `latitude`, `longitude`, `year`.

### Usage

```bash
python python_code/calculations/a_heatwave_exposure_pop_abs.py
```

Optional arguments:

```bash
python python_code/calculations/a_heatwave_exposure_pop_abs.py --trial
python python_code/calculations/a_heatwave_exposure_pop_abs.py --year 2020
```

### Output

The file is saved to:

- `FilesLocal.hw_combined_q`

## b_heatwave_exposure_pop_change.py

Calculates exposure to change in heatwave days and counts relative to the reference period.

### Output

- `FilesLocal.hw_change_combined`

### Plots

- Global weighted mean change (days and counts)
- Global total exposure change (days and counts)
- Mediterranean change maps and weighted mean
- Global histograms for a selected year

# Region Rasters (Admin0)

This script rasterizes World Bank Admin0 polygons onto the ERA5-Land grid. It creates
integer masks for country, WHO, HDI, and Lancet groupings and saves them in
`boundaries-2026/region-rasters`.

## Inputs

- Shapefile: `FilesLocal.world_bank_shapefile`
- Groupings table: `FilesLocal.country_names_groupings` (sheet: ISO3 - Name - LC - WHO - HDI)
- Grid template: `FilesLocal.pop_inf`

## Outputs

- `FilesLocal.raster_country` (variable: `country_id`)
- `FilesLocal.raster_who` (variable: `who_id`)
- `FilesLocal.raster_hdi` (variable: `hdi_id`)
- `FilesLocal.raster_lancet` (variable: `lc_id`)

## Run

```bash
python python_code/calculations/c_regions_rasters.py
```

## Notes

- Country IDs are created from sorted ISO3 values to ensure stable mapping.
- Region rasters are used by `python_code/calculations/c_aggregate_results.py`.

# Heatwave Exposure Aggregates

This script aggregates ERA5-Land heatwave exposure results by country and regional groupings
(WHO, HDI, Lancet, Country). It reads the combined absolute exposure file and writes one NetCDF per
region type into `results_<YEAR>/aggregates`.

## Inputs

- Combined exposure file: `FilesLocal.hw_combined_q`
- Population files: `FilesLocal.pop_inf`, `FilesLocal.pop_over_65`
- Boundary rasters and country groupings: see `DirsLocal.dir_admin_boundaries`

## Outputs

- `FilesAggregates.country`
- `FilesAggregates.who`
- `FilesAggregates.hdi`
- `FilesAggregates.lancet`
- `FilesAggregates.excel`
- Plots: `DirsLocal.aggregates_figures/*.png`

## Run

```bash
python python_code/calculations/d_aggregate_results.py
```

## Notes

- Outputs include totals and per-person metrics for both heatwave days and counts.
- The script validates grid alignment before aggregation.
- The Excel export includes one sheet per aggregate plus a global summary sheet.


