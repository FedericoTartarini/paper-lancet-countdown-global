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

