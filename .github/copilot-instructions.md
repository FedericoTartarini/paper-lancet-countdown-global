# Project Context

This project generates the Heatwave Indicator for the Lancet Countdown Global Report.
It runs on the NCI Gadi supercomputer and processes large climate datasets (ERA5-Land) and population data.
Some of the calculations are computationally intensive and require careful memory management and optimization.
Some other can be run locally.
When I ask you to change the code read the document docstring, the docstring of the function and the comments in the
code to understand the context and the requirements of the code.
Also review the README.md and the README-GADI.md to understand the overall structure of the code and the workflow.
If there are any questions about the code, the workflow, the data or the requirements, please ask me before making any
changes to the code.
Never make changes to the code without understanding the context and the requirements, as it can break the code and
cause issues in the workflow.
Never assume that you understand the context and the requirements without asking me, as it can lead to misunderstandings
and mistakes.
If two documents or comments contradict each other, please ask me to clarify which one is correct before making any
changes to the code.

# Aim of the project

## Weather

- Process ERA5-Land hourly data into daily summaries (e.g., max temperature).
- Calculate 95th percentile thresholds for heatwave definitions.
- Identify heatwave days based on thresholds.

# Technology Stack

- **Language**: Python 3.12+
- **Key Libraries**:
    - `xarray` (Data handling)
    - `dask` (Parallel computing)
    - `pandas`
    - `numpy`
    - `scipy`

# Best Practices

## 1. Path Management

- ALWAYS use `pathlib.Path` instead of string paths or `os.path`.
- ALWAYS import and use `Dirs` from `my_config.py` for defined directory paths.
- **Do not hardcode paths** unless absolutely necessary for local debugging (and then, use a flag).
- Do not do this:
  ```python
  from my_config import DirsLocal
  output_file = DirsLocal.dir_results_heatwaves / "output.nc"
  ```
  but instead create a file path in my_config.py:
  ```python
  output_file = Dirs.dir_results_heatwaves / "output.nc"
  ```

## 2. Infrastructure (NCI Gadi)

- The code is deployed to Gadi.
- Large input data is read-only in `/g/data`.
- Outputs must go to `/scratch`.
- Compute-intensive scripts use `dask.distributed.Client()`.
- Job submission is handled via PBS scripts (`.pbs`).

## 3. Data Handling (NetCDF/Xarray)

- **Lazy Loading**: Use `open_dataset(..., chunks={...})` or `open_mfdataset`.
- **Chunking**: Use appropriate chunk sizes (e.g., `{"time": -1, "latitude": 500, "longitude": 500}`) to avoid memory
  overflows.
- **Output Optimization**:
    - Convert `float64` to `float32` for outputs to save space.
    - Enable `zlib` compression (`complevel=5`).
    - Use `least_significant_digit` encoding for variable precision (e.g., 1 decimal for temperature).
- **Warnings**: Handle "All-NaN slice" warnings gracefully when processing land-masked data.

## 4. Coding Standards

- **Logging**: Use `logging` module. Do not use `print()` for status updates.
- **Docstrings**: Every function and module must have a descriptive docstring.
- **Type Hinting**: Use Python type hints for function arguments and returns.
- **Modularization**: Keep reusable logic (like checking file existence, verifying outputs) in helper functions.

## 5. Development Workflow

- **Local Testing**: Scripts should support a `--local_file` or `--trial` argument to allow testing on a small
  subset/local file before full HPC deployment.
- **Verification**: Always verify output file structure (vars, dims, logic checks) after generation.
- you can add script to verify outputs in `python_code/verify_output/`.
- **File Transfer**: Use `python_code/copy_files_hpc.py` to transfer files between local machine and Gadi:
    - `python python_code/copy_files_hpc.py --copy-daily 2024` - Copy specific year's daily summary
    - `python python_code/copy_files_hpc.py --copy-all-daily` - Copy all daily summaries

## 6. Git/Deployment

- Use `./deploy_to_gadi.sh` to sync code.
- Do not sync `__pycache__` or large data files.

## 7. Python

- Follow PEP 8 standards.
- Use docstrings for all functions and modules.
- Use type hints for function arguments and return types.
- Keep functions small and focused; avoid large monolithic functions.
- If possible, make sure that they can work both locally and on Gadi with minimal changes.
- Add command-line arguments for local testing (e.g., `--local_file`, `--trial`).

## 8. HPC Optimization Patterns

### Memory Management

- **Explicit cleanup**: Use `del` to delete large datasets immediately after saving to free memory sooner
- **Monthly processing**: Process data month-by-month instead of loading entire years to reduce peak memory usage
- **Interim files**: Save monthly results to disk and combine later to avoid holding all data in memory simultaneously

### Chunking Strategy

- **Dynamic chunksizes**: Use actual dimension sizes for chunking (e.g., actual days in month) instead of fixed values
- **Output chunking**: Use larger chunks for output files (e.g., 600x1200 for lat/lon) optimized for storage and access
  patterns

### Error Handling & Cleanup

- **Automatic cleanup**: Remove interim files automatically when processing fails to prevent disk space waste
- **Checkpointing**: Save progress incrementally so failed jobs can resume from last successful month

### Dask Configuration

- **Fixed workers**: Hardcode worker count to match PBS job allocation instead of making it configurable
- **Memory management**: Use spill-to-disk thresholds (target=0.6, spill=0.7, pause=0.8) to prevent memory exhaustion
- **Local directory**: Specify scratch space for Dask worker temporary files

### Data Processing Workflow

- **Trial mode**: Process only January for testing pipeline before full deployment
- **Validation**: Check input/output file existence and structure before/after processing
- **Logging**: Use structured logging with emojis for easy monitoring in HPC job logs

## 9. Paths

- Use `pathlib.Path` for all path manipulations.
- Define all directory paths in `my_config.py`. There are three classes for paths:
    - `Dirs`: For general paths used across the codebase and folders names which are used across the codebase
    - `DirsGadi`: For paths on Gadi
    - `DirsLocal`: For paths on local machine.
- I should use the same directory structure for the local machine and Gadi to avoid confusion and make it easier to
  switch between them. The only difference should be the root directory, which is defined in `my_config.py`
- All the plots and results generated by the files in `python_code/verify_output/` should be saved in the
  `results/verify_output/` folder, which is defined in `my_config.py` as `Dirs.dir_results_verify_output`. This way I
  can easily find the results of the verification and keep them organised.

## Additional Notes

### Merging Climate and Population Data

When combining climate data (ERA5-Land) with population data (WorldPop), the key challenge is aligning the grids
correctly. ERA5-Land has a coarser resolution (~9 km) compared to WorldPop's finer resolution (~1 km).

Here is the structural comparison and the recommended strategy for Gadi.

| Feature                  | **ERA5-Land (`zz93`)**              | **WorldPop**                          |
|--------------------------|-------------------------------------|---------------------------------------|
| **Source Type**          | Climate Reanalysis (NetCDF)         | Census/Satellite (GeoTIFF)            |
| **Grid Type**            | Regular Latitude/Longitude          | Regular Latitude/Longitude (WGS84)    |
| **Native Resolution**    | ~9 km ()                            | ~1 km ()                              |
| **Pixel Count (approx)** | ~6.4 Million pixels                 | ~500+ Million pixels                  |
| **Longitude Range**      | Often **0 to 360** (Greenwich is 0) | **-180 to 180** (Greenwich is 0)      |
| **Data Nature**          | Continuous (Temp varies smoothly)   | Discrete Count (People are clustered) |

**You must regrid WorldPop (1km) to match ERA5-Land (9km).**

1. **Conservation of People:** Since WorldPop is *finer* (smaller pixels), you can fit approximately **144 WorldPop
   pixels** inside a single ERA5-Land pixel. The correct mathematical operation is to **SUM** all the people in those
   144 small pixels into the one large climate pixel.
2. **Avoid False Precision:** If you went the other way (interpolating Climate to 1km), you would just be slicing one
   temperature value into 144 identical pieces. This increases your data size by 100x without adding any new climate
   information, effectively crashing your memory for no reason.

ERA5 data on Gadi usually runs `0 to 360`, where the Americas are >180. WorldPop runs `-180 to 180`, where the Americas
are negative. We need to check that before combining them, otherwise the grids won't align and the population counts
will be assigned to the wrong locations.

* **Action:** You must shift the ERA5-Land longitude coordinates to match WorldPop (`-180 to 180`) before combining
  them. Check this in the code and make sure to sort the longitude after shifting, otherwise the regridding will fail.

Since you are summing counts, you cannot use "Linear Interpolation" or "Nearest Neighbor" (which would pick 1 random
pixel and ignore the other 143). You need **Conservative Remapping** or **Raster Zonal Statistics**.

The easiest way to do this in your Python environment is using `rioxarray` (which wraps `rasterio` for xarray).

Since we didn't add this earlier, log in to Gadi and update your env:

```bash
source /g/data/PROJECT_CODE/lancet_env/bin/activate
pip install rioxarray

```

Add this function to `shared_functions.py` or your main script. It handles the longitude wrap and the aggregation sum.

```python
import xarray as xr
import numpy as np
import rioxarray


def align_and_aggregate_population(pop_path, era_template):
    """
    1. Loads WorldPop (High Res).
    2. Shifts ERA5 Longitude to match WorldPop (-180 to 180).
    3. Aggregates (Sums) WorldPop to match ERA5 grid.
    """
    # 1. Load ERA5 Template (Just one timestep to get the grid)
    # era_template should be an xarray DataArray from your heatwave data

    # FIX LONGITUDE: Convert ERA5 from 0..360 to -180..180
    # This aligns it with WorldPop
    era_shifted = era_template.assign_coords(longitude=(((era_template.longitude + 180) % 360) - 180))
    era_shifted = era_shifted.sortby('longitude')

    # 2. Load WorldPop
    # chunks={'x': 2000, 'y': 2000} allows dask to process the huge file
    pop = rioxarray.open_rasterio(pop_path, chunks={'x': 2000, 'y': 2000})

    # 3. Reproject and Aggregate (Sum)
    # "reproject_match" will automatically find the transform to align them.
    # resampling=5 is Resampling.sum in rasterio
    from rasterio.enums import Resampling

    pop_regridded = pop.rio.reproject_match(
        era_shifted,
        resampling=Resampling.sum
    )

    # Clean up dimensions (rioxarray adds 'band')
    pop_regridded = pop_regridded.squeeze('band', drop=True)

    # Rename coords to match ERA5 standard (lat/lon)
    pop_regridded = pop_regridded.rename({'x': 'longitude', 'y': 'latitude'})

    return pop_regridded

```

Create validation plots to ensure the regridding is correct. I want to make sure that:

- the total population count is preserved (sum of WorldPop should equal sum of regridded population)
- the spatial distribution looks correct (population clusters should be in the same locations, just aggregated)
- the longitude coordinates are correctly aligned (check that the Americas are in the correct hemisphere)
- the output grid matches the ERA5 grid (check dimensions and coordinate values)
- the data types are correct (population counts should be integers, not floats)
- the function can handle large files without crashing (test on a small subset first, then scale up)
- I want to see the before and after maps of population density to visually confirm the aggregation is working as
  expected. Plot only a small region (e.g., Sydney area, North Italy, and a place with islands) to check the aggregation
  visually.

### Population data

I do not need to calculate the risk for over 75.
I only need to calculate the risk for over 65, under 1 and pregenant women.
Data about pregnant women is not available in the WorldPop dataset, hence, I will possibly need to calculate them as a
function of the under 1 population and then apply a correction factor to account for stillbirths, miscarriages and
abortions.
I can remove the code related to the over 75 population and focus on the other three groups.
This will simplify the code and reduce the computational load, as I will be working with fewer population groups.
I should also update the documentation and comments in the code to reflect this change, so that it is clear which
population groups are being analyzed.

I want to create a script that first merges the different age groups and sex groups into a single population count for
each of the groups of interest (over 65, under 1).
Then, I will apply the function to regrid the data to the ERA5 grid and save the output as a NetCDF file that can be
used in the heatwave exposure calculations.
Between 2000 and 2014 the WorldPop data is split by sex (global_f_0_2000_1km.tif, global_m_0_2000_1km.tif) and age
group (0, [65, 70, 75, 80]) while after 2015 the data is only split by age group (0, [65, 70, 75, 80, 85, 90]) without
the sex (global_t_0_2015_1km.tif)