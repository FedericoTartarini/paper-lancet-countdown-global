# Project Context

This project generates the Heatwave Indicator for the Lancet Countdown Global Report.
It runs on the NCI Gadi supercomputer and processes large climate datasets (ERA5-Land) and population data.

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
  from my_config import Dirs
  output_file = Dirs.dir_results_heatwaves / "output.nc"
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
