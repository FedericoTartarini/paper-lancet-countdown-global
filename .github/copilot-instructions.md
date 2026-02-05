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
