# Heatwave Calculation on Gadi

This directory contains scripts to calculate heatwave indicators from ERA5-Land daily temperature data on NCI Gadi HPC.

## Overview

The heatwave calculation processes daily temperature summaries to identify heatwave events based on 95th percentile
thresholds from a reference period (1986-2005).
A heatwave is defined as 3+ consecutive days where both daily maximum and minimum temperatures exceed their respective
thresholds.

**Key Features:**

- Processes data in **latitude chunks** to reduce memory usage
- **Resumable**: Saves interim chunk files, so failed jobs can resume
- **Progress tracking**: Logs elapsed time and ETA for each chunk
- Uses **numpy** for fast vectorized operations (no Dask overhead)

## Prerequisites

1. **Data Sync**: Ensure quantiles and daily summary files are synced to Gadi:
   ```bash
   # From local machine
   python python_code/copy_files_hpc.py  # Sync quantiles and daily data
   ```

2. **Code Deployment**: Deploy code to Gadi by running this script from the local repository root:
   ```bash
   ./gadi/deploy_to_gadi.sh
   ```

3. **Environment**: The PBS script uses the conda environment from `/g/data/xp65/public/modules`.

## Files

- `d_calculate_heatwaves_gadi.py`: Python script to process one year (in latitude chunks)
- `job_heatwaves_single_year.pbs`: PBS job script for single year processing
- `submit_heatwaves_gadi.sh`: Shell script to submit multiple jobs

## Usage

### Submit Single Year

```bash
# From Gadi
qsub -v YEAR=2020 gadi/job_heatwaves_single_year.pbs
```

### Submit Multiple Years

```bash
cd paper-lancet-countdown-global/

# All years (1980-2024)
./gadi/submit_heatwaves_gadi.sh

# Single year
./gadi/submit_heatwaves_gadi.sh 2025

# Year range
./gadi/submit_heatwaves_gadi.sh 2020 2024
```

### Monitor Jobs

```bash
qstat -u $USER
```

## Output

Results are saved to `/scratch/mn51/ft8695/heatwaves/` as `heatwave_indicators_{year}.nc` files containing:

- `heatwave_count`: Number of heatwave events per year
- `heatwave_days`: Total heatwave days per year

## Progress Tracking

The script logs progress for each latitude chunk:

```
cat logs/calculate_heatwaves.log
```

will show output like:

```
📊 Chunk 5/10: lat[800:1000] (200 rows)
   ✅ Chunk 5/10 done in 45.2s | Elapsed: 3.8min | ETA: 3.8min
```

## Resume Capability

If a job fails or times out:

- Interim chunk files are preserved in `heatwaves/interim/{year}/`
- Resubmitting the job will skip completed chunks and continue from where it left off
- After successful completion, interim files are automatically cleaned up

## Troubleshooting

- **Job fails**: Check logs in the job output files. Interim files are preserved for resume.
- **Memory issues**: The script uses ~32GB memory and 1 CPU per job
- **Data not found**: Ensure data sync completed successfully
- **Invalid year**: Years must be between 1980-2024

## Contact

For issues, check the Lancet Countdown project documentation or contact the development team.
