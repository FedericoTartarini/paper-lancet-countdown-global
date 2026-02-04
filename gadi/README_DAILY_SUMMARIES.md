# ERA5-Land Daily Summaries Processing on Gadi

This guide explains how to process ERA5-Land hourly data into daily summaries (min, mean, max) on NCI Gadi.

## Overview

The processing pipeline:

1. **Input**: Hourly ERA5-Land data from `/g/data/zz93/era5-land/reanalysis/2t/`
2. **Processing**: Each year is processed as a separate PBS job
3. **Output**: Daily summaries saved to `/scratch/mn51/ft8695/era5-land/daily/2t/`

## Files

- `python_code/weather/b_daily_summaries_gadi.py` - Gadi-optimized Python script (processes one year)
- `gadi/job_daily_summaries_single_year.pbs` - PBS job script for a single year
- `gadi/submit_daily_summaries_gadi.sh` - Submit jobs for specified years
- `gadi/submit_missing_years_gadi.sh` - Automatically detect and submit missing years
- `my_config.py` - Contains `DirsGadi` class with all Gadi paths

## Quick Start

### Option 1: Process All Missing Years (Recommended)

```bash
./gadi/submit_missing_years_gadi.sh
```

This script will:

1. Check which years (1979-2025) are already processed
2. Show you which years are missing
3. Ask for confirmation before submitting jobs

### Option 2: Process Specific Years

```bash
# Process specific years (full year - all 12 months)
./gadi/submit_daily_summaries_gadi.sh --years 1980 1981 1982

# Process all years (1979-2025)
./gadi/submit_daily_summaries_gadi.sh
```

### Option 3: Trial Mode (Testing)

Trial mode processes **only January** for quick testing:

```bash
# Test specific year (only processes January 1980)
./gadi/submit_daily_summaries_gadi.sh --years 1980 --trial

# Test all missing years (only January for each year)
./gadi/submit_missing_years_gadi.sh --trial
```

### 🔑 Understanding the Flags

- **`--years 1980`**: Processes the **entire year 1980** (all 12 months) - takes ~1 hour
- **`--trial`**: Processes **only January** (for quick pipeline testing) - takes ~5 minutes
- **`--years 1980 --trial`**: Processes **only January 1980** (not the full year)

**Example workflows:**

```bash
# Full processing: All 12 months of 1980 and 1981
./gadi/submit_daily_summaries_gadi.sh --years 1980 1981

# Quick test: Only January 1980 (to verify the pipeline works)
./gadi/submit_daily_summaries_gadi.sh --years 1980 --trial
```

## Manual Workflow

If you prefer to sync and submit manually:

```bash
# 1. Sync code to Gadi
./gadi/deploy_to_gadi.sh

# 2. SSH to Gadi
ssh ft8695@gadi.nci.org.au

# 3. Navigate to project directory
cd ~/paper-lancet-countdown-global

# 4. Submit job for a specific year
qsub -N daily_1980 -v YEAR=1980 gadi/job_daily_summaries_single_year.pbs

# 5. Submit with trial flag (only January)
qsub -N daily_1980 -v YEAR=1980,TRIAL=1 gadi/job_daily_summaries_single_year.pbs
```

## Monitoring Jobs

### Check job status

```bash
ssh ft8695@gadi.nci.org.au 'qstat -u ft8695'
```

### View job logs

```bash
ssh ft8695@gadi.nci.org.au
cd ~/paper-lancet-countdown-global
ls -lt daily_*.o* daily_*.e*  # Output and error logs
tail -f daily_1980.o123456    # Follow specific job log
```

### Check output files

```bash
ssh ft8695@gadi.nci.org.au
ls -lh /scratch/mn51/ft8695/era5-land/daily/2t/
```

## Resource Requirements

Each job requests:

- **CPUs**: 16
- **Memory**: 64 GB
- **Walltime**: 4 hours
- **Project**: mn51
- **Storage**: `gdata/zz93+scratch/mn51`

These settings are defined in `gadi/job_daily_summaries_single_year.pbs`.

## Output Files

Each processed year creates one file:

```
/scratch/mn51/ft8695/era5-land/daily/2t/YYYY_daily_summaries.nc
```

Example: `1980_daily_summaries.nc`

### File Structure

Each output file contains:

- **Variables**: `t_min`, `t_mean`, `t_max` (daily temperature)
- **Dimensions**: `time`, `latitude`, `longitude`
- **Compression**: zlib level 1, float32, 1 decimal precision
- **Typical size**: ~700 MB per year

## Troubleshooting

### Job fails with "walltime exceeded"

Increase walltime in `gadi/job_daily_summaries_single_year.pbs`:

```bash
#PBS -l walltime=02:00:00
```

### Job fails with "memory exceeded"

Increase memory in `gadi/job_daily_summaries_single_year.pbs`:

```bash
#PBS -l mem=32GB
```

### Check detailed error messages

```bash
ssh ft8695@gadi.nci.org.au
cd ~/paper-lancet-countdown-global
cat daily_YYYY.o*  # Replace YYYY with year
```

### Re-run failed years

Simply delete the output file and resubmit:

```bash
ssh ft8695@gadi.nci.org.au
rm /scratch/mn51/ft8695/era5-land/daily/2t/1980_daily_summaries.nc
cd ~/paper-lancet-countdown-global
qsub -N daily_1980 -v YEAR=1980 gadi/job_daily_summaries_single_year.pbs
```

Or use the helper script to automatically resubmit missing years:

```bash
./gadi/submit_missing_years_gadi.sh
```

## Advanced Usage

### Process a range of years in a loop

```bash
ssh ft8695@gadi.nci.org.au
cd ~/paper-lancet-countdown-global

for YEAR in {2020..2025}; do
    qsub -N daily_$YEAR -v YEAR=$YEAR gadi/job_daily_summaries_single_year.pbs
done
```

### Check progress of all jobs

```bash
ssh ft8695@gadi.nci.org.au '
    cd ~/paper-lancet-countdown-global && 
    echo "Running jobs:" && 
    qstat -u ft8695 && 
    echo -e "\nProcessed years:" && 
    ls -1 /scratch/mn51/ft8695/era5-land/daily/2t/*.nc | wc -l
'
```

## Paths Configuration

All paths are centralized in `my_config.py` in the `DirsGadi` class:

```python
class DirsGadi:
    # Input: ERA5-Land hourly data
    dir_era_land_hourly = Path("/g/data/zz93/era5-land/reanalysis/2t")

    # Output: Daily summaries
    dir_era_daily = Path("/scratch/mn51/ft8695/era5-land/daily/2t")

    # Results: Heatwave calculations
    dir_results_heatwaves = Path("/scratch/mn51/ft8695/heatwaves")
```

## Benefits of This Approach

1. **Resilience**: If one year fails, others continue processing
2. **Parallelism**: Multiple years can run simultaneously (up to available resources)
3. **Resume capability**: Only missing years are processed
4. **No interim files**: Direct hourly → daily conversion saves disk space
5. **Clear logging**: Each year has its own log file
