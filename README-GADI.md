# Gadi Instructions for Lancet Countdown Heatwave Indicator

This document contains the instructions to run the code needed to generate the heatwave indicator for the Lancet
Countdown Global report on the Gadi supercomputer.

## Setup Environment

```
# 1. Go to your project folder (Replace PROJECT_CODE with your actual code, e.g., fp0)
cd /g/data/ua88

# 2. Load the base python module
module load python3/3.12.1

# 3. Create the virtual environment named 'lancet_global_env'
python3 -m venv lancet_global_env

# 4. Activate it
source /g/data/ua88/lancet_global_env/bin/activate

# 5. Install libraries (This takes a few minutes)
pip install --upgrade pip
pip install xarray dask netCDF4 bottleneck geopandas shapely seaborn matplotlib joblib tqdm distributed pandas numpy scipy rasterio requests fiona flox numpy_groupies h5netcdf
```

## Running the Code

```bash
./gadi/submit_on_gadi.sh python_code/weather/b_daily_summaries.py --trial
```

## Monitoring and Troubleshooting

### 1. Check Job Status

Use `qstat` to see if your job is Running (R), Queued (Q), or Held (H).

```bash
qstat -u ft8695
```

If a job is **Held (H)**, use the `-f` flag to see why (look for the `comment` field):

```bash
qstat -f <JOB_ID>
```

To quit a job, use:

```bash
qdel <JOB_ID>
```

### 2. Check Output Logs

When a job runs, it creates two files in the directory where you ran `qsub`:

* **Standard Output**: `<Job_Name>.o<Job_ID>` (e.g., `daily_sum.o12345678`)
* **Standard Error**: `<Job_Name>.e<Job_ID>` (e.g., `daily_sum.e12345678`)

**To see progress in real-time:**

```bash
tail -f daily_sum.o<JOB_ID>
```

**To check for errors:**

```bash
cat daily_sum.e<JOB_ID>
```

### 3. Verify Resource Usage

After a job finishes, checking the resource usage helps optimize future submissions (saving SUs). Open the `.o` file and
scroll to the bottom to find:

* `resources_used.mem`: Actual memory used.
* `resources_used.walltime`: Actual time taken.
* `resources_used.ncpus`: CPUs utilized.

Use this info to adjust `#PBS -l mem=...` and `#PBS -l walltime=...` in your script.
