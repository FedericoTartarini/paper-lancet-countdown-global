#!/bin/bash
# Submit heatwave calculation jobs to Gadi - one job per year
#
# Usage:
#   ./submit_heatwaves_gadi.sh              # Submit all years (1980-2024)
#   ./submit_heatwaves_gadi.sh 2020         # Submit single year
#   ./submit_heatwaves_gadi.sh 2020 2024    # Submit range of years

set -e

# Parse arguments
START_YEAR=""
END_YEAR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        *)
            if [ -z "$START_YEAR" ]; then
                START_YEAR=$1
            else
                END_YEAR=$1
            fi
            shift
            ;;
    esac
done

# Default year range if not specified
if [ -z "$START_YEAR" ]; then
    START_YEAR=1980
    END_YEAR=2024
elif [ -z "$END_YEAR" ]; then
    END_YEAR=$START_YEAR
fi

echo "======================================================================"
echo "Submitting heatwave calculation jobs to Gadi"
echo "Year range: $START_YEAR to $END_YEAR"
echo "======================================================================"

# Count jobs to submit
TOTAL_JOBS=$((END_YEAR - START_YEAR + 1))
echo "Total jobs to submit: $TOTAL_JOBS"
echo ""

# Submit jobs
JOB_COUNT=0
for YEAR in $(seq $START_YEAR $END_YEAR); do
    JOB_ID=$(qsub -v YEAR=$YEAR gadi/job_heatwaves_single_year.pbs)
    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[$JOB_COUNT/$TOTAL_JOBS] Submitted year $YEAR: $JOB_ID"
done

echo ""
echo "======================================================================"
echo "All $JOB_COUNT jobs submitted successfully!"
echo "Monitor with: qstat -u \$USER"
echo "======================================================================"
