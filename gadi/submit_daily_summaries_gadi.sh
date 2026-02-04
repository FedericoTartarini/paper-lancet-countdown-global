#!/bin/bash

# Submit daily summary jobs for multiple years to Gadi
# This script syncs the code, then submits individual PBS jobs for each year
#
# Usage:
#   ./submit_daily_summaries_gadi.sh                    # Process all years (1979-2026)
#   ./submit_daily_summaries_gadi.sh --years 1980 1981  # Process specific years
#   ./submit_daily_summaries_gadi.sh --trial            # Trial mode (only January) for all years
#   ./submit_daily_summaries_gadi.sh --years 1980 --trial  # Trial for specific year

# Configuration
GADI_USER="ft8695"
GADI_HOST="gadi.nci.org.au"
REMOTE_DIR="~/paper-lancet-countdown-global"
PBS_SCRIPT="gadi/job_daily_summaries_single_year.pbs"

# Default: process all years from 1979 to 2025
CURRENT_YEAR=2025
DEFAULT_YEARS=$(seq 1979 $CURRENT_YEAR)

# Parse arguments
TRIAL_FLAG=""
YEARS_TO_PROCESS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --trial)
            TRIAL_FLAG="TRIAL=1"
            shift
            ;;
        --years)
            shift
            YEARS_TO_PROCESS=""
            while [[ $# -gt 0 ]] && [[ ! $1 == --* ]]; do
                YEARS_TO_PROCESS="$YEARS_TO_PROCESS $1"
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--years YEAR1 YEAR2 ...] [--trial]"
            exit 1
            ;;
    esac
done

# Use specified years or default to all years
if [ -z "$YEARS_TO_PROCESS" ]; then
    YEARS_TO_PROCESS=$DEFAULT_YEARS
fi

echo "======================================================================"
echo "🚀  Submitting ERA5-Land Daily Summary Jobs to Gadi"
echo "======================================================================"
if [ ! -z "$TRIAL_FLAG" ]; then
    echo "Mode: TRIAL (January only)"
else
    echo "Mode: FULL YEAR"
fi
echo "Years to process: $YEARS_TO_PROCESS"
echo "======================================================================"

# Phase 1: Sync code to Gadi
echo ""
echo "Phase 1: Syncing code to Gadi..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

if [ -f "gadi/deploy_to_gadi.sh" ]; then
    ./gadi/deploy_to_gadi.sh
else
    echo "❌ Error: gadi/deploy_to_gadi.sh not found"
    exit 1
fi

echo "✅ Sync complete"
echo ""

# Phase 2: Submit jobs for each year
echo "======================================================================"
echo "Phase 2: Submitting PBS jobs..."
echo "======================================================================"

JOB_IDS=()

for YEAR in $YEARS_TO_PROCESS; do
    echo "Submitting job for year $YEAR..."

    # Build qsub command
    if [ ! -z "$TRIAL_FLAG" ]; then
        QSUB_CMD="cd ${REMOTE_DIR} && qsub -N daily_${YEAR} -v YEAR=${YEAR},${TRIAL_FLAG} ${PBS_SCRIPT}"
    else
        QSUB_CMD="cd ${REMOTE_DIR} && qsub -N daily_${YEAR} -v YEAR=${YEAR} ${PBS_SCRIPT}"
    fi

    # Submit job via SSH
    JOB_ID=$(ssh "${GADI_USER}@${GADI_HOST}" "source /etc/profile && ${QSUB_CMD}")

    if [ $? -eq 0 ]; then
        echo "  ✅ Job submitted: $JOB_ID (Year: $YEAR)"
        JOB_IDS+=("$JOB_ID")
    else
        echo "  ❌ Failed to submit job for year $YEAR"
    fi
done

echo ""
echo "======================================================================"
echo "✅  Submission complete!"
echo "======================================================================"
echo "Submitted ${#JOB_IDS[@]} job(s)"
echo ""
echo "To check job status on Gadi, run:"
echo "  ssh ${GADI_USER}@${GADI_HOST} 'qstat -u ${GADI_USER}'"
echo ""
echo "To view output logs (on Gadi):"
echo "  cd ${REMOTE_DIR}"
echo "  ls -lt daily_*.o* daily_*.e*"
echo ""
echo "To check output files (on Gadi):"
echo "  ls -lh /scratch/mn51/ft8695/era5-land/daily/2t/"
echo "======================================================================"
