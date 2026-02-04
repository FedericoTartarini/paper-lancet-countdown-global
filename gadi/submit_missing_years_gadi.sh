#!/bin/bash

# Helper script to check which years have already been processed on Gadi
# and submit jobs only for missing years
#
# Usage:
#   ./submit_missing_years_gadi.sh [--trial]

GADI_USER="ft8695"
GADI_HOST="gadi.nci.org.au"
REMOTE_DIR="~/paper-lancet-countdown-global"
OUTPUT_DIR="/scratch/mn51/ft8695/era5-land/daily/2t"
PBS_SCRIPT="gadi/job_daily_summaries_single_year.pbs"

# Parse trial flag
TRIAL_FLAG=""
if [[ "$1" == "--trial" ]]; then
    TRIAL_FLAG="TRIAL=1"
    echo "Running in TRIAL mode"
fi

echo "======================================================================"
echo "🔍  Checking for missing years on Gadi..."
echo "======================================================================"

# Get list of already processed years from Gadi
echo "Fetching list of processed years from Gadi..."
PROCESSED_FILES=$(ssh "${GADI_USER}@${GADI_HOST}" "ls ${OUTPUT_DIR}/*_daily_summaries.nc 2>/dev/null | xargs -n 1 basename")

PROCESSED_YEARS=()
for file in $PROCESSED_FILES; do
    # Extract year from filename (e.g., "1980_daily_summaries.nc" -> "1980")
    year=$(echo $file | cut -d'_' -f1)
    PROCESSED_YEARS+=($year)
done

echo "Found ${#PROCESSED_YEARS[@]} already processed years: ${PROCESSED_YEARS[@]}"
echo ""

# Determine which years need processing (1979-2025)
CURRENT_YEAR=2025
ALL_YEARS=$(seq 1979 $CURRENT_YEAR)
MISSING_YEARS=()

for year in $ALL_YEARS; do
    if [[ ! " ${PROCESSED_YEARS[@]} " =~ " ${year} " ]]; then
        MISSING_YEARS+=($year)
    fi
done

if [ ${#MISSING_YEARS[@]} -eq 0 ]; then
    echo "✅ All years (1979-$CURRENT_YEAR) have been processed!"
    echo "Nothing to submit."
    exit 0
fi

echo "Found ${#MISSING_YEARS[@]} missing years: ${MISSING_YEARS[@]}"
echo ""
read -p "Submit jobs for these missing years? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

# Sync code to Gadi
echo ""
echo "======================================================================"
echo "Phase 1: Syncing code to Gadi..."
echo "======================================================================"

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

# Submit jobs for missing years
echo "======================================================================"
echo "Phase 2: Submitting PBS jobs for missing years..."
echo "======================================================================"

JOB_IDS=()

for YEAR in "${MISSING_YEARS[@]}"; do
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
echo "Submitted ${#JOB_IDS[@]} job(s) for missing years"
echo ""
echo "To check job status:"
echo "  ssh ${GADI_USER}@${GADI_HOST} 'qstat -u ${GADI_USER}'"
echo "======================================================================"
