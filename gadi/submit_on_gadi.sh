#!/bin/bash

# Configuration
# Replace these if your remote directory is different
GADI_USER="ft8695"
GADI_HOST="gadi.nci.org.au"
REMOTE_DIR="~/paper-lancet-countdown-global"

# Check if an input file was provided
if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "  1. Run a generic Python script:"
    echo "     $0 <python_script.py> [optional_flags]"
    echo "     Example: $0 python_code/weather/b_daily_summaries.py --trial"
    echo ""
    echo "  2. Submit a specific PBS job:"
    echo "     $0 <job_script.pbs> [optional_flags]"
    echo "     Example: $0 gadi/custom_job.pbs"
    exit 1
fi

INPUT_FILE=$1
shift
SCRIPT_ARGS="$@"

# Detect where this script is stored (e.g., project_root/gadi/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Assume project root is one level up
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "--------------------------------------------------"
echo "🚀  Phase 1: Syncing code to Gadi..."
echo "--------------------------------------------------"

# Change to project root to ensure sync works correctly
cd "$PROJECT_ROOT" || exit 1

# Run the deploy script from the gadi folder
if [ -f "gadi/deploy_to_gadi.sh" ]; then
    ./gadi/deploy_to_gadi.sh
else
    echo "Error: gadi/deploy_to_gadi.sh not found in project root."
    exit 1
fi

echo "--------------------------------------------------"

# Determine mode based on file extension
if [[ "$INPUT_FILE" == *.py ]]; then

    # The runner is now in the gadi subdirectory
    PBS_SCRIPT="gadi/job_runner.pbs"
    TARGET_SCRIPT="$INPUT_FILE"

    # Extract filename without extension for Job Name (max 15 chars for PBS)
    JOB_NAME=$(basename "$TARGET_SCRIPT" .py | cut -c 1-15)

    echo "🚀  Phase 2: Submitting Python script '$TARGET_SCRIPT'..."
    echo "    Using generic runner: $PBS_SCRIPT"
    echo "    Job Name: $JOB_NAME"
    if [ -n "$SCRIPT_ARGS" ]; then
        echo "    With arguments: $SCRIPT_ARGS"
    fi

    # Construct qsub command
    REMOTE_CMD="cd ${REMOTE_DIR} && qsub -N ${JOB_NAME} -v TARGET_SCRIPT=\"${TARGET_SCRIPT}\",PYTHON_ARGS=\"${SCRIPT_ARGS}\" ${PBS_SCRIPT}"

else
    # Assume it is a PBS script (legacy mode)
    PBS_SCRIPT="$INPUT_FILE"

    echo "🚀  Phase 2: Submitting PBS script '$PBS_SCRIPT'..."

    if [ -n "$SCRIPT_ARGS" ]; then
        echo "    With arguments: $SCRIPT_ARGS"
        REMOTE_CMD="cd ${REMOTE_DIR} && qsub -v PYTHON_ARGS=\"${SCRIPT_ARGS}\" ${PBS_SCRIPT}"
    else
        REMOTE_CMD="cd ${REMOTE_DIR} && qsub ${PBS_SCRIPT}"
    fi

fi

echo "--------------------------------------------------"

# Executing remote command via SSH
# We force loading of /etc/profile to ensure qsub is in the PATH
echo "Executing remote command on Gadi:"
#echo "${GADI_USER}@${GADI_HOST}" "source /etc/profile && ${REMOTE_CMD}"
ssh "${GADI_USER}@${GADI_HOST}" "source /etc/profile && ${REMOTE_CMD}"

echo "--------------------------------------------------"

echo "✅  Job submission command executed."