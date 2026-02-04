#!/bin/bash

# Configuration
GADI_USER="ft8695"
GADI_HOST="gadi.nci.org.au"
REMOTE_DIR="~/paper-lancet-countdown-global"

# Exclude list to avoid syncing unnecessary large files or local envs
EXCLUDE_LIST=(
    "__pycache__"
    ".git"
    ".idea"
    ".vscode"
    "venv"
    "env"
    "*.pyc"
    ".DS_Store"
    "Lancet Countdown Heatwave 2024 v1.0.2.zip"
    "manuscript"
)

# Build the exclude arguments array
rsync_args=()
for item in "${EXCLUDE_LIST[@]}"; do
    rsync_args+=(--exclude "$item")
done

# Sync command
# We sync the current directory (.) to the remote directory
echo "Syncing code to ${GADI_USER}@${GADI_HOST}:${REMOTE_DIR}..."

# Ensure the remote directory exists
ssh ${GADI_USER}@${GADI_HOST} "mkdir -p ${REMOTE_DIR}"

# Run rsync
rsync -avzP "${rsync_args[@]}" ./ ${GADI_USER}@${GADI_HOST}:${REMOTE_DIR}/

# Clean up old log files on Gadi
echo "Cleaning up old log files on Gadi..."
ssh ${GADI_USER}@${GADI_HOST} "cd ${REMOTE_DIR} && rm -f *.o* *.e*"

echo "Sync complete."
