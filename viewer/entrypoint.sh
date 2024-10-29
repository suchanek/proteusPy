#!/bin/bash
set -e

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate proteusPy

# Check if Xvfb is already running and kill it if necessary
if pgrep Xvfb > /dev/null; then
    echo "Xvfb is already running. Killing the existing process."
    pkill Xvfb
fi

# Remove the lock file if it exists
if [ -f /tmp/.X99-lock ]; then
    echo "Removing existing /tmp/.X99-lock file."
    rm /tmp/.X99-lock
fi

# Start Xvfb in the background
Xvfb :99 -screen 0 1920x1080x24 &

# Export the DISPLAY environment variable
export DISPLAY=:99

# Optional: Print DISPLAY to verify
echo "DISPLAY set to $DISPLAY"

# Run the Panel application without attempting to open the URL
panel serve rcsb_viewer.py \
    --address 0.0.0.0 \
    --port 5006 \
    --allow-websocket-origin="*"
