#!/bin/bash
set -e

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate proteusPy

# Start Xvfb in the background
Xvfb :99 -screen 0 1920x1080x24 &

# Export the DISPLAY environment variable
export DISPLAY=:99

# Optional: Print DISPLAY to verify
echo "DISPLAY set to $DISPLAY"

# Run the Panel application
panel serve rcsb_viewer.py \
    --address 0.0.0.0 \
    --port 5006 \
    --allow-websocket-origin=* \
    --show
