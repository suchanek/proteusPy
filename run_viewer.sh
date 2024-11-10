#!/bin/bash
set -e

# Activate the Conda environment
~/miniforge3/condabin/conda activate ppydev

# Get the full path to the DBViewer.py file
DBVIEWER_PATH="~/repos/proteusPy/programs/DBViewer.py"

# Run the Panel application without attempting to open the URL
panel serve "$DBVIEWER_PATH" \
    --address 0.0.0.0 \
    --port 5006 \
    --allow-websocket-origin="*" &

