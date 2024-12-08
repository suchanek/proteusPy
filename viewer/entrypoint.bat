# Activate the Conda environment
& "C:\Miniconda3\Scripts\activate" proteusPy

# Check if Xvfb is already running and kill it if necessary
if (Get-Process -Name Xvfb -ErrorAction SilentlyContinue) {
    Write-Output "Xvfb is already running. Killing the existing process."
    Stop-Process -Name Xvfb -Force
}

# Remove the lock file if it exists
if (Test-Path "C:\tmp\.X99-lock") {
    Write-Output "Removing existing C:\tmp\.X99-lock file."
    Remove-Item "C:\tmp\.X99-lock"
}

# Start Xvfb in the background
Start-Process -FilePath "Xvfb" -ArgumentList ":99 -screen 0 1920x1080x24" -NoNewWindow -PassThru

# Export the DISPLAY environment variable
$env:DISPLAY = ":99"

# Optional: Print DISPLAY to verify
Write-Output "DISPLAY set to $env:DISPLAY"

# Run the Panel application without attempting to open the URL
Start-Process -FilePath "panel" -ArgumentList "serve rcsb_viewer.py --address 0.0.0.0 --port 5006 --allow-websocket-origin='*'" -NoNewWindow -Wait
