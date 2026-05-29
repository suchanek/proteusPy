# RCSB Disulfide Viewer - Docker Deployment

This directory contains Docker configuration for running the RCSB Disulfide Viewer with proper memory allocation.

## Quick Start

### Using Docker Compose (Recommended)

```bash
cd viewer
docker-compose up -d
```

Access the viewer at: http://localhost:5006/rcsb_viewer

### View Logs

```bash
docker-compose logs -f
```

### Stop the Container

```bash
docker-compose down
```

## Memory Configuration

The viewer loads large protein database files (`PDB_SS_ALL_LOADER.pkl`) which require significant memory. The docker-compose.yml now includes memory limits to prevent OOM (Out of Memory) kills:

```yaml
mem_limit: 8g              # Maximum memory the container can use
mem_reservation: 4g        # Soft limit - guaranteed minimum memory
memswap_limit: 12g         # Total memory including swap
shm_size: 2g               # Shared memory for VTK/PyVista operations
```

### Adjusting Memory Limits

If you encounter OOM issues (exit code 137), increase the limits in [docker-compose.yml](docker-compose.yml:14-17):

```yaml
mem_limit: 12g
mem_reservation: 6g
memswap_limit: 16g
```

**Important:** Ensure Docker Desktop has sufficient memory allocated:
- **macOS/Windows**: Docker Desktop → Settings → Resources → Memory
- Recommended: 12GB+ for Docker Desktop memory allocation

## Monitoring

### Real-time Memory Usage

```bash
docker stats rcsb_viewer
```

### Check for OOM Kills

```bash
docker inspect rcsb_viewer | grep OOMKilled
```

### View Container Details

```bash
docker inspect rcsb_viewer
```

## Troubleshooting

### Container exits with code 137

This indicates an OOM kill. The container ran out of memory while loading data.

**Solutions:**
1. Increase Docker Desktop memory allocation (Settings → Resources)
2. Increase container memory limits in docker-compose.yml
3. Reduce the size of the PKL database file if possible

**Verify OOM status:**
```bash
docker inspect rcsb_viewer | jq '.[0].State.OOMKilled'
```

### Startup Issues

Check the logs for errors:
```bash
docker-compose logs --tail=100
```

Common issues:
- Xvfb display errors (usually harmless)
- PyVista/VTK theme warnings (usually harmless)
- Database loading errors (check file permissions and paths)

## Manual Docker Commands

If you prefer not to use docker-compose:

### Build the Image

```bash
docker build -t egsuchanek/rcsb_viewer:latest .
```

### Run with Memory Limits

```bash
docker run -d \
  --name rcsb_viewer \
  --memory="8g" \
  --memory-swap="12g" \
  --shm-size="2g" \
  -p 5006:5006 \
  --restart unless-stopped \
  -v $(pwd):/app \
  egsuchanek/rcsb_viewer:latest
```

### Stop and Remove

```bash
docker stop rcsb_viewer
docker rm rcsb_viewer
```

## Files

- [docker-compose.yml](docker-compose.yml) - Docker Compose configuration with memory limits
- [dockerfile](dockerfile) - Linux container Dockerfile
- [docker_win.yml](docker_win.yml) - Windows container Dockerfile
- [entrypoint.sh](entrypoint.sh) - Linux container startup script
- [entrypoint.bat](entrypoint.bat) - Windows container startup script
- [environment.yml](environment.yml) - Conda environment specification
- [rcsb_viewer.py](rcsb_viewer.py) - Main viewer application

## Architecture

The container:
1. Uses Miniconda3 as the base image
2. Installs system dependencies (OpenGL, Mesa, Xvfb)
3. Creates a Conda environment from environment.yml
4. Installs proteusPy package
5. Runs Xvfb (virtual frame buffer) for headless rendering
6. Starts Panel/Bokeh server on port 5006
7. Loads the RCSB Disulfide Database on startup

## Environment Variables

Set in docker-compose.yml:
- `PYTHONUNBUFFERED=1` - Real-time Python output
- `DOCKER_RUNNING=true` - Signals running in Docker
- `PYVISTA_OFF_SCREEN=true` - Enables headless PyVista rendering
- `MPLCONFIGDIR=/tmp/matplotlib` - Matplotlib cache location

## Data Files

The viewer expects database files in `/app/data/`:
- `PDB_SS_ALL_LOADER.pkl` - Main disulfide bond database

Ensure data files are present before starting the container.

## Performance Notes

- First startup may take 5-10 minutes while loading the database
- Memory usage peaks during database loading (~6-8GB)
- Steady-state memory usage is typically 2-4GB
- Shared memory (shm_size) is critical for VTK operations

## Support

For issues related to:
- **proteusPy package**: https://github.com/suchanek/proteusPy
- **RCSB Viewer**: Create an issue with logs and `docker inspect` output
- **Docker configuration**: Check Docker Desktop memory allocation first
