#!/usr/bin/env bash
# Rebuild proteusPy KG indices (PyCodeKG SQLite + sqlite-vec).
# Usage: bash scripts/rebuild-kg.sh [--wipe]
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIPE=${1:-}

echo "--- PyCodeKG rebuild: SQLite ---"
poetry run pycodekg build-sqlite --repo "$REPO_ROOT" ${WIPE}

echo "--- PyCodeKG rebuild: sqlite-vec ---"
poetry run pycodekg build-index --repo "$REPO_ROOT" ${WIPE}

echo "--- PyCodeKG rebuild: complete ---"
