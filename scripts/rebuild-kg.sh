#!/usr/bin/env bash
# Rebuild proteusPy KG indices (PyCodeKG SQLite + LanceDB).
# Usage: bash scripts/rebuild-kg.sh [--wipe]
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WIPE=${1:-}

echo "--- PyCodeKG rebuild: SQLite ---"
poetry run pycodekg build-sqlite --repo "$REPO_ROOT" ${WIPE}

echo "--- PyCodeKG rebuild: LanceDB ---"
poetry run pycodekg build-lancedb --repo "$REPO_ROOT" ${WIPE}

echo "--- PyCodeKG rebuild: complete ---"
