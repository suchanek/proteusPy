#!/usr/bin/env bash
# proteusPy pre-commit hook — keeps local KG indices in sync and captures
# metrics snapshots BEFORE quality checks run.
# Install with: bash scripts/install-hooks.sh
# Skip with:   PYCODEKG_SKIP_SNAPSHOT=1 git commit ...
set -euo pipefail

[ "${PYCODEKG_SKIP_SNAPSHOT:-0}" = "1" ] && exit 0

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Capture the tree hash of the staged index NOW — before any tool modifies files.
TREE_HASH=$(git write-tree)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

PYCODEKG="$REPO_ROOT/.venv/bin/pycodekg"
DOCKG="$REPO_ROOT/.venv/bin/dockg"
FTREEKG="$REPO_ROOT/.venv/bin/ftreekg"

# --- Rebuild indices (skip gracefully if tool not installed) ---

if [ -x "$PYCODEKG" ] && [ -d "$REPO_ROOT/.pycodekg" ]; then
    "$PYCODEKG" build --repo "$REPO_ROOT" || exit 1
fi

if [ -x "$DOCKG" ] && [ -d "$REPO_ROOT/.dockg" ]; then
    "$DOCKG" build || exit 1
fi

if [ -x "$FTREEKG" ] && [ -d "$REPO_ROOT/.filetreekg" ]; then
    "$FTREEKG" build || exit 1
fi

# --- Snapshots ---

if [ -x "$PYCODEKG" ] && [ -d "$REPO_ROOT/.pycodekg" ]; then
    "$PYCODEKG" snapshot save \
        --repo . \
        --tree-hash "$TREE_HASH" \
        --branch "$BRANCH" \
      || { echo "[pycodekg] snapshot skipped (run 'pycodekg build' to initialize)" >&2; }
fi

if [ -x "$DOCKG" ] && [ -d "$REPO_ROOT/.dockg" ]; then
    "$DOCKG" snapshot save \
        --repo . \
        --tree-hash "$TREE_HASH" \
        --branch "$BRANCH" \
      || { echo "[dockg] snapshot skipped" >&2; }
fi

if [ -x "$FTREEKG" ] && [ -d "$REPO_ROOT/.filetreekg" ]; then
    "$FTREEKG" snapshot save \
        --repo . \
        --tree-hash "$TREE_HASH" \
        --branch "$BRANCH" \
      || { echo "[ftreekg] snapshot skipped" >&2; }
fi

# --- Stage snapshot directories ---

git add .pycodekg/snapshots/ 2>/dev/null || true
git add .dockg/snapshots/ 2>/dev/null || true
git add .filetreekg/snapshots/ 2>/dev/null || true

# --- Quality checks (delegates to .pre-commit-config.yaml) ---

PRECOMMIT="$REPO_ROOT/.venv/bin/pre-commit"
if [ -x "$PRECOMMIT" ]; then
    "$PRECOMMIT" run || exit 1
elif command -v pre-commit &>/dev/null; then
    pre-commit run || exit 1
fi

exit 0
