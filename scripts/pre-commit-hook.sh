#!/usr/bin/env bash
# proteusPy pre-commit hook -- runs quality checks. The KG index rebuild and
# metrics snapshots are opt-in and OFF by default.
# Install with: bash scripts/install-hooks.sh
#
#   PROTEUSPY_SNAPSHOT=1 git commit ...        opt in to a per-commit snapshot
#   PROTEUSPY_SKIP_SNAPSHOT=1 git commit ...   force snapshots off (wins)
#
# Why snapshots are off by default: a per-commit snapshot keys itself on
# `git write-tree` and is then staged into the same commit, so the recorded
# tree can never be the committed one. The fleet stopped per-commit snapshots
# on 2026-08-18 and snapshots at release instead, keyed on the tag
# (`pycodekg snapshot save <version> --subject repo:proteuspy`). See
# kgrag_priv/docs/SNAPSHOT_STRATEGY.md.
#
# Order is deliberately checks-then-index. `pre-commit run` stashes unstaged
# changes and restores them afterwards; rebuilding indices first put their
# rewritten snapshot manifests inside that stash window, where the restore can
# fail or let a staged snapshot deletion slip into the commit. And a rebuild
# is wasted on a commit the checks are about to reject.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# --- Quality checks (delegates to .pre-commit-config.yaml) ---
PRECOMMIT="$REPO_ROOT/.venv/bin/pre-commit"
if [ -x "$PRECOMMIT" ]; then
    "$PRECOMMIT" run || exit 1
elif command -v pre-commit &>/dev/null; then
    pre-commit run || exit 1
fi

# --- Opt-in index rebuild + snapshots ---
[ "${PROTEUSPY_SNAPSHOT:-0}" = "1" ] || exit 0
[ "${PROTEUSPY_SKIP_SNAPSHOT:-0}" = "1" ] && exit 0

BRANCH=$(git rev-parse --abbrev-ref HEAD)
# Snapshots are default-branch history only.
DEFAULT_BRANCH=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || true)
DEFAULT_BRANCH="${DEFAULT_BRANCH#origin/}"
[ "$BRANCH" != "${DEFAULT_BRANCH:-master}" ] && exit 0

PYCODEKG="$REPO_ROOT/.venv/bin/pycodekg"
DOCKG="$REPO_ROOT/.venv/bin/dockg"
FTREEKG="$REPO_ROOT/.venv/bin/ftreekg"

if [ -x "$PYCODEKG" ] && [ -d "$REPO_ROOT/.pycodekg" ]; then
    "$PYCODEKG" build --repo "$REPO_ROOT" || exit 1
    "$PYCODEKG" snapshot save --repo . --branch "$BRANCH" \
      || echo "[pycodekg] snapshot skipped" >&2
    git add .pycodekg/snapshots/ 2>/dev/null || true
fi

if [ -x "$DOCKG" ] && [ -d "$REPO_ROOT/.dockg" ]; then
    "$DOCKG" build || exit 1
    "$DOCKG" snapshot save --repo . --branch "$BRANCH" \
      || echo "[dockg] snapshot skipped" >&2
    git add .dockg/snapshots/ 2>/dev/null || true
fi

if [ -x "$FTREEKG" ] && [ -d "$REPO_ROOT/.filetreekg" ]; then
    "$FTREEKG" build || exit 1
    "$FTREEKG" snapshot save --repo . --branch "$BRANCH" \
      || echo "[ftreekg] snapshot skipped" >&2
    git add .filetreekg/snapshots/ 2>/dev/null || true
fi

exit 0
