#!/usr/bin/env bash
# Install proteusPy git hooks.
# Run from the repo root: bash scripts/install-hooks.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Install the KG snapshot + quality-check hook as the primary pre-commit hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Remove legacy hook from previous pre-commit migration-mode installs
rm -f .git/hooks/pre-commit.legacy

echo "Installed .git/hooks/pre-commit from scripts/pre-commit-hook.sh"
