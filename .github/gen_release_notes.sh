#!/usr/bin/env bash
# Generate categorized release notes from conventional commit messages.
# Usage: ./gen_release_notes.sh -v v2.0.1
#   Auto-finds the previous version tag and generates notes for prev..HEAD.

set -euo pipefail

version=""

while getopts "v:" opt; do
  case $opt in
    v) version=$OPTARG ;;
    \?) echo "Usage: $0 -v <version>" >&2; exit 1 ;;
  esac
done

if [ -z "$version" ]; then
  echo "Error: -v is required." >&2
  echo "Example: $0 -v v2.1.0" >&2
  exit 1
fi

REPO="https://github.com/Ladbaby/PyOmniTS"

# Auto-detect the previous version tag by sorting all v-prefixed tags.
prev=$(python -c "
import subprocess, re, sys
result = subprocess.run(['git', 'tag', '-l'], capture_output=True, text=True)
tags = [t.strip() for t in result.stdout.strip().splitlines() if t.strip().startswith('v')]
# Filter out any tag that starts with the current version (handles re-runs)
tags = [t for t in tags if not t.startswith('$version')]
if not tags:
    sys.exit(0)
def key(t):
    m = re.match(r'^v(\d+)\.(\d+)\.(\d+)', t)
    return tuple(int(x) for x in m.groups()) if m else (-1,-1,-1)
tags.sort(key=key, reverse=True)
print(tags[0])
")

# Always use HEAD as the range endpoint so re-runs capture all recent commits.
if [ -n "$prev" ]; then
  LOG_RANGE="${prev}...HEAD"
else
  LOG_RANGE="HEAD"
fi

# Helper: list commits, filtering out noise (ghrs, merge), deduplicated & sorted
list_commits() {
  local grep_pattern="$1"
  git log --pretty="- %s ([commit](${REPO}/commit/%h))" --grep="$grep_pattern" -i "$LOG_RANGE" \
    | grep -v "^.\+ghrs:" \
    | grep -v "^.\+Merge " \
    | sort -f | uniq
}

{
  # ── Features ──────────────────────────────────────────────────────
  FEAT_OUTPUT=$(list_commits "^feat")
  if [ -n "$FEAT_OUTPUT" ]; then
    echo "## ✨ New Features"
    echo ""
    echo "$FEAT_OUTPUT"
    echo ""
  fi

  # ── Bug fixes ─────────────────────────────────────────────────────
  FIX_OUTPUT=$(list_commits "^fix")
  if [ -n "$FIX_OUTPUT" ]; then
    echo "## 🐛 Bug Fixes"
    echo ""
    echo "$FIX_OUTPUT"
    echo ""
  fi

  # ── Maintenance ───────────────────────────────────────────────────
  MAINT_OUTPUT=$(list_commits "^ci\|^chore\|^docs\|^refactor\|^style\|^test")
  if [ -n "$MAINT_OUTPUT" ]; then
    echo "## 🔧 Maintenance"
    echo ""
    echo "$MAINT_OUTPUT"
    echo ""
  fi

  # ── Footer ────────────────────────────────────────────────────────
  echo "---"
  echo ""
  if [ -n "$prev" ]; then
    echo "**Full Changelog**: ${REPO}/compare/${prev}...${version}"
  else
    echo "**Full Changelog**: ${REPO}/releases/tag/${version}"
  fi
} > release.md

echo "Release notes written to release.md" >&2
cat release.md
