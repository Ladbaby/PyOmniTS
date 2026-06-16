#!/usr/bin/env bash
# Generate categorized release notes from conventional commit messages.
# Usage: ./gen_release_notes.sh -v prev_tag...curr_tag
#        ./gen_release_notes.sh -v "...v0.1.0"   (first release, no previous tag)

set -euo pipefail

version_range=""

while getopts "v:" opt; do
  case $opt in
    v) version_range=$OPTARG ;;
    \?) echo "Usage: $0 -v <prev_tag>...<curr_tag>" >&2; exit 1 ;;
  esac
done

if [ -z "$version_range" ]; then
  echo "Error: -v is required." >&2
  echo "Example: $0 -v v2.0.0...v2.1.0" >&2
  exit 1
fi

# Parse prev and curr from the range string
if [[ "$version_range" == *"..."* ]]; then
  prev="${version_range%...*}"
  curr="${version_range#*...}"
elif [[ "$version_range" == "..."* ]]; then
  prev=""
  curr="${version_range:3}"
else
  prev=""
  curr="$version_range"
fi

REPO="https://github.com/Ladbaby/PyOmniTS"

# For first release (no prev), use the tag directly.
if [ -z "$prev" ]; then
  LOG_RANGE="$curr"
else
  LOG_RANGE="${prev}...${curr}"
fi

# Helper: list commits, filtering out noise (ghrs, merge), deduplicated & sorted
list_commits() {
  local grep_pattern="$1"
  git log --pretty="- %s ([commit](${REPO}/commit/%h))" --grep="$grep_pattern" -i "$LOG_RANGE" \
    | grep -v "^- ghrs:" \
    | grep -v "^- Merge " \
    | sort -f | uniq
}

{
  echo "# Release ${curr}"
  echo ""

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
  if [ -n "$prev" ]; then
    echo "---"
    echo ""
    echo "**Full Changelog**: ${REPO}/compare/${prev}...${curr}"
  else
    echo "---"
    echo ""
    echo "**Full Changelog**: ${REPO}/releases/tag/${curr}"
  fi
} > release.md

echo "Release notes written to release.md" >&2
cat release.md
