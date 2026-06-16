#!/usr/bin/env python
"""Generate simple release notes grouped by commit type (feat/fix/chore/docs)."""

import subprocess
import sys
import re
from collections import defaultdict


def get_commits(prev_ref, head="HEAD"):
    """Get all commits between prev_ref and head, excluding noise commits."""
    result = subprocess.run(
        ["git", "log", f"{prev_ref}..{head}", "--format=%H|%s"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        return []

    commits = []
    noise_prefixes = ("ghrs:", "Merge branch")
    for line in result.stdout.strip().splitlines():
        commit_hash, subject = line.split("|", 1)
        short_hash = commit_hash[:7]
        if subject.startswith(noise_prefixes):
            continue
        commits.append((short_hash, subject))
    return commits


def parse_commit_type(subject):
    """Extract commit type from subject. Returns (type, rest) or (None, subject).

    Handles:
    - type(scope):desc  -> e.g. feat(models):ASTGI  -> ("feat", "models:ASTGI")
    - type:desc         -> e.g. chore:README        -> ("chore", "README")
    """
    m = re.match(r"^(feat|fix|chore|docs|style|refactor|test)\(([^\)]+)\):(.+)$", subject.strip())
    if m:
        return m.group(1), f"{m.group(2)}:{m.group(3).strip()}"

    m = re.match(r"^(feat|fix|chore|docs|style|refactor|test)\(([^\)]+)\):\s*$", subject.strip())
    if m:
        return m.group(1), m.group(2).strip()

    m = re.match(r"^(feat|fix|chore|docs|style|refactor|test):(.+)$", subject.strip())
    if m:
        return m.group(1), m.group(2).strip()

    return None, subject.strip()


def generate_release_notes(commits, version, repo):
    """Generate release notes grouped by commit type."""
    # Group commits by type
    groups = defaultdict(list)
    for short_hash, subject in commits:
        commit_type, rest = parse_commit_type(subject)
        if commit_type is None:
            commit_type = "other"
        groups[commit_type].append((short_hash, rest))

    # Define sections in order
    sections = [
        ("feat", "✨ New Features"),
        ("fix", "🐛 Bug Fixes"),
        ("chore", "🔧 Chores"),
        ("docs", "📚 Documentation"),
        ("refactor", "♻️ Refactoring"),
        ("style", "🎨 Style"),
        ("test", "🧪 Tests"),
        ("other", "Other"),
    ]

    lines = [f"# Release {version}\n"]

    for commit_type, title in sections:
        entries = groups.get(commit_type)
        if not entries:
            continue

        lines.append(f"## {title}\n")
        for short_hash, desc in entries:
            lines.append(f"- {desc} ([commit](https://github.com/{repo}/commit/{short_hash}))")
        lines.append("")

    if len(lines) <= 1:
        lines.append("No changes since the previous release.\n")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: generate_release_notes.py <version> <prev_tag> [repo]")
        sys.exit(1)

    version = sys.argv[1]
    prev_tag = sys.argv[2]
    repo = sys.argv[3] if len(sys.argv) > 3 else "Ladbaby/PyOmniTS"

    print(f"Generating release notes for {version} (since {prev_tag})...", file=sys.stderr)

    commits = get_commits(prev_tag)
    if not commits:
        print(f"# Release {version}\n\nNo changes since the previous release.\n")
        sys.exit(0)

    print(f"Found {len(commits)} commits", file=sys.stderr)

    notes = generate_release_notes(commits, version, repo)
    print(notes)


if __name__ == "__main__":
    main()
