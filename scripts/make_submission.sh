#!/usr/bin/env bash
# Build the submission zip.
#
# It archives exactly the git-tracked files at the last commit. Private files
# (SCRATCHPAD.md, docs/learning/) are untracked, so they cannot appear here by
# construction, not by an exclude list that could be mistyped. The grep at the
# end is a second safety net that fails loudly if that ever stops being true.
set -euo pipefail

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "NOTE: you have uncommitted changes. The zip archives the LAST COMMIT,"
  echo "      so commit first if you want those changes included."
  echo
fi

sha=$(git rev-parse --short HEAD)
out="submission-${sha}.zip"

git archive --format=zip -o "$out" HEAD

echo "Files in ${out}:"
unzip -Z1 "$out" | sort | sed 's/^/  /'
echo

if unzip -Z1 "$out" | grep -Eiq 'SCRATCHPAD|docs/learning'; then
  echo "ERROR: a private file leaked into ${out}. Aborting and deleting it." >&2
  rm -f "$out"
  exit 1
fi

echo "OK: ${out} built, and it contains no private files."
