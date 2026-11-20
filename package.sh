#!/bin/bash
# Assemble the replication package as a self-contained directory (and tarball).
#   ./package.sh [out-dir]
set -e
cd "$(dirname "$0")"
OUT=${1:-replication-package}
rm -rf "$OUT" && mkdir -p "$OUT"
cp -r src tests run_protocol.py reproduce.sh pyproject.toml README.md LICENSE "$OUT/"
cp -r study paper_out artifacts traces "$OUT/"
# Only what the manuscript uses: drop the outputs of exploratory analyses it does not report.
rm -f "$OUT"/study/*_hier.* "$OUT"/study/*_subsets.* "$OUT"/study/*_posthoc.* "$OUT"/paper_out/tab_hierarchy.tex "$OUT"/paper_out/tab_posthoc.tex
mkdir -p "$OUT/vectors" && cp -L vectors/*.npz "$OUT/vectors/"   # feature matrices only; dereferences the symlink
find "$OUT" \( -name '__pycache__' -o -name '*.egg-info' -o -name '.DS_Store' \) -prune -exec rm -rf {} +
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
du -sh "$OUT" "$OUT.tar.gz"
