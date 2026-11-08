#!/usr/bin/env bash
# Regenerate all figures into output/. Needs data/ and data025/ from Zenodo (derived_data.tar) or build_events.sh.
set -euo pipefail
cd "$(dirname "$0")/../figures"
for f in fig1 fig2 fig3 fig4 fig5 extended_data; do python "$f.py"; done
