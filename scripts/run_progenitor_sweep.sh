#!/usr/bin/env bash
set -euo pipefail

python scripts/progenitor_steady_state_parameter_sweep.py
python scripts/progenitor_sweep_analysis.py
python scripts/progenitor_sweep_plot.py
