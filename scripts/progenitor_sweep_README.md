# Progenitor Steady-State Sweep Pipeline

This pipeline runs a progenitor-only steady-state simulation across a parameter grid, computes summary statistics from the steady-state profiles, and generates heatmap figures for the output variables.

## Workflow

The pipeline is split into three stages:

1. Simulation
   - Runs the progenitor model to steady state for every parameter combination.
   - Saves raw steady-state profiles and the sweep configuration.

2. Analysis
   - Reads the saved simulation bundle.
   - Computes:
     - mean birth radius
     - birth-radius variance
     - total neuron birth rate

3. Plotting
   - Reads the analysis table.
   - Makes one figure per output variable.
   - Each figure is a grid of heatmaps over `D_P` and `v_P`, with rows indexed by `b_0` and columns indexed by `c_diff`.

## Main Files

- `progenitor_steady_state_parameter_sweep.py`
  - Simulation stage
  - Edit the `CONFIG` dictionary at the top to change sweep values and solver settings

- `progenitor_sweep_analysis.py`
  - Analysis stage
  - Converts raw simulation output into summary metrics

- `progenitor_sweep_plot.py`
  - Plotting stage
  - Makes the final figures

- `progenitor_model_core.py`
  - Shared model and helper functions

- `run_progenitor_sweep.sh`
  - Convenience wrapper that runs all three stages in sequence

## Output Structure

Each full run is written to a timestamped folder under:

`results/progenitor_param_sweep_steady_state/`

Inside each run folder you should see:

- `simulation_config.json`
- `simulation_raw.npz`
- `analysis_results.csv`
- `analysis_summary.json`
- `mean_birth_radius.png`
- `var_birth_radius.png`
- `total_birth_rate.png`

## How to Run

To run the full pipeline:

```bash
bash scripts/run_progenitor_sweep.sh
```

To run only one stage:

- Simulation only:
  ```bash
  python scripts/progenitor_steady_state_parameter_sweep.py
  ```

- Analysis only:
  ```bash
  python scripts/progenitor_sweep_analysis.py
  ```

- Plotting only:
  ```bash
  python scripts/progenitor_sweep_plot.py
  ```

## Editing the Sweep

Open `progenitor_steady_state_parameter_sweep.py` and edit `CONFIG` directly.

Important parameters:

- `b0_values`
- `D_P_values`
- `v_P_values`
- `cdiff_values`
- `chunk_time`
- `min_time`
- `max_time`
- `steady_tol`

The simulation script saves a timestamped run folder automatically, so analysis and plotting can be repeated later without rerunning the simulations.

## Notes

- The heatmap axes are formatted in scientific notation.
- The color scale is shared across all panels within a figure, so subplot sizes stay consistent.
- The analysis and plotting stages always use the latest timestamped run folder unless you point them at a specific one in `CONFIG`.
