# Progenitor-Only Parameter Grid Plan

## Goal
Build a stand-alone Python script that runs the progenitor-only PDE model across a grid of parameter values until progenitor density reaches steady state, then quantifies how these parameters shape where neuron births occur at steady state.

Parameters to sweep:
- Progenitor birth amplitude: `b_0`
- Progenitor diffusion: `D_P`
- Progenitor radial velocity: `v_P`
- Differentiation rate: `c_diff` (mapped to model parameter `d` in `simulate_progenitors_only`)

Neuron birth rate definition at radius `r` at steady state:
- `lambda_birth_ss(r) = c_diff * P_ss(r)`

## Existing Model Mapping
Use `simulate_progenitors_only(params, T_final, dt, verbose=False)` from `scripts/sim_jellyfish_eqn_clean.py`.

Relevant model components already present:
- Progenitor PDE includes diffusion (`D_P`), advection (`v_P`), source (`b_0` via `birth_rate`), and sink (`d * P`)
- Radial grid is `r in [r_mouth, R]`

For consistency with your interpretation, the sweep variable `c_diff` will be assigned to:
- `params['d'] = c_diff`

## Steady-State Protocol
For each parameter setting:
- Run the model in time chunks (for example `chunk_time = 20` to `50`)
- After each chunk, use the final profile as the initial condition for the next chunk
- Check convergence using the final few step-to-step changes:
  - `max_i ||P_{n+1} - P_n||_inf / dt < steady_tol`
- Require at least `min_time` simulated before accepting convergence
- Stop and mark as not converged if `max_time` is reached

Steady-state profile:
- `P_ss(r) = P(r, t_converged)`

## Outputs Per Parameter Setting
For each parameter tuple `(b_0, D_P, v_P, c_diff)`, compute:

1. Mean neuron birth radius
- Steady-state weighted mean:
  - Numerator: `sum_r [r * lambda_birth_ss(r) * w_r]`
  - Denominator: `sum_r [lambda_birth_ss(r) * w_r]`
  - Mean: numerator / denominator

2. Variance of neuron birth radius
- Steady-state weighted variance around the mean:
  - `Var(r_birth_ss) = sum_r [(r - mean)^2 * lambda_birth_ss(r) * w_r] / denominator`

3. Overall neuron birth rate
- Spatially integrated steady-state total birth rate:
  - `R_birth_ss = sum_r [lambda_birth_ss(r) * w_r]`

Where `w_r` is the radial area element for each ring (recommended: proportional to `r * dr`; optional constant factors like `2*pi` cancel in normalized moments and can be included or omitted consistently).

## Parameter Grid Design
Define explicit arrays for each parameter, for example:
- `b_0_values = np.linspace(..., ..., n_b)`
- `D_P_values = np.logspace(..., ..., n_D)`
- `v_P_values = np.linspace(..., ..., n_v)`
- `c_diff_values = np.linspace(..., ..., n_d)`

Use Cartesian product over all arrays.

Suggested practical start:
- Keep grid moderate first (for example 5 x 5 x 4 x 4)
- Increase resolution after runtime check

## Script Structure
New script (stand-alone) should include:

1. Imports and argument parsing
- `argparse`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `itertools`
- Import `simulate_progenitors_only` from `scripts/sim_jellyfish_eqn_clean.py`

2. Base parameter builder
- One canonical base `params` dictionary
- Helper to clone and override only swept parameters

3. Metric computation functions
- Run chunked simulation to convergence and extract `P_ss`
- Compute `lambda_birth_ss(r)` from `P_ss`
- Compute steady-state weighted mean radius, variance, and total birth rate

4. Grid runner
- Loop over all parameter combinations
- Run model and compute metrics
- Append one row per setting into a results table

5. Save outputs
- CSV table: one row per parameter combination with all metrics
- Optional NPY/PKL for full metadata

6. Visualization functions
- Heatmaps for pairs of parameters (x,y)
- Separate subplot panels (or separate figures) for slices of a third parameter
- Fixed value (or small set) for the fourth parameter

## Plotting Plan
Primary figure pattern:
- Choose two parameters for axes (for example `D_P` vs `v_P`)
- Choose one slicing parameter (for example `c_diff`)
- Fix one parameter (for example `b_0`) at one value per figure

For each metric (`mean_r_birth`, `var_r_birth`, `mean_total_birth_rate`):
- Create a panel grid where each panel is one `c_diff` slice
- In each panel, draw a heatmap over (`D_P`, `v_P`)
- Save one PNG per metric (or one multi-row figure)

Secondary plots (optional):
- Line plots vs each single parameter after averaging over others
- Distribution sanity check: radial birth profile for selected parameter settings

## Validation and Sanity Checks
1. Numerical sanity
- Ensure chunk-to-chunk `P` remains finite
- If small negatives occur from Euler error, clip only for birth-rate reporting and log this behavior
- Verify convergence criterion is robust by testing stricter `steady_tol`

2. Metric sanity
- Confirm denominator in weighted mean/variance is nonzero
- Validate one or two settings manually by direct recomputation

3. Runtime sanity
- Start with coarse grids and modest `max_time`
- Scale up once runtime is acceptable

## Deliverables
1. Stand-alone sweep script (new file)
2. Results CSV with all parameter combinations and metrics
3. Heatmap figure set showing parameter effects on:
- Mean birth radius at steady state
- Variance of birth radius at steady state
- Overall neuron birth rate at steady state

## Implementation Notes
- Keep `simulate_progenitors_only` untouched and reuse directly.
- Use reproducible deterministic runs (no randomness in current PDE solver).
- Keep all outputs in a dedicated folder, e.g. `results/progenitor_param_sweep/`.
