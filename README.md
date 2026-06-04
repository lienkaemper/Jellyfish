# Project Scripts

The core analysis and modeling workflows are in [scripts](scripts):

- `identify_jumps.ipynb`
	- Run this first.
	- Computes real-time values and identifies periods when the jellyfish is resting off-scope.
	- Also creates convenience variables (`r`, `theta`) in a coordinate system with the mouth at the origin.

- `infer_params.ipynb`
	- Estimates drift and diffusion parameters from data.
	- Generates plots of neuron and birth densities over time.

- `sim_jellyfish_eqn_clean.ipynb`
	- Simulates the PDE model of neuron and progenitor migration.

- `generate_individual_neuron_plots.ipynb`
	- Runs the individual-based model and produces neuron-level plots.

- `spatial_derivative_utiliies.py`
	- Helper functions for spatial derivatives used by the PDE model.