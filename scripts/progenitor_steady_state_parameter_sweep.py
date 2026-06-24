#!/usr/bin/env python3
"""Simulation-only steady-state parameter sweep.

This stage runs the progenitor model over a Cartesian grid of
(b_0, D_P, v_P, c_diff) until steady state and saves the raw steady-state
profiles for later analysis and plotting.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from progenitor_model_core import (
    ensure_output_run_dir,
    run_parameter_grid_simulation,
)


# Edit this dictionary directly to control runs.
# Defaults mirror the values used in sim_jellyfish_eqn_clean.py where applicable.
CONFIG = {
    # Parameter sweep values (edit lists directly)
    "b0_values": [0.5, 1.0, 2.0],
    "D_P_values": [1 * 10 ** -5, 3 * 10 ** -5, 1 * 10 ** -4],
    "v_P_values": [0.001, 0.002, 0.003],
    "cdiff_values": [0.03],

    # Steady-state solver settings
    "dt": 1e-3,
    "chunk_time": 20.0,
    "min_time": 20.0,
    "max_time": 500.0,
    "steady_tol": 1e-3,
    "check_window": 5,

    # Geometry and model defaults (from sim_jellyfish_eqn_clean.py)
    "r_mouth": 0.14,
    "R": 1.0,
    "sigma": 0.1,
    "nr_grid": 100,
    "default_b0": 0.01,
    "default_D_P": 3e-5,
    "default_v_P": 0.02,
    "default_cdiff": 0.02,
    "default_D_N": 3e-5,
    "default_v_N": 0.02,
    "default_m0": 0.003,

    # Output and behavior
    "output_dir": "results/progenitor_param_sweep_steady_state",
    "print_progress": True,
    "verbose_solver": False,
}


def main() -> None:
    config = dict(CONFIG)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_output_run_dir(config["output_dir"], timestamp)

    config_path = run_dir / "simulation_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    result = run_parameter_grid_simulation(config)

    np.savez_compressed(
        run_dir / "simulation_raw.npz",
        r=result["r"],
        steady_profiles=result["steady_profiles"],
        parameter_table=result["parameter_table"],
        converged=result["converged"],
        steady_time=result["steady_time"],
        delta_inf=result["delta_inf"],
        delta_rel=result["delta_rel"],
    )

    with (run_dir / "run_dir.txt").open("w", encoding="utf-8") as f:
        f.write(str(run_dir))

    print(f"Saved simulation bundle: {run_dir / 'simulation_raw.npz'}")
    print(f"Saved simulation config: {config_path}")
    print(f"Converged runs: {int(np.sum(result['converged']))}/{len(result['converged'])}")


if __name__ == "__main__":
    main()
