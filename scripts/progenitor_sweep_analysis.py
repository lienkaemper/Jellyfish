#!/usr/bin/env python3
"""Analysis stage for the steady-state progenitor sweep.

This script reads the raw simulation bundle, computes the three output
variables at steady state, and writes a tabular analysis file for plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from progenitor_model_core import compute_steady_birth_metrics, latest_run_dir


# Edit this if you want to analyze a specific run directory.
# If left as None, the latest timestamped run directory is used.
CONFIG = {
    "output_dir": "results/progenitor_param_sweep_steady_state",
    "run_dir": None,
    "raw_bundle_name": "simulation_raw.npz",
    "analysis_csv_name": "analysis_results.csv",
    "analysis_summary_name": "analysis_summary.json",
}


def resolve_run_dir(config: dict) -> Path:
    run_dir = config.get("run_dir")
    if run_dir:
        return Path(str(run_dir))
    return latest_run_dir(config["output_dir"])


def main() -> None:
    config = dict(CONFIG)
    run_dir = resolve_run_dir(config)
    raw_path = run_dir / str(config["raw_bundle_name"])

    bundle = np.load(raw_path, allow_pickle=True)
    r = bundle["r"]
    steady_profiles = bundle["steady_profiles"]
    parameter_table = pd.DataFrame(bundle["parameter_table"])
    converged = bundle["converged"].astype(bool)
    steady_time = bundle["steady_time"].astype(float)
    delta_inf = bundle["delta_inf"].astype(float)
    delta_rel = bundle["delta_rel"].astype(float)

    rows = []
    for idx in range(len(parameter_table)):
        row = parameter_table.iloc[idx].to_dict()
        row.update(
            {
                "converged": bool(converged[idx]),
                "steady_time": float(steady_time[idx]),
                "delta_inf": float(delta_inf[idx]),
                "delta_rel": float(delta_rel[idx]),
            }
        )

        if row["converged"]:
            metrics = compute_steady_birth_metrics(r, steady_profiles[idx], row["c_diff"])
        else:
            metrics = {
                "mean_birth_radius": np.nan,
                "var_birth_radius": np.nan,
                "total_birth_rate": np.nan,
                "max_P_ss": float(np.max(steady_profiles[idx])),
                "min_P_ss": float(np.min(steady_profiles[idx])),
            }

        row.update(metrics)
        rows.append(row)

    analysis_df = pd.DataFrame(rows)
    analysis_path = run_dir / str(config["analysis_csv_name"])
    analysis_df.to_csv(analysis_path, index=False)

    summary = {
        "run_dir": str(run_dir),
        "n_runs": int(len(analysis_df)),
        "n_converged": int(analysis_df["converged"].sum()),
        "metrics": ["mean_birth_radius", "var_birth_radius", "total_birth_rate"],
    }
    summary_path = run_dir / str(config["analysis_summary_name"])
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis table: {analysis_path}")
    print(f"Saved analysis summary: {summary_path}")
    print(f"Converged runs: {summary['n_converged']}/{summary['n_runs']}")


if __name__ == "__main__":
    main()
