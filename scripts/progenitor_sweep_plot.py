#!/usr/bin/env python3
"""Plotting stage for the steady-state progenitor sweep.

This script reads the analysis table and makes one figure per output
variable. Each figure is a grid of heatmaps over D_P and v_P, with rows
indexed by b_0 and columns indexed by c_diff.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import seaborn as sns

from progenitor_model_core import latest_run_dir


CONFIG = {
    "output_dir": "results/progenitor_param_sweep_steady_state",
    "run_dir": None,
    "analysis_csv_name": "analysis_results.csv",
    "figure_names": ["mean_birth_radius", "var_birth_radius", "total_birth_rate"],
    "x_param": "D_P",
    "y_param": "v_P",
    "row_param": "b_0",
    "col_param": "c_diff",
    "cmap": "viridis",
}


def resolve_run_dir(config: dict) -> Path:
    run_dir = config.get("run_dir")
    if run_dir:
        return Path(str(run_dir))
    return latest_run_dir(config["output_dir"])


def sorted_unique(values: np.ndarray) -> list[float]:
    return sorted(set(float(v) for v in values))


def sci_label(value: float) -> str:
    return f"{value:.1e}"


def plot_metric_grid(df: pd.DataFrame, metric: str, config: dict, run_dir: Path) -> None:
    converged = df[df["converged"]].copy()
    if converged.empty:
        print(f"Skipping {metric}: no converged runs.")
        return

    row_values = sorted_unique(converged[config["row_param"]].to_numpy())
    col_values = sorted_unique(converged[config["col_param"]].to_numpy())

    n_rows = len(row_values)
    n_cols = len(col_values)

    fig = plt.figure(figsize=(4.6 * n_cols + 0.6, 4.1 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.05],
        wspace=0.2,
        hspace=0.25,
    )

    metric_values = converged[metric].to_numpy(dtype=float)
    vmin = np.nanmin(metric_values)
    vmax = np.nanmax(metric_values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=config["cmap"])
    sm.set_array([])

    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = fig.add_subplot(gs[i, j])
            subset = converged[
                np.isclose(converged[config["row_param"]], row_val)
                & np.isclose(converged[config["col_param"]], col_val)
            ]
            pivot = subset.pivot(index=config["y_param"], columns=config["x_param"], values=metric)
            pivot = pivot.sort_index(axis=0).sort_index(axis=1)

            sns.heatmap(
                pivot,
                ax=ax,
                cmap=config["cmap"],
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                square=False,
            )

            ax.set_xticklabels([sci_label(value) for value in pivot.columns], rotation=0)
            ax.set_yticklabels([sci_label(value) for value in pivot.index], rotation=0)

            if i == 0:
                ax.set_title(f"{config['col_param']}={sci_label(col_val)}")
            if j == 0:
                ax.set_ylabel(f"{config['row_param']}={sci_label(row_val)}\n{config['y_param']}")
            else:
                ax.set_ylabel(config["y_param"])
            if i == n_rows - 1:
                ax.set_xlabel(config["x_param"])
            else:
                ax.set_xlabel("")

    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(sm, cax=cax, label=metric)
    fig.suptitle(metric, y=1.01)
    fig.savefig(run_dir / f"{metric}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = dict(CONFIG)
    run_dir = resolve_run_dir(config)
    analysis_path = run_dir / str(config["analysis_csv_name"])
    df = pd.read_csv(analysis_path)

    for metric in config["figure_names"]:
        plot_metric_grid(df, metric, config, run_dir)
        print(f"Saved {metric}.png")


if __name__ == "__main__":
    main()
