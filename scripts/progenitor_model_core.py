#!/usr/bin/env python3
"""Shared core utilities for the steady-state progenitor sweep."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def as_float_list(values: object, name: str) -> List[float]:
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"CONFIG['{name}'] must be a non-empty list.")
    return [float(v) for v in values]


def birth_rate(r: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    return params["b_0"] * np.exp(-((r - params["R"]) ** 2) / (params["sigma"] ** 2))


def differentiation_rate(r: np.ndarray, params: Dict[str, object]) -> np.ndarray:
    return np.full_like(r, params["d"], dtype=float)


def radial_velocity_P(params: Dict[str, object]) -> float:
    return float(params["v_P"])


def conservative_polar_transport(u: np.ndarray, r: np.ndarray, dr: float, D: float, v: float) -> np.ndarray:
    n = len(u)

    r_faces = np.empty(n + 1)
    r_faces[1:-1] = 0.5 * (r[:-1] + r[1:])
    r_faces[0] = r[0]
    r_faces[-1] = r[-1]

    u_faces = np.empty(n + 1)
    u_faces[1:-1] = 0.5 * (u[:-1] + u[1:])
    u_faces[0] = u[0]
    u_faces[-1] = u[-1]

    u_r_faces = np.zeros(n + 1)
    u_r_faces[1:-1] = (u[1:] - u[:-1]) / dr

    flux_faces = r_faces * (D * u_r_faces + v * u_faces)
    flux_faces[0] = 0.0
    flux_faces[-1] = 0.0

    divergence = (flux_faces[1:] - flux_faces[:-1]) / (r * dr)
    return divergence


def dP_dt(P: np.ndarray, r: np.ndarray, dr: float, params: Dict[str, object]) -> np.ndarray:
    transport = conservative_polar_transport(
        P,
        r,
        dr,
        D=float(params["D_P"]),
        v=radial_velocity_P(params),
    )
    birth = birth_rate(r, params)
    d = differentiation_rate(r, params)
    return transport + birth - d * P


def simulate_progenitors_only(
    params: Dict[str, object],
    T_final: float,
    dt: float,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    _ = verbose
    r_mouth = float(params["r_mouth"])
    R = float(params["R"])
    nr_grid = int(params["nr_grid"])

    r = np.linspace(r_mouth, R, nr_grid)
    dr = float(r[1] - r[0])

    nt = int(np.ceil(T_final / dt)) + 1
    times = np.linspace(0, T_final, nt)

    P0 = params["P_initial"]
    P = P0(r) if callable(P0) else np.array(P0, dtype=float).copy()

    P_history = np.zeros((nt, nr_grid), dtype=float)
    P_history[0, :] = P

    for n in range(1, nt):
        P = P + dt * dP_dt(P, r, dr, params)
        P_history[n, :] = P

    return {
        "r": r,
        "P_history": P_history,
        "times": times,
    }


def default_model_params(config: Dict[str, object]) -> Dict[str, object]:
    return {
        "r_mouth": float(config["r_mouth"]),
        "R": float(config["R"]),
        "sigma": float(config["sigma"]),
        "nr_grid": int(config["nr_grid"]),
        "D_P": float(config["default_D_P"]),
        "D_N": float(config["default_D_N"]),
        "v_P": float(config["default_v_P"]),
        "v_N": float(config["default_v_N"]),
        "b_0": float(config["default_b0"]),
        "d": float(config["default_cdiff"]),
        "m_0": float(config["default_m0"]),
        "P_initial": lambda r: np.zeros_like(r),
        "N_initial": lambda r: np.zeros_like(r),
    }


def build_params(base: Dict[str, object], b0: float, D_P: float, v_P: float, c_diff: float) -> Dict[str, object]:
    params = dict(base)
    params["b_0"] = b0
    params["D_P"] = D_P
    params["v_P"] = v_P
    params["d"] = c_diff
    return params


def run_until_steady_state(
    params: Dict[str, object],
    dt: float,
    chunk_time: float,
    min_time: float,
    max_time: float,
    steady_tol: float,
    check_window: int,
    verbose: bool,
) -> Dict[str, object]:
    total_time = 0.0
    current_params = dict(params)

    last_delta_inf = math.inf
    last_delta_rel = math.inf
    r = None
    P_final = None

    while total_time < max_time:
        solution = simulate_progenitors_only(
            current_params,
            T_final=chunk_time,
            dt=dt,
            verbose=verbose,
        )

        r = solution["r"]
        P_history = solution["P_history"]
        P_final = P_history[-1].copy()
        total_time += chunk_time

        if P_history.shape[0] < 2:
            raise RuntimeError("Simulation chunk produced insufficient time points for convergence check.")

        window = min(check_window, P_history.shape[0] - 1)
        recent = P_history[-(window + 1) :]
        diffs = np.diff(recent, axis=0)

        per_step_inf = np.max(np.abs(diffs), axis=1) / dt
        per_step_l2 = np.linalg.norm(diffs, axis=1)
        denom = np.linalg.norm(recent[1:], axis=1)
        per_step_rel = per_step_l2 / np.maximum(denom, 1e-12)

        last_delta_inf = float(np.max(per_step_inf))
        last_delta_rel = float(np.max(per_step_rel))

        if not np.all(np.isfinite(P_final)):
            return {
                "converged": False,
                "r": r,
                "P_ss": P_final,
                "steady_time": total_time,
                "delta_inf": last_delta_inf,
                "delta_rel": last_delta_rel,
                "reason": "non_finite_values",
            }

        if total_time >= min_time and last_delta_inf < steady_tol:
            return {
                "converged": True,
                "r": r,
                "P_ss": P_final,
                "steady_time": total_time,
                "delta_inf": last_delta_inf,
                "delta_rel": last_delta_rel,
                "reason": "steady_tol_reached",
            }

        current_params["P_initial"] = P_final

    return {
        "converged": False,
        "r": r,
        "P_ss": P_final,
        "steady_time": total_time,
        "delta_inf": last_delta_inf,
        "delta_rel": last_delta_rel,
        "reason": "max_time_reached",
    }


def ensure_output_run_dir(base_output_dir: object, timestamp: str) -> Path:
    run_dir = Path(str(base_output_dir)) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir(base_output_dir: object) -> Path:
    base_dir = Path(str(base_output_dir))
    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {base_dir}")

    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {base_dir}")

    return sorted(candidates)[-1]


def compute_steady_birth_metrics(r: np.ndarray, P_ss: np.ndarray, c_diff: float) -> Dict[str, float]:
    dr = float(r[1] - r[0])
    P_nonneg = np.clip(P_ss, 0.0, None)
    lambda_birth_ss = c_diff * P_nonneg

    weights = r * dr
    weighted_birth = lambda_birth_ss * weights
    total_birth_rate = float(np.sum(weighted_birth))

    if total_birth_rate <= 0.0:
        return {
            "mean_birth_radius": np.nan,
            "var_birth_radius": np.nan,
            "total_birth_rate": 0.0,
            "max_P_ss": float(np.max(P_nonneg)),
            "min_P_ss": float(np.min(P_ss)),
        }

    mean_birth_radius = float(np.sum(r * weighted_birth) / total_birth_rate)
    var_birth_radius = float(np.sum(((r - mean_birth_radius) ** 2) * weighted_birth) / total_birth_rate)

    return {
        "mean_birth_radius": mean_birth_radius,
        "var_birth_radius": var_birth_radius,
        "total_birth_rate": total_birth_rate,
        "max_P_ss": float(np.max(P_nonneg)),
        "min_P_ss": float(np.min(P_ss)),
    }


def run_parameter_grid_simulation(config: Dict[str, object]) -> Dict[str, object]:
    b0_values = as_float_list(config["b0_values"], "b0_values")
    D_P_values = as_float_list(config["D_P_values"], "D_P_values")
    v_P_values = as_float_list(config["v_P_values"], "v_P_values")
    c_diff_values = as_float_list(config["cdiff_values"], "cdiff_values")

    base = default_model_params(config)

    rows = []
    profiles = []
    r = None
    grid = list(itertools.product(b0_values, D_P_values, v_P_values, c_diff_values))

    for idx, (b0, D_P, v_P, c_diff) in enumerate(grid, start=1):
        if bool(config["print_progress"]):
            print(
                f"[{idx}/{len(grid)}] b_0={b0:.4g}, D_P={D_P:.4g}, v_P={v_P:.4g}, c_diff={c_diff:.4g}",
                flush=True,
            )

        params = build_params(base, b0=b0, D_P=D_P, v_P=v_P, c_diff=c_diff)
        steady = run_until_steady_state(
            params=params,
            dt=float(config["dt"]),
            chunk_time=float(config["chunk_time"]),
            min_time=float(config["min_time"]),
            max_time=float(config["max_time"]),
            steady_tol=float(config["steady_tol"]),
            check_window=int(config["check_window"]),
            verbose=bool(config["verbose_solver"]),
        )

        r = steady["r"]
        profiles.append(np.array(steady["P_ss"], dtype=float))

        rows.append(
            {
                "b_0": b0,
                "D_P": D_P,
                "v_P": v_P,
                "c_diff": c_diff,
                "converged": bool(steady["converged"]),
                "steady_time": float(steady["steady_time"]),
                "delta_inf": float(steady["delta_inf"]),
                "delta_rel": float(steady["delta_rel"]),
                "stop_reason": steady["reason"],
            }
        )

    parameter_table = pd.DataFrame(rows).to_records(index=False)

    return {
        "r": np.array(r, dtype=float),
        "steady_profiles": np.vstack(profiles),
        "parameter_table": parameter_table,
        "converged": parameter_table["converged"],
        "steady_time": parameter_table["steady_time"],
        "delta_inf": parameter_table["delta_inf"],
        "delta_rel": parameter_table["delta_rel"],
    }
