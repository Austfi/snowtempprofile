#!/usr/bin/env python3
"""
Outlier diagnostics for notebooks/snowmodel_USGS.ipynb.

What it does:
1) Runs the measured-forcing scenario (Measured SW out, Measured LW in)
2) Computes model-observed surface temperature error
3) Flags outliers where |error| >= threshold (default 4 C)
4) Groups consecutive outlier points into events
5) Summarizes surrounding forcing, modeled fluxes, and solver step size
6) Writes CSV outputs for point-level and event-level review
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze large model error outlier events for USGS snow model notebook."
    )
    parser.add_argument(
        "--station",
        default="senator_beck",
        choices=["senator_beck", "independence_pass", "berthoud_pass"],
        help="Station key used in usgs_collector STATIONS.",
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--threshold-c",
        type=float,
        default=4.0,
        help="Absolute error threshold in C for outlier points.",
    )
    parser.add_argument(
        "--event-gap-hours",
        type=float,
        default=2.0,
        help="Maximum gap between outlier points to keep them in one event.",
    )
    parser.add_argument(
        "--skin",
        choices=["on", "off"],
        default="on",
        help="Use skin diagnostic (on) or top-layer center (off).",
    )
    parser.add_argument(
        "--skin-beta",
        type=float,
        default=0.20,
        help="Skin extrapolation factor if --skin on.",
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/snowmodel_USGS.ipynb",
        help="Notebook path.",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory for CSV outputs.",
    )
    return parser.parse_args()


def load_notebook_code(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    # Keep only setup/function definitions before scenario auto-run blocks.
    marker = "# Define scenarios"
    if marker in code:
        code = code.split(marker, 1)[0]
    return code


def apply_runtime_overrides(code: str, station: str, start: str, end: str) -> str:
    code = re.sub(r'STATION\s*=\s*"[^"]+"', f'STATION = "{station}"', code)
    code = re.sub(r'START_DATE\s*=\s*"[^"]+"', f'START_DATE = "{start}"', code)
    code = re.sub(r'END_DATE\s*=\s*"[^"]+"', f'END_DATE = "{end}"', code)
    code = re.sub(r"show_plots\s*=\s*True", "show_plots = False", code)
    code = re.sub(r"generate_animation\s*=\s*True", "generate_animation = False", code)
    return code


def model_surface_series_c(
    t_layers_k: np.ndarray, use_skin: bool, beta: float
) -> np.ndarray:
    if (not use_skin) or (t_layers_k.shape[0] < 2):
        return t_layers_k[-1, :] - 273.15
    t_top = t_layers_k[-1, :]
    t_below = t_layers_k[-2, :]
    grad = np.clip(t_top - t_below, -2.0, 2.0)
    t_skin = np.clip(t_top + beta * grad, 223.15, 273.15)
    return t_skin - 273.15


def group_outlier_events(outlier_times_sec: np.ndarray, gap_hours: float) -> list[list[int]]:
    if len(outlier_times_sec) == 0:
        return []
    gap_sec = gap_hours * 3600.0
    groups: list[list[int]] = [[0]]
    for i in range(1, len(outlier_times_sec)):
        if (outlier_times_sec[i] - outlier_times_sec[i - 1]) <= gap_sec:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def summarize_point_conditions(ns: dict, t_sec: float, model_c: float, obs_c: float, flux_idx: int, fluxes: dict) -> dict:
    tair_c = float(ns["measured_air_temp"](t_sec) - 273.15)
    rh = float(ns["RH_arr_interp"](t_sec))
    wind_ms = float(ns["get_wind_speed"](t_sec))
    sw_in = float(ns["SW_in_interp"](t_sec))
    sw_out = float(ns["SW_out_interp"](t_sec))
    lw_in = float(ns["LW_in_interp"](t_sec))
    h_snow = float(ns["snow_depth_interp"](t_sec))

    qrad = float(fluxes["net_radiative"][flux_idx])
    qsen = float(fluxes["sensible"][flux_idx])
    qlat = float(fluxes["latent"][flux_idx])
    qnet = float(fluxes["net_energy"][flux_idx])

    return {
        "error_c": float(model_c - obs_c),
        "model_c": float(model_c),
        "obs_c": float(obs_c),
        "air_temp_c": tair_c,
        "rh_pct": rh,
        "wind_ms": wind_ms,
        "sw_in_wm2": sw_in,
        "sw_out_wm2": sw_out,
        "lw_in_wm2": lw_in,
        "snow_depth_m": h_snow,
        "Qrad_wm2": qrad,
        "Qsen_wm2": qsen,
        "Qlat_wm2": qlat,
        "Qnet_wm2": qnet,
    }


def nearest_solver_index(solver_t: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """
    Map query times to nearest solver index.
    """
    pos = np.searchsorted(solver_t, t_query, side="left")
    pos = np.clip(pos, 0, len(solver_t) - 1)
    left = np.maximum(pos - 1, 0)
    right = pos
    choose_right = np.abs(solver_t[right] - t_query) <= np.abs(solver_t[left] - t_query)
    return np.where(choose_right, right, left)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = (repo_root / args.notebook).resolve()

    if not notebook_path.exists():
        print(f"ERROR: notebook not found: {notebook_path}")
        return 1

    notebooks_dir = repo_root / "notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    code = load_notebook_code(notebook_path)
    code = apply_runtime_overrides(code, args.station, args.start, args.end)

    ns: dict = {}
    try:
        exec(code, ns, ns)
    except Exception as exc:
        print("ERROR: failed while executing notebook setup/functions.")
        print(type(exc).__name__, exc)
        return 2

    required = [
        "run_snow_model",
        "usgs_temp_obs_interp",
        "measured_air_temp",
        "RH_arr_interp",
        "get_wind_speed",
        "SW_in_interp",
        "SW_out_interp",
        "LW_in_interp",
        "snow_depth_interp",
        "df",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        print("ERROR: missing required objects:", missing)
        return 3

    use_skin = args.skin == "on"

    # Run measured-forcing scenario
    t_solver, t_layers, fluxes_solver = ns["run_snow_model"](False, False)
    model_solver_c = model_surface_series_c(t_layers, use_skin=use_skin, beta=args.skin_beta)

    # Evaluate diagnostics on observation timestamps for cleaner event counting
    if "times_sec" not in ns:
        print("ERROR: times_sec not found from notebook setup.")
        return 4
    t_eval = np.asarray(ns["times_sec"], dtype=float)
    obs_c = ns["usgs_temp_obs_interp"](t_eval)
    model_c = np.interp(t_eval, t_solver, model_solver_c)
    err_c = model_c - obs_c

    qrad = np.interp(t_eval, t_solver, np.asarray(fluxes_solver["net_radiative"], dtype=float))
    qsen = np.interp(t_eval, t_solver, np.asarray(fluxes_solver["sensible"], dtype=float))
    qlat = np.interp(t_eval, t_solver, np.asarray(fluxes_solver["latent"], dtype=float))
    qnet = np.interp(t_eval, t_solver, np.asarray(fluxes_solver["net_energy"], dtype=float))
    fluxes = {
        "net_radiative": qrad,
        "sensible": qsen,
        "latent": qlat,
        "net_energy": qnet,
    }

    # Build timestamp base from loaded dataframe
    t0 = ns["df"].index[0]
    timestamps = pd.to_datetime(t0) + pd.to_timedelta(t_eval, unit="s")

    # Solver step-size context near each eval point
    dt_prev_solver = np.full_like(t_solver, np.nan, dtype=float)
    dt_next_solver = np.full_like(t_solver, np.nan, dtype=float)
    if len(t_solver) > 1:
        dt_prev_solver[1:] = np.diff(t_solver)
        dt_next_solver[:-1] = np.diff(t_solver)
    idx_near = nearest_solver_index(t_solver, t_eval)
    dt_prev = dt_prev_solver[idx_near]
    dt_next = dt_next_solver[idx_near]

    # Outlier points
    out_mask = np.abs(err_c) >= args.threshold_c
    out_idx = np.where(out_mask)[0]

    print("\nOutlier Threshold Summary")
    print("=" * 80)
    print(f"station={args.station} window={args.start} to {args.end}")
    print(f"skin={args.skin} skin_beta={args.skin_beta:.2f} threshold={args.threshold_c:.2f} C")
    print(f"total points={len(t_eval)} outlier points={len(out_idx)} ({100.0*len(out_idx)/len(t_eval):.2f}%)")

    point_rows = []
    for i in out_idx:
        row = summarize_point_conditions(ns, float(t_eval[i]), float(model_c[i]), float(obs_c[i]), int(i), fluxes)
        row["i"] = int(i)
        row["time_sec"] = float(t_eval[i])
        row["timestamp"] = timestamps[i]
        row["hour"] = float((t_eval[i] % (24 * 3600.0)) / 3600.0)
        row["day_or_night"] = "day" if 6.0 <= row["hour"] <= 18.0 else "night"
        row["dt_prev_s"] = float(dt_prev[i]) if np.isfinite(dt_prev[i]) else np.nan
        row["dt_next_s"] = float(dt_next[i]) if np.isfinite(dt_next[i]) else np.nan
        point_rows.append(row)

    points_df = pd.DataFrame(point_rows)

    # Event grouping
    event_rows = []
    if len(out_idx) > 0:
        out_times = t_eval[out_idx]
        groups = group_outlier_events(out_times, args.event_gap_hours)
        for ev_id, g in enumerate(groups, start=1):
            idx_local = out_idx[g]
            sub = points_df[points_df["i"].isin(idx_local)].sort_values("i")
            start_ts = sub["timestamp"].iloc[0]
            end_ts = sub["timestamp"].iloc[-1]
            dur_h = (sub["time_sec"].iloc[-1] - sub["time_sec"].iloc[0]) / 3600.0
            event_rows.append(
                {
                    "event_id": ev_id,
                    "n_points": int(len(sub)),
                    "start": start_ts,
                    "end": end_ts,
                    "duration_h": float(dur_h),
                    "mean_error_c": float(sub["error_c"].mean()),
                    "max_abs_error_c": float(sub["error_c"].abs().max()),
                    "mean_hour": float(sub["hour"].mean()),
                    "frac_day": float((sub["day_or_night"] == "day").mean()),
                    "mean_air_temp_c": float(sub["air_temp_c"].mean()),
                    "mean_rh_pct": float(sub["rh_pct"].mean()),
                    "mean_wind_ms": float(sub["wind_ms"].mean()),
                    "mean_sw_in_wm2": float(sub["sw_in_wm2"].mean()),
                    "mean_lw_in_wm2": float(sub["lw_in_wm2"].mean()),
                    "mean_snow_depth_m": float(sub["snow_depth_m"].mean()),
                    "mean_Qrad_wm2": float(sub["Qrad_wm2"].mean()),
                    "mean_Qsen_wm2": float(sub["Qsen_wm2"].mean()),
                    "mean_Qlat_wm2": float(sub["Qlat_wm2"].mean()),
                    "mean_Qnet_wm2": float(sub["Qnet_wm2"].mean()),
                    "mean_dt_prev_s": float(sub["dt_prev_s"].mean()),
                    "mean_dt_next_s": float(sub["dt_next_s"].mean()),
                }
            )

    events_df = pd.DataFrame(event_rows)

    # Compare outlier vs non-outlier conditions
    full_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "time_sec": t_eval,
            "error_c": err_c,
            "abs_error_c": np.abs(err_c),
            "air_temp_c": [float(ns["measured_air_temp"](t) - 273.15) for t in t_eval],
            "rh_pct": [float(ns["RH_arr_interp"](t)) for t in t_eval],
            "wind_ms": [float(ns["get_wind_speed"](t)) for t in t_eval],
            "sw_in_wm2": [float(ns["SW_in_interp"](t)) for t in t_eval],
            "sw_out_wm2": [float(ns["SW_out_interp"](t)) for t in t_eval],
            "lw_in_wm2": [float(ns["LW_in_interp"](t)) for t in t_eval],
            "snow_depth_m": [float(ns["snow_depth_interp"](t)) for t in t_eval],
            "Qrad_wm2": qrad,
            "Qsen_wm2": qsen,
            "Qlat_wm2": qlat,
            "Qnet_wm2": qnet,
            "dt_prev_s": dt_prev,
            "dt_next_s": dt_next,
        }
    )
    full_df["is_outlier"] = np.abs(full_df["error_c"]) >= args.threshold_c
    full_df["hour"] = (full_df["time_sec"] % (24.0 * 3600.0)) / 3600.0
    full_df["day_or_night"] = np.where(
        (full_df["hour"] >= 6.0) & (full_df["hour"] <= 18.0), "day", "night"
    )

    cols_cmp = [
        "air_temp_c",
        "rh_pct",
        "wind_ms",
        "sw_in_wm2",
        "lw_in_wm2",
        "snow_depth_m",
        "Qrad_wm2",
        "Qsen_wm2",
        "Qlat_wm2",
        "Qnet_wm2",
        "dt_prev_s",
        "dt_next_s",
    ]
    cmp_rows = []
    for c in cols_cmp:
        out_mean = float(full_df.loc[full_df["is_outlier"], c].mean()) if full_df["is_outlier"].any() else np.nan
        non_mean = float(full_df.loc[~full_df["is_outlier"], c].mean()) if (~full_df["is_outlier"]).any() else np.nan
        cmp_rows.append(
            {
                "variable": c,
                "outlier_mean": out_mean,
                "non_outlier_mean": non_mean,
                "difference": out_mean - non_mean if np.isfinite(out_mean) and np.isfinite(non_mean) else np.nan,
            }
        )
    cmp_df = pd.DataFrame(cmp_rows)

    # Print quick interpretation helpers
    print("\nOutlier Timing")
    print("=" * 80)
    if len(points_df) == 0:
        print("No outliers found at this threshold.")
    else:
        by_hour = points_df.groupby(points_df["hour"].round().astype(int)).size().reindex(range(24), fill_value=0)
        print("outliers by hour:")
        print(by_hour.to_string())
        print("\nTop 8 largest |error| points:")
        print(
            points_df.assign(abs_error_c=np.abs(points_df["error_c"]))
            .sort_values("abs_error_c", ascending=False)
            .head(8)[
                [
                    "timestamp",
                    "error_c",
                    "model_c",
                    "obs_c",
                    "air_temp_c",
                    "wind_ms",
                    "sw_in_wm2",
                    "lw_in_wm2",
                    "Qrad_wm2",
                    "Qsen_wm2",
                    "Qlat_wm2",
                    "Qnet_wm2",
                    "dt_prev_s",
                    "dt_next_s",
                ]
            ].to_string(index=False)
        )

    if len(events_df) > 0:
        print("\nEvent Summary (largest events first)")
        print("=" * 80)
        print(
            events_df.sort_values(["n_points", "max_abs_error_c"], ascending=[False, False])
            .head(12)
            .to_string(index=False)
        )

    print("\nOutlier vs Non-Outlier Mean Conditions")
    print("=" * 80)
    print(cmp_df.to_string(index=False))

    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.station}_{args.start}_{args.end}_thr{args.threshold_c:.1f}_skin{args.skin}_b{args.skin_beta:.2f}"

    points_path = outdir / f"outlier_points_{tag}.csv"
    events_path = outdir / f"outlier_events_{tag}.csv"
    compare_path = outdir / f"outlier_compare_{tag}.csv"
    full_path = outdir / f"full_series_{tag}.csv"

    points_df.to_csv(points_path, index=False)
    events_df.to_csv(events_path, index=False)
    cmp_df.to_csv(compare_path, index=False)
    full_df.to_csv(full_path, index=False)

    print("\nSaved Files")
    print("=" * 80)
    print(points_path)
    print(events_path)
    print(compare_path)
    print(full_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
