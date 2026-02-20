#!/usr/bin/env python3
"""
ISSW-oriented radiation and gradient sensitivity playground.

Purpose
-------
1) Save a reproducible baseline snapshot for a station/time window
2) Run simple one-variable-at-a-time scenario tests:
   - Albedo sensitivity (constant alpha vs measured SW out baseline)
   - LW down sensitivity (parameterized and scaled measured LW down)
   - LW out nighttime sensitivity (reduced nighttime radiative loss)
3) Export ISSW-ready tables/figures:
   - Scenario metrics table (error + gradient metrics)
   - Measured albedo time series plot
   - Meteogram-style context plot with cloud/outlier windows
   - Scenario comparison bar plot
   - Baseline/full timeseries CSV + cloud-window CSV

This script does not edit the notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib cache writable in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path(__file__).resolve().parents[1] / ".mplconfig").resolve())
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str((Path(__file__).resolve().parents[1] / ".cache").resolve())
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Scenario:
    name: str
    group: str
    use_idealized_sw: bool
    use_idealized_lw: bool
    alpha: float | None = None
    lw_down_scale: float = 1.0
    lw_out_night_factor: float = 1.0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ISSW-style radiation sensitivity playground.")
    p.add_argument(
        "--station",
        default="senator_beck",
        choices=["senator_beck", "independence_pass", "berthoud_pass"],
    )
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--notebook", default="notebooks/snowmodel_USGS.ipynb")
    p.add_argument("--skin", choices=["on", "off"], default="on")
    p.add_argument("--skin-beta", type=float, default=0.20)
    p.add_argument(
        "--cloud-eps-threshold",
        type=float,
        default=0.85,
        help="Cloud-like condition from eps_atm_obs = LWdown / (sigma*Tair^4).",
    )
    p.add_argument(
        "--night-sw-threshold",
        type=float,
        default=20.0,
        help="SW down threshold (W/m2) for nighttime LW-out sensitivity.",
    )
    p.add_argument(
        "--outlier-threshold-c",
        type=float,
        default=4.0,
        help="Absolute baseline error threshold for outlier highlighting.",
    )
    p.add_argument("--outdir", default="results/issw_playground")
    return p.parse_args()


def load_notebook_code(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )
    marker = "# Define scenarios"
    if marker in code:
        code = code.split(marker, 1)[0]
    return code


def apply_runtime_overrides(code: str, station: str, start: str, end: str) -> str:
    code = re.sub(r'STATION\s*=\s*"[^"]+"', f'STATION = "{station}"', code)
    code = re.sub(r'START_DATE\s*=\s*"[^"]+"', f'START_DATE = "{start}"', code)
    code = re.sub(r'END_DATE\s*=\s*"[^"]+"', f'END_DATE = "{end}"', code)
    code = code.replace(
        "from usgs_collector import fetch_usgs_iv, simplify_columns, STATIONS",
        "from usgs_collector import fetch_usgs_iv, fetch_usgs_iv_cached, simplify_columns, STATIONS",
    )
    # Force cached data path for stable offline/reproducible runs.
    code = re.sub(
        r"df_raw\s*=\s*fetch_usgs_iv\(\s*STATIONS\[STATION\]\s*,\s*START_DATE\s*,\s*END_DATE\s*\)",
        'df_raw = fetch_usgs_iv_cached(STATIONS[STATION], START_DATE, END_DATE, cache_dir="data/usgs_cache", refresh=False)',
        code,
    )
    code = re.sub(r"show_plots\s*=\s*True", "show_plots = False", code)
    code = re.sub(r"generate_animation\s*=\s*True", "generate_animation = False", code)
    return code


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="baseline_measured",
            group="baseline",
            use_idealized_sw=False,
            use_idealized_lw=False,
            notes="Measured SW out and measured LW down.",
        ),
        Scenario(
            name="albedo_const_0p70",
            group="albedo",
            use_idealized_sw=True,
            use_idealized_lw=False,
            alpha=0.70,
            notes="Constant albedo alpha=0.70.",
        ),
        Scenario(
            name="albedo_const_0p80",
            group="albedo",
            use_idealized_sw=True,
            use_idealized_lw=False,
            alpha=0.80,
            notes="Constant albedo alpha=0.80.",
        ),
        Scenario(
            name="albedo_const_0p90",
            group="albedo",
            use_idealized_sw=True,
            use_idealized_lw=False,
            alpha=0.90,
            notes="Constant albedo alpha=0.90.",
        ),
        Scenario(
            name="lwdown_parameterized",
            group="lwdown",
            use_idealized_sw=False,
            use_idealized_lw=True,
            notes="Parameterized LW down from Tair/RH.",
        ),
        Scenario(
            name="lwdown_scaled_0p85",
            group="lwdown",
            use_idealized_sw=False,
            use_idealized_lw=False,
            lw_down_scale=0.85,
            notes="Measured LW down scaled to 85%.",
        ),
        Scenario(
            name="lwdown_scaled_1p15",
            group="lwdown",
            use_idealized_sw=False,
            use_idealized_lw=False,
            lw_down_scale=1.15,
            notes="Measured LW down scaled to 115%.",
        ),
        Scenario(
            name="lwout_night_factor_0p95",
            group="lwout_night",
            use_idealized_sw=False,
            use_idealized_lw=False,
            lw_out_night_factor=0.95,
            notes="Reduce nighttime LW out by 5%.",
        ),
        Scenario(
            name="lwout_night_factor_0p90",
            group="lwout_night",
            use_idealized_sw=False,
            use_idealized_lw=False,
            lw_out_night_factor=0.90,
            notes="Reduce nighttime LW out by 10%.",
        ),
    ]


def to_surface_series_c(t_layers_k: np.ndarray, use_skin: bool, beta: float) -> np.ndarray:
    if (not use_skin) or (t_layers_k.shape[0] < 2):
        return t_layers_k[-1, :] - 273.15
    t_top = t_layers_k[-1, :]
    t_below = t_layers_k[-2, :]
    grad = np.clip(t_top - t_below, -2.0, 2.0)
    t_skin = np.clip(t_top + beta * grad, 223.15, 273.15)
    return t_skin - 273.15


def contiguous_windows(mask: np.ndarray, timestamps: pd.DatetimeIndex, max_gap_h: float = 2.0) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    windows: list[tuple[int, int]] = []
    i0 = idx[0]
    last = idx[0]
    for i in idx[1:]:
        gap_h = (timestamps[i] - timestamps[last]).total_seconds() / 3600.0
        if gap_h <= max_gap_h:
            last = i
            continue
        windows.append((i0, last))
        i0 = i
        last = i
    windows.append((i0, last))
    return windows


def make_radiative_function(
    ns: dict,
    *,
    lw_source: str,  # "measured" or "idealized"
    lw_down_scale: float,
    lw_out_night_factor: float,
    night_sw_threshold: float,
):
    def _func(t_sec, t_surf_k):
        if lw_source == "idealized":
            t_air_k = float(ns["measured_air_temp"](t_sec))
            eps_atm = float(ns["compute_atmospheric_emissivity"](t_sec))
            lw_down = eps_atm * ns["sigma"] * (t_air_k**4)
        else:
            lw_down = float(ns["LW_in_interp"](t_sec))

        lw_down = max(0.0, lw_down * lw_down_scale)

        lw_factor = 1.0
        if float(ns["SW_in_interp"](t_sec)) < night_sw_threshold:
            lw_factor *= lw_out_night_factor

        lw_up = float(ns["eps_snow"]) * ns["sigma"] * (t_surf_k**4) * lw_factor
        lw_net = lw_down - lw_up

        # Returned SW is not used by dT_dt for conduction source, but keep signature.
        sw_net = max(float(ns["SW_in_interp"](t_sec) - ns["SW_out_interp"](t_sec)), 0.0)
        return lw_net, sw_net

    return _func


def evaluate_scenario(
    ns: dict,
    scenario: Scenario,
    *,
    use_skin: bool,
    skin_beta: float,
    t_eval: np.ndarray,
    obs_c: np.ndarray,
    cloud_mask: np.ndarray,
    night_sw_threshold: float,
):
    orig_alpha = ns["alpha_snow"]
    orig_compute_rad = ns["compute_radiative_fluxes"]
    orig_compute_ideal = ns["compute_idealized_radiative_fluxes"]

    try:
        if scenario.alpha is not None:
            ns["alpha_snow"] = float(scenario.alpha)

        # Patch both measured/idealized LW functions so scenario toggles stay safe.
        ns["compute_radiative_fluxes"] = make_radiative_function(
            ns,
            lw_source="measured",
            lw_down_scale=float(scenario.lw_down_scale),
            lw_out_night_factor=float(scenario.lw_out_night_factor),
            night_sw_threshold=night_sw_threshold,
        )
        ns["compute_idealized_radiative_fluxes"] = make_radiative_function(
            ns,
            lw_source="idealized",
            lw_down_scale=float(scenario.lw_down_scale),
            lw_out_night_factor=float(scenario.lw_out_night_factor),
            night_sw_threshold=night_sw_threshold,
        )

        t_solver, t_layers, fluxes = ns["run_snow_model"](
            scenario.use_idealized_sw, scenario.use_idealized_lw
        )
        model_solver_c = to_surface_series_c(t_layers, use_skin=use_skin, beta=skin_beta)
        model_eval_c = np.interp(t_eval, t_solver, model_solver_c)
        err = model_eval_c - obs_c

        hours = (t_eval % (24.0 * 3600.0)) / 3600.0
        day_mask = (hours >= 6.0) & (hours <= 18.0)
        night_mask = ~day_mask
        morning_mask = (hours >= 9.0) & (hours <= 11.0)
        clear_mask = ~cloud_mask

        top_eval_k = np.interp(t_eval, t_solver, t_layers[-1, :])
        bottom_eval_k = np.interp(t_eval, t_solver, t_layers[0, :])
        if t_layers.shape[0] > 1:
            below_top_eval_k = np.interp(t_eval, t_solver, t_layers[-2, :])
        else:
            below_top_eval_k = top_eval_k.copy()
        h_eval = np.maximum(np.asarray(ns["snow_depth_interp"](t_eval), dtype=float), 0.10)
        dz_eval = np.maximum(h_eval / float(t_layers.shape[0]), 0.01)

        bulk_grad_cpm = (top_eval_k - bottom_eval_k) / h_eval
        surf_grad_cpm = (top_eval_k - below_top_eval_k) / dz_eval

        qrad = np.interp(t_eval, t_solver, np.asarray(fluxes["net_radiative"], dtype=float))
        qsen = np.interp(t_eval, t_solver, np.asarray(fluxes["sensible"], dtype=float))
        qlat = np.interp(t_eval, t_solver, np.asarray(fluxes["latent"], dtype=float))
        qnet = np.interp(t_eval, t_solver, np.asarray(fluxes["net_energy"], dtype=float))

        metrics = {
            "scenario": scenario.name,
            "group": scenario.group,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "Bias": float(np.mean(err)),
            "Correlation": float(np.corrcoef(obs_c, model_eval_c)[0, 1]),
            "Day_Bias": float(np.mean(err[day_mask])),
            "Night_Bias": float(np.mean(err[night_mask])),
            "Morning_9_11_Bias": float(np.mean(err[morning_mask])) if morning_mask.any() else np.nan,
            "Night_RMSE": float(np.sqrt(np.mean(err[night_mask] ** 2))),
            "Cloud_RMSE": float(np.sqrt(np.mean(err[cloud_mask] ** 2))) if cloud_mask.any() else np.nan,
            "Clear_RMSE": float(np.sqrt(np.mean(err[clear_mask] ** 2))) if clear_mask.any() else np.nan,
            "Mean_abs_bulk_grad_Cpm": float(np.mean(np.abs(bulk_grad_cpm))),
            "P95_abs_bulk_grad_Cpm": float(np.nanpercentile(np.abs(bulk_grad_cpm), 95)),
            "Max_abs_bulk_grad_Cpm": float(np.nanmax(np.abs(bulk_grad_cpm))),
            "Pct_abs_bulk_grad_ge_10Cpm": float(100.0 * np.mean(np.abs(bulk_grad_cpm) >= 10.0)),
            "Mean_abs_surf_grad_Cpm": float(np.mean(np.abs(surf_grad_cpm))),
            "P95_abs_surf_grad_Cpm": float(np.nanpercentile(np.abs(surf_grad_cpm), 95)),
            "notes": scenario.notes,
        }

        series = {
            "t_solver": t_solver,
            "t_eval": t_eval,
            "model_eval_c": model_eval_c,
            "obs_c": obs_c,
            "error_c": err,
            "bulk_grad_cpm": bulk_grad_cpm,
            "surf_grad_cpm": surf_grad_cpm,
            "qrad_wm2": qrad,
            "qsen_wm2": qsen,
            "qlat_wm2": qlat,
            "qnet_wm2": qnet,
        }
        return metrics, series
    finally:
        ns["alpha_snow"] = orig_alpha
        ns["compute_radiative_fluxes"] = orig_compute_rad
        ns["compute_idealized_radiative_fluxes"] = orig_compute_ideal


def plot_measured_albedo(
    timestamps: pd.DatetimeIndex,
    sw_down: np.ndarray,
    sw_up: np.ndarray,
    albedo_measured: np.ndarray,
    out_png: Path,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(timestamps, sw_down, color="goldenrod", linewidth=1.8, label="SW down (W/m²)")
    ax1.plot(timestamps, sw_up, color="orange", linewidth=1.2, label="SW up (W/m²)")
    ax1.set_ylabel("SW Radiation (W/m²)")
    ax1.set_title("Measured Shortwave Radiation")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(timestamps, albedo_measured, color="black", linewidth=1.0, label="Measured albedo SWup/SWdown")
    ax2.axhline(0.8, color="tab:blue", linestyle="--", linewidth=1, label="alpha=0.80")
    ax2.set_ylabel("Albedo")
    ax2.set_xlabel("Time")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title("Measured Albedo Evolution")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_meteogram(
    timestamps: pd.DatetimeIndex,
    *,
    tair_c: np.ndarray,
    tsurf_obs_c: np.ndarray,
    tsurf_model_c: np.ndarray,
    err_c: np.ndarray,
    sw_down: np.ndarray,
    sw_up: np.ndarray,
    lw_down: np.ndarray,
    eps_atm_obs: np.ndarray,
    wind_ms: np.ndarray,
    rh_pct: np.ndarray,
    snow_depth_m: np.ndarray,
    albedo_measured: np.ndarray,
    cloud_mask: np.ndarray,
    outlier_mask: np.ndarray,
    out_png: Path,
):
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)

    # 1) Temperature panel
    ax = axes[0]
    ax.plot(timestamps, tair_c, color="tab:blue", linewidth=1.2, label="Air temp (C)")
    ax.plot(timestamps, tsurf_obs_c, color="black", linewidth=1.4, label="Surface obs (C)")
    ax.plot(timestamps, tsurf_model_c, color="tab:red", linewidth=1.2, alpha=0.9, label="Surface model (C)")
    ax.set_ylabel("Temp (C)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    ax.set_title("Meteogram Context: Forcing + Baseline Model Response")

    # 2) Shortwave + albedo
    ax = axes[1]
    ax.plot(timestamps, sw_down, color="goldenrod", linewidth=1.2, label="SW down")
    ax.plot(timestamps, sw_up, color="orange", linewidth=1.0, label="SW up")
    ax.set_ylabel("SW (W/m²)")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.plot(timestamps, albedo_measured, color="gray", linewidth=0.9, alpha=0.8, label="Measured albedo")
    axr.set_ylabel("Albedo")
    axr.set_ylim(-0.05, 1.15)

    # 3) Longwave + cloud proxy
    ax = axes[2]
    ax.plot(timestamps, lw_down, color="purple", linewidth=1.2, label="LW down")
    ax.set_ylabel("LW down (W/m²)")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.plot(timestamps, eps_atm_obs, color="tab:green", linewidth=1.0, label="eps_atm_obs")
    axr.axhline(0.85, color="tab:green", linestyle="--", linewidth=1)
    axr.set_ylabel("eps_atm_obs")
    axr.set_ylim(0.5, 1.2)

    # 4) Wind + RH
    ax = axes[3]
    ax.plot(timestamps, wind_ms, color="tab:cyan", linewidth=1.1, label="Wind (m/s)")
    ax.set_ylabel("Wind (m/s)")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.plot(timestamps, rh_pct, color="tab:olive", linewidth=1.0, alpha=0.9, label="RH (%)")
    axr.set_ylabel("RH (%)")
    axr.set_ylim(0, 100)

    # 5) Snow depth + error
    ax = axes[4]
    ax.plot(timestamps, snow_depth_m, color="tab:blue", linewidth=1.4, label="Snow depth (m)")
    ax.set_ylabel("Snow depth (m)")
    ax.grid(True, alpha=0.3)
    axr = ax.twinx()
    axr.plot(timestamps, err_c, color="tab:red", linewidth=1.0, alpha=0.8, label="Model - Obs (C)")
    axr.axhline(0.0, color="k", linestyle="--", linewidth=0.9, alpha=0.7)
    axr.set_ylabel("Error (C)")
    axes[4].set_xlabel("Time")

    # Shade cloudy and outlier windows across all panels.
    for i in range(len(timestamps)):
        if cloud_mask[i]:
            for ax in axes:
                ax.axvspan(
                    timestamps[i] - pd.Timedelta(minutes=20),
                    timestamps[i] + pd.Timedelta(minutes=20),
                    color="lightgray",
                    alpha=0.05,
                    linewidth=0,
                )
        if outlier_mask[i]:
            axes[0].axvspan(
                timestamps[i] - pd.Timedelta(minutes=20),
                timestamps[i] + pd.Timedelta(minutes=20),
                color="red",
                alpha=0.08,
                linewidth=0,
            )

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_scenario_comparison(df: pd.DataFrame, out_png: Path):
    d = df.copy()
    base = d.loc[d["scenario"] == "baseline_measured"].iloc[0]
    d["RMSE_delta"] = d["RMSE"] - float(base["RMSE"])
    d["Night_Bias_delta"] = d["Night_Bias"] - float(base["Night_Bias"])
    d["P95_bulk_grad_delta"] = d["P95_abs_bulk_grad_Cpm"] - float(base["P95_abs_bulk_grad_Cpm"])

    d = d.sort_values("RMSE_delta")
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    axes[0].barh(x, d["RMSE_delta"], color="tab:blue", alpha=0.8)
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("RMSE Change vs Baseline")
    axes[0].set_xlabel("Delta RMSE (C)")

    axes[1].barh(x, d["Night_Bias_delta"], color="tab:purple", alpha=0.8)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Night Bias Change vs Baseline")
    axes[1].set_xlabel("Delta Night Bias (C)")

    axes[2].barh(x, d["P95_bulk_grad_delta"], color="tab:green", alpha=0.8)
    axes[2].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[2].set_title("P95 |Bulk Gradient| Change")
    axes[2].set_xlabel("Delta P95 |dT/dz| (C/m)")

    axes[0].set_yticks(x, d["scenario"])
    axes[1].set_yticks(x, [])
    axes[2].set_yticks(x, [])

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    nb_path = (repo_root / args.notebook).resolve()
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not nb_path.exists():
        print(f"ERROR: notebook not found: {nb_path}")
        return 1

    notebooks_dir = repo_root / "notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    code = load_notebook_code(nb_path)
    code = apply_runtime_overrides(code, args.station, args.start, args.end)

    ns: dict = {}
    try:
        exec(code, ns, ns)
    except Exception as exc:
        print("ERROR: failed executing notebook setup.")
        print(type(exc).__name__, exc)
        return 2

    required = [
        "df",
        "times_sec",
        "run_snow_model",
        "usgs_temp_obs_interp",
        "measured_air_temp",
        "SW_in_interp",
        "SW_out_interp",
        "LW_in_interp",
        "RH_arr_interp",
        "get_wind_speed",
        "snow_depth_interp",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        print(f"ERROR: missing required notebook objects: {missing}")
        return 3

    use_skin = args.skin == "on"
    t_eval = np.asarray(ns["times_sec"], dtype=float)
    timestamps = pd.to_datetime(ns["df"].index[0]) + pd.to_timedelta(t_eval, unit="s")
    obs_c = np.asarray(ns["usgs_temp_obs_interp"](t_eval), dtype=float)

    tair_c = np.asarray(ns["measured_air_temp"](t_eval), dtype=float) - 273.15
    rh_pct = np.asarray(ns["RH_arr_interp"](t_eval), dtype=float)
    wind_ms = np.asarray([float(ns["get_wind_speed"](t)) for t in t_eval], dtype=float)
    sw_down = np.asarray(ns["SW_in_interp"](t_eval), dtype=float)
    sw_up = np.asarray(ns["SW_out_interp"](t_eval), dtype=float)
    lw_down = np.asarray(ns["LW_in_interp"](t_eval), dtype=float)
    snow_depth_m = np.asarray(ns["snow_depth_interp"](t_eval), dtype=float)

    albedo_measured = np.full_like(sw_down, np.nan, dtype=float)
    np.divide(sw_up, sw_down, out=albedo_measured, where=sw_down > 20.0)
    albedo_measured = np.clip(albedo_measured, -0.2, 1.5)

    t_air_k = tair_c + 273.15
    eps_atm_obs = lw_down / (float(ns["sigma"]) * (t_air_k**4))
    eps_atm_obs = np.clip(eps_atm_obs, 0.0, 1.5)
    cloud_mask = eps_atm_obs >= float(args.cloud_eps_threshold)

    # Save baseline config snapshot first.
    constants = {}
    for key in [
        "CH",
        "CE",
        "k",
        "rho_snow",
        "eps_snow",
        "alpha_snow",
        "N",
        "snow_depth_m",
        "z_ref",
        "dt_max",
        "p_air",
        "rho_air",
        "c_pa",
        "use_twoBand",
        "SKIN_BETA",
    ]:
        if key in ns:
            val = ns[key]
            if isinstance(val, (int, float, bool, str)):
                constants[key] = val
            else:
                constants[key] = str(val)

    scenario_defs = [s.__dict__ for s in build_scenarios()]
    baseline_snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "station": args.station,
        "start_date": args.start,
        "end_date": args.end,
        "notebook_path": str(nb_path),
        "notebook_sha256": hashlib.sha256(nb_path.read_bytes()).hexdigest(),
        "n_rows_qc": int(len(ns["df"])),
        "time_start": str(ns["df"].index.min()),
        "time_end": str(ns["df"].index.max()),
        "use_skin": use_skin,
        "skin_beta": float(args.skin_beta),
        "cloud_eps_threshold": float(args.cloud_eps_threshold),
        "night_sw_threshold": float(args.night_sw_threshold),
        "constants": constants,
        "scenarios": scenario_defs,
    }
    baseline_json = outdir / f"baseline_snapshot_{args.station}_{args.start}_{args.end}.json"
    baseline_json.write_text(json.dumps(baseline_snapshot, indent=2))

    # Run scenario suite
    metrics_rows: list[dict] = []
    series_by_scenario: dict[str, dict] = {}
    for sc in build_scenarios():
        print(f"Running scenario: {sc.name}")
        metrics, series = evaluate_scenario(
            ns,
            sc,
            use_skin=use_skin,
            skin_beta=float(args.skin_beta),
            t_eval=t_eval,
            obs_c=obs_c,
            cloud_mask=cloud_mask,
            night_sw_threshold=float(args.night_sw_threshold),
        )
        metrics_rows.append(metrics)
        series_by_scenario[sc.name] = series

    metrics_df = pd.DataFrame(metrics_rows)
    if "baseline_measured" not in set(metrics_df["scenario"].values):
        print("ERROR: baseline scenario missing from scenario table.")
        return 4
    base = metrics_df.loc[metrics_df["scenario"] == "baseline_measured"].iloc[0]
    for col in [
        "MAE",
        "RMSE",
        "Bias",
        "Night_Bias",
        "Cloud_RMSE",
        "Clear_RMSE",
        "P95_abs_bulk_grad_Cpm",
    ]:
        metrics_df[f"{col}_delta_from_baseline"] = metrics_df[col] - float(base[col])
    metrics_df = metrics_df.sort_values(["group", "RMSE"]).reset_index(drop=True)

    metrics_csv = outdir / f"scenario_metrics_{args.station}_{args.start}_{args.end}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Build baseline timeseries output
    baseline_series = series_by_scenario["baseline_measured"]
    outlier_mask = np.abs(baseline_series["error_c"]) >= float(args.outlier_threshold_c)

    ts_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "time_sec": t_eval,
            "air_temp_c": tair_c,
            "surface_temp_obs_c": obs_c,
            "surface_temp_model_baseline_c": baseline_series["model_eval_c"],
            "error_baseline_c": baseline_series["error_c"],
            "sw_down_wm2": sw_down,
            "sw_up_wm2": sw_up,
            "albedo_measured": albedo_measured,
            "lw_down_wm2": lw_down,
            "eps_atm_obs": eps_atm_obs,
            "cloud_like": cloud_mask.astype(int),
            "wind_ms": wind_ms,
            "rh_pct": rh_pct,
            "snow_depth_m": snow_depth_m,
            "bulk_grad_baseline_cpm": baseline_series["bulk_grad_cpm"],
            "surf_grad_baseline_cpm": baseline_series["surf_grad_cpm"],
            "qrad_baseline_wm2": baseline_series["qrad_wm2"],
            "qsen_baseline_wm2": baseline_series["qsen_wm2"],
            "qlat_baseline_wm2": baseline_series["qlat_wm2"],
            "qnet_baseline_wm2": baseline_series["qnet_wm2"],
            "outlier_baseline": outlier_mask.astype(int),
        }
    )
    if "pressure_mmhg" in ns["df"].columns:
        ts_df["pressure_mmhg"] = np.asarray(ns["df"]["pressure_mmhg"].values, dtype=float)

    for sc_name, s in series_by_scenario.items():
        ts_df[f"error_{sc_name}_c"] = s["error_c"]

    timeseries_csv = outdir / f"timeseries_{args.station}_{args.start}_{args.end}.csv"
    ts_df.to_csv(timeseries_csv, index=False)

    # Daily albedo table (midday only for robust SW geometry)
    hour = pd.to_datetime(ts_df["timestamp"]).dt.hour
    midday = (hour >= 10) & (hour <= 14) & (ts_df["sw_down_wm2"] > 100.0)
    albedo_daily = (
        ts_df.loc[midday, ["timestamp", "albedo_measured"]]
        .assign(date=lambda d: pd.to_datetime(d["timestamp"]).dt.date)
        .groupby("date", as_index=False)["albedo_measured"]
        .median()
        .rename(columns={"albedo_measured": "albedo_midday_median"})
    )
    albedo_daily_csv = outdir / f"albedo_daily_{args.station}_{args.start}_{args.end}.csv"
    albedo_daily.to_csv(albedo_daily_csv, index=False)

    # Cloud windows summary
    cloud_windows = contiguous_windows(cloud_mask, pd.DatetimeIndex(timestamps), max_gap_h=2.0)
    cloud_rows = []
    for idx, (i0, i1) in enumerate(cloud_windows, start=1):
        sub = ts_df.iloc[i0 : i1 + 1]
        dur_h = (pd.to_datetime(sub["timestamp"].iloc[-1]) - pd.to_datetime(sub["timestamp"].iloc[0])).total_seconds() / 3600.0
        cloud_rows.append(
            {
                "cloud_window_id": idx,
                "start": sub["timestamp"].iloc[0],
                "end": sub["timestamp"].iloc[-1],
                "duration_h": float(dur_h),
                "mean_eps_atm_obs": float(sub["eps_atm_obs"].mean()),
                "mean_lw_down_wm2": float(sub["lw_down_wm2"].mean()),
                "mean_air_temp_c": float(sub["air_temp_c"].mean()),
                "mean_wind_ms": float(sub["wind_ms"].mean()),
                "mean_error_baseline_c": float(sub["error_baseline_c"].mean()),
                "rmse_baseline_c": float(np.sqrt(np.mean(sub["error_baseline_c"] ** 2))),
            }
        )
    cloud_df = pd.DataFrame(cloud_rows)
    cloud_csv = outdir / f"cloud_windows_{args.station}_{args.start}_{args.end}.csv"
    cloud_df.to_csv(cloud_csv, index=False)

    # Hourly error summary by scenario
    hourly_rows = []
    for sc_name, s in series_by_scenario.items():
        h = (t_eval % (24.0 * 3600.0)) / 3600.0
        for hour_i in range(24):
            m = (h >= hour_i) & (h < hour_i + 1)
            if not np.any(m):
                continue
            hourly_rows.append(
                {
                    "scenario": sc_name,
                    "hour_local": hour_i,
                    "mean_error_c": float(np.mean(s["error_c"][m])),
                    "rmse_c": float(np.sqrt(np.mean((s["error_c"][m]) ** 2))),
                }
            )
    hourly_df = pd.DataFrame(hourly_rows)
    hourly_csv = outdir / f"hourly_error_summary_{args.station}_{args.start}_{args.end}.csv"
    hourly_df.to_csv(hourly_csv, index=False)

    # Figures
    albedo_png = outdir / f"albedo_measured_{args.station}_{args.start}_{args.end}.png"
    meteogram_png = outdir / f"meteogram_{args.station}_{args.start}_{args.end}.png"
    scenario_png = outdir / f"scenario_comparison_{args.station}_{args.start}_{args.end}.png"

    plot_measured_albedo(
        pd.DatetimeIndex(timestamps),
        sw_down=sw_down,
        sw_up=sw_up,
        albedo_measured=albedo_measured,
        out_png=albedo_png,
    )
    plot_meteogram(
        pd.DatetimeIndex(timestamps),
        tair_c=tair_c,
        tsurf_obs_c=obs_c,
        tsurf_model_c=baseline_series["model_eval_c"],
        err_c=baseline_series["error_c"],
        sw_down=sw_down,
        sw_up=sw_up,
        lw_down=lw_down,
        eps_atm_obs=eps_atm_obs,
        wind_ms=wind_ms,
        rh_pct=rh_pct,
        snow_depth_m=snow_depth_m,
        albedo_measured=albedo_measured,
        cloud_mask=cloud_mask,
        outlier_mask=outlier_mask,
        out_png=meteogram_png,
    )
    plot_scenario_comparison(metrics_df, scenario_png)

    print("\nTop scenarios by RMSE")
    print("=" * 80)
    print(
        metrics_df.sort_values("RMSE")[
            [
                "scenario",
                "group",
                "MAE",
                "RMSE",
                "Bias",
                "Correlation",
                "Day_Bias",
                "Night_Bias",
                "Morning_9_11_Bias",
                "P95_abs_bulk_grad_Cpm",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved Files")
    print("=" * 80)
    for p in [
        baseline_json,
        metrics_csv,
        timeseries_csv,
        albedo_daily_csv,
        cloud_csv,
        hourly_csv,
        albedo_png,
        meteogram_png,
        scenario_png,
    ]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
