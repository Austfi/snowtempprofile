#!/usr/bin/env python3
"""
Compare multiple simple physics options for snowmodel_USGS notebook.

Uses cached USGS data through notebook configuration (fetch_usgs_iv_cached).
Focuses on measured forcing scenario and evaluates:
- MAE, RMSE, Bias, Correlation
- Day/Night bias
- Morning bias (09-11)
- Outlier count (|error| >= threshold)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Config:
    name: str
    k: float
    rho_snow: float
    kappa_vis: float
    stability_mode: str  # current, no_split, solar_boost, strong_night, neutral


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep simple physics options.")
    p.add_argument("--station", default="senator_beck", choices=["senator_beck", "independence_pass", "berthoud_pass"])
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-01-31")
    p.add_argument("--skin", choices=["on", "off"], default="on")
    p.add_argument("--skin-beta", type=float, default=0.20)
    p.add_argument("--outlier-threshold", type=float, default=4.0)
    p.add_argument("--notebook", default="notebooks/snowmodel_USGS.ipynb")
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
    code = re.sub(r"show_plots\s*=\s*True", "show_plots = False", code)
    code = re.sub(r"generate_animation\s*=\s*True", "generate_animation = False", code)
    return code


def model_surface_series_c(t_layers_k: np.ndarray, use_skin: bool, beta: float) -> np.ndarray:
    if (not use_skin) or (t_layers_k.shape[0] < 2):
        return t_layers_k[-1, :] - 273.15
    t_top = t_layers_k[-1, :]
    t_below = t_layers_k[-2, :]
    grad = np.clip(t_top - t_below, -2.0, 2.0)
    t_skin = np.clip(t_top + beta * grad, 223.15, 273.15)
    return t_skin - 273.15


def build_shortwave_func(ns: dict, kappa_vis: float):
    def shortwave_absorption_two_band(t_sec, z_faces):
        sw_net = ns["compute_shortwave_net"](t_sec)
        f_nir = 0.4
        f_vis = 1.0 - f_nir
        kappa_nir = 35.0
        kv = kappa_vis

        sw_nir_top = sw_net * f_nir
        sw_vis_top = sw_net * f_vis

        h_curr = z_faces[-1]
        d_faces = h_curr - z_faces
        d_top = d_faces[1:]
        d_bottom = d_faces[:-1]

        fin_nir = sw_nir_top * np.exp(-kappa_nir * d_top)
        fout_nir = sw_nir_top * np.exp(-kappa_nir * d_bottom)
        abs_nir = fin_nir - fout_nir

        fin_vis = sw_vis_top * np.exp(-kv * d_top)
        fout_vis = sw_vis_top * np.exp(-kv * d_bottom)
        abs_vis = fin_vis - fout_vis

        return np.maximum(abs_nir + abs_vis, 0.0)

    return shortwave_absorption_two_band


def build_turbulent_func(ns: dict, mode: str):
    def turbulent_fluxes(t_sec, T_surfK):
        T_airK = ns["measured_air_temp"](t_sec)
        U = max(float(ns["get_wind_speed"](t_sec)), 0.1)
        g = 9.81
        z_ref_local = float(ns["z_ref"])
        Ri_b = (g * z_ref_local * (T_airK - T_surfK)) / (T_airK * (U**2))

        if mode == "neutral":
            stability_factor = 1.0
        else:
            if T_airK > T_surfK:
                sw_in = float(ns["SW_in_interp"](t_sec))
                if mode == "current":
                    if sw_in > 50.0:
                        stability_factor = 1.0 / (1.0 + 5.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.35)
                    else:
                        stability_factor = 1.0 / (1.0 + 10.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.20)
                elif mode == "no_split":
                    stability_factor = 1.0 / (1.0 + 5.0 * max(Ri_b, 0.0))
                    stability_factor = max(stability_factor, 0.35)
                elif mode == "solar_boost":
                    if sw_in > 200.0:
                        stability_factor = 1.0 / (1.0 + 3.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.50)
                    elif sw_in > 50.0:
                        stability_factor = 1.0 / (1.0 + 5.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.35)
                    else:
                        stability_factor = 1.0 / (1.0 + 10.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.20)
                elif mode == "strong_night":
                    if sw_in > 50.0:
                        stability_factor = 1.0 / (1.0 + 5.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.35)
                    else:
                        stability_factor = 1.0 / (1.0 + 12.0 * max(Ri_b, 0.0))
                        stability_factor = max(stability_factor, 0.15)
                else:
                    stability_factor = 1.0
            else:
                stability_factor = 1.0

        q_air = ns["get_air_specific_humidity"](t_sec)
        q_snow = ns["get_snow_specific_humidity"](T_surfK)
        qsen = ns["rho_air"] * ns["c_pa"] * ns["CH"] * U * (T_airK - T_surfK) * stability_factor
        qlat = ns["rho_air"] * ns["Lv_subl"] * ns["CE"] * U * (q_air - q_snow) * stability_factor
        return qsen, qlat

    return turbulent_fluxes


def evaluate_config(ns: dict, cfg: Config, use_skin: bool, skin_beta: float, outlier_thr: float) -> dict:
    # preserve originals
    orig_k = ns["k"]
    orig_rho = ns["rho_snow"]
    orig_turb = ns["turbulent_fluxes"]
    orig_sw = ns["shortwave_absorption_twoBand"]

    try:
        ns["k"] = float(cfg.k)
        ns["rho_snow"] = float(cfg.rho_snow)
        ns["turbulent_fluxes"] = build_turbulent_func(ns, cfg.stability_mode)
        ns["shortwave_absorption_twoBand"] = build_shortwave_func(ns, cfg.kappa_vis)

        t_solver, t_layers, _ = ns["run_snow_model"](False, False)
        model_solver_c = model_surface_series_c(t_layers, use_skin=use_skin, beta=skin_beta)
        t_eval = np.asarray(ns["times_sec"], dtype=float)
        model_eval_c = np.interp(t_eval, t_solver, model_solver_c)
        obs_c = ns["usgs_temp_obs_interp"](t_eval)
        err = model_eval_c - obs_c

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        bias = float(np.mean(err))
        corr = float(np.corrcoef(obs_c, model_eval_c)[0, 1])

        hours = (t_eval % (24.0 * 3600.0)) / 3600.0
        day_mask = (hours >= 6.0) & (hours <= 18.0)
        night_mask = ~day_mask
        morning_mask = (hours >= 9.0) & (hours <= 11.0)
        aft_mask = (hours >= 15.0) & (hours <= 18.0)

        day_bias = float(np.mean(err[day_mask]))
        night_bias = float(np.mean(err[night_mask]))
        morning_bias = float(np.mean(err[morning_mask])) if morning_mask.any() else np.nan
        aft_bias = float(np.mean(err[aft_mask])) if aft_mask.any() else np.nan

        outlier_count = int(np.sum(np.abs(err) >= outlier_thr))
        outlier_pct = float(100.0 * outlier_count / len(err))

        score = (
            rmse
            + 0.20 * abs(bias)
            + 0.20 * abs(day_bias)
            + 0.20 * abs(night_bias)
            + 0.20 * abs(morning_bias if np.isfinite(morning_bias) else 0.0)
            + 0.02 * outlier_count
        )

        return {
            "name": cfg.name,
            "k": cfg.k,
            "rho_snow": cfg.rho_snow,
            "kappa_vis": cfg.kappa_vis,
            "stability_mode": cfg.stability_mode,
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "Corr": corr,
            "Day_Bias": day_bias,
            "Night_Bias": night_bias,
            "Morning_9_11_Bias": morning_bias,
            "Afternoon_15_18_Bias": aft_bias,
            "Outliers_abs_ge_thr": outlier_count,
            "Outlier_pct": outlier_pct,
            "Score": score,
        }
    finally:
        ns["k"] = orig_k
        ns["rho_snow"] = orig_rho
        ns["turbulent_fluxes"] = orig_turb
        ns["shortwave_absorption_twoBand"] = orig_sw


def build_configs() -> list[Config]:
    cfgs: list[Config] = []

    # baseline/current
    cfgs.append(Config("baseline_current", 0.22, 200.0, 8.0, "current"))

    # Thermal sensitivity
    cfgs.extend(
        [
            Config("thermal_rho180", 0.22, 180.0, 8.0, "current"),
            Config("thermal_rho170", 0.22, 170.0, 8.0, "current"),
            Config("thermal_k020", 0.20, 200.0, 8.0, "current"),
            Config("thermal_k024", 0.24, 200.0, 8.0, "current"),
            Config("thermal_k020_rho180", 0.20, 180.0, 8.0, "current"),
        ]
    )

    # SW absorption depth sensitivity
    cfgs.extend(
        [
            Config("sw_kv6", 0.22, 200.0, 6.0, "current"),
            Config("sw_kv10", 0.22, 200.0, 10.0, "current"),
            Config("sw_kv12", 0.22, 200.0, 12.0, "current"),
        ]
    )

    # Stability sensitivity
    cfgs.extend(
        [
            Config("stab_no_split", 0.22, 200.0, 8.0, "no_split"),
            Config("stab_solar_boost", 0.22, 200.0, 8.0, "solar_boost"),
            Config("stab_strong_night", 0.22, 200.0, 8.0, "strong_night"),
            Config("stab_neutral", 0.22, 200.0, 8.0, "neutral"),
        ]
    )

    # Combined plausible candidates
    cfgs.extend(
        [
            Config("combo_solarBoost_rho180", 0.22, 180.0, 8.0, "solar_boost"),
            Config("combo_solarBoost_kv10", 0.22, 200.0, 10.0, "solar_boost"),
            Config("combo_solarBoost_rho180_kv10", 0.22, 180.0, 10.0, "solar_boost"),
            Config("combo_noSplit_rho180_kv10", 0.22, 180.0, 10.0, "no_split"),
            Config("combo_strongNight_rho180", 0.22, 180.0, 8.0, "strong_night"),
        ]
    )
    return cfgs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    nb_path = (repo_root / args.notebook).resolve()
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

    use_skin = args.skin == "on"
    cfgs = build_configs()

    rows = []
    for c in cfgs:
        rows.append(
            evaluate_config(
                ns,
                cfg=c,
                use_skin=use_skin,
                skin_beta=args.skin_beta,
                outlier_thr=args.outlier_threshold,
            )
        )

    df = pd.DataFrame(rows).sort_values("Score").reset_index(drop=True)
    print("\nPhysics Option Comparison")
    print("=" * 140)
    print(df.to_string(index=False))

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"physics_option_compare_{args.station}_{args.start}_{args.end}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
