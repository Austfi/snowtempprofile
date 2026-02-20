#!/usr/bin/env python3
"""
Standalone assessment for notebook model variants + skin-temperature diagnostics.

It compares combinations of:
- Physics variant:
  - baseline
  - A_stability  (milder stable suppression)
  - B_sw_depth   (kappa_vis = 8.0)
  - C_both       (A + B)
- Surface diagnostic:
  - skin off  -> use top layer center temperature
  - skin beta -> T_skin = T_top + beta * (T_top - T_below)

Outputs:
- MAE, RMSE, Bias, Corr, Day Bias, Night Bias
- best lag (hours) and corr over +/- 3 h
- mean daily amplitude ratio (model/obs)
- balanced score and recommended config

This script does not edit your notebook.
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
class Variant:
    label: str
    use_a: bool
    use_b: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess low-risk model + skin variants from snowmodel_USGS notebook."
    )
    parser.add_argument(
        "--station",
        default="senator_beck",
        choices=["senator_beck", "independence_pass", "berthoud_pass"],
        help="Station key used in notebook/usgs_collector.",
    )
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-01-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--notebook",
        default="notebooks/snowmodel_USGS.ipynb",
        help="Path to notebook file",
    )
    parser.add_argument(
        "--skin-betas",
        default="off,0.10,0.15,0.20",
        help='Comma list, e.g. "off,0.10,0.15,0.20"',
    )
    parser.add_argument(
        "--no-a-variant",
        action="store_true",
        help="Skip A_stability and C_both variants (optional quick mode).",
    )
    return parser.parse_args()


def parse_skin_betas(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t in {"off", "none", "0"}:
            out.append(None)
        else:
            out.append(float(t))
    # preserve order, remove duplicates
    seen = set()
    uniq: list[float | None] = []
    for b in out:
        key = "off" if b is None else f"{b:.6f}"
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq


def load_notebook_code(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text())
    code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    # We only need function/data definitions up to run_snow_model.
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


def build_variant_functions(ns: dict):
    def turbulent_fluxes_change_a(t_sec, T_surfK):
        T_airK = ns["measured_air_temp"](t_sec)
        U = max(float(ns["get_wind_speed"](t_sec)), 0.1)

        z_ref_local = 2.0
        g_local = 9.81
        Ri_b = (g_local * z_ref_local * (T_airK - T_surfK)) / (T_airK * (U**2))

        if T_airK > T_surfK:
            stability_factor = 1.0 / (1.0 + 5.0 * max(Ri_b, 0.0))
            stability_factor = max(stability_factor, 0.35)
        else:
            stability_factor = 1.0

        Q_sensible = (
            ns["rho_air"]
            * ns["c_pa"]
            * ns["CH"]
            * U
            * (T_airK - T_surfK)
            * stability_factor
        )
        q_air = ns["get_air_specific_humidity"](t_sec)
        q_snow = ns["get_snow_specific_humidity"](T_surfK)
        Q_latent = (
            ns["rho_air"]
            * ns["Lv_subl"]
            * ns["CE"]
            * U
            * (q_air - q_snow)
            * stability_factor
        )
        return Q_sensible, Q_latent

    def shortwave_absorption_change_b(t_sec, z_faces):
        sw_net = ns["compute_shortwave_net"](t_sec)
        f_nir = 0.4
        f_vis = 1.0 - f_nir
        kappa_nir = 35.0
        kappa_vis = 8.0

        sw_nir_top = sw_net * f_nir
        sw_vis_top = sw_net * f_vis
        h_curr = z_faces[-1]
        d_faces = h_curr - z_faces
        d_top = d_faces[1:]
        d_bottom = d_faces[:-1]

        flux_in_nir = sw_nir_top * np.exp(-kappa_nir * d_top)
        flux_out_nir = sw_nir_top * np.exp(-kappa_nir * d_bottom)
        sw_abs_nir = flux_in_nir - flux_out_nir

        flux_in_vis = sw_vis_top * np.exp(-kappa_vis * d_top)
        flux_out_vis = sw_vis_top * np.exp(-kappa_vis * d_bottom)
        sw_abs_vis = flux_in_vis - flux_out_vis

        return np.maximum(sw_abs_nir + sw_abs_vis, 0.0)

    return turbulent_fluxes_change_a, shortwave_absorption_change_b


def model_surface_series_c(t_sec: np.ndarray, t_layers_k: np.ndarray, beta: float | None) -> np.ndarray:
    if beta is None:
        return t_layers_k[-1, :] - 273.15
    if t_layers_k.shape[0] < 2:
        return t_layers_k[-1, :] - 273.15

    t_top = t_layers_k[-1, :]
    t_below = t_layers_k[-2, :]
    grad = np.clip(t_top - t_below, -2.0, 2.0)
    t_skin = t_top + beta * grad
    t_skin = np.clip(t_skin, 223.15, 273.15)
    return t_skin - 273.15


def compute_best_lag_and_corr(
    t_sec: np.ndarray, model_c: np.ndarray, obs_c: np.ndarray, lag_hours: float = 3.0
) -> tuple[float, float]:
    grid = np.arange(float(t_sec[0]), float(t_sec[-1]) + 1.0, 3600.0)
    mod_h = np.interp(grid, t_sec, model_c)
    obs_h = np.interp(grid, t_sec, obs_c)

    best_lag = 0.0
    best_corr = -np.inf
    for lag in np.arange(-lag_hours, lag_hours + 0.001, 0.25):
        shifted = np.interp(grid, grid + lag * 3600.0, mod_h, left=np.nan, right=np.nan)
        m = np.isfinite(shifted) & np.isfinite(obs_h)
        if m.sum() < 10:
            continue
        corr = float(np.corrcoef(shifted[m], obs_h[m])[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)
    return best_lag, best_corr


def compute_amplitude_ratio(t_sec: np.ndarray, model_c: np.ndarray, obs_c: np.ndarray) -> float:
    t0 = pd.Timestamp("2000-01-01 00:00:00")
    dt_index = t0 + pd.to_timedelta(t_sec, unit="s")
    df = pd.DataFrame({"model": model_c, "obs": obs_c}, index=dt_index)
    g = df.groupby(df.index.floor("D"))
    amp_mod = g["model"].max() - g["model"].min()
    amp_obs = g["obs"].max() - g["obs"].min()
    mask = amp_obs > 0.1
    if mask.sum() == 0:
        return float("nan")
    return float((amp_mod[mask] / amp_obs[mask]).mean())


def compute_stats(t_sec: np.ndarray, model_c: np.ndarray, obs_c: np.ndarray) -> dict:
    err = model_c - obs_c
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(obs_c, model_c)[0, 1])

    hours = (t_sec % (24 * 3600.0)) / 3600.0
    day = (hours >= 6.0) & (hours <= 18.0)
    night = ~day
    day_bias = float(np.mean(err[day])) if day.any() else float("nan")
    night_bias = float(np.mean(err[night])) if night.any() else float("nan")

    lag_h, lag_corr = compute_best_lag_and_corr(t_sec, model_c, obs_c, lag_hours=3.0)
    amp_ratio = compute_amplitude_ratio(t_sec, model_c, obs_c)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Correlation": corr,
        "Day Bias": day_bias,
        "Night Bias": night_bias,
        "Lag_h": lag_h,
        "LagCorr": lag_corr,
        "AmpRatio": amp_ratio,
    }


def run_variant(ns: dict, variant: Variant, skin_beta: float | None) -> dict:
    original_turb = ns["turbulent_fluxes"]
    original_sw2 = ns["shortwave_absorption_twoBand"]
    turb_a, sw_b = build_variant_functions(ns)

    ns["turbulent_fluxes"] = turb_a if variant.use_a else original_turb
    ns["shortwave_absorption_twoBand"] = sw_b if variant.use_b else original_sw2

    try:
        t_run, t_layers, _ = ns["run_snow_model"](False, False)
        obs = ns["usgs_temp_obs_interp"](t_run)
        mod = model_surface_series_c(t_run, t_layers, skin_beta)
        s = compute_stats(t_run, mod, obs)
        return {
            "variant": variant.label,
            "skin_beta": "off" if skin_beta is None else round(float(skin_beta), 3),
            **s,
        }
    finally:
        ns["turbulent_fluxes"] = original_turb
        ns["shortwave_absorption_twoBand"] = original_sw2


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_balanced"] = (
        out["RMSE"]
        + 0.20 * out["Bias"].abs()
        + 0.25 * out["Day Bias"].abs()
        + 0.25 * out["Night Bias"].abs()
        + 0.20 * out["Lag_h"].abs()
        + 0.20 * (out["AmpRatio"] - 1.0).abs()
    )
    return out


def choose_recommended(df: pd.DataFrame) -> pd.Series:
    # Prefer physically balanced solutions first.
    filt = (
        (df["Bias"].abs() <= 0.35)
        & (df["Night Bias"].abs() <= 0.75)
        & (df["Lag_h"].abs() <= 1.0)
    )
    if filt.any():
        return df.loc[filt].sort_values("score_balanced").iloc[0]
    return df.sort_values("score_balanced").iloc[0]


def main() -> int:
    args = parse_args()
    skin_betas = parse_skin_betas(args.skin_betas)

    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = (repo_root / args.notebook).resolve()
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        return 1

    notebooks_dir = repo_root / "notebooks"
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    code = load_notebook_code(notebook_path)
    code = apply_runtime_overrides(code, station=args.station, start=args.start, end=args.end)

    ns: dict = {}
    try:
        exec(code, ns, ns)
    except Exception as exc:
        print("ERROR: failed while executing notebook code in script mode.")
        print(type(exc).__name__, exc)
        return 2

    required = ["run_snow_model", "usgs_temp_obs_interp", "turbulent_fluxes", "shortwave_absorption_twoBand"]
    missing = [k for k in required if k not in ns]
    if missing:
        print("ERROR: Missing required notebook objects:", missing)
        print("If data load failed, check station/date and network.")
        return 3

    variants = [Variant("baseline", False, False), Variant("B_sw_depth", False, True)]
    if not args.no_a_variant:
        variants = [
            Variant("baseline", False, False),
            Variant("A_stability", True, False),
            Variant("B_sw_depth", False, True),
            Variant("C_both", True, True),
        ]

    rows = []
    for v in variants:
        for beta in skin_betas:
            rows.append(run_variant(ns, variant=v, skin_beta=beta))

    df = pd.DataFrame(rows)
    df = add_scores(df)
    df_sorted = df.sort_values("score_balanced").reset_index(drop=True)
    rec = choose_recommended(df_sorted)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("\nFull Comparison")
    print("=" * 120)
    print(df.to_string(index=False))

    print("\nTop Ranked (lower score_balanced is better)")
    print("=" * 120)
    print(df_sorted.head(12).to_string(index=False))

    print("\nRecommended")
    print("=" * 120)
    print(rec.to_frame().T.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
