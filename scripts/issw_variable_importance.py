#!/usr/bin/env python3
"""
ISSW variable-importance experiment runner.

Goal:
- Quantify how much error is introduced when key forcings are simplified/removed.
- Break impacts out by weather windows (cloud/clear/day/night/morning).

Primary use:
python scripts/issw_variable_importance.py --station senator_beck --start 2026-01-01 --end 2026-01-31
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

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
class Experiment:
    name: str
    family: str
    use_idealized_sw: bool = False
    use_idealized_lw: bool = False
    sw_mode: str = "measured"  # measured | net_hourly_clim | net_day_clim | net_cloud_day_clim | alpha_fixed_no_up
    sw_alpha: float = 0.80
    turb_mode: str = "normal"  # normal | no_sensible | no_latent | no_turb
    lw_mode: str = "measured"  # measured | hourly_clim | zero | cloud_zero | cloud_to_clear
    qsen_scale: float = 1.0
    qlat_scale: float = 1.0
    lw_down_scale: float = 1.0
    lw_scale_regime: str = "all"  # all | cloud_night | cloud_day | clear_night
    wind_scale: float = 1.0
    snow_depth_scale: float = 1.0
    airtemp_scale: float = 1.0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Variable importance runner for ISSW-style diagnostics.")
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
    p.add_argument("--cloud-eps-threshold", type=float, default=0.85)
    p.add_argument("--night-sw-threshold", type=float, default=20.0)
    p.add_argument("--event-window-hours", type=float, default=6.0)
    p.add_argument("--event-top-k", type=int, default=5)
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
    code = re.sub(
        r"df_raw\s*=\s*fetch_usgs_iv\(\s*STATIONS\[STATION\]\s*,\s*START_DATE\s*,\s*END_DATE\s*\)",
        'df_raw = fetch_usgs_iv_cached(STATIONS[STATION], START_DATE, END_DATE, cache_dir="data/usgs_cache", refresh=False)',
        code,
    )
    code = re.sub(r"show_plots\s*=\s*True", "show_plots = False", code)
    code = re.sub(r"generate_animation\s*=\s*True", "generate_animation = False", code)
    return code


def to_surface_series_c(t_layers_k: np.ndarray, use_skin: bool, beta: float) -> np.ndarray:
    if (not use_skin) or (t_layers_k.shape[0] < 2):
        return t_layers_k[-1, :] - 273.15
    t_top = t_layers_k[-1, :]
    t_below = t_layers_k[-2, :]
    grad = np.clip(t_top - t_below, -2.0, 2.0)
    t_skin = np.clip(t_top + beta * grad, 223.15, 273.15)
    return t_skin - 273.15


def build_experiments() -> list[Experiment]:
    return [
        Experiment(
            name="baseline_measured",
            family="baseline",
            notes="Measured SW and LW forcing.",
        ),
        Experiment(
            name="sw_hourly_climatology",
            family="shortwave",
            sw_mode="net_hourly_clim",
            notes="Net SW replaced with hourly climatology (no cloud-event variability).",
        ),
        Experiment(
            name="sw_daytime_climatology",
            family="shortwave",
            sw_mode="net_day_clim",
            notes="Net SW replaced by diurnal climatology during daytime only.",
        ),
        Experiment(
            name="sw_cloud_day_climatology",
            family="shortwave",
            sw_mode="net_cloud_day_clim",
            notes="Net SW replaced by diurnal climatology during cloudy daytime only.",
        ),
        Experiment(
            name="sw_no_up_alpha_0p80",
            family="shortwave",
            sw_mode="alpha_fixed_no_up",
            sw_alpha=0.80,
            notes="No SW-up sensor case: SWnet = SWdown * (1 - 0.80).",
        ),
        Experiment(
            name="sensible_scale_0p7",
            family="turbulent",
            qsen_scale=0.7,
            notes="Sensible flux scaled to 70%.",
        ),
        Experiment(
            name="sensible_scale_1p3",
            family="turbulent",
            qsen_scale=1.3,
            notes="Sensible flux scaled to 130%.",
        ),
        Experiment(
            name="latent_scale_0p7",
            family="turbulent",
            qlat_scale=0.7,
            notes="Latent flux scaled to 70%.",
        ),
        Experiment(
            name="latent_scale_1p3",
            family="turbulent",
            qlat_scale=1.3,
            notes="Latent flux scaled to 130%.",
        ),
        Experiment(
            name="wind_scale_0p8",
            family="turbulent",
            wind_scale=0.8,
            notes="Wind speed forcing scaled to 80%.",
        ),
        Experiment(
            name="wind_scale_1p2",
            family="turbulent",
            wind_scale=1.2,
            notes="Wind speed forcing scaled to 120%.",
        ),
        Experiment(
            name="wind_scale_0p75",
            family="turbulent",
            wind_scale=0.75,
            notes="Wind speed forcing scaled to 75%.",
        ),
        Experiment(
            name="wind_scale_0p5",
            family="turbulent",
            wind_scale=0.5,
            notes="Wind speed forcing scaled to 50%.",
        ),
        Experiment(
            name="wind_scale_0p25",
            family="turbulent",
            wind_scale=0.25,
            notes="Wind speed forcing scaled to 25%.",
        ),
        Experiment(
            name="wind_scale_0p0",
            family="turbulent",
            wind_scale=0.0,
            notes="Wind speed forcing scaled to 0% (extreme no-wind limit).",
        ),
        Experiment(
            name="airtemp_scale_0p75",
            family="airtemp",
            airtemp_scale=0.75,
            notes="Air-temperature variability scaled to 75% around monthly mean.",
        ),
        Experiment(
            name="airtemp_scale_0p5",
            family="airtemp",
            airtemp_scale=0.5,
            notes="Air-temperature variability scaled to 50% around monthly mean.",
        ),
        Experiment(
            name="airtemp_scale_0p25",
            family="airtemp",
            airtemp_scale=0.25,
            notes="Air-temperature variability scaled to 25% around monthly mean.",
        ),
        Experiment(
            name="airtemp_scale_0p0",
            family="airtemp",
            airtemp_scale=0.0,
            notes="Air-temperature variability scaled to 0% (constant monthly-mean air temp).",
        ),
        Experiment(
            name="no_sensible",
            family="turbulent",
            turb_mode="no_sensible",
            notes="Sensible flux disabled.",
        ),
        Experiment(
            name="no_latent",
            family="turbulent",
            turb_mode="no_latent",
            notes="Latent flux disabled.",
        ),
        Experiment(
            name="no_turbulent",
            family="turbulent",
            turb_mode="no_turb",
            notes="Both sensible and latent disabled.",
        ),
        Experiment(
            name="lwdown_parameterized",
            family="longwave",
            use_idealized_lw=True,
            notes="Use parameterized LW down instead of measured.",
        ),
        Experiment(
            name="lw_hourly_climatology",
            family="longwave",
            lw_mode="hourly_clim",
            notes="LW down replaced with hourly climatology (no cloud-event variability).",
        ),
        Experiment(
            name="lwdown_scale_0p9",
            family="longwave",
            lw_down_scale=0.9,
            notes="Measured LW down scaled to 90%.",
        ),
        Experiment(
            name="lwdown_scale_1p1",
            family="longwave",
            lw_down_scale=1.1,
            notes="Measured LW down scaled to 110%.",
        ),
        Experiment(
            name="lwdown_scale_0p9_cloud_night",
            family="longwave",
            lw_down_scale=0.9,
            lw_scale_regime="cloud_night",
            notes="LW down scaled to 90% during cloudy nights only.",
        ),
        Experiment(
            name="lwdown_scale_1p1_cloud_night",
            family="longwave",
            lw_down_scale=1.1,
            lw_scale_regime="cloud_night",
            notes="LW down scaled to 110% during cloudy nights only.",
        ),
        Experiment(
            name="lwdown_scale_0p9_cloud_day",
            family="longwave",
            lw_down_scale=0.9,
            lw_scale_regime="cloud_day",
            notes="LW down scaled to 90% during cloudy daytime only.",
        ),
        Experiment(
            name="lwdown_scale_1p1_cloud_day",
            family="longwave",
            lw_down_scale=1.1,
            lw_scale_regime="cloud_day",
            notes="LW down scaled to 110% during cloudy daytime only.",
        ),
        Experiment(
            name="lwdown_scale_0p9_clear_night",
            family="longwave",
            lw_down_scale=0.9,
            lw_scale_regime="clear_night",
            notes="LW down scaled to 90% during clear nights only.",
        ),
        Experiment(
            name="lwdown_scale_1p1_clear_night",
            family="longwave",
            lw_down_scale=1.1,
            lw_scale_regime="clear_night",
            notes="LW down scaled to 110% during clear nights only.",
        ),
        Experiment(
            name="lwdown_zero_all",
            family="longwave",
            lw_mode="zero",
            notes="Set LW down to zero (extreme stress test).",
        ),
        Experiment(
            name="lwdown_zero_when_cloudy",
            family="longwave",
            lw_mode="cloud_zero",
            notes="Set LW down to zero only in cloud-like periods.",
        ),
        Experiment(
            name="lwdown_cloud_to_clear",
            family="longwave",
            lw_mode="cloud_to_clear",
            notes="Replace cloudy LW down with clear-sky hourly baseline.",
        ),
        Experiment(
            name="snowdepth_scale_0p9",
            family="snow_depth",
            snow_depth_scale=0.9,
            notes="Snow depth forcing scaled to 90% (shallower pack).",
        ),
        Experiment(
            name="snowdepth_scale_1p1",
            family="snow_depth",
            snow_depth_scale=1.1,
            notes="Snow depth forcing scaled to 110% (deeper pack).",
        ),
    ]


def make_hourly_curve(t_eval: np.ndarray, values: np.ndarray, start_offset_sec: int = 0) -> np.ndarray:
    hours = hour_index_from_seconds(np.asarray(t_eval, dtype=float), start_offset_sec)
    s = pd.Series(values).groupby(hours).mean()
    s = s.reindex(range(24))
    s = s.interpolate(limit_direction="both")
    s = s.fillna(float(np.nanmean(values)))
    return s.values


def hour_index_from_seconds(t_sec, start_offset_sec: int):
    arr = np.asarray(t_sec, dtype=float)
    hours = (((arr + float(start_offset_sec)) % 86400.0) / 3600.0).astype(int)
    return hours


def _eval_scalar_func_on_array(func, t_sec):
    arr = np.asarray(t_sec, dtype=float)
    if np.isscalar(t_sec):
        return float(func(float(arr)))
    out = np.array([float(func(float(v))) for v in arr.ravel()], dtype=float)
    return out.reshape(arr.shape)


def make_hourly_func(hourly_curve: np.ndarray):
    def _f(t_sec):
        arr = np.asarray(t_sec, dtype=float)
        hours = ((arr % 86400.0) / 3600.0).astype(int)
        out = hourly_curve[hours]
        if np.isscalar(t_sec):
            return float(out)
        return out

    return _f


def make_sw_net_func(
    ns: dict,
    *,
    sw_mode: str,
    sw_net_hourly_curve: np.ndarray,
    cloud_interp,
    start_offset_sec: int,
    night_sw_threshold: float,
    sw_alpha: float,
):
    base_sw_in = ns["SW_in_interp"]
    base_sw_net = ns["compute_shortwave_net"]

    def _f(t_sec):
        arr = np.asarray(t_sec, dtype=float)
        hour_i = hour_index_from_seconds(arr, start_offset_sec)
        sw_net_clim = sw_net_hourly_curve[hour_i]
        sw_net_meas = _eval_scalar_func_on_array(base_sw_net, arr)
        sw_in_meas = np.asarray(base_sw_in(arr), dtype=float)
        cloudy = np.asarray(cloud_interp(arr), dtype=float) >= 0.5
        day = sw_in_meas > float(night_sw_threshold)

        if sw_mode == "net_hourly_clim":
            out = sw_net_clim
        elif sw_mode == "net_day_clim":
            out = np.where(day, sw_net_clim, sw_net_meas)
        elif sw_mode == "net_cloud_day_clim":
            out = np.where(day & cloudy, sw_net_clim, sw_net_meas)
        elif sw_mode == "alpha_fixed_no_up":
            out = np.maximum(sw_in_meas * (1.0 - float(sw_alpha)), 0.0)
        else:
            out = sw_net_meas

        out = np.maximum(np.asarray(out, dtype=float), 0.0)
        if np.isscalar(t_sec):
            return float(np.asarray(out))
        return out

    return _f


def make_lw_radiative_func(
    ns: dict,
    *,
    lw_mode: str,
    cloud_interp,
    lw_clear_hourly: np.ndarray,
    lw_hourly_curve: np.ndarray,
    lw_down_scale: float,
    lw_scale_regime: str,
    start_offset_sec: int,
    night_sw_threshold: float,
):
    def _func(t_sec, t_surf_k):
        if lw_mode == "zero":
            lw_down = 0.0
        else:
            if lw_mode == "hourly_clim":
                hour_i = int(hour_index_from_seconds(float(t_sec), start_offset_sec))
                lw_down = float(lw_hourly_curve[hour_i])
            else:
                lw_down = float(ns["LW_in_interp"](t_sec))
            cloudy = float(cloud_interp(t_sec)) >= 0.5
            if lw_mode == "cloud_zero" and cloudy:
                lw_down = 0.0
            elif lw_mode == "cloud_to_clear" and cloudy:
                hour_i = int(hour_index_from_seconds(float(t_sec), start_offset_sec))
                lw_down = float(lw_clear_hourly[hour_i])
        sw_now = float(ns["SW_in_interp"](t_sec))
        day = sw_now > float(night_sw_threshold)
        night = not day
        cloudy = float(cloud_interp(t_sec)) >= 0.5

        apply_scale = False
        if lw_scale_regime == "all":
            apply_scale = True
        elif lw_scale_regime == "cloud_night":
            apply_scale = cloudy and night
        elif lw_scale_regime == "cloud_day":
            apply_scale = cloudy and day
        elif lw_scale_regime == "clear_night":
            apply_scale = (not cloudy) and night

        if apply_scale:
            lw_down = lw_down * lw_down_scale
        lw_down = max(0.0, lw_down)
        lw_up = float(ns["eps_snow"]) * ns["sigma"] * (t_surf_k**4)
        lw_net = lw_down - lw_up
        sw_net = max(float(ns["compute_shortwave_net"](t_sec)), 0.0)
        return lw_net, sw_net

    return _func


def make_turbulent_func(ns: dict, mode: str):
    base = ns["turbulent_fluxes"]

    def _f(t_sec, t_surf_k):
        qsen, qlat = base(t_sec, t_surf_k)
        if mode == "no_sensible":
            return 0.0, qlat
        if mode == "no_latent":
            return qsen, 0.0
        if mode == "no_turb":
            return 0.0, 0.0
        return qsen, qlat

    return _f


def make_scaled_turbulent_func(ns: dict, qsen_scale: float, qlat_scale: float):
    base = ns["turbulent_fluxes"]

    def _f(t_sec, t_surf_k):
        qsen, qlat = base(t_sec, t_surf_k)
        return qsen * qsen_scale, qlat * qlat_scale

    return _f


def make_scaled_airtemp_func(ns: dict, scale: float, t_eval: np.ndarray):
    base = ns["measured_air_temp"]
    t_ref = np.asarray(base(t_eval), dtype=float)
    t_mean = float(np.nanmean(t_ref))

    def _f(t_sec):
        t = np.asarray(base(t_sec), dtype=float)
        out = t_mean + scale * (t - t_mean)
        if np.isscalar(t_sec):
            return float(np.asarray(out))
        return out

    return _f


def compute_metrics(err: np.ndarray, obs: np.ndarray, model: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        return np.nan, np.nan
    e = err[mask]
    rmse = float(np.sqrt(np.mean(e**2)))
    bias = float(np.mean(e))
    return rmse, bias


def gradient_top_depth_cpm(
    t_layers_k: np.ndarray,
    t_solver: np.ndarray,
    t_eval: np.ndarray,
    h_eval: np.ndarray,
    depth_m: float,
) -> np.ndarray:
    """
    Compute near-surface gradient: (T_surface - T_at_depth) / depth in C/m.
    """
    n_layers, _ = t_layers_k.shape
    depth_m = max(0.01, float(depth_m))

    # Interpolate each layer to evaluation times.
    layer_eval_c = np.vstack(
        [np.interp(t_eval, t_solver, t_layers_k[i, :] - 273.15) for i in range(n_layers)]
    )
    t_top_c = layer_eval_c[-1, :]

    dz = np.maximum(h_eval / float(n_layers), 0.002)
    k_down = np.clip(np.round(depth_m / dz).astype(int), 1, n_layers - 1)
    j_idx = n_layers - 1 - k_down
    t_depth_c = np.array([layer_eval_c[j_idx[i], i] for i in range(len(t_eval))], dtype=float)
    return (t_top_c - t_depth_c) / depth_m


def run_experiment(
    ns: dict,
    exp: Experiment,
    *,
    use_skin: bool,
    skin_beta: float,
    t_eval: np.ndarray,
    obs_c: np.ndarray,
    window_masks: dict[str, np.ndarray],
    sw_net_hourly_curve: np.ndarray,
    cloud_interp,
    lw_clear_hourly: np.ndarray,
    lw_hourly_curve: np.ndarray,
    start_offset_sec: int,
    night_sw_threshold: float,
):
    orig_sw_in = ns["SW_in_interp"]
    orig_sw_net = ns["compute_shortwave_net"]
    orig_turb = ns["turbulent_fluxes"]
    orig_rad = ns["compute_radiative_fluxes"]
    orig_rad_ideal = ns["compute_idealized_radiative_fluxes"]
    orig_get_wind = ns["get_wind_speed"]
    orig_air_temp = ns["measured_air_temp"]
    orig_snow_h = ns["snow_depth_interp"]

    try:
        if exp.sw_mode in {"net_hourly_clim", "net_day_clim", "net_cloud_day_clim", "alpha_fixed_no_up"}:
            ns["compute_shortwave_net"] = make_sw_net_func(
                ns,
                sw_mode=exp.sw_mode,
                sw_net_hourly_curve=sw_net_hourly_curve,
                cloud_interp=cloud_interp,
                start_offset_sec=start_offset_sec,
                night_sw_threshold=night_sw_threshold,
                sw_alpha=exp.sw_alpha,
            )

        if exp.turb_mode != "normal":
            ns["turbulent_fluxes"] = make_turbulent_func(ns, exp.turb_mode)
        elif not np.isclose(exp.qsen_scale, 1.0) or not np.isclose(exp.qlat_scale, 1.0):
            ns["turbulent_fluxes"] = make_scaled_turbulent_func(ns, exp.qsen_scale, exp.qlat_scale)

        if not np.isclose(exp.wind_scale, 1.0):
            def _wind_scaled(t_sec):
                v = np.asarray(orig_get_wind(t_sec), dtype=float) * exp.wind_scale
                if np.isscalar(t_sec):
                    return float(v)
                return v
            ns["get_wind_speed"] = _wind_scaled

        if not np.isclose(exp.airtemp_scale, 1.0):
            ns["measured_air_temp"] = make_scaled_airtemp_func(ns, exp.airtemp_scale, t_eval)

        if not np.isclose(exp.snow_depth_scale, 1.0):
            def _snow_scaled(t_sec):
                h = np.asarray(orig_snow_h(t_sec), dtype=float) * exp.snow_depth_scale
                h = np.maximum(h, 0.10)
                if np.isscalar(t_sec):
                    return float(h)
                return h
            ns["snow_depth_interp"] = _snow_scaled

        if (
            exp.lw_mode != "measured"
            or not np.isclose(exp.lw_down_scale, 1.0)
            or exp.lw_scale_regime != "all"
        ):
            patched = make_lw_radiative_func(
                ns,
                lw_mode=exp.lw_mode,
                cloud_interp=cloud_interp,
                lw_clear_hourly=lw_clear_hourly,
                lw_hourly_curve=lw_hourly_curve,
                lw_down_scale=exp.lw_down_scale,
                lw_scale_regime=exp.lw_scale_regime,
                start_offset_sec=start_offset_sec,
                night_sw_threshold=night_sw_threshold,
            )
            ns["compute_radiative_fluxes"] = patched
            ns["compute_idealized_radiative_fluxes"] = patched

        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            t_solver, t_layers, _ = ns["run_snow_model"](exp.use_idealized_sw, exp.use_idealized_lw)
            model_solver_c = to_surface_series_c(t_layers, use_skin=use_skin, beta=skin_beta)
            model_eval_c = np.interp(t_eval, t_solver, model_solver_c)
            err = model_eval_c - obs_c

        # Bulk snowpack gradient across modeled snow depth (surface - bottom) / H.
        t_top_c = np.interp(t_eval, t_solver, t_layers[-1, :] - 273.15)
        t_bot_c = np.interp(t_eval, t_solver, t_layers[0, :] - 273.15)
        h_eval = np.maximum(np.asarray(ns["snow_depth_interp"](t_eval), dtype=float), 0.10)
        grad_bulk_cpm = (t_top_c - t_bot_c) / h_eval
        grad_top10_cpm = gradient_top_depth_cpm(t_layers, t_solver, t_eval, h_eval, depth_m=0.10)
        grad_top20_cpm = gradient_top_depth_cpm(t_layers, t_solver, t_eval, h_eval, depth_m=0.20)
        grad_top60_cpm = gradient_top_depth_cpm(t_layers, t_solver, t_eval, h_eval, depth_m=0.60)

        out = {
            "experiment": exp.name,
            "family": exp.family,
            "notes": exp.notes,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "Bias": float(np.mean(err)),
            "Correlation": float(np.corrcoef(obs_c, model_eval_c)[0, 1]),
            "Mean_abs_bulk_grad_Cpm": float(np.mean(np.abs(grad_bulk_cpm))),
            "P95_abs_bulk_grad_Cpm": float(np.nanpercentile(np.abs(grad_bulk_cpm), 95)),
            "Max_abs_bulk_grad_Cpm": float(np.nanmax(np.abs(grad_bulk_cpm))),
            "Mean_abs_top10_grad_Cpm": float(np.mean(np.abs(grad_top10_cpm))),
            "Mean_abs_top20_grad_Cpm": float(np.mean(np.abs(grad_top20_cpm))),
            "Mean_abs_top60_grad_Cpm": float(np.mean(np.abs(grad_top60_cpm))),
            "P95_abs_top20_grad_Cpm": float(np.nanpercentile(np.abs(grad_top20_cpm), 95)),
        }
        for wname, m in window_masks.items():
            rmse_w, bias_w = compute_metrics(err, obs_c, model_eval_c, m)
            out[f"RMSE_{wname}"] = rmse_w
            out[f"Bias_{wname}"] = bias_w

        grad_pack = {
            "bulk": grad_bulk_cpm,
            "top10": grad_top10_cpm,
            "top20": grad_top20_cpm,
            "top60": grad_top60_cpm,
        }
        return out, model_eval_c, err, grad_pack
    finally:
        ns["SW_in_interp"] = orig_sw_in
        ns["compute_shortwave_net"] = orig_sw_net
        ns["turbulent_fluxes"] = orig_turb
        ns["compute_radiative_fluxes"] = orig_rad
        ns["compute_idealized_radiative_fluxes"] = orig_rad_ideal
        ns["get_wind_speed"] = orig_get_wind
        ns["measured_air_temp"] = orig_air_temp
        ns["snow_depth_interp"] = orig_snow_h


def plot_importance(df: pd.DataFrame, out_png: Path):
    d = df[df["experiment"] != "baseline_measured"].copy()
    d = d.sort_values("ImpactScore", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    y = np.arange(len(d))

    axes[0].barh(y, d["dRMSE_all"], color="tab:blue", alpha=0.8)
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Delta RMSE (All)")
    axes[0].set_xlabel("C")

    axes[1].barh(y, d["dRMSE_cloud"], color="tab:purple", alpha=0.8)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Delta RMSE (Cloud)")
    axes[1].set_xlabel("C")

    axes[2].barh(y, d["dRMSE_night"], color="tab:green", alpha=0.8)
    axes[2].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[2].set_title("Delta RMSE (Night)")
    axes[2].set_xlabel("C")

    if "dGRADRMSE_all" in d.columns:
        axes[3].barh(y, d["dGRADRMSE_all"], color="tab:orange", alpha=0.8)
    else:
        axes[3].barh(y, np.zeros_like(y, dtype=float), color="tab:orange", alpha=0.8)
    axes[3].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[3].set_title("Delta Gradient RMSE")
    axes[3].set_xlabel("C/m")

    axes[0].set_yticks(y, d["experiment"])
    axes[1].set_yticks(y, [])
    axes[2].set_yticks(y, [])
    axes[3].set_yticks(y, [])
    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_surface_gradient_scatter(df: pd.DataFrame, out_png: Path):
    d = df[df["experiment"] != "baseline_measured"].copy()
    if d.empty:
        return
    color_map = {
        "longwave": "tab:purple",
        "shortwave": "tab:orange",
        "turbulent": "tab:blue",
        "snow_depth": "tab:green",
        "airtemp": "tab:brown",
        "heuristic": "tab:gray",
        "baseline": "gray",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for fam, sub in d.groupby("family"):
        ax.scatter(
            sub["dRMSE_all"].values,
            sub["dGRADRMSE_all"].values,
            s=70,
            alpha=0.85,
            label=fam,
            color=color_map.get(fam, "black"),
        )
    # Annotate top-impact experiments for readability.
    top = d.sort_values("ImpactScore", ascending=False).head(10)
    for _, r in top.iterrows():
        ax.annotate(
            r["experiment"],
            (r["dRMSE_all"], r["dGRADRMSE_all"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Surface Delta RMSE (C)")
    ax.set_ylabel("Gradient Delta RMSE (C/m)")
    ax.set_title("Surface vs Gradient Sensitivity by Experiment")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_regime_heatmap(df: pd.DataFrame, out_png: Path):
    d = df[df["experiment"] != "baseline_measured"].copy()
    if d.empty:
        return
    cols = [
        "dRMSE_all",
        "dRMSE_day",
        "dRMSE_night",
        "dRMSE_cloud",
        "dRMSE_clear",
        "dRMSE_morning",
        "dGRADRMSE_all",
        "dGRADRMSE_cloud",
        "dGRADRMSE_night",
    ]
    # Some columns may not exist if upstream changes.
    cols = [c for c in cols if c in d.columns]
    h = d.sort_values("ImpactScore", ascending=False).head(18).copy()
    mat = h[cols].to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(mat), 95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(h) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(h)), h["experiment"].tolist())
    ax.set_xticks(np.arange(len(cols)), cols, rotation=45, ha="right")
    ax.set_title("Regime Sensitivity Heatmap (Delta Metrics)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Delta value")
    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_event_composite(composite_df: pd.DataFrame, out_png: Path):
    if composite_df.empty:
        return
    hi = composite_df[composite_df["group"] == "high_error"].set_index("variable")["value"]
    lo = composite_df[composite_df["group"] == "low_error"].set_index("variable")["value"]
    vars_order = [v for v in hi.index if v in lo.index]
    x = np.arange(len(vars_order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, hi[vars_order].values, width=w, label="High-error windows", color="tab:red", alpha=0.8)
    ax.bar(x + w / 2, lo[vars_order].values, width=w, label="Low-error windows", color="tab:blue", alpha=0.8)
    ax.set_xticks(x, vars_order, rotation=35, ha="right")
    ax.set_title("Forcing/State Composite: High-error vs Low-error Windows")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_baseline_meteogram(
    timestamps: pd.DatetimeIndex,
    *,
    tair_c: np.ndarray,
    obs_c: np.ndarray,
    model_c: np.ndarray,
    err_c: np.ndarray,
    sw_down: np.ndarray,
    lw_down: np.ndarray,
    wind_ms: np.ndarray,
    cloud_mask: np.ndarray,
    out_png: Path,
):
    """Compact meteogram with baseline model error strip for practitioner interpretation."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    ax = axes[0]
    ax.plot(timestamps, obs_c, color="black", linewidth=1.4, label="Observed surface temp")
    ax.plot(timestamps, model_c, color="tab:red", linewidth=1.2, alpha=0.9, label="Model surface temp")
    ax.plot(timestamps, tair_c, color="tab:blue", linewidth=1.0, alpha=0.85, label="Air temp")
    ax.set_ylabel("Temp (C)")
    ax.set_title("Baseline Meteogram: Forcing And Surface Response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3, fontsize=9)

    ax = axes[1]
    ax.plot(timestamps, sw_down, color="goldenrod", linewidth=1.2, label="SW down")
    ax.plot(timestamps, lw_down, color="purple", linewidth=1.2, label="LW down")
    ax.set_ylabel("Radiation (W/m2)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    ax = axes[2]
    ax.plot(timestamps, wind_ms, color="tab:cyan", linewidth=1.1, label="Wind speed")
    ax.set_ylabel("Wind (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    ax.plot(timestamps, err_c, color="tab:red", linewidth=1.1, label="Model - Obs error")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("Error (C)")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Shade cloudy periods so radiation/weather regime is easy to connect to error behavior.
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

    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def build_decision_impact_table(error_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build practitioner-facing hour-count impacts relative to baseline."""
    if "model_baseline_measured_c" not in error_df.columns:
        return pd.DataFrame()

    if "grad_top20_baseline_measured_cpm" in error_df.columns:
        grad_prefix = "grad_top20"
        grad_thresholds = (10.0, 20.0)
    elif "bulk_grad_baseline_measured_cpm" in error_df.columns:
        grad_prefix = "bulk_grad"
        grad_thresholds = (20.0, 30.0)
    else:
        return pd.DataFrame()

    if "dt_hours" in error_df.columns:
        dt_hours = error_df["dt_hours"].values.astype(float)
    else:
        if "timestamp" in error_df.columns:
            t = pd.to_datetime(error_df["timestamp"]).astype("int64").values / 1e9
            dt_sec = np.diff(t)
        else:
            dt_sec = np.array([], dtype=float)
        if len(dt_sec) == 0:
            dt_sec = np.array([3600.0], dtype=float)
        dt_sec = np.r_[dt_sec, dt_sec[-1]]
        dt_hours = dt_sec / 3600.0
    dt_hours = np.maximum(dt_hours, 0.0)

    def _hours(mask):
        return float(np.sum(mask.astype(float) * dt_hours))

    g1, g2 = grad_thresholds
    base_grad = np.abs(error_df[f"{grad_prefix}_baseline_measured_cpm"].values.astype(float))
    base_surf = error_df["model_baseline_measured_c"].values.astype(float)
    experiments = [str(x) for x in metrics_df["experiment"].tolist()]

    rows = []
    for exp in experiments:
        gcol = f"{grad_prefix}_{exp}_cpm"
        scol = f"model_{exp}_c"
        if gcol not in error_df.columns or scol not in error_df.columns:
            continue
        grad = np.abs(error_df[gcol].values.astype(float))
        surf = error_df[scol].values.astype(float)

        b20 = base_grad >= g1
        s20 = grad >= g1
        b30 = base_grad >= g2
        s30 = grad >= g2

        bc15 = base_surf <= -15.0
        sc15 = surf <= -15.0
        bc20 = base_surf <= -20.0
        sc20 = surf <= -20.0

        rows.append(
            {
                "experiment": exp,
                "gradient_metric": grad_prefix,
                "gradient_threshold_1_cpm": float(g1),
                "gradient_threshold_2_cpm": float(g2),
                "grad_ge10_h": _hours(grad >= 10.0),
                "grad_ge20_h": _hours(s20),
                "grad_ge30_h": _hours(s30),
                "surf_le15_h": _hours(sc15),
                "surf_le20_h": _hours(sc20),
                "flip_grad20_h": _hours(b20 != s20),
                "extra_grad20_h": _hours((~b20) & s20),
                "missed_grad20_h": _hours(b20 & (~s20)),
                "flip_grad30_h": _hours(b30 != s30),
                "extra_grad30_h": _hours((~b30) & s30),
                "missed_grad30_h": _hours(b30 & (~s30)),
                "flip_cold15_h": _hours(bc15 != sc15),
                "extra_cold15_h": _hours((~bc15) & sc15),
                "missed_cold15_h": _hours(bc15 & (~sc15)),
                "flip_cold20_h": _hours(bc20 != sc20),
                "extra_cold20_h": _hours((~bc20) & sc20),
                "missed_cold20_h": _hours(bc20 & (~sc20)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    base = out.loc[out["experiment"] == "baseline_measured"]
    if base.empty:
        return out
    b = base.iloc[0]
    for c in ["grad_ge10_h", "grad_ge20_h", "grad_ge30_h", "surf_le15_h", "surf_le20_h"]:
        out[f"d_{c}"] = out[c] - float(b[c])

    out["DecisionImpactScore"] = (
        np.abs(out["d_grad_ge20_h"])
        + 0.7 * np.abs(out["d_surf_le15_h"])
        + 0.5 * np.abs(out["flip_grad20_h"])
        + 0.3 * np.abs(out["flip_cold15_h"])
    )
    out = out.merge(metrics_df[["experiment", "family"]], on="experiment", how="left")
    out = out.sort_values(["DecisionImpactScore", "experiment"], ascending=[False, True]).reset_index(drop=True)
    return out


def plot_decision_impact(decision_df: pd.DataFrame, out_png: Path, top_n: int = 16):
    if decision_df.empty:
        return
    d = decision_df[decision_df["experiment"] != "baseline_measured"].copy()
    if d.empty:
        return
    d = d.sort_values("DecisionImpactScore", ascending=False).head(top_n).copy()
    y = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, 0.45 * len(d) + 2)), sharey=True)

    colors_grad = np.where(d["d_grad_ge20_h"] >= 0, "tab:red", "tab:blue")
    colors_cold = np.where(d["d_surf_le15_h"] >= 0, "tab:red", "tab:blue")
    axes[0].barh(y, d["d_grad_ge20_h"].values, color=colors_grad, alpha=0.85)
    axes[1].barh(y, d["d_surf_le15_h"].values, color=colors_cold, alpha=0.85)

    for ax in axes:
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax.grid(True, axis="x", alpha=0.3)

    g1 = float(decision_df.get("gradient_threshold_1_cpm", pd.Series([20.0])).iloc[0])
    axes[0].set_title(f"Change In Hours With |Gradient| >= {g1:.0f} C/m")
    axes[0].set_xlabel("Hours (scenario - baseline)")
    axes[0].set_yticks(y, d["experiment"].tolist())
    axes[1].set_title("Change In Hours With Surface Temp <= -15 C")
    axes[1].set_xlabel("Hours (scenario - baseline)")
    axes[1].set_yticks(y, [])

    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def build_implementation_status(
    *,
    metrics_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    has_meteogram: bool,
) -> pd.DataFrame:
    """Pass/fail checklist for requested ISSW improvements."""
    exps = set(metrics_df["experiment"].astype(str).tolist())
    cols = set(metrics_df.columns.tolist())
    decision_cols = set(decision_df.columns.tolist()) if len(decision_df) else set()

    checks = [
        (
            "1_decision_error_framing",
            ("d_grad_ge20_h" in decision_cols) and ("flip_grad20_h" in decision_cols),
            "Decision-impact hour deltas/flips produced (time-weighted).",
        ),
        (
            "2_cloud_lw_pathway",
            {"lw_hourly_climatology", "lwdown_cloud_to_clear"}.issubset(exps),
            "Cloud/LW pathway scenarios included.",
        ),
        (
            "3_air_temp_proxy_benchmark",
            "air_temp_proxy" in exps,
            "Naive air-temperature proxy benchmark included.",
        ),
        (
            "5_surface_plus_gradient_error",
            {"RMSE", "dGRADRMSE_all"}.issubset(cols),
            "Surface and gradient errors reported together.",
        ),
        (
            "6_clear_cloud_split",
            {"RMSE_cloud", "RMSE_clear", "dRMSE_cloud", "dRMSE_clear"}.issubset(cols),
            "Clear/cloud regime split metrics present.",
        ),
        (
            "7_variable_importance_ranking",
            "ImpactScore" in cols,
            "Ranked variable-importance score computed.",
        ),
        (
            "8_meteogram_with_error_strip",
            has_meteogram and isinstance(windows_df, pd.DataFrame),
            "Meteogram + error-strip figure generated.",
        ),
    ]

    rows = []
    for key, ok, desc in checks:
        rows.append(
            {
                "improvement": key,
                "status": "PASS" if ok else "FAIL",
                "implemented": int(bool(ok)),
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


def _fmt_signed(x: float, nd: int = 2) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.{nd}f}"


def _get_val(df: pd.DataFrame, exp: str, col: str) -> float:
    sub = df.loc[df["experiment"] == exp, col]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0])


def _safe_ratio(num: float, den: float) -> float:
    if (not np.isfinite(num)) or (not np.isfinite(den)) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def plot_mental_model_compare(
    timestamps: pd.DatetimeIndex,
    *,
    obs_surface_c: np.ndarray,
    model_surface_c: np.ndarray,
    air_c: np.ndarray,
    grad_obs_proxy_cpm: np.ndarray,
    grad_model_cpm: np.ndarray,
    grad_naive_cpm: np.ndarray,
    out_png: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(timestamps, obs_surface_c, color="black", linewidth=1.4, label="Observed surface temp")
    ax.plot(timestamps, model_surface_c, color="tab:red", linewidth=1.2, label="Physics model surface temp")
    ax.plot(timestamps, air_c, color="tab:blue", linewidth=1.0, alpha=0.85, label="Air temp (naive proxy)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Surface Temperature: Physics Model vs Naive Air-Temp Proxy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(timestamps, grad_obs_proxy_cpm, color="black", linewidth=1.3, label="Gradient obs-proxy (surface-soil)/H")
    ax.plot(timestamps, grad_model_cpm, color="tab:red", linewidth=1.1, label="Physics model gradient")
    ax.plot(timestamps, grad_naive_cpm, color="tab:blue", linewidth=1.0, alpha=0.85, label="Naive gradient (air-soil)/H")
    ax.set_ylabel("Gradient (C/m)")
    ax.set_xlabel("Time")
    ax.set_title("Snowpack Gradient: Physics Model vs Naive Air-Ground Proxy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def build_so_what_summary(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    mental_df: pd.DataFrame | None,
    *,
    station: str,
    start: str,
    end: str,
) -> str:
    base = df.loc[df["experiment"] == "baseline_measured"].iloc[0]
    lines: list[str] = []
    lines.append(f"# {station} So-What Summary ({start} to {end})")
    lines.append("")
    lines.append("## Baseline")
    lines.append(
        f"- Baseline surface RMSE is `{base['RMSE']:.2f} C` and baseline P95 bulk gradient is `{base['P95_abs_bulk_grad_Cpm']:.1f} C/m` "
        f"(`{0.1*base['P95_abs_bulk_grad_Cpm']:.2f} C per 10 cm`)."
    )
    lines.append(
        "- So what: this is the reference skill. Every scenario below is interpreted as added error relative to this baseline."
    )
    lines.append("")

    # "How much more impactful?" statements for non-physics audiences.
    lw10_s = abs(_get_val(df, "lwdown_scale_0p9", "dRMSE_all"))
    lw10_g = abs(_get_val(df, "lwdown_scale_0p9", "dGRADRMSE_all"))
    sen_s = abs(_get_val(df, "sensible_scale_0p7", "dRMSE_all"))
    sen_g = abs(_get_val(df, "sensible_scale_0p7", "dGRADRMSE_all"))
    lat_s = abs(_get_val(df, "latent_scale_0p7", "dRMSE_all"))
    lat_g = abs(_get_val(df, "latent_scale_0p7", "dGRADRMSE_all"))
    sw_s = abs(_get_val(df, "sw_daytime_climatology", "dRMSE_all"))
    sw_g = abs(_get_val(df, "sw_daytime_climatology", "dGRADRMSE_all"))
    wind_s = abs(_get_val(df, "wind_scale_0p8", "dRMSE_all"))
    wind_g = abs(_get_val(df, "wind_scale_0p8", "dGRADRMSE_all"))

    lines.append("## Big Plain-Language Numbers")
    lines.append(
        f"- A 10% LWdown error (`lwdown_scale_0p9`) has `{_safe_ratio(lw10_s, sen_s):.1f}x` the surface impact and "
        f"`{_safe_ratio(lw10_g, sen_g):.1f}x` the gradient impact of a 30% sensible-flux error (`sensible_scale_0p7`)."
    )
    lines.append(
        f"- The same 10% LWdown error has `{_safe_ratio(lw10_s, lat_s):.1f}x` the surface impact and "
        f"`{_safe_ratio(lw10_g, lat_g):.1f}x` the gradient impact of a 30% latent-flux error (`latent_scale_0p7`)."
    )
    lines.append(
        f"- Cloud-averaged daytime net-SW simplification (`sw_daytime_climatology`) changes gradients by `{sw_g:.2f} C/m`, "
        f"while a 20% wind error (`wind_scale_0p8`) changes gradients by `{wind_g:.2f} C/m`."
    )
    lines.append(
        "- So what: forgetting LW does not just nudge surface temperature; it changes the internal snowpack thermal gradient enough to alter stability interpretation."
    )
    lines.append("")

    if mental_df is not None and len(mental_df):
        m = mental_df.iloc[0]
        lines.append("## Mental Model Check (Air + Sun + Wind Only)")
        lines.append(
            f"- Treating surface temp as air temp gives RMSE `{m['surface_rmse_air_proxy_c']:.2f} C` versus "
            f"`{m['surface_rmse_model_c']:.2f} C` for the physics model."
        )
        lines.append(
            f"- Treating gradient as `(air - ground)/H` gives RMSE `{m['grad_rmse_airground_proxy_cpm']:.2f} C/m` versus "
            f"`{m['grad_rmse_model_vs_obsproxy_cpm']:.2f} C/m` for the physics model."
        )
        lines.append(
            f"- Physics model reduces gradient error by `{m['grad_error_reduction_pct_vs_airground_proxy']:.1f}%` relative to the air-ground proxy."
        )
        lines.append(
            "- So what: the common beginner mental model misses important radiation-driven structure in the snowpack."
        )
        lines.append("")

    def add_stmt(exp: str, headline: str, why: str):
        sub = df.loc[df["experiment"] == exp]
        if sub.empty:
            return
        r = sub.iloc[0]
        lines.append(f"## {headline}")
        lines.append(
            f"- `{exp}` changes surface RMSE by `{_fmt_signed(r['dRMSE_all'])} C` and gradient RMSE by "
            f"`{_fmt_signed(r['dGRADRMSE_all'])} C/m` (`{_fmt_signed(0.1*r['dGRADRMSE_all'])} C/10 cm`)."
        )
        lines.append(
            f"- Cloud-window impact: surface `{_fmt_signed(r['dRMSE_cloud'])} C`, gradient `{_fmt_signed(r['dGRADRMSE_cloud'])} C/m`."
        )
        lines.append(f"- So what: {why}")
        lines.append("")

    # Core statements aligned with user goals.
    add_stmt(
        "sw_hourly_climatology",
        "Using Climatological Net SW Instead Of Measured Net SW",
        "removing observed cloud-driven SW variability in absorbed shortwave materially degrades both surface temperature and gradient skill; "
        "daytime cloud effects should be represented in operational interpretation.",
    )
    add_stmt(
        "lw_hourly_climatology",
        "Using Climatological LW Instead Of Measured LW",
        "removing event-scale cloud LW variability strongly degrades snow temperature and gradient skill, even when daily weather 'feels' similar.",
    )
    add_stmt(
        "sw_cloud_day_climatology",
        "Using Climatological Net SW Only In Cloudy Daytime",
        "even partial SW simplification during cloudy daytime changes gradients, showing cloud-radiation timing matters for snowpack thermal structure.",
    )
    add_stmt(
        "sw_no_up_alpha_0p80",
        "No SW-Up Sensor Case (Fixed Albedo 0.80)",
        "assuming fixed albedo when SW-up is unavailable can materially shift absorbed shortwave and therefore surface and gradient interpretation.",
    )
    add_stmt(
        "no_sensible",
        "Leaving Out Sensible Heat Flux",
        "sensible exchange is a first-order control on snow temperature and gradient evolution, especially during transition periods.",
    )
    add_stmt(
        "no_latent",
        "Leaving Out Latent Heat Flux",
        "latent exchange affects results but is lower impact than LW/SW and sensible in this dry winter window.",
    )
    add_stmt(
        "lwdown_parameterized",
        "Replacing Measured LWdown With Parameterized LWdown",
        "LWdown representation is one of the strongest error drivers; inaccurate cloud/atmospheric longwave forcing strongly propagates into snow temperatures and gradients.",
    )
    add_stmt(
        "lwdown_cloud_to_clear",
        "Removing Cloud LW Enhancement (Cloud Converted To Clear-LW)",
        "cloud longwave forcing is critical; assuming clear-sky LW during cloudy periods causes large errors in both surface and gradient estimates.",
    )
    add_stmt(
        "lwdown_scale_0p9_cloud_night",
        "Reducing LWdown During Cloudy Nights (90%)",
        "nighttime cloud longwave input is highly influential for overnight cooling control and near-surface gradient persistence.",
    )
    add_stmt(
        "lwdown_scale_1p1_cloud_night",
        "Increasing LWdown During Cloudy Nights (110%)",
        "small LW adjustments in cloudy nights shift thermal gradients enough to matter for snow stability interpretation.",
    )
    add_stmt(
        "snowdepth_scale_0p9",
        "Snow Depth Bias Test (-10%)",
        "snow depth uncertainty can have modest surface impact but meaningful gradient impact, so depth quality matters for gradient-focused products.",
    )
    add_stmt(
        "airtemp_scale_0p0",
        "Flattening Air-Temperature Variability To Monthly Mean (0%)",
        "air temperature alone cannot reproduce observed snow thermal evolution; removing air variability creates large interpretive errors.",
    )
    add_stmt(
        "wind_scale_0p0",
        "Removing Wind Influence (0%)",
        "wind effects can be large and nonlinear; this extreme test is memorable for training because it shows how quickly flux balance can drift.",
    )

    # Realistic ranking section (exclude extreme stress tests).
    realistic = [
        "baseline_measured",
        "sw_hourly_climatology",
        "sw_daytime_climatology",
        "sw_cloud_day_climatology",
        "sw_no_up_alpha_0p80",
        "lw_hourly_climatology",
        "lwdown_scale_0p9",
        "lwdown_scale_1p1",
        "lwdown_scale_0p9_cloud_night",
        "lwdown_scale_1p1_cloud_night",
        "lwdown_scale_0p9_cloud_day",
        "lwdown_scale_1p1_cloud_day",
        "lwdown_scale_0p9_clear_night",
        "lwdown_scale_1p1_clear_night",
        "sensible_scale_0p7",
        "sensible_scale_1p3",
        "latent_scale_0p7",
        "latent_scale_1p3",
        "airtemp_scale_0p75",
        "airtemp_scale_0p5",
        "airtemp_scale_0p25",
        "airtemp_scale_0p0",
        "wind_scale_0p8",
        "wind_scale_1p2",
        "wind_scale_0p75",
        "wind_scale_0p5",
        "wind_scale_0p25",
        "wind_scale_0p0",
        "snowdepth_scale_0p9",
        "snowdepth_scale_1p1",
        "air_temp_proxy",
    ]
    rr = (
        df[df["experiment"].isin(realistic)]
        .sort_values("ImpactScore", ascending=False)
        .loc[:, ["experiment", "family", "dRMSE_all", "dGRADRMSE_all", "ImpactScore"]]
        .head(8)
    )
    lines.append("## Top Realistic Uncertainty Drivers")
    for _, r in rr.iterrows():
        lines.append(
            f"- `{r['experiment']}` ({r['family']}): surface `{_fmt_signed(r['dRMSE_all'])} C`, "
            f"gradient `{_fmt_signed(r['dGRADRMSE_all'])} C/m`."
        )
    lines.append("")

    if len(windows_df):
        top = windows_df.head(5)
        tag_counts = top["weather_tag"].value_counts().to_dict()
        lines.append("## Weather Windows That Matter Most")
        lines.append(
            f"- Highest-error 6-hour windows are mostly tagged as `{tag_counts}` with typical "
            f"LWdown `{top['mean_lw_down_wm2'].mean():.1f} W/m2`, SWdown `{top['mean_sw_down_wm2'].mean():.1f} W/m2`, "
            f"wind `{top['mean_wind_ms'].mean():.2f} m/s`."
        )
        lines.append(
            "- So what: forecast/measurement uncertainty during these window types will have outsized effects on both snow surface temperature and gradient diagnostics."
        )
        lines.append("")

    if len(composite_df):
        hi = composite_df[composite_df["group"] == "high_error"].set_index("variable")["value"].to_dict()
        lo = composite_df[composite_df["group"] == "low_error"].set_index("variable")["value"].to_dict()
        if hi and lo:
            lines.append("## High-Error vs Low-Error Composite")
            lines.append(
                f"- High-error windows: cloud fraction `{hi.get('cloud_frac', np.nan):.2f}`, LWdown `{hi.get('mean_lw_down_wm2', np.nan):.1f} W/m2`, "
                f"wind `{hi.get('mean_wind_ms', np.nan):.2f} m/s`."
            )
            lines.append(
                f"- Low-error windows: cloud fraction `{lo.get('cloud_frac', np.nan):.2f}`, LWdown `{lo.get('mean_lw_down_wm2', np.nan):.1f} W/m2`, "
                f"wind `{lo.get('mean_wind_ms', np.nan):.2f} m/s`."
            )
            lines.append(
                "- So what: this gives a concrete operational context for when radiation and turbulent forcings are most likely to cause model interpretation risk."
            )
            lines.append("")

    lines.append("## Practical Takeaway For Snow Safety")
    lines.append(
        "- For Senator Beck in this month, LW/SW forcing fidelity controls much of the combined surface+gradient skill; "
        "sensible is important but secondary under realistic perturbations, and latent is usually smaller."
    )
    lines.append(
        "- This supports prioritizing high-quality radiation observations (especially cloud-modulated LW and SW) when gradient-based decisions are important."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outdir = (repo_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

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
        print("ERROR: notebook setup execution failed.")
        print(type(exc).__name__, exc)
        return 2

    required = [
        "run_snow_model",
        "usgs_temp_obs_interp",
        "times_sec",
        "measured_air_temp",
        "SW_in_interp",
        "compute_shortwave_net",
        "LW_in_interp",
        "RH_arr_interp",
        "get_wind_speed",
    ]
    missing = [k for k in required if k not in ns]
    if missing:
        print(f"ERROR: missing required notebook objects: {missing}")
        return 3

    use_skin = args.skin == "on"
    t_eval = np.asarray(ns["times_sec"], dtype=float)
    obs_c = np.asarray(ns["usgs_temp_obs_interp"](t_eval), dtype=float)
    timestamps = pd.to_datetime(ns["df"].index[0]) + pd.to_timedelta(t_eval, unit="s")
    start_ts = pd.to_datetime(ns["df"].index[0])
    start_offset_sec = int(start_ts.hour * 3600 + start_ts.minute * 60 + start_ts.second)
    h_eval = np.maximum(np.asarray(ns["snow_depth_interp"](t_eval), dtype=float), 0.10)

    tair_k = np.asarray(ns["measured_air_temp"](t_eval), dtype=float)
    sw_down = np.asarray(ns["SW_in_interp"](t_eval), dtype=float)
    lw_down = np.asarray(ns["LW_in_interp"](t_eval), dtype=float)
    wind_ms = np.asarray([float(ns["get_wind_speed"](t)) for t in t_eval], dtype=float)

    # Optional ground/soil observation for gradient proxy diagnostics.
    has_soil_obs = False
    soil_obs_c = np.full_like(obs_c, np.nan, dtype=float)
    for soil_col in ("soil_temp_c", "soil_temp_c_2"):
        if soil_col in ns["df"].columns:
            soil_raw = pd.to_numeric(ns["df"][soil_col], errors="coerce").values.astype(float)
            if np.isfinite(soil_raw).sum() >= max(10, int(0.2 * len(soil_raw))):
                soil_obs_c = (
                    pd.Series(soil_raw)
                    .interpolate(limit_direction="both")
                    .ffill()
                    .bfill()
                    .values
                )
                has_soil_obs = np.isfinite(soil_obs_c).sum() > 0
                if has_soil_obs:
                    break
    if has_soil_obs:
        grad_obs_proxy_cpm = (obs_c - soil_obs_c) / h_eval
    else:
        grad_obs_proxy_cpm = np.full_like(obs_c, np.nan, dtype=float)

    local_hour = (
        pd.DatetimeIndex(timestamps).hour
        + pd.DatetimeIndex(timestamps).minute / 60.0
        + pd.DatetimeIndex(timestamps).second / 3600.0
    )
    day_mask = sw_down > float(args.night_sw_threshold)
    night_mask = ~day_mask
    morning_mask = (local_hour >= 9.0) & (local_hour <= 11.0)

    eps_atm_obs = lw_down / (float(ns["sigma"]) * (tair_k**4))
    eps_atm_obs = np.clip(eps_atm_obs, 0.0, 1.5)
    cloud_mask = eps_atm_obs >= float(args.cloud_eps_threshold)
    clear_mask = ~cloud_mask
    cloud_night = cloud_mask & night_mask
    low_wind = wind_ms < 2.0
    high_sw = (sw_down > 300.0) & day_mask

    window_masks = {
        "all": np.ones_like(day_mask, dtype=bool),
        "day": day_mask,
        "night": night_mask,
        "morning_9_11": morning_mask,
        "cloud": cloud_mask,
        "clear": clear_mask,
        "cloud_night": cloud_night,
        "low_wind": low_wind,
        "high_sw_day": high_sw,
    }

    lw_hourly_curve = make_hourly_curve(t_eval, lw_down, start_offset_sec=start_offset_sec)
    sw_net_eval = _eval_scalar_func_on_array(ns["compute_shortwave_net"], t_eval)
    sw_net_hourly_curve = make_hourly_curve(t_eval, sw_net_eval, start_offset_sec=start_offset_sec)
    cloud_interp = ns["interp1d"](
        t_eval,
        cloud_mask.astype(float),
        kind="nearest",
        bounds_error=False,
        fill_value=(float(cloud_mask[0]), float(cloud_mask[-1])),
    )
    clear_df = pd.DataFrame(
        {
            "hour": hour_index_from_seconds(t_eval, start_offset_sec),
            "lw": lw_down,
            "cloud": cloud_mask,
        }
    )
    lw_clear_hourly_s = clear_df.loc[~clear_df["cloud"]].groupby("hour")["lw"].median()
    lw_clear_hourly_s = lw_clear_hourly_s.reindex(range(24))
    lw_clear_hourly_s = lw_clear_hourly_s.interpolate(limit_direction="both")
    lw_clear_hourly_s = lw_clear_hourly_s.fillna(float(np.nanmedian(lw_down)))
    lw_clear_hourly = lw_clear_hourly_s.values

    rows = []
    grad_by_experiment: dict[str, np.ndarray] = {}
    grad_top20_by_experiment: dict[str, np.ndarray] = {}
    error_df = pd.DataFrame({"timestamp": timestamps, "obs_c": obs_c})
    dt_sec = np.diff(t_eval)
    if len(dt_sec) == 0:
        dt_sec = np.array([3600.0], dtype=float)
    dt_sec = np.r_[dt_sec, dt_sec[-1]]
    error_df["dt_hours"] = dt_sec / 3600.0
    experiments = build_experiments()
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        row, model_eval_c, err, grad_pack = run_experiment(
            ns,
            exp,
            use_skin=use_skin,
            skin_beta=float(args.skin_beta),
            t_eval=t_eval,
            obs_c=obs_c,
            window_masks=window_masks,
            sw_net_hourly_curve=sw_net_hourly_curve,
            cloud_interp=cloud_interp,
            lw_clear_hourly=lw_clear_hourly,
            lw_hourly_curve=lw_hourly_curve,
            start_offset_sec=start_offset_sec,
            night_sw_threshold=float(args.night_sw_threshold),
        )
        rows.append(row)
        grad_by_experiment[exp.name] = grad_pack["bulk"]
        grad_top20_by_experiment[exp.name] = grad_pack["top20"]
        error_df[f"model_{exp.name}_c"] = model_eval_c
        error_df[f"error_{exp.name}_c"] = err
        error_df[f"bulk_grad_{exp.name}_cpm"] = grad_pack["bulk"]
        error_df[f"grad_top20_{exp.name}_cpm"] = grad_pack["top20"]
        error_df[f"grad_top10_{exp.name}_cpm"] = grad_pack["top10"]
        error_df[f"grad_top60_{exp.name}_cpm"] = grad_pack["top60"]
        if has_soil_obs:
            error_df[f"grad_proxy_err_{exp.name}_cpm"] = grad_pack["bulk"] - grad_obs_proxy_cpm

    # Add a direct heuristic benchmark as an explicit scenario row:
    # surface temp proxy = air temp.
    air_proxy_c = tair_k - 273.15
    if has_soil_obs:
        soil_for_proxy_c = soil_obs_c
    else:
        t_ground_c = float(ns.get("T_ground", 273.15) - 273.15)
        soil_for_proxy_c = np.full_like(air_proxy_c, t_ground_c, dtype=float)
    air_proxy_grad_cpm = (air_proxy_c - soil_for_proxy_c) / h_eval
    air_proxy_err = air_proxy_c - obs_c
    air_proxy_row = {
        "experiment": "air_temp_proxy",
        "family": "heuristic",
        "notes": "Direct surface proxy using air temperature only (no energy-balance model).",
        "MAE": float(np.mean(np.abs(air_proxy_err))),
        "RMSE": float(np.sqrt(np.mean(air_proxy_err**2))),
        "Bias": float(np.mean(air_proxy_err)),
        "Correlation": float(np.corrcoef(obs_c, air_proxy_c)[0, 1]),
        "Mean_abs_bulk_grad_Cpm": float(np.mean(np.abs(air_proxy_grad_cpm))),
        "P95_abs_bulk_grad_Cpm": float(np.nanpercentile(np.abs(air_proxy_grad_cpm), 95)),
        "Max_abs_bulk_grad_Cpm": float(np.nanmax(np.abs(air_proxy_grad_cpm))),
    }
    for wname, m in window_masks.items():
        rmse_w, bias_w = compute_metrics(air_proxy_err, obs_c, air_proxy_c, m)
        air_proxy_row[f"RMSE_{wname}"] = rmse_w
        air_proxy_row[f"Bias_{wname}"] = bias_w
    rows.append(air_proxy_row)
    grad_by_experiment["air_temp_proxy"] = air_proxy_grad_cpm
    grad_top20_by_experiment["air_temp_proxy"] = air_proxy_grad_cpm
    error_df["model_air_temp_proxy_c"] = air_proxy_c
    error_df["error_air_temp_proxy_c"] = air_proxy_err
    error_df["bulk_grad_air_temp_proxy_cpm"] = air_proxy_grad_cpm
    error_df["grad_top20_air_temp_proxy_cpm"] = air_proxy_grad_cpm
    error_df["grad_top10_air_temp_proxy_cpm"] = air_proxy_grad_cpm
    error_df["grad_top60_air_temp_proxy_cpm"] = air_proxy_grad_cpm
    if has_soil_obs:
        error_df["grad_proxy_err_air_temp_proxy_cpm"] = air_proxy_grad_cpm - grad_obs_proxy_cpm

    df = pd.DataFrame(rows)
    if "baseline_measured" not in set(df["experiment"]):
        print("ERROR: baseline_measured missing.")
        return 4

    base = df.loc[df["experiment"] == "baseline_measured"].iloc[0]
    df["dRMSE_all"] = df["RMSE"] - float(base["RMSE"])
    df["dRMSE_day"] = df["RMSE_day"] - float(base["RMSE_day"])
    df["dRMSE_cloud"] = df["RMSE_cloud"] - float(base["RMSE_cloud"])
    df["dRMSE_clear"] = df["RMSE_clear"] - float(base["RMSE_clear"])
    df["dRMSE_night"] = df["RMSE_night"] - float(base["RMSE_night"])
    df["dRMSE_morning"] = df["RMSE_morning_9_11"] - float(base["RMSE_morning_9_11"])

    # Gradient sensitivity vs baseline modeled gradient.
    base_grad = grad_by_experiment["baseline_measured"]
    grad_rows = []
    for exp_name, g in grad_by_experiment.items():
        dg = g - base_grad
        row = {
            "experiment": exp_name,
            "GRADRMSE_change_all": float(np.sqrt(np.mean(dg**2))),
            "GRADRMSE_change_cloud": float(np.sqrt(np.mean((dg[cloud_mask]) ** 2))) if np.any(cloud_mask) else np.nan,
            "GRADRMSE_change_night": float(np.sqrt(np.mean((dg[night_mask]) ** 2))) if np.any(night_mask) else np.nan,
            "GRADRMSE_change_morning": float(np.sqrt(np.mean((dg[morning_mask]) ** 2))) if np.any(morning_mask) else np.nan,
            "Mean_abs_dGrad_cpm": float(np.mean(np.abs(dg))),
            "P95_abs_dGrad_cpm": float(np.nanpercentile(np.abs(dg), 95)),
        }
        if exp_name in grad_top20_by_experiment and "baseline_measured" in grad_top20_by_experiment:
            dg20 = grad_top20_by_experiment[exp_name] - grad_top20_by_experiment["baseline_measured"]
            row["GRAD20RMSE_change_all"] = float(np.sqrt(np.mean(dg20**2)))
            row["GRAD20RMSE_change_cloud"] = (
                float(np.sqrt(np.mean((dg20[cloud_mask]) ** 2))) if np.any(cloud_mask) else np.nan
            )
            row["GRAD20RMSE_change_night"] = (
                float(np.sqrt(np.mean((dg20[night_mask]) ** 2))) if np.any(night_mask) else np.nan
            )
        if has_soil_obs:
            ge = g - grad_obs_proxy_cpm
            row["GRADRMSE_vs_obsproxy_all"] = float(np.sqrt(np.nanmean(ge**2)))
            row["GRADBias_vs_obsproxy_all"] = float(np.nanmean(ge))
            row["GRADRMSE_vs_obsproxy_cloud"] = (
                float(np.sqrt(np.nanmean((ge[cloud_mask]) ** 2))) if np.any(cloud_mask) else np.nan
            )
            row["GRADRMSE_vs_obsproxy_night"] = (
                float(np.sqrt(np.nanmean((ge[night_mask]) ** 2))) if np.any(night_mask) else np.nan
            )
        grad_rows.append(row)
    grad_df = pd.DataFrame(grad_rows)
    df = df.merge(grad_df, on="experiment", how="left")
    # Rename for simpler plotting labels
    df["dGRADRMSE_all"] = df["GRADRMSE_change_all"]
    df["dGRADRMSE_cloud"] = df["GRADRMSE_change_cloud"]
    df["dGRADRMSE_night"] = df["GRADRMSE_change_night"]
    if "GRAD20RMSE_change_all" in df.columns:
        df["dGRAD20RMSE_all"] = df["GRAD20RMSE_change_all"]

    grad_score = (
        np.abs(df["dGRADRMSE_all"])
        + 0.6 * np.abs(df["dGRADRMSE_cloud"])
        + 0.6 * np.abs(df["dGRADRMSE_night"])
    )
    if {"GRAD20RMSE_change_all", "GRAD20RMSE_change_cloud", "GRAD20RMSE_change_night"}.issubset(df.columns):
        grad_score = grad_score + 0.8 * (
            np.abs(df["GRAD20RMSE_change_all"])
            + 0.6 * np.abs(df["GRAD20RMSE_change_cloud"])
            + 0.6 * np.abs(df["GRAD20RMSE_change_night"])
        )

    df["ImpactScore"] = (
        np.abs(df["dRMSE_all"])
        + 0.6 * np.abs(df["dRMSE_cloud"])
        + 0.6 * np.abs(df["dRMSE_night"])
        + 0.4 * np.abs(df["dRMSE_morning"])
        + 0.5 * grad_score
    )
    df = df.sort_values(["ImpactScore", "RMSE"], ascending=[False, True]).reset_index(drop=True)

    metrics_csv = outdir / f"variable_importance_{args.station}_{args.start}_{args.end}.csv"
    errors_csv = outdir / f"variable_importance_timeseries_{args.station}_{args.start}_{args.end}.csv"
    decision_csv = outdir / f"decision_impact_hours_{args.station}_{args.start}_{args.end}.csv"
    decision_png = outdir / f"decision_impact_plot_{args.station}_{args.start}_{args.end}.png"
    decision_md = outdir / f"decision_impact_summary_{args.station}_{args.start}_{args.end}.md"
    fig_png = outdir / f"variable_importance_plot_{args.station}_{args.start}_{args.end}.png"
    fig_scatter_png = outdir / f"variable_importance_surface_gradient_scatter_{args.station}_{args.start}_{args.end}.png"
    fig_heat_png = outdir / f"variable_importance_regime_heatmap_{args.station}_{args.start}_{args.end}.png"
    fig_event_png = outdir / f"variable_importance_event_composite_{args.station}_{args.start}_{args.end}.png"
    fig_mental_png = outdir / f"mental_model_compare_{args.station}_{args.start}_{args.end}.png"
    fig_meteogram_png = outdir / f"variable_importance_meteogram_{args.station}_{args.start}_{args.end}.png"
    impl_csv = outdir / f"implementation_status_{args.station}_{args.start}_{args.end}.csv"
    impl_md = outdir / f"implementation_status_{args.station}_{args.start}_{args.end}.md"
    df.to_csv(metrics_csv, index=False)
    error_df.to_csv(errors_csv, index=False)
    decision_df = build_decision_impact_table(error_df, df)
    decision_df.to_csv(decision_csv, index=False)
    plot_decision_impact(decision_df, decision_png)
    plot_importance(df, fig_png)
    plot_surface_gradient_scatter(df, fig_scatter_png)
    plot_regime_heatmap(df, fig_heat_png)

    # Weather-window count file for quick reporting.
    counts = {
        "all_n": int(window_masks["all"].sum()),
        "day_n": int(window_masks["day"].sum()),
        "night_n": int(window_masks["night"].sum()),
        "morning_9_11_n": int(window_masks["morning_9_11"].sum()),
        "cloud_n": int(window_masks["cloud"].sum()),
        "clear_n": int(window_masks["clear"].sum()),
        "cloud_night_n": int(window_masks["cloud_night"].sum()),
        "low_wind_n": int(window_masks["low_wind"].sum()),
        "high_sw_day_n": int(window_masks["high_sw_day"].sum()),
        "cloud_eps_threshold": float(args.cloud_eps_threshold),
        "night_sw_threshold": float(args.night_sw_threshold),
        "has_soil_obs": int(has_soil_obs),
        "all_h": float(np.sum(error_df["dt_hours"])),
        "day_h": float(np.sum(error_df["dt_hours"] * window_masks["day"].astype(float))),
        "night_h": float(np.sum(error_df["dt_hours"] * window_masks["night"].astype(float))),
        "cloud_h": float(np.sum(error_df["dt_hours"] * window_masks["cloud"].astype(float))),
    }
    pd.DataFrame([counts]).to_csv(
        outdir / f"variable_importance_window_counts_{args.station}_{args.start}_{args.end}.csv",
        index=False,
    )

    # Baseline weather-window ranking (6-hour blocks) for "focus on step 2" workflows.
    wdf = pd.DataFrame(
        {
            "timestamp": timestamps,
            "error_baseline_c": error_df["error_baseline_measured_c"].values,
            "cloud_like": cloud_mask.astype(int),
            "sw_down_wm2": sw_down,
            "lw_down_wm2": lw_down,
            "wind_ms": wind_ms,
            "rh_pct": np.asarray(ns["RH_arr_interp"](t_eval), dtype=float),
            "air_temp_c": tair_k - 273.15,
        }
    )
    wr = []
    freq_td = pd.Timedelta(hours=float(args.event_window_hours))
    for t0, g in wdf.set_index("timestamp").groupby(pd.Grouper(freq=freq_td)):
        if len(g) < 4:
            continue
        err = g["error_baseline_c"].values
        wr.append(
            {
                "start": g.index.min(),
                "end": g.index.max(),
                "n_points": int(len(g)),
                "rmse_c": float(np.sqrt(np.mean(err**2))),
                "bias_c": float(np.mean(err)),
                "abs_err_p95_c": float(np.percentile(np.abs(err), 95)),
                "cloud_frac": float(g["cloud_like"].mean()),
                "mean_lw_down_wm2": float(g["lw_down_wm2"].mean()),
                "mean_sw_down_wm2": float(g["sw_down_wm2"].mean()),
                "mean_wind_ms": float(g["wind_ms"].mean()),
                "mean_rh_pct": float(g["rh_pct"].mean()),
                "mean_air_temp_c": float(g["air_temp_c"].mean()),
            }
        )
    windows_df = pd.DataFrame(wr)
    if len(windows_df):
        windows_df["weather_tag"] = "mixed"
        windows_df.loc[windows_df["cloud_frac"] >= 0.7, "weather_tag"] = "cloudy"
        windows_df.loc[windows_df["cloud_frac"] <= 0.2, "weather_tag"] = "clear"
        windows_df.loc[
            (windows_df["mean_sw_down_wm2"] > 250.0) & (windows_df["cloud_frac"] < 0.5),
            "weather_tag",
        ] = "sunny_clear"
        windows_df.loc[
            (windows_df["mean_wind_ms"] < 2.0) & (windows_df["weather_tag"] == "clear"),
            "weather_tag",
        ] = "clear_calm"
        windows_df.loc[windows_df["mean_wind_ms"] >= 5.0, "weather_tag"] = "windy"
        windows_df = windows_df.sort_values("rmse_c", ascending=False).reset_index(drop=True)
    windows_csv = outdir / f"variable_importance_weather_windows_{args.station}_{args.start}_{args.end}.csv"
    windows_df.to_csv(windows_csv, index=False)

    # Composite high-vs-low error window forcing/stability conditions.
    composite_csv = outdir / f"variable_importance_event_composite_{args.station}_{args.start}_{args.end}.csv"
    composite_df = pd.DataFrame(columns=["group", "variable", "value"])
    if len(windows_df) >= max(2, args.event_top_k):
        top_k = int(args.event_top_k)
        hi = windows_df.head(top_k)
        lo = windows_df.tail(top_k)

        def _agg(sub: pd.DataFrame, label: str) -> list[dict]:
            return [
                {"group": label, "variable": "rmse_c", "value": float(sub["rmse_c"].mean())},
                {"group": label, "variable": "abs_err_p95_c", "value": float(sub["abs_err_p95_c"].mean())},
                {"group": label, "variable": "cloud_frac", "value": float(sub["cloud_frac"].mean())},
                {"group": label, "variable": "mean_lw_down_wm2", "value": float(sub["mean_lw_down_wm2"].mean())},
                {"group": label, "variable": "mean_sw_down_wm2", "value": float(sub["mean_sw_down_wm2"].mean())},
                {"group": label, "variable": "mean_wind_ms", "value": float(sub["mean_wind_ms"].mean())},
                {"group": label, "variable": "mean_rh_pct", "value": float(sub["mean_rh_pct"].mean())},
                {"group": label, "variable": "mean_air_temp_c", "value": float(sub["mean_air_temp_c"].mean())},
            ]

        composite_rows = _agg(hi, "high_error") + _agg(lo, "low_error")
        composite_df = pd.DataFrame(composite_rows)
    composite_df.to_csv(composite_csv, index=False)
    plot_event_composite(composite_df, fig_event_png)

    # Mental-model benchmark table + figure.
    mental_csv = outdir / f"mental_model_benchmark_{args.station}_{args.start}_{args.end}.csv"
    tair_c = tair_k - 273.15
    model_surface_baseline = error_df["model_baseline_measured_c"].values.astype(float)
    grad_model_baseline = error_df["bulk_grad_baseline_measured_cpm"].values.astype(float)
    if has_soil_obs:
        grad_naive = (tair_c - soil_obs_c) / h_eval
        grad_obs = grad_obs_proxy_cpm
    else:
        t_ground_c = float(ns.get("T_ground", 273.15) - 273.15)
        soil_obs_c = np.full_like(tair_c, t_ground_c, dtype=float)
        grad_naive = (tair_c - soil_obs_c) / h_eval
        grad_obs = np.full_like(grad_naive, np.nan, dtype=float)

    surf_rmse_air = float(np.sqrt(np.mean((tair_c - obs_c) ** 2)))
    surf_rmse_model = float(np.sqrt(np.mean((model_surface_baseline - obs_c) ** 2)))
    surf_bias_air = float(np.mean(tair_c - obs_c))
    surf_bias_model = float(np.mean(model_surface_baseline - obs_c))

    m = np.isfinite(grad_obs) & np.isfinite(grad_model_baseline) & np.isfinite(grad_naive)
    if np.any(m):
        grad_rmse_model = float(np.sqrt(np.mean((grad_model_baseline[m] - grad_obs[m]) ** 2)))
        grad_rmse_naive = float(np.sqrt(np.mean((grad_naive[m] - grad_obs[m]) ** 2)))
        grad_bias_model = float(np.mean(grad_model_baseline[m] - grad_obs[m]))
        grad_bias_naive = float(np.mean(grad_naive[m] - grad_obs[m]))
        grad_corr_model = float(np.corrcoef(grad_model_baseline[m], grad_obs[m])[0, 1])
        grad_corr_naive = float(np.corrcoef(grad_naive[m], grad_obs[m])[0, 1])
        grad_reduction_pct = float(100.0 * (grad_rmse_naive - grad_rmse_model) / grad_rmse_naive) if grad_rmse_naive > 0 else np.nan
    else:
        grad_rmse_model = np.nan
        grad_rmse_naive = np.nan
        grad_bias_model = np.nan
        grad_bias_naive = np.nan
        grad_corr_model = np.nan
        grad_corr_naive = np.nan
        grad_reduction_pct = np.nan

    mental_df = pd.DataFrame(
        [
            {
                "station": args.station,
                "start": args.start,
                "end": args.end,
                "surface_rmse_air_proxy_c": surf_rmse_air,
                "surface_rmse_model_c": surf_rmse_model,
                "surface_bias_air_proxy_c": surf_bias_air,
                "surface_bias_model_c": surf_bias_model,
                "grad_rmse_airground_proxy_cpm": grad_rmse_naive,
                "grad_rmse_model_vs_obsproxy_cpm": grad_rmse_model,
                "grad_bias_airground_proxy_cpm": grad_bias_naive,
                "grad_bias_model_vs_obsproxy_cpm": grad_bias_model,
                "grad_corr_airground_proxy": grad_corr_naive,
                "grad_corr_model_vs_obsproxy": grad_corr_model,
                "grad_error_reduction_pct_vs_airground_proxy": grad_reduction_pct,
                "has_soil_obs": int(has_soil_obs),
            }
        ]
    )
    mental_df.to_csv(mental_csv, index=False)
    plot_mental_model_compare(
        pd.DatetimeIndex(timestamps),
        obs_surface_c=obs_c,
        model_surface_c=model_surface_baseline,
        air_c=tair_c,
        grad_obs_proxy_cpm=grad_obs,
        grad_model_cpm=grad_model_baseline,
        grad_naive_cpm=grad_naive,
        out_png=fig_mental_png,
    )
    plot_baseline_meteogram(
        pd.DatetimeIndex(timestamps),
        tair_c=tair_c,
        obs_c=obs_c,
        model_c=model_surface_baseline,
        err_c=error_df["error_baseline_measured_c"].values.astype(float),
        sw_down=sw_down,
        lw_down=lw_down,
        wind_ms=wind_ms,
        cloud_mask=cloud_mask,
        out_png=fig_meteogram_png,
    )

    impl_df = build_implementation_status(
        metrics_df=df,
        decision_df=decision_df,
        windows_df=windows_df,
        has_meteogram=fig_meteogram_png.exists(),
    )
    impl_df.to_csv(impl_csv, index=False)
    n_pass = int((impl_df["status"] == "PASS").sum())
    n_total = int(len(impl_df))
    impl_lines = [
        f"# Implementation Status ({args.station}, {args.start} to {args.end})",
        "",
        f"- Passed `{n_pass}/{n_total}` requested improvements.",
        "",
        "## Requested Improvements",
    ]
    for _, r in impl_df.iterrows():
        mark = "PASS" if r["status"] == "PASS" else "FAIL"
        impl_lines.append(f"- `{r['improvement']}`: **{mark}** - {r['description']}")
    impl_lines.append("")
    impl_lines.append("## Evidence Files")
    impl_lines.append(f"- Metrics table: `{metrics_csv.name}`")
    impl_lines.append(f"- Decision-impact table: `{decision_csv.name}`")
    impl_lines.append(f"- Ranking plot: `{fig_png.name}`")
    impl_lines.append(f"- Meteogram: `{fig_meteogram_png.name}`")
    impl_lines.append(f"- Mental-model benchmark: `{mental_csv.name}`")
    impl_md.write_text("\n".join(impl_lines) + "\n")

    # Plain-language "so what?" summary for abstract writing support.
    so_what_md = outdir / f"so_what_summary_{args.station}_{args.start}_{args.end}.md"
    so_what_text = build_so_what_summary(
        df,
        windows_df=windows_df,
        composite_df=composite_df,
        mental_df=mental_df,
        station=args.station,
        start=args.start,
        end=args.end,
    )
    so_what_md.write_text(so_what_text)

    # Simple scenario-first summary for slides/teaching.
    key_scenarios = [
        "baseline_measured",
        "sw_hourly_climatology",
        "lw_hourly_climatology",
        "lwdown_cloud_to_clear",
        "airtemp_scale_0p0",
        "wind_scale_0p0",
        "air_temp_proxy",
    ]
    lines = []
    lines.append(f"# Decision Impact Summary ({args.station}, {args.start} to {args.end})")
    lines.append("")
    lines.append("## What Changes The Most?")
    if not decision_df.empty:
        for exp in key_scenarios:
            sub = decision_df.loc[decision_df["experiment"] == exp]
            if sub.empty:
                continue
            r = sub.iloc[0]
            grad_metric = r.get("gradient_metric", "bulk_grad")
            lines.append(
                f"- `{exp}` ({grad_metric}): d|grad|>=thr1h `{float(r['d_grad_ge20_h']):+.1f}`, dT<=-15h `{float(r['d_surf_le15_h']):+.1f}`, "
                f"grad flips `{float(r['flip_grad20_h']):.1f}`h, cold15 flips `{float(r['flip_cold15_h']):.1f}`h."
            )
    lines.append("")
    lines.append("## Presentation Use")
    lines.append("- Use `decision_impact_plot_*.png` first to show practical consequences in hours.")
    lines.append("- Use `variable_importance_plot_*.png` second for technical cross-check.")
    lines.append("- Use `mental_model_compare_*.png` last to bridge to practitioner education.")
    decision_md.write_text("\n".join(lines) + "\n")

    print("\nVariable Importance Ranking (high impact first)")
    print("=" * 100)
    print(
        df[
            [
                "experiment",
                "family",
                "RMSE",
                "dRMSE_all",
                "dRMSE_cloud",
                "dRMSE_night",
                "dRMSE_morning",
                "dGRADRMSE_all",
                "dGRADRMSE_cloud",
                "dGRADRMSE_night",
                "P95_abs_bulk_grad_Cpm",
                "ImpactScore",
                "notes",
            ]
        ].to_string(index=False)
    )
    print("\nSaved Files")
    print("=" * 100)
    print(metrics_csv)
    print(errors_csv)
    print(fig_png)
    print(fig_scatter_png)
    print(fig_heat_png)
    print(composite_csv)
    print(fig_event_png)
    print(windows_csv)
    print(decision_csv)
    print(decision_png)
    print(decision_md)
    print(mental_csv)
    print(fig_mental_png)
    print(fig_meteogram_png)
    print(impl_csv)
    print(impl_md)
    print(so_what_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
