# senator_beck So-What Summary (2026-01-01 to 2026-01-31)

## Baseline
- Baseline surface RMSE is `1.64 C` and baseline P95 bulk gradient is `31.0 C/m` (`3.10 C per 10 cm`).
- So what: this is the reference skill. Every scenario below is interpreted as added error relative to this baseline.

## Big Plain-Language Numbers
- A 10% LWdown error (`lwdown_scale_0p9`) has `3.7x` the surface impact and `2.1x` the gradient impact of a 30% sensible-flux error (`sensible_scale_0p7`).
- The same 10% LWdown error has `12.3x` the surface impact and `8.7x` the gradient impact of a 30% latent-flux error (`latent_scale_0p7`).
- Cloud-averaged daytime SW simplification (`sw_daytime_climatology`) changes gradients by `2.92 C/m`, while a 20% wind error (`wind_scale_0p8`) changes gradients by `1.36 C/m`.
- So what: forgetting LW does not just nudge surface temperature; it changes the internal snowpack thermal gradient enough to alter stability interpretation.

## Mental Model Check (Air + Sun + Wind Only)
- Treating surface temp as air temp gives RMSE `7.22 C` versus `1.64 C` for the physics model.
- Treating gradient as `(air - ground)/H` gives RMSE `9.44 C/m` versus `2.25 C/m` for the physics model.
- Physics model reduces gradient error by `76.2%` relative to the air-ground proxy.
- So what: the common beginner mental model misses important radiation-driven structure in the snowpack.

## Using Climatological SW Instead Of Measured SW
- `sw_hourly_climatology` changes surface RMSE by `+1.59 C` and gradient RMSE by `+2.92 C/m` (`+0.29 C/10 cm`).
- Cloud-window impact: surface `+1.76 C`, gradient `+2.58 C/m`.
- So what: removing observed cloud-driven SW variability materially degrades both surface temperature and gradient skill; daytime cloud effects should be represented in operational interpretation.

## Using Climatological LW Instead Of Measured LW
- `lw_hourly_climatology` changes surface RMSE by `+2.22 C` and gradient RMSE by `+4.90 C/m` (`+0.49 C/10 cm`).
- Cloud-window impact: surface `+4.64 C`, gradient `+7.82 C/m`.
- So what: removing event-scale cloud LW variability strongly degrades snow temperature and gradient skill, even when daily weather 'feels' similar.

## Using Climatological SW Only In Cloudy Daytime
- `sw_cloud_day_climatology` changes surface RMSE by `+0.42 C` and gradient RMSE by `+1.26 C/m` (`+0.13 C/10 cm`).
- Cloud-window impact: surface `+1.71 C`, gradient `+2.52 C/m`.
- So what: even partial SW simplification during cloudy daytime changes gradients, showing cloud-radiation timing matters for snowpack thermal structure.

## Leaving Out Sensible Heat Flux
- `no_sensible` changes surface RMSE by `+4.59 C` and gradient RMSE by `+7.53 C/m` (`+0.75 C/10 cm`).
- Cloud-window impact: surface `+0.94 C`, gradient `+2.19 C/m`.
- So what: sensible exchange is a first-order control on snow temperature and gradient evolution, especially during transition periods.

## Leaving Out Latent Heat Flux
- `no_latent` changes surface RMSE by `+0.50 C` and gradient RMSE by `+1.28 C/m` (`+0.13 C/10 cm`).
- Cloud-window impact: surface `+0.62 C`, gradient `+1.16 C/m`.
- So what: latent exchange affects results but is lower impact than LW/SW and sensible in this dry winter window.

## Replacing Measured LWdown With Parameterized LWdown
- `lwdown_parameterized` changes surface RMSE by `+3.23 C` and gradient RMSE by `+6.02 C/m` (`+0.60 C/10 cm`).
- Cloud-window impact: surface `+6.50 C`, gradient `+9.57 C/m`.
- So what: LWdown representation is one of the strongest error drivers; inaccurate cloud/atmospheric longwave forcing strongly propagates into snow temperatures and gradients.

## Removing Cloud LW Enhancement (Cloud Converted To Clear-LW)
- `lwdown_cloud_to_clear` changes surface RMSE by `+2.37 C` and gradient RMSE by `+5.19 C/m` (`+0.52 C/10 cm`).
- Cloud-window impact: surface `+6.70 C`, gradient `+10.50 C/m`.
- So what: cloud longwave forcing is critical; assuming clear-sky LW during cloudy periods causes large errors in both surface and gradient estimates.

## Reducing LWdown During Cloudy Nights (90%)
- `lwdown_scale_0p9_cloud_night` changes surface RMSE by `+0.08 C` and gradient RMSE by `+0.94 C/m` (`+0.09 C/10 cm`).
- Cloud-window impact: surface `+0.54 C`, gradient `+1.89 C/m`.
- So what: nighttime cloud longwave input is highly influential for overnight cooling control and near-surface gradient persistence.

## Increasing LWdown During Cloudy Nights (110%)
- `lwdown_scale_1p1_cloud_night` changes surface RMSE by `+0.18 C` and gradient RMSE by `+0.80 C/m` (`+0.08 C/10 cm`).
- Cloud-window impact: surface `+1.01 C`, gradient `+1.62 C/m`.
- So what: small LW adjustments in cloudy nights shift thermal gradients enough to matter for snow stability interpretation.

## Snow Depth Bias Test (-10%)
- `snowdepth_scale_0p9` changes surface RMSE by `-0.00 C` and gradient RMSE by `+2.06 C/m` (`+0.21 C/10 cm`).
- Cloud-window impact: surface `+0.01 C`, gradient `+1.36 C/m`.
- So what: snow depth uncertainty can have modest surface impact but meaningful gradient impact, so depth quality matters for gradient-focused products.

## Flattening Air-Temperature Variability To Monthly Mean (0%)
- `airtemp_scale_0p0` changes surface RMSE by `+1.64 C` and gradient RMSE by `+3.01 C/m` (`+0.30 C/10 cm`).
- Cloud-window impact: surface `+2.51 C`, gradient `+3.67 C/m`.
- So what: air temperature alone cannot reproduce observed snow thermal evolution; removing air variability creates large interpretive errors.

## Removing Wind Influence (0%)
- `wind_scale_0p0` changes surface RMSE by `+5.74 C` and gradient RMSE by `+8.92 C/m` (`+0.89 C/10 cm`).
- Cloud-window impact: surface `+1.29 C`, gradient `+2.26 C/m`.
- So what: wind effects can be large and nonlinear; this extreme test is memorable for training because it shows how quickly flux balance can drift.

## Top Realistic Uncertainty Drivers
- `wind_scale_0p0` (turbulent): surface `+5.74 C`, gradient `+8.92 C/m`.
- `air_temp_proxy` (heuristic): surface `+5.58 C`, gradient `+9.15 C/m`.
- `wind_scale_0p25` (turbulent): surface `+4.27 C`, gradient `+7.09 C/m`.
- `lw_hourly_climatology` (longwave): surface `+2.22 C`, gradient `+4.90 C/m`.
- `wind_scale_0p5` (turbulent): surface `+2.15 C`, gradient `+4.32 C/m`.
- `airtemp_scale_0p0` (airtemp): surface `+1.64 C`, gradient `+3.01 C/m`.
- `lwdown_scale_0p9` (longwave): surface `+1.05 C`, gradient `+2.72 C/m`.
- `sw_hourly_climatology` (shortwave): surface `+1.59 C`, gradient `+2.92 C/m`.

## Weather Windows That Matter Most
- Highest-error 6-hour windows are mostly tagged as `{'clear_calm': 2, 'clear': 2, 'sunny_clear': 1}` with typical LWdown `176.1 W/m2`, SWdown `187.6 W/m2`, wind `2.44 m/s`.
- So what: forecast/measurement uncertainty during these window types will have outsized effects on both snow surface temperature and gradient diagnostics.

## High-Error vs Low-Error Composite
- High-error windows: cloud fraction `0.00`, LWdown `176.1 W/m2`, wind `2.44 m/s`.
- Low-error windows: cloud fraction `0.90`, LWdown `248.4 W/m2`, wind `6.33 m/s`.
- So what: this gives a concrete operational context for when radiation and turbulent forcings are most likely to cause model interpretation risk.

## Practical Takeaway For Snow Safety
- For Senator Beck in this month, LW/SW forcing fidelity controls much of the combined surface+gradient skill; sensible is important but secondary under realistic perturbations, and latent is usually smaller.
- This supports prioritizing high-quality radiation observations (especially cloud-modulated LW and SW) when gradient-based decisions are important.
