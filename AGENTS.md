# AGENTS.md

This file is the long-term project playbook for `snowtempprofile-1`.
It is written for both humans and AI agents.

Purpose:
- Keep the project scoped and finishable for ISSW 2026.
- Keep code beginner-friendly and scientifically defensible.
- Keep results useful for both researchers and avalanche practitioners.

---

## 0) Project Snapshot

Project:
- Minimal 1-D multilayer snow temperature model (surface energy balance + conduction).

Core question:
- How well can a minimal SEB snow model reproduce observed snow surface temperature?
- Which forcing assumptions (especially LW and cloud/radiation handling) drive error and decision-relevant interpretation changes?

Main deliverable:
- Reproducible pipeline that:
  1) pulls station data,
  2) runs controlled scenarios,
  3) reports both standard skill metrics and practitioner-facing impact metrics.

---

## 1) ISSW 2026 Target

Target event:
- ISSW 2026, Whistler, BC.

Working deadlines (verify on official site before submission):
- Abstract deadline: May 1, 2026
- Presenting author registration: July 15, 2026
- Proceedings paper deadline: August 24, 2026

Review priorities to satisfy:
- Quality
- Theory-practice integration
- Relevance
- Innovation/contribution

Agent rule:
- Do not invent formatting requirements.
- Use official ISSW templates and guidance when finalizing abstract/paper.

---

## 2) Scope Control

### In Scope (minimum viable science)
- 1-D conduction + SEB top boundary.
- Forcing from USGS station data.
- Primary validation against non-contact snow surface temperature (`72405`).
- Controlled forcing/scenario perturbations for attribution.
- Multi-window/site evaluation (expand evaluation before expanding physics).

### Out of Scope (for abstract phase)
- Full SNOWPACK-style physics (settlement, percolation/refreezing, grain evolution, preferential flow).
- Strong claims about weak-layer prediction without in-snow thermistor validation.
- Large architecture rewrites that slow iteration.

Principle:
- Expand evaluation breadth (windows/sites/scenarios), not model complexity.

---

## 3) Data Source of Truth

Primary source:
- USGS Water Services IV endpoint.

Primary station IDs currently used:
- `senator_beck`: `375429107433201`
- `independence_pass`: `390622106343001`
- `berthoud_pass`: `394759105464101`
- `ptarmigan` (legacy support if needed): `392954106162501`

Current collector module:
- `notebooks/usgs_collector.py`

Cache policy:
- Prefer cached fetches to reduce API fragility and speed experiments.
- Cache path: `data/usgs_cache/`

Example cache command:
```bash
python scripts/cache_usgs_dataset.py --station senator_beck --start 2026-01-01 --end 2026-01-31
```

---

## 4) Required Variables and Units

Use consistent names from `simplify_columns()`:

Atmospheric:
- `air_temp_c` (`00020`)
- `rh_pct` (`00052`)
- `wind_speed_mph` (`00035`)
- `wind_dir_deg` (`00036`)
- `pressure_mmhg` (`00025`)
- `wind_gust_mph` (`61728`, optional)

Radiation:
- `sw_down_wm2` (`72186`)
- `sw_up_wm2` (`72185`)
- `lw_down_wm2` (`72175`)
- `lw_up_wm2` (`72174`)

Snow/soil:
- `snow_depth_m` (`72189`)
- `surface_temp_c` (`72405`) PRIMARY validation
- `soil_temp_c`, `soil_temp_c_2` (`72253` depths)
- `lwc_pct` (`72393`, optional)
- `drifting_snow` (`72394`, optional)

Unit rules:
- Keep dataframe temperature in C; convert to K inside physics.
- Wind mph -> m/s with `0.44704`.
- Pressure mmHg -> Pa with `133.322`.
- Radiation remains W/m2.

---

## 5) Physics Conventions (Non-Negotiable)

Surface flux sign convention:
- Positive flux means downward into snow.

Surface terms:
- `LWnet = LWdown - LWup`
- `SWnet = SWdown - SWup` (or albedo form when scenario requires)
- `Qsen` positive when air warms surface.
- `Qlat` positive when moisture flux warms surface.

Layer indexing:
- `T[0]` = bottom (ground-adjacent)
- `T[-1]` = top (surface-adjacent)

Boundary conditions:
- Top: SEB closure.
- Bottom: fixed or measured soil-temp option; document which is used in each run.

Snow depth handling:
- Preferred path: `H(t)` from sensor with fixed `N` layers and `dz(t)=H(t)/N`.
- Minimal robust method: piecewise integration + profile remapping between intervals.

---

## 6) Data QC Rules

Mandatory QC before any solve:
- Remove duplicates and sort time.
- Ensure time monotonicity.
- Clip radiation lower bound: `SWdown >= 0`, `SWup >= 0`.
- Handle missing required columns with explicit error.
- Interpolate only after QC and with documented method.

Recommended QC diagnostics to print each run:
- Rows before/after QC.
- Min SWdown/SWup after QC.
- Time coverage start/end and expected model duration.

---

## 7) Scenario Framework

### Baseline
- `baseline_measured`: measured SW + measured LW with current tuned baseline params.

### Core attribution scenarios
- `sw_hourly_climatology`: average SW, no cloud-event variability.
- `sw_daytime_climatology`
- `sw_cloud_day_climatology`
- `lw_hourly_climatology`: average LW, no cloud-event variability.
- `lwdown_parameterized`
- `lwdown_cloud_to_clear`
- `lwdown_scale_0p9`, `lwdown_scale_1p1`

### Turbulence / wind scenarios
- `sensible_scale_0p7`, `sensible_scale_1p3`
- `latent_scale_0p7`, `latent_scale_1p3`
- `no_sensible`, `no_latent`, `no_turbulent`
- `wind_scale_0p8`, `wind_scale_1p2`
- Extreme teaching set: `wind_scale_0p75`, `wind_scale_0p5`, `wind_scale_0p25`, `wind_scale_0p0`

### Air-temperature influence scenarios
- `airtemp_scale_0p75`, `airtemp_scale_0p5`, `airtemp_scale_0p25`, `airtemp_scale_0p0`

### Heuristic benchmark
- `air_temp_proxy`: surface temp = air temp (explicit benchmark row).

### Snow-depth sensitivity
- `snowdepth_scale_0p9`, `snowdepth_scale_1p1`

Agent rule:
- Keep diagnostic calculations consistent with the active scenario forcing.

---

## 8) Metrics to Report

### Standard scientific metrics
- MAE, RMSE, Bias, Correlation (surface temp)
- Optional splits: day/night/cloud/clear/morning.

### Practitioner-facing decision metrics (required)
Relative to baseline, report scenario deltas in hours for:
- `|bulk gradient| >= 20 C/m`
- `|bulk gradient| >= 30 C/m`
- `surface temp <= -15 C`
- `surface temp <= -20 C`

Also report:
- category flip hours (extra vs missed) for gradient and cold-surface thresholds.

Interpretation rule:
- For presentations, start with hour-based impacts first.
- Use RMSE/MAE as supporting technical validation.

---

## 9) Required Outputs

From `scripts/issw_variable_importance.py`, keep producing:
- `variable_importance_*.csv`
- `variable_importance_timeseries_*.csv`
- `variable_importance_plot_*.png`
- `variable_importance_surface_gradient_scatter_*.png`
- `variable_importance_regime_heatmap_*.png`
- `variable_importance_weather_windows_*.csv`
- `variable_importance_event_composite_*.csv`
- `mental_model_benchmark_*.csv`
- `mental_model_compare_*.png`
- `so_what_summary_*.md`

Decision-focused outputs:
- `decision_impact_hours_*.csv`
- `decision_impact_plot_*.png`
- `decision_impact_summary_*.md`

Scenario/communication helpers:
- `scenario_pack_memorable.md`
- `presentation_storyboard.md`

---

## 10) Recommended Visual Story Order

Use this order for mixed audiences:
1. Baseline credibility (observed vs modeled surface temp).
2. Error windows by weather type (clear/calm vs cloudy/windy).
3. Decision-impact hours plot (category changes in hours).
4. Scenario comparison table (memorable tests).
5. Technical sensitivity figures (for research audience).
6. Plain-language takeaways and limits.

---

## 11) Coding Standards

Do:
- Small pure functions.
- Clear variable names.
- Explicit unit conversions.
- Reproducible outputs from scripts.
- Incremental edits and checkpoints.

Avoid:
- Hidden notebook state dependencies.
- Global behavior toggles without logging.
- Overfitting one station/window.
- Large refactors during abstract phase.

Beginner-first rule:
- Prefer simple readable code over clever code.

---

## 12) Git and Workflow Rules

- Commit in small, reversible steps.
- Do not bundle unrelated changes.
- Keep one frozen baseline config for comparison.
- Recompute all deltas against the same baseline.
- Document any parameter change with reason and expected effect.

Suggested commit rhythm:
1) scenario logic
2) metrics
3) visuals
4) summary text

---

## 13) Runbook (Primary Commands)

Cache data:
```bash
python scripts/cache_usgs_dataset.py --station senator_beck --start 2026-01-01 --end 2026-01-31
```

Run variable-importance suite:
```bash
python scripts/issw_variable_importance.py --station senator_beck --start 2026-01-01 --end 2026-01-31 --event-window-hours 6 --event-top-k 5
```

Optional outlier analysis:
```bash
python scripts/analyze_outlier_events.py --station senator_beck --start 2026-01-01 --end 2026-01-31 --threshold-c 4.0
```

---

## 14) ISSW Writing Guidance

Abstract strategy:
- Keep one clear claim.
- Use 2-4 strong numbers only.
- Include one practical consequence statement in plain language.
- Include one limitations sentence.

Safe framing:
- "Field cues are necessary but not sufficient."
- "Radiation-aware tools reduce interpretation error."

Avoid:
- Claiming education is "wrong".
- Over-generalizing beyond tested sites/windows.

---

## 15) GPT Pro Extended Thinking Review Prompt (Reusable)

```text
You are a senior reviewer for a snow energy-balance modeling project. Your job is to audit, improve, and simplify the workflow for both researchers and avalanche practitioners.

Context:
- Repo: snowtempprofile-1
- Key files:
  - AGENTS.md
  - notebooks/snowmodel_USGS.ipynb
  - scripts/issw_variable_importance.py
  - results/issw_playground/*
- Current focus window: Senator Beck station, 2026-01-01 to 2026-01-31.
- Goal: use a low-error baseline model as a "playground" to test forcing assumptions and show practical "so what" impacts for avalanche interpretation.
- Audience: mixed (ISSW researchers + field practitioners + beginner Python users).
- Constraints:
  - Keep changes low-risk and incremental.
  - No large refactors.
  - Beginner-readable code and explanations.
  - Prioritize practical impact and interpretability over complexity.

What I want from you:
1) Full technical + science review
- Check physics assumptions (SEB sign conventions, SW/LW handling, turbulent terms, boundary conditions, unit conversions, interpolation/timing assumptions, solver behavior).
- Identify likely error sources and rank by impact.
- Flag anything scientifically weak, inconsistent, or overfit.

2) Scenario framework review
- Evaluate whether scenario set is educational and defensible.
- Specifically assess these scenarios:
  - avg SW with no cloud-event variability
  - avg LW with no cloud-event variability
  - reduce air-temp influence from 0-100%
  - reduce wind influence from 0-100%
  - air-temp-only surface proxy
- Recommend additional high-value scenarios if they improve memorability and practitioner relevance.

3) Metrics + outputs review
- Evaluate current metrics (RMSE/MAE/Bias/Correlation + decision-impact hours).
- Propose improvements for so-what communication.
- Prioritize category-based metrics (example: hours in strong-gradient range, hours in very-cold-surface range, flips vs baseline).

4) Visualization/storytelling review
- Design a clear figure set for ISSW + practitioner audience.
- Include one recommended slide sequence with figure purpose and caption text.
- Make sure visuals tell a clear decision story, not just model diagnostics.

5) Deliverable quality check for abstract/paper
- Check fit with ISSW 2026 expectations (quality, relevance, theory-practice bridge).
- Suggest exact wording improvements that avoid sounding accusatory to avalanche education systems.
- Keep claims defensible and scoped to tested windows/sites.

Output format (strict):
A) Executive Summary (max 10 bullets)
B) Critical Findings (ordered by severity; include file/line references when possible)
C) Top 5 Low-Risk Improvements (each with why, expected impact, risk)
D) Patch Plan (small incremental edits only; commit-sized chunks)
E) Copy/Paste Code Snippets (minimal, beginner-friendly)
F) Revised Figure/Story Plan
G) Abstract Language Revision (practitioner-friendly, clear so-what)
H) Validation Checklist (what to rerun after each change)

Rules:
- If you make factual claims about avalanche education or LW importance, provide source links and state confidence.
- Prefer primary sources for technical/scientific claims.
- Do not recommend overengineering.
- Explicitly separate must-do-now vs nice-later.
- If uncertain, say what you infer and why.
```

Prompt use notes:
- Replace station and date window before running.
- Ask for both practitioner-facing and technical conclusion versions.
- Keep one baseline configuration fixed while comparing scenarios.

---

## 16) Official Links

ISSW 2026:
- https://www.issw2026.com/submit-an-abstract
- https://www.issw2026.com/guidelines-for-papers
- https://www.issw2026.com/importantdates

USGS docs:
- https://waterservices.usgs.gov/docs/instantaneous-values/instantaneous-values-details/
- https://api.waterdata.usgs.gov/docs/ogcapi/migration/

Avalanche education context:
- https://avtraining.org/
- https://avalanche.ca/pages/avaluator
- https://avalanche.org/

---

## 17) Definition of Done (Abstract Phase)

Minimum done criteria:
1. Reproducible baseline + scenario suite for at least one full month window.
2. Decision-impact hour metrics generated and explained.
3. Clear/calm vs cloudy/windy regime story demonstrated.
4. Practitioner-readable abstract draft with defensible quantitative support.
5. At least one transfer check window or second station run completed.

If these are met, move to abstract finalization and paper expansion.
