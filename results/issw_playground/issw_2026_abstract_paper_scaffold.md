# ISSW 2026 Scaffold: Abstract -> Proceedings Paper

This is a practical pathway from your current Senator Beck model outputs to a submission-ready ISSW abstract and then a 2-8 page proceedings paper.

## 1) Fit To ISSW 2026

Based on ISSW 2026 submission guidance and theme list, this project fits best under:
- `Avalanche Education` (primary)
- `Avalanche Forecasting` (primary)
- `Avalanche Formation` (secondary; gradient/faceting relevance)
- `Modelling and Quantitative Forecasting` (secondary)
- `Decision Making` (secondary)

Why this fits:
- You quantify model error in operationally relevant terms.
- You translate physics into practitioner decisions (not just equations).
- You explicitly bridge theory and practice, matching ISSW review criteria.

## 2) One Clear Research Claim

Working claim for the abstract:

In a month-long alpine case study, realistic errors in longwave forcing produced larger degradation in snow surface temperature and snowpack gradient skill than comparable turbulent-flux perturbations, and simple air-ground heuristics missed key gradient behavior.

Current evidence from your run (Senator Beck, 2026-01-01 to 2026-01-31):
- Baseline surface RMSE: `1.64 C`
- Baseline gradient RMSE: `2.25 C/m` (obs-proxy based)
- Air-only surface proxy RMSE: `7.22 C`
- Air-ground gradient proxy RMSE: `9.44 C/m`
- Gradient error reduction vs naive proxy: `76.2%`
- 10% LWdown decrease increased:
  - surface RMSE by `+1.05 C` (`+64%` of baseline RMSE)
  - gradient RMSE by `+2.72 C/m` (`+121%` of baseline gradient RMSE)

## 3) Abstract Scaffold (Sentence-by-Sentence)

Keep to max 350 words.

1. Context/problem:
   Snow surface temperature and near-surface thermal gradients influence weak-layer evolution, but operational mental models often over-weight air temperature and under-weight radiative forcing.
2. Practical gap:
   Practitioners need simple, defensible guidance on which forcings matter most for temperature and gradient interpretation.
3. Method:
   We ran a minimal 1-D multilayer snow model forced by USGS observations, validated primarily against non-contact surface temperature.
4. Design:
   We used a controlled scenario suite (measured vs simplified LW/SW, turbulent perturbations, and weather-regime perturbations) for Senator Beck (January 2026), with gradient diagnostics included.
5. Baseline skill:
   Baseline performance was RMSE `[x] C` with correlation `[x]` against measured surface temperature.
6. Main finding:
   A 10% LW forcing perturbation increased surface RMSE by `[x] C` and gradient RMSE by `[x] C/m`, exceeding impacts from comparable latent/sensible perturbations.
7. Education finding:
   A naive air-ground gradient proxy had RMSE `[x] C/m` versus `[x] C/m` for the physics model, indicating `[x]%` reduction in gradient error.
8. Operational takeaway:
   Cloud-modulated longwave forcing should be treated as a first-order variable in avalanche education and field interpretation, not a secondary correction.
9. Scope/limits:
   Findings are from dry-snow winter windows with a minimal model and should be generalized across additional windows/sites before broad transfer.

## 4) Suggested Titles (all under 140 chars)

1. `What We Miss Without Longwave: Quantifying Snow Surface and Gradient Error in a Minimal USGS-Forced Snow Model`
2. `Air Temp Is Not Enough: Longwave Control of Snow Surface Temperature and Gradient Skill for Avalanche Decision Support`
3. `Radiation Fidelity and Snowpack Gradients: A Minimal Model Benchmark for Avalanche Forecasting and Education`

## 5) Paper Scaffold (2-8 pages)

## 5.1 Recommended format
- Use `Technical paper` template if emphasis is methods + quantified evaluation.
- Use `Practical paper` template if emphasis is decision support + training implications.
- You can still include both science and practitioner guidance in either format.

## 5.2 Section structure
1. Introduction
   - Operational motivation for Tsfc and gradient.
   - Gap in simple mental models.
2. Data and Study Design
   - USGS station, variables, QC rules, case windows.
3. Minimal Model
   - 1-D conduction + SEB, sign conventions, validation variable.
4. Scenario Suite
   - LW/SW/turbulent/depth perturbations and regime splits.
5. Results
   - Surface skill, gradient skill, scenario ranking, weather windows.
6. Operational + Education Interpretation
   - AIARE/AST-compatible takeaways.
7. Limitations
   - Minimal physics, proxy gradient, single-window emphasis.
8. Conclusions
   - Clear "measure this first" guidance.

## 5.3 Figures and tables to reuse now
- Figure 1: `results/issw_playground/variable_importance_plot_senator_beck_2026-01-01_2026-01-31.png`
- Figure 2: `results/issw_playground/mental_model_compare_senator_beck_2026-01-01_2026-01-31.png`
- Figure 3: `results/issw_playground/variable_importance_regime_heatmap_senator_beck_2026-01-01_2026-01-31.png`
- Figure 4: `results/issw_playground/variable_importance_event_composite_senator_beck_2026-01-01_2026-01-31.png`
- Table 1: `results/issw_playground/variable_importance_senator_beck_2026-01-01_2026-01-31.csv`
- Table 2: `results/issw_playground/mental_model_benchmark_senator_beck_2026-01-01_2026-01-31.csv`

## 6) Avalanche-Education Translation Layer

Frame the paper in language consistent with major rec-education systems:
- AIARE emphasizes a repeatable risk-management framework and daily/seasonal routines.
- Avalanche Canada AST/Avaluator structure links conditions and terrain (danger + ATES) for practical choices.

Use this bridge statement:

`This work does not replace terrain-based decision tools; it improves the weather/snow interpretation feeding those tools by quantifying when radiation uncertainty materially shifts snow surface and gradient estimates.`

## 7) Timeline From Today To ISSW Deadlines

Today: `Thursday, February 19, 2026`

Milestone A (by March 5, 2026):
- Freeze Senator Beck baseline.
- Finalize abstract figure set and key numbers.

Milestone B (by March 20, 2026):
- Add second station/window set (at least one additional month-window).
- Recompute scenario ranking across windows.

Milestone C (by April 5, 2026):
- Draft abstract v1 and internal 3-person review.
- Ensure wording explicitly addresses ISSW criteria:
  - quality
  - theory-practice bridge
  - relevance
  - innovation/contribution

Milestone D (by April 20, 2026):
- Lock abstract v2 + title + theme tags.

Submission deadline:
- Abstract due `May 1, 2026`.

Paper development:
- Expand methods/results after abstract acceptance.
- Proceedings paper due `August 24, 2026`.

## 8) Minimum Additional Work Before Abstract Submission

1. Run one more station or two more windows to show transferability.
2. Keep exactly one baseline configuration frozen for all comparisons.
3. Pre-register your "headline metric family":
   - surface RMSE delta
   - gradient RMSE delta
   - cloud-window sensitivity delta
4. Avoid adding new heavy physics before abstract submission.

## 9) What Not To Overclaim

- Do not claim direct weak-layer prediction without in-snow thermistor validation.
- Do not claim universal ranking across all climates from one station/month.
- Do claim:
  - forcing sensitivity ranking for tested windows
  - practical implications for field interpretation and education framing
  - clear uncertainty windows (e.g., clear/calm vs cloudy/windy)

## 10) Immediate Next Commands (Reproducible)

```bash
python scripts/cache_usgs_dataset.py --station senator_beck --start 2026-01-01 --end 2026-01-31
python scripts/issw_variable_importance.py --station senator_beck --start 2026-01-01 --end 2026-01-31 --event-window-hours 6 --event-top-k 5
```

For transfer check:

```bash
python scripts/cache_usgs_dataset.py --station independence_pass --start 2026-01-01 --end 2026-01-31
python scripts/issw_variable_importance.py --station independence_pass --start 2026-01-01 --end 2026-01-31 --event-window-hours 6 --event-top-k 5
```

## 11) Source Alignment Notes

This scaffold is aligned to:
- ISSW 2026 abstract constraints, themes, and deadlines.
- ISSW 2026 paper format/accessibility requirements.
- AIARE framework language (decision-making and repeatable routines).
- Avalanche Canada Avaluator/Trip Planner condition+terrain framing.

Primary source links:
- ISSW Submit an Abstract: https://www.issw2026.com/submit-an-abstract
- ISSW Important Dates: https://www.issw2026.com/importantdates
- ISSW Guidelines for Papers: https://www.issw2026.com/guidelines-for-papers
- AIARE home + framework language: https://avtraining.org/
- Avalanche Canada Avaluator: https://avalanche.ca/pages/avaluator
- Avaluator Trip Planner: https://avysavvy.avalanche.ca/en-ca/the-avaluator-trip-planner
