# Presentation Storyboard (ISSW + Practitioner Audience)

Use this flow to keep the message simple and memorable.

## Slide 1: Why This Matters
- Field users read air temp, wind, and sun.
- But the snowpack also responds to sky radiation we cannot directly feel.
- Question: what errors do we introduce if we ignore that?

## Slide 2: Baseline Credibility
- Show baseline model-vs-observed surface temperature skill.
- Keep one number only (typical miss or RMSE) to establish trust.
- Figure:
  - `results/issw_playground/mental_model_compare_senator_beck_2026-01-01_2026-01-31.png`

## Slide 3: Conditions Where Model Struggles
- Show that high errors cluster in clear/calm windows.
- Show that cloudy/windy windows are much lower error.
- Figure:
  - `results/issw_playground/variable_importance_weather_windows_senator_beck_2026-01-01_2026-01-31.csv` (table excerpt)

## Slide 4: Decision Consequence (Most Important)
- Do not start with RMSE.
- Start with hours changed in key interpretation categories:
  - change in hours with |gradient| >= 20 C/m
  - change in hours with surface temp <= -15 C
- Figure:
  - `results/issw_playground/decision_impact_plot_senator_beck_2026-01-01_2026-01-31.png`

## Slide 5: Scenario Ladder (Memorable Tests)
- Avg SW (no cloud events)
- Avg LW (no cloud events)
- Air temp variability reduced to 0%
- Wind reduced to 0%
- Air temp as direct surface proxy
- Table:
  - `results/issw_playground/decision_impact_hours_senator_beck_2026-01-01_2026-01-31.csv`

## Slide 6: Technical Cross-Check
- Show standard sensitivity panel for researchers.
- Figure:
  - `results/issw_playground/variable_importance_plot_senator_beck_2026-01-01_2026-01-31.png`

## Slide 7: Clear "So What?" Statements
- Field cues are necessary, but not sufficient.
- Radiation-aware tools reduce wrong calls about cold surface states and strong gradients.
- Biggest risk windows in this case: clear/calm transitions.

## Slide 8: Limits + Next Step
- Single station/month here.
- Repeat at second site/window to test transferability.
- Keep physics minimal; expand evaluation windows.
