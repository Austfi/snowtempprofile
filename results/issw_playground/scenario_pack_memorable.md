# Memorable Scenario Pack (Current Runner)

File: `scripts/issw_variable_importance.py`

These scenarios are now included in `build_experiments()` and are designed for educational impact, including extremes.

## Requested Scenarios (now implemented)

1. Average SW, no cloud-event changes:
- `sw_hourly_climatology`

2. Average LW, no cloud-event changes:
- `lw_hourly_climatology`

3. Reduce air-temperature influence (0-100% variability around monthly mean):
- `airtemp_scale_0p75`
- `airtemp_scale_0p5`
- `airtemp_scale_0p25`
- `airtemp_scale_0p0` (extreme: no air-temp variability)

4. Reduce wind impact / cooling (0-100% wind forcing):
- `wind_scale_0p75`
- `wind_scale_0p5`
- `wind_scale_0p25`
- `wind_scale_0p0` (extreme: no wind)

5. Use air temperature as direct surface proxy:
- `air_temp_proxy` (explicit benchmark row in output table)

## Other high-impact education scenarios already included

- Cloud LW removed:
  - `lwdown_cloud_to_clear`
- LW reduced/increased by 10%:
  - `lwdown_scale_0p9`
  - `lwdown_scale_1p1`
- SW daytime simplification:
  - `sw_daytime_climatology`
- No sensible / no latent:
  - `no_sensible`
  - `no_latent`
- Extreme LW tests:
  - `lwdown_zero_all`
  - `lwdown_zero_when_cloudy`

## Run command

```bash
python scripts/issw_variable_importance.py \
  --station senator_beck \
  --start 2026-01-01 \
  --end 2026-01-31 \
  --event-window-hours 6 \
  --event-top-k 5
```

## Output table to inspect

- `results/issw_playground/variable_importance_senator_beck_2026-01-01_2026-01-31.csv`

Use this table to rank scenarios by:
- `dRMSE_all` (surface skill change)
- `dGRADRMSE_all` (gradient change)
- `ImpactScore` (combined impact score)

## Next high-value scenarios to add (not yet in code)

1. Forcing timing error (clock mismatch):
- Shift SW/LW by +1 hour and -1 hour.
- Educational value: "A one-hour timing mistake can create false morning/afternoon interpretations."

2. Cloud classifier sensitivity:
- Repeat with cloud emissivity threshold 0.80 vs 0.90.
- Educational value: shows uncertainty in cloud-regime assumptions.

3. Sensor loss experiments:
- "No LW sensor available" fallback branch.
- "No SW up sensor available" fallback branch.
- Educational value: directly mirrors common station limitations.
