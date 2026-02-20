# ISSW + AIARE Plain-Language Story (Senator Beck, 2026-01-01 to 2026-01-31)

## The Big Message
If you ignore longwave radiation (LW), you can be very wrong about snow temperature and snowpack gradient, even when air temperature and sun look reasonable.

## One-Sentence "So What?"
For this month at Senator Beck, a realistic 10% LW error created much larger model damage than comparable turbulent-flux uncertainty, and simple "air-ground" mental models missed most of the real gradient behavior.

## Your Strongest Numbers (from this run)
- Baseline model: surface RMSE `1.64 C`, gradient RMSE `2.25 C/m`.
- Air-only surface proxy (`Tsurf = Tair`) RMSE: `7.22 C`.
- Naive gradient proxy (`(Tair - Tground)/H`) RMSE: `9.44 C/m`.
- Gradient error reduction from physics model vs naive proxy: `76.2%`.
- 10% LWdown decrease (`lwdown_scale_0p9`) increased:
  - surface RMSE by `+1.05 C` (`+64%` relative to baseline RMSE),
  - gradient RMSE by `+2.72 C/m` (`+121%` relative to baseline gradient RMSE).
- 30% sensible reduction (`sensible_scale_0p7`) increased:
  - surface RMSE by `+0.29 C`,
  - gradient RMSE by `+1.32 C/m`.
- 30% latent reduction (`latent_scale_0p7`) increased:
  - surface RMSE by `+0.09 C`,
  - gradient RMSE by `+0.31 C/m`.
- Cloud LW removal test (`lwdown_cloud_to_clear`) in cloud windows:
  - surface RMSE `+6.70 C`,
  - gradient RMSE `+10.50 C/m`.

## Plain-Language Teaching Translation
- "Air temp drives snow temp" is incomplete.
- "Sun drives warming" is incomplete.
- "Wind cools the snow" is true, but LW often sets the thermal floor during key windows.
- The snowpack can gain/lose large thermal energy from radiation while air temperature changes little.

## Why This Matters For Rec 1 / Rec 2 Logic
- Students often use a simple mental model: "gradient is just air minus ground."
- In this dataset, that shortcut produced very large gradient error (`9.44 C/m` RMSE).
- Weak-layer metamorphism and faceting are gradient-sensitive; if gradient is wrong, hazard reasoning can drift.
- Practical implication: cloud/radiation observations should be part of interpretation, not optional context.

## ISSW-Style Claim You Can Defend
In this month-long alpine case, longwave forcing fidelity was a first-order control on both surface temperature skill and internal gradient skill, and neglecting LW led to materially larger errors than comparable latent/sensible perturbations.

## Suggested Figure Trio For Abstract/Slides
1. Variable importance bar chart (surface + gradient deltas side by side).
2. Mental model benchmark panel (air-only / air-ground proxy vs physics).
3. Weather-window composite (high-error vs low-error windows showing cloud fraction, LWdown, wind).

## External Evidence That Supports This Story
- Rudisill et al. (2025): Process-based snow model better captured avalanche-site snow temperatures than air/dewpoint-only indicators; radiative fluxes (including LW) were key contributors.
  - https://doi.org/10.1029/2024JD042366
  - https://repository.library.noaa.gov/view/noaa/69385/noaa_69385_DS1.pdf
- HESS discussion of simplified snow-surface energy-balance modeling and radiation/turbulence partitioning:
  - https://hess.copernicus.org/articles/27/3051/2023/hess-27-3051-2023.html
- Avalanche education references on temperature-gradient metamorphism (faceting context):
  - https://avalanche.org/avalanche-encyclopedia/snowpack/snowpack-formation/faceted-snow-crystals/
  - https://avalanche.ca/avalanche-encyclopedia/snowpack/snowpack-formation/faceting
- ISSW example highlighting shortwave penetration relevance to near-surface/subsurface snow temperatures:
  - https://arc.lib.montana.edu/snow-science/objects/issw-2008-962-968.pdf

## Bottom Line
For your current model playground, LW is not a "small correction." It is a core control that can change both snow surface error and gradient interpretation enough to matter for operational snow safety messaging.
