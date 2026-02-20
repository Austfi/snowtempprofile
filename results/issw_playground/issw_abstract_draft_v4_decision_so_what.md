# ISSW 2026 Abstract Draft (v4, Decision-Focused "So What")

## Proposed Title (140 chars)
Invisible Sky Heat, Visible Consequences: How Longwave and Clouds Change Snow Temperature and Gradient Interpretation for Avalanche Practice

## Proposed Themes
- Avalanche Education
- Avalanche Forecasting
- Modelling and Quantitative Forecasting
- Decision Making

## Abstract (Draft)
Avalanche practitioners are trained to read what they can observe in real time: air temperature, wind, sun, and recent weather. That works well, but it misses one major input that humans cannot directly sense: incoming longwave radiation (LW) from the sky and clouds. We tested what that omission costs in practical snow-temperature interpretation.

We used a minimal 1-D snow temperature model, forced by USGS observations at Senator Beck Basin (Colorado), for 1-31 January 2026. The baseline run reproduced measured non-contact surface temperature with low error (typical miss about 1.2 C; RMSE 1.64 C), which gave us a reliable "playground" to test forcing scenarios.

The biggest misses clustered in clear/calm transition windows, where 6-hour RMSE reached about 5.3 C. The best windows were cloudy/windy, often near 0.1-0.3 C RMSE.

The key result is not just "LW matters," but how much it changes interpretation categories. In our baseline, |bulk gradient| >= 20 C/m occurred 247 hours. If we reduced LW by only 10%, that jumped to 330 hours (+83 hours). If cloudy LW was replaced with clear-sky-like LW, it jumped to 342 hours (+95 hours). The same cloudy-LW simplification also created 107 additional hours with modeled surface temperature <= -15 C. By comparison, a daytime SW simplification produced 42 strong-gradient category flips, mainly in daytime windows.

A simple by-feel benchmark was much weaker: using air temperature as a surface proxy gave 7.22 C RMSE, and using (air-ground)/snow depth as a gradient proxy gave 9.44 C/m RMSE, versus 2.25 C/m for the physics model.

For this winter alpine case, the operational takeaway is clear: field cues remain essential, but they are not enough for thermal interpretation. Low-complexity data tools that include sky radiation can materially reduce wrong calls about how cold the surface is and how strong the snowpack gradient is.
