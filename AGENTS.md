# AGENTS.md

This file is a **project playbook** for anyone (human or AI agent) working on the
`snowtempprofile` notebook/codebase. It defines the **ISSW goal**, the **minimum viable scope**,
the **data source**, the **model/validation conventions**, and the **highest‑leverage coding tasks**
to make the ISSW abstract and proceedings paper credible and finishable.

---

## 0) Project snapshot

**Project name:** Minimal 1‑D multilayer snow temperature model (surface energy balance + conduction)

**Primary ISSW research question (A‑pathway):**  
> How well can a minimal surface energy balance model reproduce observed **snow surface temperature**,
> and which flux parameterizations (especially incoming longwave) drive the error?

**Core deliverable for ISSW:**  
A reproducible pipeline that pulls real station forcing, runs a controlled scenario suite, and reports
surface temperature skill + error attribution in regimes relevant to avalanche practice.

---

## 1) ISSW 2026 context (what we are building toward)

This project is targeting **ISSW 2026** (Whistler, BC).

### Abstract rules (must meet)
From the ISSW 2026 abstract guidelines page:
- Language: English  
- Title: max **140 characters**  
- Abstract text: max **350 words**  
- Choose presentation type: oral / poster / either  
- No submission fees  
- Abstracts reviewed anonymously; scoring emphasizes:
  1) Quality  
  2) Merging theory and practice  
  3) Relevance  
  4) Innovation and contribution  
- Abstracts are only rejected for being off-topic, inflammatory, purely promotional, or duplicative.

### Deadlines (hard constraints)
From ISSW 2026 "Submit an Abstract" + "Important Dates":
- Abstract deadline: **May 1, 2026**
- Presenting authors register by: **July 15, 2026**
- Proceedings paper deadline: **August 24, 2026** (2–8 pages)

### Proceedings paper formats + accessibility
ISSW offers **two paper formats** (technical vs practical), with Word and LaTeX templates.
The MSU Library archive requires **WCAG 2.1 Level AA accessibility**, and the ISSW 2026 templates
have been updated accordingly.

**Agent rule:** Do not invent formatting rules. Use the official ISSW 2026 templates.

---

## 2) Scope: what is in / out (do not let scope creep kill the abstract)

### In-scope (minimum viable science)
- 1‑D multilayer conduction model with a surface energy balance boundary condition
- Run model using **USGS station forcing**
- Validate primarily against **measured non-contact snow surface temperature**
- Controlled scenario suite to attribute error:
  - measured vs parameterized **LW↓**
  - measured vs simplified **SW/albedo**
  - optional: neutral vs simple stability correction for turbulent fluxes
- Run across multiple **7–14 day case windows** (and ideally 2 sites)

### Out-of-scope (explicitly)
- Full SNOWPACK physics: evolving layering, densification/settlement, water percolation/refreezing,
  grain type evolution, preferential flow, etc.
- Blowing snow transport/sublimation physics (we may *filter* drifting-snow periods instead)
- Claiming validated near-surface gradients without in-snow thermistor observations

**Principle:** Expand the *evaluation* (more windows/sites), not the physics.

---

## 3) Data: USGS forcing and observations (single source of truth)

### Station IDs (initial targets)
- **Senator Beck Meteorological Station**: `USGS 375429107433201`
- **Ptarmigan Meteorological Station**: `USGS 392954106162501`

These NWIS pages list all key parameters and show they are "provisional data subject to revision."

### USGS Data Collector
Use the standalone collector: `notebooks/usgs_collector.py`

```python
from usgs_collector import fetch_usgs_iv, simplify_columns, STATIONS

df = simplify_columns(fetch_usgs_iv(STATIONS["senator_beck"], "2025-01-01", "2025-01-10"))
```

### Verified parameter codes (from USGS API header 2026-02-05)
These are the parameter codes confirmed available at Senator Beck and Ptarmigan:

#### Atmospheric Forcing
| Code | DataFrame Column | Description | Units |
|------|------------------|-------------|-------|
| `00020` | `air_temp_c` | Air temperature | °C |
| `00052` | `rh_pct` | Relative humidity | % |
| `00035` | `wind_speed_mph` | Wind speed | mph |
| `00036` | `wind_dir_deg` | Wind direction (from true north) | degrees |
| `00025` | `pressure_mmhg` | Barometric pressure | mmHg |
| `61728` | `wind_gust_mph` | Wind gust speed | mph |

#### Radiation (CRITICAL for energy balance)
| Code | DataFrame Column | Description | Units |
|------|------------------|-------------|-------|
| `72186` | `sw_down_wm2` | Shortwave radiation, downward (incoming) | W/m² |
| `72185` | `sw_up_wm2` | Shortwave radiation, upward (reflected) | W/m² |
| `72175` | `lw_down_wm2` | Longwave radiation, downward (incoming) | W/m² |
| `72174` | `lw_up_wm2` | Longwave radiation, upward (emitted) | W/m² |

#### Snow Properties
| Code | DataFrame Column | Description | Units |
|------|------------------|-------------|-------|
| `72189` | `snow_depth_m` | Snow depth | meters |
| `72405` | `surface_temp_c` | Surface temperature (non-contact) **PRIMARY VALIDATION** | °C |
| `72393` | `lwc_pct` | Liquid water content | % volume |
| `72394` | `drifting_snow` | Drifting snow mass flux | g/m²/s |

#### Soil (Bottom Boundary)
| Code | DataFrame Column | Description | Units |
|------|------------------|-------------|-------|
| `72253` | `soil_temp_c` | Soil temperature (5 cm depth) | °C |
| `72253` | `soil_temp_c_2` | Soil temperature (20 cm depth) | °C |

### How to retrieve data (simple, reproducible)
Use USGS Water Services **Instantaneous Values (iv)** endpoint. It supports tab-delimited "RDB"
format (easy to parse in pandas).

Recommended: request only the parameter codes needed for the current experiment window.
Avoid scraping NWIS webpages.

**Note on modernization:** Some NWIS pages are undergoing modernization with expected
decommissioning; prefer API retrieval and keep an eye on migration guidance.

---

## 4) Model + validation conventions (avoid silent sign/unit bugs)

### Units (must be consistent)
- Temperature: °C in DataFrame; convert to K inside physics
- Wind: mph → m/s using `0.44704`
- Pressure: mm Hg → Pa using `133.322`
- Radiation: W/m² (as provided)

### Surface temperature observation hierarchy
1) Primary: `72405` (non-contact snow surface temperature)  
2) Secondary: derive Ts from `LW↑` using Stefan–Boltzmann (diagnostic only)

### Flux sign convention (standardize everywhere)
Define **positive flux as downward into snow** at the surface.
- `LWnet = LW↓ − LW↑`
- `SWnet = SW↓ − SW↑` (or `SW↓*(1−α)` if α constant)
- `Qs` positive downward when air warms surface
- `Ql` positive downward when condensation/deposition warms surface (sign from q_air − q_snow)

### Layer indexing (recommended)
For readability:
- `T[0]` = top (surface-adjacent layer)
- `T[-1]` = bottom (ground-adjacent layer)

### Snow depth H(t)
Target requirement: **H adjusts with the sensor**.

Keep it undergraduate-simple:
- Fix the number of layers `N`
- Use `H(t)` from `72189` (snow depth)
- Layer thickness is `dz(t) = H(t)/N`

**Minimal viable way to implement H(t) without heavy math**
Instead of a continuously deforming grid inside one long ODE solve:
1) Define a regular time grid using station timestamps (e.g., hourly).
2) Integrate the ODE one interval at a time with constant `H` over that interval.
3) If `H` changes between intervals, remap the temperature profile from old depths to new depths using
   simple 1‑D interpolation on depth coordinates.

This keeps the physics "good enough" and prevents hidden instability.

**Important limitation:** This approach ignores explicit advection from snowfall/settlement. We will
either (a) choose windows with relatively small dH/dt for the abstract, or (b) quantify dH/dt and
discuss it as a limitation.

---

## 5) Scenario suite (the core of "what drives error?")

Run the same suite for every case window.

### Scenario toggles
- **LW↓ option**
  - measured: use measured LW↓ parameter (code TBD)
  - parameterized: compute LW↓ from Tair/RH (simple emissivity model)
- **SW/albedo option**
  - measured: use measured SW↓−SW↑ (codes TBD)
  - simplified: constant albedo α (e.g., 0.80), `SW↓*(1−α)`

### Minimum scenario set (4 runs)
1) Measured LW↓, measured SW↑  
2) Parameterized LW↓, measured SW↑  
3) Measured LW↓, constant α  
4) Parameterized LW↓, constant α

**Agent rule:** Any diagnostic flux plots must use the **same** scenario settings as the solve.
No "measured LW diagnostics" for an "idealized LW run".

---

## 6) Evaluation outputs (Definition of Done for the abstract)

For each site + case window:
- Plot: Tsfc_obs vs Tsfc_model for all scenarios
- Table: MAE, RMSE, Bias, Correlation for each scenario
- Optional: split metrics by day/night and/or clear/cloudy proxy
  - Example clear/cloudy proxy: observed effective atmospheric emissivity
    `eps_atm_obs = LW↓ / (σ * Tair^4)` (diagnostic only)

Across multiple windows:
- One summary table across windows (mean/median performance per scenario)
- One summary plot highlighting which simplification hurts most (ΔMAE from baseline)

**Minimum number of case windows:** 4–8 total is enough for a strong abstract if chosen well.

---

## 7) Case-window selection rules (keep physics honest)

We can filter windows to match model assumptions.
Suggested simple filters (tunable):
- Snow depth: `Hs > 0.20 m` for the full window
- Dry-snow focus (optional): `LWC < 0.5%` (if using LWC)  
- Avoid strong drifting-snow events (optional): drifting snow flux below threshold

If a window violates assumptions (wet snow, major drifting, melt-out), either:
- exclude it from the abstract analysis, or
- include it explicitly as an "out of scope / failure mode" example (only if time allows)

---

## 8) Coding standards (keep it simple and readable)

This repo is an undergraduate-friendly scientific codebase.

**Do**
- Use small pure functions
- Use plain dictionaries for config/scenarios
- Keep variable names explicit (`LWdown_Wm2`, `Tsfc_obs_C`)
- Write short docstrings that describe inputs/outputs
- Centralize unit conversions in one place
- Make every figure reproducible from raw inputs

**Avoid**
- Complex class hierarchies
- Global mutable flags (prefer passing a `scenario` dict into functions)
- "Notebook-only" hidden state that changes results based on execution order
- Overfitting the model to one window/site (ISSW wants generality)

---

## 9) Priority task list (high leverage first)

### P0 — Must fix before writing an ISSW abstract
1) ✅ **USGS data loader**: `notebooks/usgs_collector.py` — fetch iv data + clean units + consistent column naming
2) ✅ **Verify radiation parameter codes**: Confirmed SW↓/SW↑ (72186/72185) and LW↓/LW↑ (72175/72174)
3) **Scenario consistency**: solver + diagnostics use identical LW/SW assumptions
4) **H(t) implementation**: piecewise integration with profile remapping
5) **Primary validation**: use non-contact Tsfc (72405) as main observation
6) **Reproducible outputs**: one command/notebook cell produces plots + metrics table for a window

### P1 — Strongly recommended for a credible story
7) Use measured soil temperature as bottom boundary *if it simplifies and improves realism*
8) Simple stability correction for turbulent fluxes (or explicitly state "neutral" and remove stability)
9) Automated case-window loop: run N windows and write results to CSV

### P2 — Nice-to-have (only if time remains)
10) Two-band SW penetration (keep, but only if it demonstrably changes results)
11) Clear/cloudy regime classification and regime-specific error attribution plots

---

## 10) Repository hygiene (so the work survives beyond the notebook)

Current structure:
```
snowtempprofile/
├── notebooks/
│   ├── usgs_collector.py      # USGS data fetching (standalone module)
│   ├── 02_snowmodel_realdata.ipynb
│   └── 03_data_collection.ipynb
├── data/
│   ├── basin_collect.txt
│   ├── beck_collect.txt
│   └── boss_collect.txt
├── figures/
└── AGENTS.md                   # This file
```

Suggested additions:
- `src/` — core physics functions (solver, metrics) when they stabilize
- `results/` — CSV tables for metrics
- Keep notebooks lightweight by importing from standalone modules

**Agent rule:** If you add a new function used in the paper pipeline, consider putting it in a
standalone `.py` file and importing it into the notebook.

---

## 11) Quick abstract outline (for later; do not write until P0 tasks are done)

A strong ISSW abstract usually reads like:
1) Problem: why Tsfc accuracy matters (practice + science)
2) Method: minimal model + USGS forcing + scenario tests + multi-window evaluation
3) Results: quantitative skill + which simplification hurts most (numbers)
4) Takeaway: what practitioners should measure/assume (LW↓ importance) + limitations

---

## 12) Links (official references)

### ISSW 2026
- Submit an Abstract: https://www.issw2026.com/submit-an-abstract
- Guidelines for Papers + templates: https://www.issw2026.com/guidelines-for-papers
- Important Dates: https://www.issw2026.com/importantdates

### USGS station pages (parameter availability, quick plots)
- Senator Beck (legacy): https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=375429107433201
- Ptarmigan (legacy): https://waterdata.usgs.gov/nwis/uv?legacy=1&site_no=392954106162501

### USGS Water Services docs
- Instantaneous Values (iv): https://waterservices.usgs.gov/docs/instantaneous-values/instantaneous-values-details/
- Migration to modernized APIs: https://api.waterdata.usgs.gov/docs/ogcapi/migration/
