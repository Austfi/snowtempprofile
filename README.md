# Snowpack Temperature Profile Model

A simple 1D heat diffusion model for snowpack temperature evolution.  
Developed for Numerical Modeling course at CU Boulder.

## Overview

This model simulates vertical temperature profiles through a layered snowpack using surface energy balance and measured meteorological inputs. It demonstrates core concepts in:

- 1D heat diffusion with multi-layer discretization
- Surface energy balance (shortwave, longwave, sensible heat)
- Beer-Lambert shortwave penetration

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_snowmodel_intro.ipynb` | Core model with idealized conditions |
| 2 | `02_snowmodel_realdata.ipynb` | Integration with real CAIC station data |
| 3 | `03_data_collection.ipynb` | Tools for scraping weather station data |

## Data Sources

- [Colorado Avalanche Information Center (CAIC)]
- USGS National Water Information System (NWIS)

## Repository Structure

```
├── notebooks/     # Jupyter notebooks
├── data/          # Weather station data files
└── figures/       # Output visualizations
```
