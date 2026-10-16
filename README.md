# Where Should Restoration Go First? Desertification Susceptibility for Northwest China (AI4Climate Track 6)

Repository: https://github.com/123123213weqw/ai4climate-desertification-nwchina

Open-source pipeline for **Track 6 (Desertification control and climate resilience,
Northwest China)** of the [AI4Climate Global Youth Climate Solutions
Challenge](https://www.ai4climate.world/en). The competition question for this
track: limited people and money — which places should be prioritized, and what
should be done there? We answer the first half with a reproducible,
high-resolution susceptibility map.

## What it does

1. **Labels** — ESA WorldCover 2021 (10 m, open): a site is a *desertified-state*
   positive when mapped as bare/sparse vegetation (class 60) and a negative when
   mapped as grass/shrub/cropland/tree, kept only where the class is stable over
   a ~300 m neighbourhood. 600/600 balanced sample over 99–105°E, 36–42°N
   (Tengger, Mu Us, Hexi corridor, Gobi fringe).
2. **Features (14, all keyless)** — 8 terrain descriptors from SRTM-based
   Terrarium tiles (~34 m/px); 5 climate covariates from NASA POWER (annual and
   Jun–Sep precipitation, precipitation variability, mean 2 m wind speed, mean
   temperature); distance to rivers (Natural Earth) as the oasis-corridor
   factor.
3. **Models** — logistic regression, gradient boosting, MLP on the 14 scalars,
   and a residual CNN over 8×33×33 terrain patches fused with the scalar head —
   all evaluated with grouped 5-fold CV on 0.5° spatial blocks.
4. **Product** — sliding-window CNN inference (~200 m) over the
   Tengger–Hexi corridor ranks every cell by predicted desertification
   susceptibility; among cells that are still vegetated, the highest-risk ones
   are the watchlist for monitoring and restoration siting.

## Results (5-fold spatial block CV, 1200 sites)

| Model | AUC-ROC | AP |
|---|---|---|
| **MLP (14 factors)** | **0.988 ± 0.006** | 0.990 |
| **HybridCNN (patches + scalars)** | **0.988 ± 0.004** | 0.989 |
| Gradient boosting | 0.986 ± 0.008 | 0.988 |
| Logistic regression | 0.985 ± 0.006 | 0.986 |
| CNN, patches only (ablation) | 0.982 ± 0.011 | 0.985 |

The signal is strong and nearly linear: whether land holds vegetation in this
arid belt is decided primarily by **growing-season (Jun–Sep) rainfall** —
permutation importance ranks it 15× ahead of every other factor (ΔAUC 0.140 vs
≤0.009), with wind speed and elevation distant seconds. That is itself the
actionable finding: susceptibility tracks the monsoon-rain isohyet, so rainfall
trends shift the risk map, and intervention siting should follow.

## Quickstart

```bash
pip install -r requirements.txt   # + tifffile for WorldCover decoding
bash scripts/run_all.sh           # data -> models -> map (downloads WorldCover tiles ~160 MB, terrain tiles on demand)
```

Outputs land in `runs/`: `metrics.json`, `susceptibility_map_cnn_default.png`,
`importance.png`, trained models, GIS-ready `susceptibility.npz`.

## Data & licenses

- ESA WorldCover 2021 v200 (10 m), CC-BY 4.0, via AWS Open Data.
- NASA POWER climatology (PRECTOTCORR_SUM, WS2M, T2M) — keyless API.
- Terrarium terrain tiles (SRTM-derived) via AWS Open Data.
- Natural Earth 10m rivers, public domain.
- Code: MIT.

## Known limitations (kept honest)

- Labels are a **degradation-state proxy** (bare vs vegetated in 2021), not an
  observed transition; the map ranks susceptibility of the current state, and
  temporal NDVI trends would be needed to claim change.
- Sampling is balanced, so mapped probabilities are relative ranks, not
  calibrated areal fractions.
- POWER climate is ~0.5°: it captures the regional monsoon gradient, not
  oasis-scale humidity; lithology, groundwater and human water use are absent.
- One snapshot year; multi-year WorldCover/MODIS would separate drought years
  from structural degradation.

## Roadmap

1. Temporal labels (MODIS NDVI trends 2000–2025) → predicted *change*, not state.
2. Add lithology/soil (SoilGrids, open) and groundwater proxies.
3. Calibrate probabilities per county for budget-allocation use.
4. Intervention recommender: match top-risk cells to measures (shelterbelt,
   grazing closure, straw checkerboard) using published effectiveness rules.
5. 90-day pilot: hand the watchlist to one county forestry/grassland bureau.
