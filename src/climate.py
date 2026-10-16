"""Climate covariates from the NASA POWER API (no key required).

Three single-parameter regional requests cover the study area on a
~0.625 deg lon x 0.5 deg lat grid (monthly climatology). Sites are matched to
the nearest grid cell.

Covariates per site:
  annual_precip_mm  - annual precipitation total (mm)
  monsoon_precip_mm - Jun-Sep total (the NW-China growing-season rains)
  precip_cv         - inter-monthly std/mean of precipitation
  wind_speed_ms     - annual mean 2 m wind speed (aeolian erosion driver)
  temp_mean_c       - annual mean 2 m air temperature
"""
import json

import numpy as np
import requests

from .config import AOI_BBOX, DATA_DIR

CLIMATE_FEATURES = ["annual_precip_mm", "monsoon_precip_mm", "precip_cv", "wind_speed_ms", "temp_mean_c"]
MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
CACHE_DIR = DATA_DIR / "climate"


def _url(param):
    return (
        "https://power.larc.nasa.gov/api/temporal/climatology/regional"
        f"?latitude-min={AOI_BBOX[1]}&latitude-max={AOI_BBOX[3]}"
        f"&longitude-min={AOI_BBOX[0]}&longitude-max={AOI_BBOX[2]}"
        f"&parameters={param}&community=AG&format=JSON"
    )


class ClimateSampler:
    """Nearest-cell sampling of POWER monthly climatology for precip/wind/temp."""

    PARAMS = ["PRECTOTCORR_SUM", "WS2M", "T2M"]  # 1 parameter per POWER request

    def __init__(self):
        self.grids = []
        for param in self.PARAMS:
            cache = CACHE_DIR / f"{param}.json"
            if cache.exists():
                payload = json.loads(cache.read_text())
            else:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                r = requests.get(_url(param), timeout=300)
                r.raise_for_status()
                payload = r.json()
                cache.write_text(json.dumps(payload))
            lons, lats, months = [], [], []
            for f in payload["features"]:
                lon, lat, _ = f["geometry"]["coordinates"]
                m = f["properties"]["parameter"][param]
                lons.append(lon)
                lats.append(lat)
                months.append([m[k] for k in MONTH_KEYS])
            self.grids.append(
                (np.asarray(lons), np.asarray(lats), np.asarray(months, dtype=np.float32))
            )

    def sample(self, lons, lats) -> np.ndarray:
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        out = np.zeros((len(lons), 5), dtype=np.float32)
        for gi, (glons, glats, months) in enumerate(self.grids):
            d2 = (glons[None, :] - lons[:, None]) ** 2 + (glats[None, :] - lats[:, None]) ** 2
            vals = months[d2.argmin(axis=1)]  # (n, 12)
            if gi == 0:  # precipitation
                out[:, 0] = vals.sum(axis=1)
                out[:, 1] = vals[:, 5:9].sum(axis=1)
                out[:, 2] = vals.std(axis=1) / (vals.mean(axis=1) + 1e-6)
            elif gi == 1:  # wind speed (m/s), annual mean
                out[:, 3] = vals.mean(axis=1)
            else:  # temperature (degC), annual mean
                out[:, 4] = vals.mean(axis=1)
        return out
