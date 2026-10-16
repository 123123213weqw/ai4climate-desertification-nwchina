"""Geographic covariate: distance to nearest river line (oasis corridors).

Rivers from the Natural Earth 10m river centerlines (public); distance in km
on an equirectangular projection around the study-area central latitude.
"""
import json
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree

from .config import AOI_BBOX, DATA_DIR

GEO_FEATURES = ["dist_river_km"]
GEO_DIR = DATA_DIR / "geo"
NE_RIVERS_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_rivers_lake_centerlines.geojson"
)


def _scale(lons, lats, lat0):
    return np.column_stack(
        [np.asarray(lons) * 111.320 * np.cos(np.radians(lat0)), np.asarray(lats) * 110.574]
    )


class GeoSampler:
    def __init__(self):
        cache = GEO_DIR / "ne_rivers.geojson"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            r = requests.get(NE_RIVERS_URL, timeout=660)
            r.raise_for_status()
            payload = r.json()
            GEO_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
        lon_min, lat_min, lon_max, lat_max = AOI_BBOX
        pts = []
        for f in payload.get("features", []):
            g = f.get("geometry") or {}
            lines = [g["coordinates"]] if g.get("type") == "LineString" else g.get("coordinates", [])
            for line in lines:
                for p in line:
                    if lon_min <= p[0] <= lon_max and lat_min <= p[1] <= lat_max:
                        pts.append((p[0], p[1]))
        pts = np.asarray(pts, dtype=float)
        lat0 = 0.5 * (lat_min + lat_max)
        self.lat0 = lat0
        self.tree = cKDTree(_scale(pts[:, 0], pts[:, 1], lat0)) if len(pts) else None
        print(f"[geo] rivers: {len(pts)} vertices in bbox")

    def sample(self, lons, lats) -> np.ndarray:
        X = _scale(lons, lats, self.lat0)
        d = self.tree.query(X)[0] if self.tree is not None else np.zeros(len(lons))
        return d.reshape(-1, 1).astype(np.float32)


_SCALAR = None


def scalar_covariates(lons, lats) -> np.ndarray:
    """CNN scalar head inputs: 5 climate covariates + river distance."""
    global _SCALAR
    from .climate import ClimateSampler

    if _SCALAR is None:
        _SCALAR = (ClimateSampler(), GeoSampler())
    clim, geo = _SCALAR
    return np.concatenate([clim.sample(lons, lats), geo.sample(lons, lats)], axis=1)
