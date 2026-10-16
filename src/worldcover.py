"""ESA WorldCover 2021 (10 m) as the land-degradation label source.

Class codes: 10 tree, 20 shrub, 30 grass, 40 cropland, 50 built, 60 bare/sparse,
70 snow/ice, 80 water, 90 wetland, 95 mangrove.

Label definition (honest proxy): a site is a positive when WorldCover maps it as
bare / sparse vegetation (60) and a negative when it maps grass / shrub /
cropland / tree cover (10/20/30/40). Sites are kept only when the class is
stable across a ~300 m neighbourhood (centre + four probes), which removes
boundary noise. Water, built, snow and wetland sites are excluded.

Tiles are 3x3 deg, 36000x36000 px, ~30-300 MB COGs; one tile is decoded at a
time (~1.3 GB RAM peak).
"""
import gc

import numpy as np
import requests
import tifffile

from .config import LABEL_BBOX, WORLDCOVER_DIR, WORLDCOVER_TILES, WORLDCOVER_URL

VEGETATED = {10, 20, 30, 40}
BARE = 60
TILE_PX = 36000
PROBE = 15  # +-150 m probes around the centre pixel


def _origin(name):
    n, e = name.split("E")
    return float(e), float(n[1:])  # (lon0, lat0)


class WorldCoverSampler:
    def sample(self, lons, lats):
        """Class code per site (centre pixel when the +-150 m probes agree, else 255)."""
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        out = np.full(len(lons), 255, dtype=np.uint8)
        for name in WORLDCOVER_TILES:
            lon0, lat0 = _origin(name)
            inside = (lons >= lon0) & (lons < lon0 + 3.0) & (lats >= lat0) & (lats < lat0 + 3.0)
            if not inside.any():
                continue
            path = WORLDCOVER_DIR / f"{name}.tif"
            if not path.exists():
                WORLDCOVER_DIR.mkdir(parents=True, exist_ok=True)
                r = requests.get(WORLDCOVER_URL.format(tile=name), timeout=900)
                r.raise_for_status()
                path.write_bytes(r.content)
            a = tifffile.imread(path)
            res = 3.0 / a.shape[1]
            cols = np.clip(((lons[inside] - lon0) / res).astype(int), 0, a.shape[1] - 1)
            rows = np.clip(((lat0 + 3.0 - lats[inside]) / res).astype(int), 0, a.shape[0] - 1)
            up = np.clip(rows - PROBE, 0, a.shape[0] - 1)
            dn = np.clip(rows + PROBE, 0, a.shape[0] - 1)
            lf = np.clip(cols - PROBE, 0, a.shape[1] - 1)
            rt = np.clip(cols + PROBE, 0, a.shape[1] - 1)
            probes = [a[rows, cols], a[up, cols], a[dn, cols], a[rows, lf], a[rows, rt]]
            stable = np.ones(len(cols), dtype=bool)
            for p in probes[1:]:
                stable &= p == probes[0]
            out_idx = np.where(inside)[0]
            out[out_idx[stable]] = probes[0][stable]
            del a
            gc.collect()
            print(f"[worldcover] {name}: {inside.sum()} candidates, {stable.sum()} stable")
        return out


def in_label_bbox(lons, lats):
    return (
        (lons >= LABEL_BBOX[0]) & (lons <= LABEL_BBOX[2])
        & (lats >= LABEL_BBOX[1]) & (lats <= LABEL_BBOX[3])
    )
