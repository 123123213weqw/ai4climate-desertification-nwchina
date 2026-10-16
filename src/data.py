"""Build the NW-China desertification dataset (labels: ESA WorldCover 2021).

Usage:
  python -m src.data build            # scalar dataset (14 features)
  python -m src.data build-patches    # + 8-channel terrain patches for the CNN
"""
import argparse
import json

import numpy as np
import pandas as pd

from .climate import ClimateSampler
from .config import DATA_DIR, FEATURES, LABEL_BBOX, RANDOM_SEED, RUNS_DIR, SPATIAL_BLOCK_DEG
from .elevation import TerrainSampler
from .geo import GEO_FEATURES, GeoSampler
from .patches import CHANNELS as TERRAIN
from .patches import PatchSampler
from .worldcover import BARE, VEGETATED, WorldCoverSampler

N_CANDIDATES = 6000
N_PER_CLASS = 600


def build():
    rng = np.random.default_rng(RANDOM_SEED)
    print("sampling label candidates from WorldCover ...")
    cand = rng.uniform(LABEL_BBOX[:2], LABEL_BBOX[2:], size=(N_CANDIDATES, 2))
    classes = WorldCoverSampler().sample(cand[:, 0], cand[:, 1])
    bare = np.where(classes == BARE)[0]
    veg = np.where(np.isin(classes, list(VEGETATED)))[0]
    print(f"stable candidates: bare={len(bare)} vegetated={len(veg)}")
    n = min(len(bare), len(veg), N_PER_CLASS)
    keep = np.concatenate([rng.choice(bare, n, replace=False), rng.choice(veg, n, replace=False)])
    lons, lats = cand[keep, 0], cand[keep, 1]
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)

    sampler = TerrainSampler()
    feats, ok = sampler.sample(lons, lats)
    print(f"terrain features ok: {ok.sum()}/{len(ok)}")
    climate = ClimateSampler().sample(lons, lats)
    geo = GeoSampler().sample(lons, lats)

    rows = pd.DataFrame(feats[ok], columns=TERRAIN)
    rows["label"] = y[ok]
    rows["longitude"] = lons[ok]
    rows["latitude"] = lats[ok]
    for k, name in enumerate(["annual_precip_mm", "monsoon_precip_mm", "precip_cv", "wind_speed_ms", "temp_mean_c"]):
        rows[name] = climate[ok][:, k]
    for k, name in enumerate(GEO_FEATURES):
        rows[name] = geo[ok][:, k]
    rows["block_id"] = (
        (np.floor(rows["longitude"] / SPATIAL_BLOCK_DEG) * SPATIAL_BLOCK_DEG).astype(str)
        + "_"
        + (np.floor(rows["latitude"] / SPATIAL_BLOCK_DEG) * SPATIAL_BLOCK_DEG).astype(str)
    )
    out = DATA_DIR / "dataset.parquet"
    rows.to_parquet(out)
    print(f"saved {len(rows)} rows ({rows['label'].sum()} bare / {len(rows) - rows['label'].sum()} vegetated) -> {out}")

    RUNS_DIR.mkdir(exist_ok=True)
    summary = {
        "n_total": len(rows),
        "n_positive_bare": int(rows["label"].sum()),
        "features": FEATURES,
        "labels": "ESA WorldCover 2021: bare/sparse (60) vs vegetated (10/20/30/40), ~300m-stable sites",
        "notes": [
            "Positives: bare/sparse vegetation sites; Negatives: grass/shrub/cropland/tree sites",
            "Label window 99-105E / 36-42N; spatial CV blocks 0.5 deg",
        ],
    }
    (RUNS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))


def build_patches(patch_size: int = 33):
    from .geo import scalar_covariates

    df = pd.read_parquet(DATA_DIR / "dataset.parquet")
    sampler = PatchSampler()
    patches, ok = sampler.sample(
        df["longitude"].to_numpy().astype(float), df["latitude"].to_numpy().astype(float), size=patch_size
    )
    scalars = scalar_covariates(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    out = DATA_DIR / "patches.npz"
    np.savez_compressed(
        out,
        patches=patches[ok].astype(np.float16),
        scalars=scalars[ok],
        label=df["label"].to_numpy()[ok],
        longitude=df["longitude"].to_numpy()[ok],
        latitude=df["latitude"].to_numpy()[ok],
        block_id=df["block_id"].astype(str).to_numpy()[ok],
        patch_size=patch_size,
    )
    print(f"saved {int(ok.sum())} patches {patches.shape[1:]} + scalars {scalars.shape[1]} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build-patches"])
    ap.add_argument("--patch-size", type=int, default=33)
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        build_patches(args.patch_size)
