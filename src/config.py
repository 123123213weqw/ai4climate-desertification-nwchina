"""Shared configuration for the Northwest-China desertification susceptibility pipeline (Track 6)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
TILE_CACHE = DATA_DIR / "tiles"

# Study area: arid Gansu / Inner Mongolia belt (Tengger, Mu Us, Hexi corridor, Gobi fringe).
AOI_BBOX = (97.5, 35.8, 106.5, 42.5)  # lon_min, lat_min, lon_max, lat_max (POWER regional cap: <=10 deg)

# Label window (ESA WorldCover tiles 99-105E / 36-42N): sampling of bare vs vegetated sites.
LABEL_BBOX = (99.2, 36.2, 104.8, 41.8)

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_ZOOM = 12

FEATURES = [
    # terrain (Terrarium / SRTM-derived)
    "elevation",
    "slope_deg",
    "northness",
    "eastness",
    "curvature",
    "roughness",
    "relief_9x9",
    "twi_proxy",
    # climate (NASA POWER climatology)
    "annual_precip_mm",
    "monsoon_precip_mm",
    "precip_cv",
    "wind_speed_ms",
    "temp_mean_c",
    # hydrology
    "dist_river_km",
]

# Map AOI: Tengger Desert - Hexi corridor restoration belt.
MAP_AOI_DEFAULT = (102.8, 37.7, 105.3, 39.6)
MAP_AOI_SMOKE = (103.8, 38.3, 104.3, 38.7)

SPATIAL_BLOCK_DEG = 0.5
RANDOM_SEED = 42

WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
WORLDCOVER_TILES = ["N36E099", "N36E102", "N39E099", "N39E102"]  # 3x3 deg tiles covering LABEL_BBOX
WORLDCOVER_DIR = DATA_DIR / "worldcover"
