"""Paths, verified NASA endpoints and candidate sites."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
SPICE_DIR = DATA_DIR / "spice"
LOLA_DIR = DATA_DIR / "lola_dem"
GLOBAL_DEM_PATH = LOLA_DIR / "LDEM_64.img"  # whole-Moon LOLA, 64 px/deg (optional: smooth sphere without it)
CACHE_DIR = DATA_DIR / "cache"

NASA_API_KEY = os.getenv("NASA_API_KEY", "")

NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
SPICE_KERNELS = {  # local name -> url (load order matters: LSK, FK, PCK, SPK)
    "naif0012.tls": f"{NAIF}/lsk/naif0012.tls",
    "moon_de440_250416.tf": f"{NAIF}/fk/satellites/moon_de440_250416.tf",
    "pck00010.tpc": f"{NAIF}/pck/pck00010.tpc",
    "moon_pa_de440_200625.bpc": f"{NAIF}/pck/moon_pa_de440_200625.bpc",
    "de440s.bsp": f"{NAIF}/spk/planets/de440s.bsp",
}

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

LOLA_BASE = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/polar/img"
)
LOLA_GLOBAL_BASE = (  # whole-Moon LDEM_64 (64 px/deg, ~470 m at the equator, 531 MB): used wherever the polar grid does not reach
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img"
)
# 85S-90S, 40 m/pix, 7584x7584 int16 (~115 MB): covers all candidate regions.
LOLA_PRODUCT = {"name": "ldem_85s_40m", "size": 7584, "pixel_m": 40.0}

# Feature centers located in the LOLA DEM itself (planetocentric lat, lon east):
# crater = lowest point of 360 m-smoothed terrain, massif = highest point, searched around literature seeds.
# de Gerlache is omitted: its literature coordinates did not match a crater floor in this DEM.
CANDIDATE_SITES = {
    "shackleton": (-89.625, 119.13),  # floor -2848 m
    "haworth": (-87.496, 347.64),     # floor -4018 m
    "malapert": (-85.987, 2.08),      # summit +5169 m
}
