"""LRO LOLA polar DEM downloader (PDS Geosciences Node)."""
from computation_core.dem import LolaDEM

from . import config
from .net import download


def ensure_lola(product=None):
    """Download the IMG + label if missing; return path to the .img."""
    p = product or config.LOLA_PRODUCT
    for ext in ("lbl", "img"):
        path = download(f"{config.LOLA_BASE}/{p['name']}.{ext}", config.LOLA_DIR / f"{p['name']}.{ext}")
    return path


def ensure_global_lola():
    """Download the whole-Moon LDEM_64 grid (531 MB) + label if missing; return path to the .img."""
    for ext in ("lbl", "img"):
        path = download(f"{config.LOLA_GLOBAL_BASE}/ldem_64.{ext}", config.LOLA_DIR / f"LDEM_64.{ext}")
    return path


def load_dem(product=None):
    p = product or config.LOLA_PRODUCT
    return LolaDEM.from_img(ensure_lola(p), p["size"], p["pixel_m"])
