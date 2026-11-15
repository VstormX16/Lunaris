"""JPL Horizons REST client: independent topocentric Az/El for cross-verification."""
import requests

from . import config

BODY_IDS = {"SUN": "10", "EARTH": "399"}


def topocentric_azel(body, lat_deg, lon_deg, start_utc, stop_utc, step="1h", alt_km=0.0):
    """Return list of (time_str, az_deg, el_deg) seen from a lunar surface site."""
    params = {
        "format": "json",
        "COMMAND": f"'{BODY_IDS[body]}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "CENTER": "'coord@301'",
        "COORD_TYPE": "'GEODETIC'",
        "SITE_COORD": f"'{lon_deg},{lat_deg},{alt_km}'",
        "START_TIME": f"'{start_utc}'",
        "STOP_TIME": f"'{stop_utc}'",
        "STEP_SIZE": f"'{step}'",
        "QUANTITIES": "'4'",
        "ANG_FORMAT": "'DEG'",
        "CSV_FORMAT": "'YES'",
    }
    r = requests.get(config.HORIZONS_URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.json()["result"]
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons error: {text[:300]}")
    rows = []
    for line in text.split("$$SOE")[1].split("$$EOE")[0].strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        rows.append((f[0], float(f[3]), float(f[4])))
    return rows
