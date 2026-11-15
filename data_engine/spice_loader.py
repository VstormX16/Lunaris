"""NAIF SPICE access: Sun/Earth vectors in the Moon ME frame and topocentric Az/El."""
import numpy as np
import spiceypy as sp

from computation_core.constants import MOON_RADIUS_M
from computation_core.coordinates import latlon_to_unit, vector_to_azel

from . import config
from .net import download

_loaded = False


def load_kernels():
    """Download (if needed) and furnish all kernels once."""
    global _loaded
    if _loaded:
        return
    for name, url in config.SPICE_KERNELS.items():
        sp.furnsh(str(download(url, config.SPICE_DIR / name)))
    _loaded = True


def utc_to_et(utc_iso):
    load_kernels()
    return sp.str2et(utc_iso.rstrip("Z"))  # SPICE parses ISO strings as UTC by default


def et_to_utc(et):
    """ET seconds -> ISO UTC string (minute precision)."""
    load_kernels()
    return sp.et2utc(float(et), "ISOC", 0)[:16] + "Z"


def et_to_utc_s(et):
    """ET seconds -> ISO UTC string (second precision), e.g. 2026-11-24T14:53:07Z."""
    load_kernels()
    return sp.et2utc(float(et), "ISOC", 0) + "Z"


def rel_positions(target, observer, frame, et, abcorr="LT+S"):
    """Vectors observer -> target (km) in `frame` at each ET; (N, 3)."""
    load_kernels()
    pos, _ = sp.spkpos(target, np.atleast_1d(et).tolist(), frame, abcorr, observer)
    return np.asarray(pos)


def time_grid(start_utc_iso, days, step_minutes):
    """ET array covering [start, start+days) at a fixed cadence."""
    et0 = utc_to_et(start_utc_iso)
    return et0 + np.arange(0.0, days * 86400.0, step_minutes * 60.0)


def body_vectors_me(body, et):
    """Light-time + stellar-aberration corrected vector Moon center -> body, km, MOON_ME frame."""
    load_kernels()
    pos, _ = sp.spkpos(body, np.atleast_1d(et).tolist(), "MOON_ME", "LT+S", "MOON")
    return np.asarray(pos)


def topocentric(body, lat_deg, lon_deg, et, elevation_m=0.0, body_vec=None):
    """Az/El (deg) and range (km) of a body as seen from a lunar surface site.

    `body_vec` (km, from body_vectors_me) lets many sites reuse one ephemeris query.
    """
    site_km = latlon_to_unit(lat_deg, lon_deg) * (MOON_RADIUS_M + elevation_m) / 1000.0
    rel = (body_vectors_me(body, et) if body_vec is None else body_vec) - site_km
    az, el = vector_to_azel(rel, lat_deg, lon_deg)
    return az, el, np.linalg.norm(rel, axis=-1)


def get_sun_earth_positions(target_lat, target_lon, utc_iso_time, elevation_m=0.0):
    """Single-epoch Sun/Earth Az/El/distance for a site (PLAN.md Module A interface)."""
    et = utc_to_et(utc_iso_time)
    out = {}
    for key, body in (("sun", "SUN"), ("earth", "EARTH")):
        az, el, d = topocentric(body, target_lat, target_lon, et, elevation_m)
        out[key] = {"azimuth_deg": float(az[0]), "elevation_deg": float(el[0]), "distance_km": float(d[0])}
    sub = body_vectors_me("EARTH", et)[0]
    out["earth"]["sub_earth_lat"] = float(np.degrees(np.arcsin(sub[2] / np.linalg.norm(sub))))
    out["earth"]["sub_earth_lon"] = float(np.degrees(np.arctan2(sub[1], sub[0])))
    return out
