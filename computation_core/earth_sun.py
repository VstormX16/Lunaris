"""Sunrise and sunset on Earth from NASA SPICE (DE440 Sun/Earth positions, IAU Earth orientation).

Definition (the astronomical-almanac one): sunrise/sunset is when the Sun's upper limb is at the
horizon with standard refraction, i.e. the Sun's centre is at -(34' + its true semi-diameter).
Real refraction varies with weather, so any almanac time can differ by about a minute from what
you would see outside.
"""
import numpy as np
import spiceypy as sp

from data_engine import spice_loader as sl

from .space_time import DAY, _transitions

SUN_RADIUS_KM = 695_700.0
REFRACTION_DEG = 34.0 / 60.0


def _radii():
    sl.load_kernels()
    a, _, c = sp.bodvrd("EARTH", "RADII", 3)[1]
    return a, (a - c) / a  # equatorial radius (km), flattening


def _geometry(lat_deg, lon_deg, et):
    """Sun elevation of the centre (deg) and the horizon threshold (deg) at each ET, for a geodetic site."""
    a, f = _radii()
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    site = np.array(sp.georec(lon, lat, 0.0, a, f))
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])  # geodetic normal
    rel = sl.rel_positions("SUN", "EARTH", "IAU_EARTH", et) - site
    dist = np.linalg.norm(rel, axis=1)
    elevation = np.degrees(np.arcsin(np.clip(rel @ up / dist, -1, 1)))
    threshold = -(REFRACTION_DEG + np.degrees(np.arcsin(SUN_RADIUS_KM / dist)))
    return elevation, threshold


def earth_sun(lat_deg, lon_deg, et0, days=3.0, step_min=5.0, max_events=6):
    """State of the Sun now and its next rises and sets at a place on Earth."""
    def up(et):
        el, thr = _geometry(lat_deg, lon_deg, et)
        return el >= thr

    grid = et0 + np.arange(0.0, days * DAY, step_min * 60.0)
    up_now, transitions = _transitions(up, grid, max_events)
    el_now = float(_geometry(lat_deg, lon_deg, np.array([et0]))[0][0])
    s = sl.rel_positions("SUN", "EARTH", "IAU_EARTH", np.array([et0]))[0]
    return {
        "sun_up_now": up_now,
        "elevation_deg": el_now,
        "transitions": transitions,
        "searched_days": days,
        "subsolar": {"lat": float(np.degrees(np.arcsin(s[2] / np.linalg.norm(s)))),
                     "lon": float(np.degrees(np.arctan2(s[1], s[0])))},
    }
