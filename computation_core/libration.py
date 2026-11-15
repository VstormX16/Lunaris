"""What the Moon's libration does to Earth's visibility at one site.

The Moon keeps (on average) one face to Earth, but it rocks: its orientation (NASA's lunar orientation kernel, which includes the
physical libration), its elliptical orbit and Earth's own offset from the mean direction make Earth wander about +-7 deg in
latitude and +-8 deg in longitude over the sky of the Moon. Near the poles and the limb that wobble is the whole story: Earth
rises above the horizon and sets again, month after month.

Here the real Earth direction (SPICE, light-time corrected, in the Moon's own frame) is compared with the direction a perfectly
synchronous, non-librating Moon would give, which is the fixed mean direction (the +x axis of the mean-Earth frame). The difference
in line-of-sight time at the site is what libration alone contributes.
"""
import numpy as np

from data_engine import spice_loader

from .earth_visibility import dte_stats, dte_visible
from .horizon import compute_horizon


def libration_report(dem, lat, lon, start_utc, days=30, step_minutes=60):
    et = spice_loader.time_grid(start_utc, days, step_minutes)
    earth = spice_loader.body_vectors_me("EARTH", et)
    dist = np.linalg.norm(earth, axis=1)
    sub_lat = np.degrees(np.arcsin(earth[:, 2] / dist))
    sub_lon = np.degrees(np.arctan2(earth[:, 1], earth[:, 0]))
    fixed = np.column_stack([dist, np.zeros_like(dist), np.zeros_like(dist)])  # the mean Earth direction: lat 0, lon 0

    hz = compute_horizon(dem, lat, lon, 2.0)
    az, el, _ = spice_loader.topocentric("EARTH", lat, lon, et, body_vec=earth)
    az0, el0, _ = spice_loader.topocentric("EARTH", lat, lon, et, body_vec=fixed)
    vis, vis0 = dte_visible(az, el, hz), dte_visible(az0, el0, hz)
    h = step_minutes / 60.0
    real, still = dte_stats(vis, h), dte_stats(vis0, h)
    diff = real["dte_fraction"] - still["dte_fraction"]
    role = "enables" if still["dte_fraction"] < 0.02 and real["dte_fraction"] > 0.05 else "adds" if diff > 0.05 else "hides" if diff < -0.05 else "little"
    return {
        "site": {"lat": lat, "lon": lon},
        "hours": (np.arange(et.size) * h).tolist(),
        "sub_earth_lat_deg": sub_lat.tolist(),
        "sub_earth_lon_deg": sub_lon.tolist(),
        "earth_el_deg": el.tolist(),
        "earth_el_no_libration_deg": el0.tolist(),
        "terrain_toward_earth_deg": np.maximum(hz.at(az), 0.0).tolist(),
        "dte_fraction": real["dte_fraction"],
        "dte_fraction_no_libration": still["dte_fraction"],
        "longest_outage_hours": real["longest_outage_hours"],
        "longest_outage_hours_no_libration": still["longest_outage_hours"],
        "hours_gained": float(((vis) & ~vis0).sum() * h),
        "hours_lost": float((~vis & vis0).sum() * h),
        "role": role,
        "swing": {"lat_deg": [float(sub_lat.min()), float(sub_lat.max())], "lon_deg": [float(sub_lon.min()), float(sub_lon.max())]},
        "method": "NASA SPICE: DE440 ephemeris + lunar orientation kernel moon_pa_de440 (physical libration), light-time and aberration corrected",
    }
