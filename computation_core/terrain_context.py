"""Terrain context of an arbitrary lat/lon: crater floor, summit/massif, rim/ridge, slope or plain.

Rule-based on real LOLA heights: site height vs. surrounding rings plus the
terrain horizon (a crater floor is walled in on all sides, a summit looks
down on everything).
"""
import numpy as np

from .coordinates import polar_stereo_forward
from .dem import LolaDEM
from .horizon import compute_horizon

RING_RADII_M = (2_000.0, 5_000.0, 10_000.0)
N_RING_POINTS = 72


def _ring_heights(dem: LolaDEM, x0, y0, radius_m):
    th = np.linspace(0.0, 2 * np.pi, N_RING_POINTS, endpoint=False)
    h = dem.height_nearest(x0 + radius_m * np.cos(th), y0 + radius_m * np.sin(th))
    return h[~np.isnan(h)]


def _global_ring(g, lat, lon, radius_m):
    from .site_dem import tangent_points
    th = np.linspace(0.0, 2 * np.pi, N_RING_POINTS, endpoint=False)
    la, lo = tangent_points(lat, lon, radius_m * np.cos(th), radius_m * np.sin(th))
    return g.height(la, lo)


def classify_terrain(dem: LolaDEM, lat_deg, lon_deg, relief_threshold_m=150.0):
    g = getattr(dem, "global_dem", None)
    whole_moon = g is not None and not dem.covers(lat_deg, lon_deg)  # outside the detailed polar grid: the 470 m whole-Moon grid
    site_h = float(dem.height_at_latlon(lat_deg, lon_deg))
    if np.isnan(site_h):
        raise ValueError("location outside DEM coverage")
    if whole_moon:
        ring = np.concatenate([_global_ring(g, lat_deg, lon_deg, r) for r in RING_RADII_M])
    else:
        x0, y0 = polar_stereo_forward(lat_deg, lon_deg)
        ring = np.concatenate([_ring_heights(dem, float(x0), float(y0), r) for r in RING_RADII_M])
    ring_median = float(np.median(ring))
    relief = site_h - ring_median  # >0: above surroundings
    below = float((ring > site_h + 50.0).mean())   # share of surroundings clearly higher
    above = float((ring < site_h - 50.0).mean())   # share clearly lower
    hz = compute_horizon(dem, lat_deg, lon_deg, n_az=180, max_range_m=20_000.0)
    hz_mean = float(hz.el_deg.mean())

    if relief < -relief_threshold_m and below > 0.7 and hz_mean > 1.0:
        kind = "crater_floor_or_basin"
    elif relief > relief_threshold_m and above > 0.7 and hz_mean < 2.0:
        kind = "summit_or_massif"
    elif below > 0.35 and above > 0.35:
        kind = "slope_or_rim"
    else:
        kind = "plain_or_gentle_terrain"
    return {
        "class": kind,
        "site_height_m": site_h,
        "relief_vs_surroundings_m": relief,
        "fraction_surroundings_higher": below,
        "fraction_surroundings_lower": above,
        "mean_horizon_deg": hz_mean,
        "data_source": "ldem_470m" if whole_moon else "lola_40m",
    }
