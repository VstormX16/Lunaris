"""One elevation source for the whole Moon: the 40 m LOLA grid at the south pole, the 470 m LDEM_64 grid everywhere else.

The analysis (horizon, illumination, Earth link, slope, terrain class) used to work only below 85 S because that is where the
detailed grid ends. `SiteDEM` wraps both, behaves exactly like the polar grid where that exists (every attribute not defined here
is the polar grid's own), and gives the functions that need terrain a global alternative elsewhere.

The global grid is coarser: slopes are regional (about 2 km) and hills or crater walls narrower than a few hundred metres are
smoothed away. Callers can ask `source(lat, lon)` and say so.
"""
import numpy as np

from .constants import MOON_RADIUS_M
from .coordinates import enu_basis
from .global_dem import GlobalDEM
from .terrain_slope import slope_map


class SiteDEM:
    def __init__(self, polar, global_dem: GlobalDEM | None):
        self.polar = polar
        self.global_dem = global_dem

    def __getattr__(self, name):  # only reached for names not defined on SiteDEM: the polar grid's own (dn, n, pixel_m, patch, ...)
        if name in ("polar", "global_dem"):
            raise AttributeError(name)
        return getattr(self.polar, name)

    def covers(self, lat, lon) -> bool:
        """True where the detailed polar grid has data."""
        return not np.isnan(float(self.polar.height_at_latlon(lat, lon)))

    def source(self, lat, lon) -> str:
        if self.covers(lat, lon):
            return "lola_40m"
        return "ldem_470m" if self.global_dem is not None else "none"

    def has_data(self, lat, lon) -> bool:
        return self.source(lat, lon) != "none"

    def height_at_latlon(self, lat, lon):
        h = float(self.polar.height_at_latlon(lat, lon))
        if np.isnan(h) and self.global_dem is not None:
            return float(self.global_dem.height(lat, lon))
        return h


def tangent_points(lat, lon, east_m, north_m):
    """(lat, lon) of points offset from a site along its local east and north, on the sphere (arrays of the same shape)."""
    east, north, up = enu_basis(lat, lon)
    e, n = np.asarray(east_m, float)[..., None], np.asarray(north_m, float)[..., None]
    p = up * MOON_RADIUS_M + e * east + n * north
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1))), np.degrees(np.arctan2(p[..., 1], p[..., 0])) % 360.0


def global_hazard_stats(g: GlobalDEM, lat, lon, radius_m=500.0, max_slope_deg=10.0, with_slopes=False):
    """Landing-area slope statistics from the global grid. Same keys as the polar version.

    The grid is 470 m per pixel, so the area is never smaller than 1.5 km in radius: below that it would be a handful of pixels."""
    px = 250.0  # sample spacing (bilinear between the 470 m pixels)
    reach = max(radius_m, 1500.0)
    half = int((reach + 2 * px) // px)
    offs = np.arange(-half, half + 1) * px
    ee, nn = np.meshgrid(offs, offs)
    la, lo = tangent_points(lat, lon, ee, nn)
    heights = g.height(la, lo)
    slopes = slope_map(heights, px)
    yy, xx = np.mgrid[0:heights.shape[0], 0:heights.shape[1]]
    disk = np.hypot(yy - half, xx - half) * px <= reach
    s, h = slopes[disk], heights[disk]
    plane = np.column_stack([xx[disk], yy[disk], np.ones(h.size)])
    coef, *_ = np.linalg.lstsq(plane, h, rcond=None)
    out = {
        "mean_slope_deg": float(s.mean()),
        "max_slope_deg": float(s.max()),
        "safe_area_fraction": float((s <= max_slope_deg).mean()),
        "relief_m": float(h.max() - h.min()),
        "roughness_rms_m": float(np.std(h - plane @ coef)),
        "site_elevation_m": float(g.height(lat, lon)),
        "terrain_source": "ldem_470m",
        "area_radius_m": float(reach),
    }
    if with_slopes:
        out["slopes"] = s
    return out
