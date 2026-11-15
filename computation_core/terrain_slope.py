"""DEM gradient and landing-safety slope metrics."""
import numpy as np

from .dem import LolaDEM


def slope_map(heights_m, pixel_m):
    """Slope (deg) per pixel from central differences on a height window (m)."""
    gy, gx = np.gradient(heights_m.astype(np.float64), pixel_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def site_hazard_stats(dem: LolaDEM, lat_deg, lon_deg, radius_m=500.0, max_slope_deg=10.0, with_slopes=False):
    """Slope/roughness statistics inside a landing-area disk centered on the site.

    Gradients use a window padded by 1 pixel so edge pixels are valid. `with_slopes` adds every pixel's slope in the
    disk (the touchdown points the mission risk samples from).
    """
    g = getattr(dem, "global_dem", None)
    if g is not None and not dem.covers(lat_deg, lon_deg):
        from .site_dem import global_hazard_stats  # imported here: site_dem needs slope_map from this module
        return global_hazard_stats(g, lat_deg, lon_deg, radius_m, max_slope_deg, with_slopes)
    pad = radius_m + 2 * dem.pixel_m
    heights, px = dem.patch(lat_deg, lon_deg, pad)
    slopes = slope_map(heights, px)
    n = heights.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    disk = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) * px <= radius_m
    s, h = slopes[disk], heights[disk]
    plane = np.column_stack([xx[disk], yy[disk], np.ones(h.size)])
    coef, *_ = np.linalg.lstsq(plane, h, rcond=None)
    out = {
        "mean_slope_deg": float(s.mean()),
        "max_slope_deg": float(s.max()),
        "safe_area_fraction": float((s <= max_slope_deg).mean()),
        "relief_m": float(h.max() - h.min()),
        "roughness_rms_m": float(np.std(h - plane @ coef)),  # RMS about best-fit plane
        "site_elevation_m": float(dem.height_at_latlon(lat_deg, lon_deg)),
    }
    if with_slopes:
        out["slopes"] = s
    return out
