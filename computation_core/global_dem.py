"""Whole-Moon elevation from NASA LRO LOLA LDEM_64 (64 pixels per degree, about 470 m at the equator).

Much coarser than the 40 m south-pole grid, so slopes are regional (over ~2 km) and small hills or crater walls
narrower than a few hundred metres are smoothed away. It is real LOLA data, used wherever the polar grid does not reach.
"""
from pathlib import Path

import numpy as np

from .constants import MOON_RADIUS_M
from .coordinates import enu_basis
from .horizon import HorizonProfile

PPD = 64
ROWS, COLS = 180 * PPD, 360 * PPD
SCALE_M = 0.5  # metres per data number, relative to the 1737.4 km reference sphere
SLOPE_BASELINE_M = 1000.0


class GlobalDEM:
    def __init__(self, path):
        self.dn = np.memmap(path, dtype="<i2", mode="r", shape=(ROWS, COLS))

    def height(self, lat, lon):
        """Bilinear height (m). lat/lon in degrees (any shape); longitude east, wrapped."""
        lat = np.asarray(lat, float)
        r = np.clip((90.0 - lat) * PPD - 0.5, 0.0, ROWS - 1.001)
        c = (np.asarray(lon, float) % 360.0) * PPD - 0.5
        r0 = np.floor(r).astype(int)
        c0 = np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        c0 %= COLS
        c1 = (c0 + 1) % COLS
        g = lambda rr, cc: self.dn[rr, cc].astype(np.float64)  # noqa: E731
        h = (g(r0, c0) * (1 - fr) * (1 - fc) + g(r0, c1) * (1 - fr) * fc
             + g(r0 + 1, c0) * fr * (1 - fc) + g(r0 + 1, c1) * fr * fc)
        return h * SCALE_M

    def _lat_lon_at(self, lat, lon, direction, distance_m):
        _, _, up = enu_basis(lat, lon)
        d = np.asarray(distance_m, float) / MOON_RADIUS_M
        q = up[None, :] * np.cos(d)[:, None] + direction[None, :] * np.sin(d)[:, None]
        return np.degrees(np.arcsin(np.clip(q[:, 2], -1, 1))), np.degrees(np.arctan2(q[:, 1], q[:, 0]))

    def surface(self, lat, lon):
        """Height, slope and downhill azimuth, with the same gradient maths as the polar grid."""
        east, north, _ = enu_basis(lat, lon)
        b = SLOPE_BASELINE_M
        out = {}
        for name, v in (("e", east), ("n", north)):
            la, lo = self._lat_lon_at(lat, lon, v, np.array([b, -b]))
            hp, hm = self.height(la, lo)
            out[name] = (hp - hm) / (2 * b)
        g_e, g_n = float(out["e"]), float(out["n"])
        return {
            "height_m": float(self.height(lat, lon)),
            "slope_deg": float(np.degrees(np.arctan(np.hypot(g_e, g_n)))),
            "downhill_azimuth_deg": float(np.degrees(np.arctan2(-g_e, -g_n)) % 360.0),
            "grad_east": g_e, "grad_north": g_n, "baseline_m": SLOPE_BASELINE_M,
        }

    def horizon(self, lat, lon, observer_height_m=2.0, n_az=720, max_range_m=100_000.0):
        """Skyline elevation for every azimuth, from exact 3D geometry on the sphere (like the polar version)."""
        east, north, up = enu_basis(lat, lon)
        r_site = MOON_RADIUS_M + float(self.height(lat, lon)) + observer_height_m
        az = np.linspace(0.0, 360.0, n_az, endpoint=False)
        azr = np.radians(az)
        tang = north[None, :] * np.cos(azr)[:, None] + east[None, :] * np.sin(azr)[:, None]
        ranges = np.concatenate([np.arange(300.0, 5000.0, 300.0), 5000.0 * 1.04 ** np.arange(int(np.log(max_range_m / 5000.0) / np.log(1.04)) + 1)])
        delta = ranges / MOON_RADIUS_M
        q = up[None, None, :] * np.cos(delta)[None, :, None] + tang[:, None, :] * np.sin(delta)[None, :, None]
        la = np.degrees(np.arcsin(np.clip(q[..., 2], -1, 1)))
        lo = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
        h = self.height(la, lo)
        d = (MOON_RADIUS_M + h)[..., None] * q - r_site * up
        sin_el = (d @ up) / np.linalg.norm(d, axis=-1)
        el = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
        idx = el.argmax(axis=1)
        best = el[np.arange(n_az), idx]
        dip = -np.degrees(np.arccos((r_site - observer_height_m) / r_site))
        dist = np.where(best > dip, ranges[idx], np.nan)
        return HorizonProfile(az, np.maximum(best, dip), dist)


_cache: dict = {}


def load_global_dem(path: Path):
    """The global grid, or None when the file has not been downloaded (callers then fall back to a smooth sphere)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size != ROWS * COLS * 2:
        return None
    if path not in _cache:
        _cache[path] = GlobalDEM(path)
    return _cache[path]
