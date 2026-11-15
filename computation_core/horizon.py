"""Terrain horizon profile (max obstruction elevation vs azimuth) from a LOLA DEM."""
from dataclasses import dataclass

import numpy as np

from .constants import MOON_RADIUS_M
from .coordinates import enu_basis, unit_to_polar_xy
from .dem import LolaDEM


@dataclass
class HorizonProfile:
    az_deg: np.ndarray  # (A,) ascending, [0, 360)
    el_deg: np.ndarray  # (A,) terrain elevation angle
    dist_m: np.ndarray | None = None  # (A,) distance to the highest-angle terrain point, NaN if none

    def blocker_distance_m(self, az):
        """Distance (m) to the terrain that defines the horizon at the nearest sampled azimuth."""
        if self.dist_m is None:
            return np.nan
        i = np.rint((np.asarray(az) % 360.0) / (360.0 / self.az_deg.size)).astype(int) % self.az_deg.size
        return self.dist_m[i]

    def at(self, az):
        """Horizon elevation (deg) at arbitrary azimuth(s), periodic linear interpolation."""
        return np.interp(np.asarray(az) % 360.0, self.az_deg, self.el_deg, period=360.0)


def _ranges(pixel_m, max_range_m, near_m=4000.0, growth=1.01):
    """Pixel-spaced steps near the site, geometric growth farther out."""
    near = np.arange(pixel_m, near_m, pixel_m)
    n_far = int(np.ceil(np.log(max_range_m / near_m) / np.log(growth)))
    far = near_m * growth ** np.arange(n_far + 1)
    return np.concatenate([near, far])


def compute_horizon(dem: LolaDEM, lat_deg, lon_deg, observer_height_m=0.0,
                    n_az=720, max_range_m=100_000.0):
    """Vectorized horizon profile: all azimuths x all range steps in one NumPy pass.

    Rays follow great circles on the sphere; each sample is projected into the
    polar-stereographic grid and compared via exact 3D elevation geometry
    (includes lunar curvature). Samples outside the DEM are ignored.
    """
    g = getattr(dem, "global_dem", None)  # a SiteDEM: outside the detailed polar grid the whole-Moon grid takes over
    if g is not None and not dem.covers(lat_deg, lon_deg):
        return g.horizon(lat_deg, lon_deg, observer_height_m, n_az, max_range_m)
    east, north, up = enu_basis(lat_deg, lon_deg)
    site_h = float(dem.height_at_latlon(lat_deg, lon_deg))
    if np.isnan(site_h):
        raise ValueError("site outside DEM coverage")
    r_site = MOON_RADIUS_M + site_h + observer_height_m
    az = np.linspace(0.0, 360.0, n_az, endpoint=False)
    azr = np.radians(az)
    tang = north[None, :] * np.cos(azr)[:, None] + east[None, :] * np.sin(azr)[:, None]  # (A,3)
    ranges = _ranges(dem.pixel_m, max_range_m)
    delta = ranges / MOON_RADIUS_M  # (S,)
    q = up[None, None, :] * np.cos(delta)[None, :, None] + tang[:, None, :] * np.sin(delta)[None, :, None]
    x, y = unit_to_polar_xy(q)
    h = dem.height_nearest(x, y).astype(np.float64)  # (A,S), NaN outside
    d = (MOON_RADIUS_M + h)[..., None] * q - r_site * up  # terrain point relative to observer
    sin_el = (d @ up) / np.linalg.norm(d, axis=-1)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    el = np.where(np.isnan(h), -np.inf, el)
    idx = el.argmax(axis=1)
    best = el[np.arange(n_az), idx]
    dip = -np.degrees(np.arccos((r_site - observer_height_m) / r_site))  # geometric horizon if no terrain seen
    dist = np.where(np.isfinite(best) & (best > dip), ranges[idx], np.nan)
    return HorizonProfile(az, np.where(np.isfinite(best), np.maximum(best, dip), dip), dist)
