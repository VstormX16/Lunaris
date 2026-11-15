"""Selenographic geometry in the Moon Mean-Earth (ME) body-fixed frame.

Frame: +X toward (0N, 0E), +Z toward the north pole. Latitude is
planetocentric, longitude positive east. Azimuth is measured clockwise
from local north.
"""
import numpy as np

from .constants import MOON_RADIUS_M


def latlon_to_unit(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon)],
        axis=-1,
    )


def enu_basis(lat_deg, lon_deg):
    """Return (east, north, up) unit vectors at a site, each shape (3,)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


def enu_basis_many(lat_deg, lon_deg):
    """(east, north, up) for N sites, each (N, 3); same formulas as enu_basis."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    z = np.zeros_like(lon)
    east = np.stack([-np.sin(lon), np.cos(lon), z], axis=-1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    up = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    return east, north, up


def vector_to_azel(vec, lat_deg, lon_deg):
    """Topocentric azimuth/elevation (deg) of direction vectors (..., 3)."""
    east, north, up = enu_basis(lat_deg, lon_deg)
    vec = np.asarray(vec, dtype=float)
    e, n, u = vec @ east, vec @ north, vec @ up
    az = np.degrees(np.arctan2(e, n)) % 360.0
    el = np.degrees(np.arctan2(u, np.hypot(e, n)))
    return az, el


def polar_stereo_forward(lat_deg, lon_deg, radius_m=MOON_RADIUS_M):
    """South-polar stereographic (true scale at the pole) -> (x, y) meters.

    x = rho*sin(lon), y = rho*cos(lon), rho = 2R*tan(colat/2).
    """
    colat = np.radians(90.0 + np.asarray(lat_deg, dtype=float))
    lon = np.radians(lon_deg)
    rho = 2.0 * radius_m * np.tan(colat / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def polar_stereo_inverse(x, y, radius_m=MOON_RADIUS_M):
    """(x, y) meters -> (lat_deg, lon_deg east)."""
    rho = np.hypot(x, y)
    colat = 2.0 * np.arctan(rho / (2.0 * radius_m))
    return np.degrees(colat) - 90.0, np.degrees(np.arctan2(x, y)) % 360.0


def unit_to_polar_xy(q, radius_m=MOON_RADIUS_M):
    """Same projection applied directly to unit vectors (..., 3); cheaper than lat/lon."""
    k = 2.0 * radius_m / (1.0 - q[..., 2])
    return q[..., 1] * k, q[..., 0] * k
