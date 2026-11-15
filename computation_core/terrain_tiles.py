"""Height tiles for the 3D Moon viewer, cut from the project's own NASA LOLA LDEM_64 grid (no extra data, no synthetic values).

A tile is 91 x 91 vertices; five planes: the vertex itself plus the heights 1/64 deg north, south, east and west of it
(the viewer builds exact surface normals from them). Heights go out as uint16 of (height_m + 10000) * 2.
"""
import numpy as np

from .global_dem import PPD, GlobalDEM

N = 90
MAX_Z = 7
STEP_DEG = 1.0 / PPD


def _neighbours(lat, lon, step_deg=STEP_DEG):
    """Coordinates one step north/south/east/west along great circles; finite at the poles too."""
    phi, lam = np.radians(lat), np.radians(lon)
    cp, sp, cl, sl = np.cos(phi), np.sin(phi), np.cos(lam), np.sin(lam)
    p = np.stack([cp * cl, cp * sl, sp])                       # unit vector (x toward lon 0, z north)
    east = np.stack([-sl, cl, np.zeros_like(sl)])
    north = np.stack([-sp * cl, -sp * sl, cp])
    polar = np.abs(sp) >= 0.9995
    if polar.any():  # at the pole "east" is undefined: pick a fixed pair of tangent directions
        sign = np.where(sp >= 0, 1.0, -1.0)
        east = np.where(polar, np.stack([np.zeros_like(sl), np.ones_like(sl), np.zeros_like(sl)]), east)
        north = np.where(polar, np.stack([-sign, np.zeros_like(sl), np.zeros_like(sl)]), north)
    a = np.radians(step_deg)

    def to_ll(v):
        return np.degrees(np.arcsin(np.clip(v[2], -1, 1))), np.degrees(np.arctan2(v[1], v[0]))

    return [to_ll(np.cos(a) * p + np.sin(a) * d) for d in (north, -north, east, -east)]


def tile_bytes(dem: GlobalDEM, z: int, x: int, y: int) -> bytes:
    if not (0 <= z <= MAX_Z and 0 <= x < 2 ** (z + 1) and 0 <= y < 2 ** z):
        raise ValueError("invalid tile")
    size = 180.0 / 2 ** z
    lons = np.linspace(-180 + x * size, -180 + (x + 1) * size, N + 1)
    lats = np.linspace(90 - y * size, 90 - (y + 1) * size, N + 1)
    la, lo = lats[:, None] * np.ones((1, N + 1)), np.ones((N + 1, 1)) * lons[None, :]
    centre = dem.height(la, lo)
    if y == 0:
        centre[0, :] = float(dem.height(90.0, 0.0))        # one value on the pole row, so the tile has no seam there
    if y == 2 ** z - 1:
        centre[-1, :] = float(dem.height(-90.0, 0.0))
    planes = [centre] + [dem.height(a, b) for a, b in _neighbours(la, lo)]
    return np.round((np.stack(planes) + 10000.0) * 2).astype("<u2").tobytes()
