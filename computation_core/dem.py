"""LOLA polar DEM container (PDS3 IMG, int16, south-polar stereographic)."""
from pathlib import Path

import numpy as np

from .constants import MOON_RADIUS_M
from .coordinates import polar_stereo_forward


class LolaDEM:
    """Height (m, relative to the 1737.4 km reference sphere) = DN * scale."""

    def __init__(self, dn, pixel_m, scale=0.5):
        self.dn = dn
        self.n = dn.shape[0]
        self.pixel_m = float(pixel_m)
        self.scale = scale
        self.c0 = (self.n - 1) / 2.0  # pixel-registered grid, pole between pixels

    @classmethod
    def from_img(cls, path, size, pixel_m, scale=0.5, mmap=False):
        """mmap=True shares one OS page cache between worker processes instead of copying 115 MB each."""
        if mmap:
            dn = np.memmap(Path(path), dtype="<i2", mode="r", shape=(size, size))
        else:
            dn = np.fromfile(Path(path), dtype="<i2").reshape(size, size)
        return cls(dn, pixel_m, scale)

    def xy_to_rc(self, x, y):
        return self.c0 - y / self.pixel_m, self.c0 + x / self.pixel_m

    def height_nearest(self, x, y):
        """Heights (m) at projected coordinates; NaN outside the grid."""
        r, c = self.xy_to_rc(x, y)
        ri, ci = np.rint(r).astype(np.int64), np.rint(c).astype(np.int64)
        ok = (ri >= 0) & (ri < self.n) & (ci >= 0) & (ci < self.n)
        out = np.full(np.shape(x), np.nan, dtype=np.float32)
        out[ok] = self.dn[ri[ok], ci[ok]] * self.scale
        return out

    def height_at_latlon(self, lat_deg, lon_deg):
        x, y = polar_stereo_forward(lat_deg, lon_deg, MOON_RADIUS_M)
        return self.height_nearest(np.asarray(x), np.asarray(y))

    def patch(self, lat_deg, lon_deg, half_size_m):
        """Float32 height window (m) around a site; returns (array, pixel_m)."""
        x, y = polar_stereo_forward(lat_deg, lon_deg, MOON_RADIUS_M)
        r, c = self.xy_to_rc(x, y)
        h = int(half_size_m / self.pixel_m)
        r0, c0 = int(round(r)) - h, int(round(c)) - h
        if r0 < 0 or c0 < 0 or r0 + 2 * h + 1 > self.n or c0 + 2 * h + 1 > self.n:
            raise ValueError("patch extends outside the DEM")
        win = self.dn[r0 : r0 + 2 * h + 1, c0 : c0 + 2 * h + 1]
        return win.astype(np.float32) * self.scale, self.pixel_m
