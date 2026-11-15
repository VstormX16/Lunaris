"""Pictures of the south pole rendered from the real LOLA elevation grid.

Three layers share one geometry: a square image of the 85S-90S region in the DEM's own grid frame
(+y up = longitude 0, +x right), masked to the 85S circle, so a click maps back to (x, y) metres and
from there to latitude/longitude with the polar-stereographic inverse.

  relief     shaded relief (low Sun from the upper left)
  elevation  height above the 1737.4 km reference sphere, colour ramp between the 1st and 99th percentile
  slope      local slope in degrees (80 m baseline), ramp 0-25 deg
"""
import io

import numpy as np
from PIL import Image

from .dem import LolaDEM

LAYERS = ("relief", "elevation", "slope")
STEP = 2  # 40 m -> 80 m per pixel working resolution (3792 px) before resampling to the output size

# (position 0-1, RGB) colour stops
ELEVATION_STOPS = [(0.0, (48, 40, 92)), (0.2, (46, 92, 140)), (0.4, (70, 150, 130)),
                   (0.6, (170, 190, 110)), (0.8, (225, 200, 130)), (1.0, (250, 245, 235))]
SLOPE_STOPS = [(0.0, (30, 44, 62)), (0.2, (60, 120, 150)), (0.4, (240, 205, 85)),
               (0.6, (232, 130, 55)), (1.0, (200, 45, 45))]
SLOPE_MAX_DEG = 25.0


def _ramp(t, stops):
    t = np.clip(t, 0.0, 1.0)
    pos = [p for p, _ in stops]
    return np.stack([np.interp(t, pos, [c[i] for _, c in stops]) for i in range(3)], axis=-1).astype(np.uint8)


def _resize(arr, size):
    return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize((size, size), Image.BILINEAR))


def _gradients(h, px):
    d_row, d_col = np.gradient(h, px)  # rows run toward -y, columns toward +x
    return d_col, -d_row               # east, north


def render_layer(dem: LolaDEM, layer: str, size: int = 2048, fmt: str = "png", azimuth_deg=315.0, altitude_deg=28.0, z_exaggeration=3.0):
    """Returns (image_bytes, meta); fmt is "png" or "webp" (lossy, ~10x smaller). meta has the geometry and legend range."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}")
    h = dem.dn[::STEP, ::STEP].astype(np.float32) * dem.scale
    px = dem.pixel_m * STEP
    east, north = _gradients(h, px)

    yy, xx = np.mgrid[0:size, 0:size]
    inside = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2) <= (size - 1) / 2  # the 85S circle
    meta = {"size_px": int(size), "extent_m": float(dem.n * dem.pixel_m / 2.0), "layer": layer}

    if layer == "relief":
        nrm = np.sqrt((z_exaggeration * east) ** 2 + (z_exaggeration * north) ** 2 + 1.0)
        az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
        light = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
        shade = (-z_exaggeration * east * light[0] - z_exaggeration * north * light[1] + light[2]) / nrm
        g = (40 + 205 * np.clip(_resize(np.clip(shade, 0, 1) ** 0.8, size), 0, 1)).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    elif layer == "elevation":
        small = _resize(h, size)
        lo, hi = np.percentile(small[inside], [1, 99])
        rgb = _ramp((small - lo) / (hi - lo), ELEVATION_STOPS)
        meta["legend"] = {"min": float(lo), "max": float(hi), "unit": "m", "stops": ELEVATION_STOPS}
    else:
        slope = np.degrees(np.arctan(np.hypot(east, north)))
        rgb = _ramp(_resize(slope, size) / SLOPE_MAX_DEG, SLOPE_STOPS)
        meta["legend"] = {"min": 0.0, "max": SLOPE_MAX_DEG, "unit": "deg", "stops": SLOPE_STOPS}

    rgba = np.dstack([rgb, np.where(inside, 255, 0).astype(np.uint8)])
    buf = io.BytesIO()
    img = Image.fromarray(rgba, "RGBA")
    if fmt == "webp":
        img.save(buf, "WEBP", quality=90, method=4)
    else:
        img.save(buf, "PNG", compress_level=3)
    return buf.getvalue(), meta


def hillshade_png(dem: LolaDEM, size: int = 1084):
    """Backwards-compatible shaded relief at the size the Space Time page uses."""
    return render_layer(dem, "relief", size)
