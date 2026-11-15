"""A square patch of real terrain around one point, as seen by someone standing there (the 360 view).

Heights come from the project's own NASA LOLA data: the 40 m south-pole grid inside 85 S, the whole-Moon LDEM_64 grid
(about 470 m per pixel) elsewhere. Returned as heights above the observer's local horizontal plane, so the Moon's curvature
is already in them (about 65 m of drop at 15 km). Nothing is invented: where the data is coarse the patch is smooth.
"""
import numpy as np

from . import surface_detail as sd
from .constants import MOON_RADIUS_M
from .coordinates import enu_basis, polar_stereo_forward

POLAR_LIMIT_DEG = -85.0


def _polar_height(dem, lat, lon):
    x, y = polar_stereo_forward(lat, lon % 360.0, MOON_RADIUS_M)
    return dem.height_nearest(np.asarray(x), np.asarray(y))


def heights(dem, gdem, lat, lon):
    """Real LOLA height (m) at points (any shape): polar grid where it covers, whole-Moon grid elsewhere; NaN with neither."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    out = np.full(lat.shape, np.nan)
    if gdem is not None:
        out = gdem.height(lat, lon)
    polar = lat <= POLAR_LIMIT_DEG
    if polar.any():
        hp = _polar_height(dem, lat[polar], lon[polar])
        out[polar] = np.where(np.isnan(hp), out[polar], hp)
    return out


def resolution_note(dem, gdem, lat, lon) -> str:
    if lat <= POLAR_LIMIT_DEG and not np.isnan(float(_polar_height(dem, np.array([lat]), np.array([lon]))[0])):
        return "lola_40m"
    return "ldem_470m" if gdem is not None else "none"


def point_height(dem, gdem, lat, lon) -> float:
    return float(heights(dem, gdem, np.array([lat]), np.array([lon]))[0])


def terrain_patch(dem, gdem, lat, lon, radius_m=15000.0, n=769):
    """(n, n) float32 heights (m) above the local plane through the point at its own ground height.

    Row 0 is the north edge, column 0 the west edge; the point is the centre cell. Spacing is 2 * radius / (n - 1).
    """
    east, north, up = enu_basis(lat, lon)
    e = np.linspace(-radius_m, radius_m, n)
    nn = np.linspace(radius_m, -radius_m, n)
    E, N = np.meshgrid(e, nn)
    dist = np.hypot(E, N)
    ang = dist / MOON_RADIUS_M
    safe = np.where(dist > 0, dist, 1.0)
    horiz = (E / safe)[..., None] * east + (N / safe)[..., None] * north
    q = up * np.cos(ang)[..., None] + horiz * np.sin(ang)[..., None]
    la = np.degrees(np.arcsin(np.clip(q[..., 2], -1, 1)))
    lo = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
    h = heights(dem, gdem, la, lo)
    h0 = float(h[n // 2, n // 2])
    if np.isnan(h0):
        raise ValueError("no elevation data here")
    h = np.where(np.isnan(h), h0, h)  # a hole in the data (outside the grid edge): flat at the observer's height
    z = (MOON_RADIUS_M + h) * np.cos(ang) - (MOON_RADIUS_M + h0)
    return z.astype(np.float32)


GLOBAL_PIXEL_M = MOON_RADIUS_M * np.pi / 180.0 / 64.0     # LDEM_64 north-south pixel, ~474 m


def _enu_to_height(dem, gdem, lat, lon, e, n):
    """Real height at plane offsets (e east, n north, m) from the observer, and the arc angle of each point."""
    east, north, up = enu_basis(lat, lon)
    dist = np.hypot(e, n)
    ang = dist / MOON_RADIUS_M
    safe = np.where(dist > 0, dist, 1.0)
    horiz = (e / safe)[..., None] * east + (n / safe)[..., None] * north
    q = up * np.cos(ang)[..., None] + horiz * np.sin(ang)[..., None]
    la = np.degrees(np.arcsin(np.clip(q[..., 2], -1, 1)))
    lo = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
    return heights(dem, gdem, la, lo), ang


def _bicubic(g, r, c):
    """Catmull-Rom interpolation of grid g at fractional (row, col) arrays; edges repeat."""
    gp = np.pad(g, 2, mode="edge")
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    fr, fc = r - r0, c - c0

    def w(t):
        return np.stack([-0.5 * t ** 3 + t ** 2 - 0.5 * t, 1.5 * t ** 3 - 2.5 * t ** 2 + 1, -1.5 * t ** 3 + 2 * t ** 2 + 0.5 * t, 0.5 * t ** 3 - 0.5 * t ** 2])

    wr, wc = w(fr), w(fc)
    out = np.zeros(r.shape)
    for i in range(4):
        for j in range(4):
            out += wr[i] * wc[j] * gp[np.clip(r0 + i + 1, 0, gp.shape[0] - 1), np.clip(c0 + j + 1, 0, gp.shape[1] - 1)]
    return out


def _real_window(dem, gdem, lat, lon, note):
    """A window of the real heights around the point at the data's own pixel size: (array m, pixel m, both axes usable)."""
    if note == "lola_40m":
        try:
            win, pixel = dem.patch(lat, lon, 4000.0)
            return win.astype(float), float(pixel), True
        except ValueError:
            pass
    from .global_dem import COLS, PPD, ROWS  # whole-Moon grid: rows run north to south, one pixel is ~474 m
    r0 = int(round((90.0 - lat) * PPD))
    rows = np.clip(np.arange(r0 - 9, r0 + 10), 0, ROWS - 1)
    cols = (int(round((lon % 360.0) * PPD)) + np.arange(-9, 10)) % COLS
    return gdem.dn[np.ix_(rows, cols)].astype(float) * 0.5, GLOBAL_PIXEL_M, False


def _with_craters(h, e_axis, n_axis, craters, weight=None):
    """add_craters wants ascending axes; our rows run north (high n) to south, so flip around the call."""
    if len(craters) == 0:
        return h
    flipped = h[::-1].copy()
    sd.add_craters(flipped, e_axis, n_axis[::-1], craters, None if weight is None else weight[::-1])
    return flipped[::-1]


def terrain_tiers(dem, gdem, lat, lon, far_radius=15000.0, far_n=769, near_radius=1000.0, near_n=801, detail=True):
    """(far with detail, far real-only, near with detail): float32 heights above the observer's plane, north row first.

    The real-only far grid is what skyline and visibility numbers must use; the detail layers (surface_detail) are for
    the picture. The near grid is the 1 km around the observer at ~2.5 m. Both detailed grids share one definition, so they
    agree where they meet (the near grid's finest layers fade out toward its edge).
    """
    note = resolution_note(dem, gdem, lat, lon)
    if note == "none":
        raise ValueError("no elevation data here")
    b = float(dem.pixel_m) if note == "lola_40m" else GLOBAL_PIXEL_M
    half = far_radius + 3 * b
    nb = int(2 * half / b) + 1
    half = (nb - 1) * b / 2
    axis = np.linspace(-half, half, nb)
    be, bn = np.meshgrid(axis, axis[::-1])
    base, _ = _enu_to_height(dem, gdem, lat, lon, be, bn)
    base = np.where(np.isnan(base), np.nanmean(base), base)

    def axes(radius, n):
        return np.linspace(-radius, radius, n), np.linspace(radius, -radius, n)

    def base_at(E, N):
        return _bicubic(base, (half - N) / b, (E + half) / b)

    ef, nf = axes(far_radius, far_n)
    en, nn = axes(near_radius, near_n)
    EF, NF = np.meshgrid(ef, nf)
    EN, NN = np.meshgrid(en, nn)
    hf, hn = base_at(EF, NF), base_at(EN, NN)
    angf, angn = np.hypot(EF, NF) / MOON_RADIUS_M, np.hypot(EN, NN) / MOON_RADIUS_M
    h_base_center = float(base_at(np.zeros(1), np.zeros(1))[0])
    far_base = (MOON_RADIUS_M + hf) * np.cos(angf) - (MOON_RADIUS_M + h_base_center)

    if not detail:
        near_plain = (MOON_RADIUS_M + hn) * np.cos(angn) - (MOON_RADIUS_M + h_base_center)
        return far_base.astype(np.float32), far_base.astype(np.float32), near_plain.astype(np.float32)

    seed = sd.seed_for(lat, lon)
    window, pixel, both = _real_window(dem, gdem, lat, lon, note)
    hurst, v = sd.roughness(window, pixel, both)
    lam0, lam_min = 2.0 * b, 5.0
    sigma0 = sd.calibrate(v, b, hurst, lam0, lam_min, seed)
    sigma0 = sd.cap_by_slope(sigma0, lam0, hurst, lam_min, seed)
    d_split = 80.0
    d_max = max(2.0 * b, d_split)
    big = sd.crater_list(far_radius, d_split, d_max, seed + 1, skip_radius_m=25.0)
    small = sd.crater_list(near_radius, sd.MIN_CRATER_M, min(d_split, 2.0 * b), seed + 2, skip_radius_m=25.0)

    def coarse(E, N):
        return sd.fractal(E, N, sigma0, lam0, hurst, sd.COARSE_CUT_M, lam0, seed)

    far_h = _with_craters(hf + coarse(EF, NF), ef, nf, big)
    fade = np.clip((near_radius - np.maximum(np.abs(EN), np.abs(NN))) / (0.2 * near_radius), 0.0, 1.0)
    fine = sd.fractal(EN, NN, sigma0, lam0, hurst, lam_min, sd.COARSE_CUT_M - 1e-6, seed)
    near_h = _with_craters(hn + coarse(EN, NN) + fade * fine, en, nn, big)
    near_h = _with_craters(near_h, en, nn, small, weight=fade)

    centre = near_h[near_n // 2, near_n // 2]
    far_z = (MOON_RADIUS_M + far_h) * np.cos(angf) - (MOON_RADIUS_M + centre)
    near_z = (MOON_RADIUS_M + near_h) * np.cos(angn) - (MOON_RADIUS_M + centre)
    return far_z.astype(np.float32), far_base.astype(np.float32), near_z.astype(np.float32)
