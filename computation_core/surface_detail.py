"""Representative fine relief below the resolution of the real elevation data (the 360 view).

The real heights stop at 40 m (south pole) or ~470 m (elsewhere). Under that, nobody has measured the ground, so this module
adds relief that is *statistically like the Moon* and *scaled to how rough the real data around the point is*. It is a
looks-only layer: it is never used for slope, skyline or any number the app reports.

 * fractal roughness: octave value noise whose height grows with wavelength as a power law, exponent H estimated from the
   real data itself (the Moon measures H ~ 0.76 on maria to ~ 0.95 on highlands); amplitude matched to the real data's
   second differences and extrapolated downward with that exponent;
 * small craters: power-law size distribution with cumulative density ~ D^-2 at a few percent of geometric saturation
   (the equilibrium the Moon shows), simple bowl + rim profile, random degradation.

Everything is deterministic for a place (seeded from its coordinates), so the same point always looks the same.
"""
import numpy as np

H_MIN, H_MAX, H_DEFAULT = 0.70, 0.95, 0.85
GEOMETRIC_SATURATION_KM = 1.54      # N(>D) = 1.54 * D^-2 per km^2, D in km
EQUILIBRIUM_FRACTION = 0.03         # "a few percent" of geometric saturation
MIN_CRATER_M = 6.0
MAX_SLOPE_AT_FINEST = 0.5           # tan of ~27 deg: no detail rougher than this at the finest scale
COARSE_CUT_M = 160.0                # octaves longer than this are shared by the far and near meshes


# ---------------------------------------------------------------- value noise -------------------------------------
def _hash01(ix, iy, seed):
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(np.float32) / 32767.5 - 1.0


def value_noise(x, y, seed):
    """Smooth noise in [-1, 1] at lattice coordinates x, y (arrays)."""
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = (x - ix).astype(np.float32), (y - iy).astype(np.float32)
    u, v = fx * fx * fx * (fx * (fx * 6 - 15) + 10), fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a, b = _hash01(ix, iy, seed), _hash01(ix + 1, iy, seed)
    c, d = _hash01(ix, iy + 1, seed), _hash01(ix + 1, iy + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fractal(e, n, sigma0, lam0, hurst, lam_lo, lam_hi, seed):
    """Sum of octaves with wavelengths in [lam_lo, lam_hi] (metres), amplitude sigma0 * (lam / lam0) ** H. e, n: arrays (m)."""
    out = np.zeros(np.shape(e), np.float32)
    j = 0
    while True:
        lam = lam0 / 2 ** j
        if lam < lam_lo:
            break
        if lam <= lam_hi:
            ang = (seed * 7919 + j * 104729) % 628 / 100.0   # each octave turned by its own angle: no grid-aligned look
            c, s = np.cos(ang), np.sin(ang)
            x = (e * c - n * s) / lam + j * 17.31
            y = (e * s + n * c) / lam + j * 5.77
            out += (sigma0 * (lam / lam0) ** hurst * 1.7) * value_noise(x, y, seed + j)
        j += 1
    return out


def calibrate(v_data, b_data, hurst, lam0, lam_min, seed):
    """sigma0 so the fractal's height increments at b/4 match the real data's roughness extrapolated down with exponent H."""
    h0 = b_data / 4.0
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-6000, 6000, 20000), rng.uniform(-6000, 6000, 20000)
    th = rng.uniform(0, 2 * np.pi, 20000)
    unit = fractal(np.stack([x, x + h0 * np.cos(th)]), np.stack([y, y + h0 * np.sin(th)]), 1.0, lam0, hurst, lam_min, lam0, seed)
    v_syn = float(np.mean((unit[1] - unit[0]) ** 2))
    target = v_data ** 2 * (h0 / b_data) ** (2 * hurst)
    sigma0 = float(np.sqrt(target / max(v_syn, 1e-12)))
    finest = v_data * (lam_min / b_data) ** hurst             # rms height change over the shortest wavelength
    return min(sigma0, sigma0 * MAX_SLOPE_AT_FINEST * lam_min / max(finest, 1e-9))


def cap_by_slope(sigma0, lam0, hurst, lam_min, seed, median_limit_deg=9.0):
    """Never let the invented relief be steeper than the Moon's own metre-scale ground (median slope ~ 3 deg on maria, ~ 10 on highlands)."""
    ax = np.arange(200) * 2.5
    e, n = np.meshgrid(ax, ax)
    f = fractal(e, n, sigma0, lam0, hurst, lam_min, lam0, seed)
    gy, gx = np.gradient(f, 2.5)
    median = float(np.median(np.hypot(gx, gy)))
    limit = float(np.tan(np.radians(median_limit_deg)))
    return sigma0 if median <= limit else sigma0 * limit / median


def roughness(window, lag_m, both_axes):
    """(H, v_data) from the real heights: second differences at one and two pixels. window: 2D array of metres, lag_m: pixel."""
    def d2(a, s, axis):
        a = np.moveaxis(a, axis, 0)
        return (a[2 * s:] + a[: len(a) - 2 * s] - 2 * a[s: len(a) - s]).ravel()
    axes = (0, 1) if both_axes else (0,)
    d1 = np.concatenate([d2(window, 1, ax) for ax in axes])
    d2s = np.concatenate([d2(window, 2, ax) for ax in axes])
    r1, r2 = float(np.sqrt(np.mean(d1 ** 2))), float(np.sqrt(np.mean(d2s ** 2)))
    hurst = H_DEFAULT if r1 < 0.05 or r2 < 0.05 else float(np.clip(np.log2(r2 / r1), H_MIN, H_MAX))
    v = r1 / np.sqrt(4 - 2 ** (2 * hurst))                    # fBm: rms 2nd difference = v * sqrt(4 - 2^(2H)) at lag b
    return hurst, v


# ----------------------------------------------------------------- craters ---------------------------------------
def crater_list(area_half_m, d_lo, d_hi, seed, skip_radius_m=0.0):
    """Craters over a square [-area_half, area_half]^2 with D in [d_lo, d_hi]: columns x, y, D, degradation."""
    if d_hi <= d_lo:
        return np.zeros((0, 4))
    km2 = (2 * area_half_m / 1000.0) ** 2
    per_km2 = EQUILIBRIUM_FRACTION * GEOMETRIC_SATURATION_KM * ((d_lo / 1000.0) ** -2 - (d_hi / 1000.0) ** -2)
    count = int(min(per_km2 * km2, 12000))
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=count)
    inv2 = d_lo ** -2 - u * (d_lo ** -2 - d_hi ** -2)          # inverse CDF of N(>D) ~ D^-2 truncated to [d_lo, d_hi]
    d = inv2 ** -0.5
    x, y = rng.uniform(-area_half_m, area_half_m, (2, count))
    deg = rng.uniform(0.15, 1.0, count) ** 1.6   # most are old and softened; few are fresh
    keep = np.hypot(x, y) > d * 1.5 + skip_radius_m            # nobody stands inside a made-up crater
    return np.stack([x, y, d, deg], axis=1)[keep]


def add_craters(z, e_axis, n_axis, craters, weight=None):
    """Add bowl+rim craters into z (rows follow n_axis, columns e_axis). weight: optional (rows, cols) fade in [0, 1]."""
    de = abs(e_axis[1] - e_axis[0])
    for x, y, d, deg in craters:
        r = d / 2
        reach = 3 * r
        c0, c1 = np.searchsorted(e_axis, [x - reach, x + reach])
        rows = np.where((n_axis >= y - reach) & (n_axis <= y + reach))[0]
        if c1 - c0 < 1 or len(rows) == 0:
            continue
        r0, r1 = rows[0], rows[-1] + 1
        ee, nn = np.meshgrid(e_axis[c0:c1], n_axis[r0:r1])
        q = np.hypot(ee - x, nn - y) / r
        depth, rim = 0.18 * d * deg, 0.035 * d * deg
        h = np.where(q <= 1, (rim + depth) * q ** 2 - depth, rim * np.maximum(q, 1) ** -3)
        h = np.where(q <= 3, h, 0.0)
        if weight is not None:
            h = h * weight[r0:r1, c0:c1]
        z[r0:r1, c0:c1] += h.astype(z.dtype)
    return z


def seed_for(lat, lon):
    return int(abs(round(lat * 1e4) * 73856093 + round(lon * 1e4) * 19349663)) % 2_000_000_011
