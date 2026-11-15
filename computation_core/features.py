"""Surrogate model inputs (v2): one vectorized code path for training AND inference.

Design
- Directions live in the polar-stereographic GRID frame (angle clockwise from grid +y),
  so a site's terrain and the Sun/Earth path share one frame: grid_angle = azimuth + longitude.
- A fine but cheap "proxy horizon" (64 directions x 94 distances, straight grid rays with
  lunar curvature drop; ~0.6 ms/site) is ~65x cheaper than the exact ray-tracer (720 x ~270
  great-circle samples, ~38 ms/site) and within ~0.2 deg of it on average. From it we compute physics proxies of every target (visible fraction, longest
  outage, number of outages) plus their sensitivity to a +/-0.5 deg horizon error. The network
  learns to correct the proxy, which is where its accuracy comes from.
"""
import numpy as np

from .constants import MOON_RADIUS_M
from .coordinates import enu_basis_many, latlon_to_unit, polar_stereo_forward, polar_stereo_inverse
from .timeseries import count_runs_rows, longest_run_rows

WINDOW_DAYS, STEP_MIN = 30, 60  # the surrogate is trained and served on this window only
OBSERVER_HEIGHT_M = 2.0         # same as the exact engine
N_DIR = 64                      # proxy-horizon directions
N_SECTORS = 16                  # pooled sectors exposed to the network
# 40 m, then 80 m x 1.08^k up to the exact engine's 100 km range (94 distances). Chosen by
# experiment: denser rings helped up to ~1.08, more directions (128) did not.
DIST_M = np.concatenate([[40.0], 80.0 * 1.08 ** np.arange(int(np.log(100e3 / 80) / np.log(1.08)) + 1)])
TPI_RADII_M = (1_000.0, 5_000.0, 20_000.0)
SHIFT_DEG = 0.5                 # horizon-error sensitivity probe
NEAR_DEG = 0.5                  # "close to the horizon" band
_DIR_TH = np.radians(np.arange(N_DIR) * 360.0 / N_DIR)
_RING_TH = np.radians(np.arange(16) * 22.5)
_DIP_DEG = -np.degrees(np.arccos(MOON_RADIUS_M / (MOON_RADIUS_M + OBSERVER_HEIGHT_M)))


def _body_names(p):
    return ([f"{p}_secfrac_{k}" for k in range(N_SECTORS)] + [f"{p}_secel_{k}" for k in range(N_SECTORS)]
            + [f"{p}_above0", f"{p}_proxy_frac", f"{p}_proxy_frac_hi", f"{p}_proxy_frac_lo",
               f"{p}_proxy_longest_off", f"{p}_proxy_longest_off_hi", f"{p}_proxy_off_runs_per_day",
               f"{p}_near_horizon_frac", f"{p}_mean_margin"])


TERRAIN_NAMES = (["height_km", "slope_deg"] + [f"tpi_{int(r / 1000)}km" for r in TPI_RADII_M]
                 + [f"hz_sector_{k}" for k in range(N_SECTORS)] + ["hz_mean", "hz_std"])
FEATURE_NAMES = TERRAIN_NAMES + _body_names("sun") + _body_names("earth")
TARGET_NAMES = ["sunlit_fraction", "dte_fraction", "longest_dark_frac", "longest_outage_frac"]


def _terrain(dem, x, y):
    """Terrain features (n, 23) and proxy horizon (n, 64) in degrees."""
    h0 = dem.height_nearest(x, y).astype(np.float64)
    p = dem.pixel_m
    gx = (dem.height_nearest(x + p, y) - dem.height_nearest(x - p, y)) / (2 * p)
    gy = (dem.height_nearest(x, y + p) - dem.height_nearest(x, y - p)) / (2 * p)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    tpis = [h0 - np.nanmean(dem.height_nearest(x[:, None] + r * np.sin(_RING_TH),
                                                y[:, None] + r * np.cos(_RING_TH)), axis=1)
            for r in TPI_RADII_M]

    d = DIST_M[None, None, :]
    ht = dem.height_nearest(x[:, None, None] + d * np.sin(_DIR_TH)[None, :, None],
                            y[:, None, None] + d * np.cos(_DIR_TH)[None, :, None]).astype(np.float64)
    rise = ht - (h0 + OBSERVER_HEIGHT_M)[:, None, None] - d ** 2 / (2 * MOON_RADIUS_M)
    ang = np.degrees(np.arctan2(rise, d))
    hz = np.maximum(np.where(np.isnan(ang), -90.0, ang).max(axis=2), _DIP_DEG)  # as the exact engine

    sectors = hz.reshape(len(x), N_SECTORS, N_DIR // N_SECTORS).max(axis=2)
    feats = np.column_stack([h0 / 1000.0, slope, *tpis, sectors, hz.mean(axis=1), hz.std(axis=1)])
    return np.nan_to_num(feats, nan=0.0), hz


def _azel(body_vec_km, lat, lon):
    """Topocentric az/el (n, T) of one body for n sites at once (sites on the reference sphere,
    exactly like spice_loader.topocentric's default)."""
    east, north, up = enu_basis_many(lat, lon)
    rel = body_vec_km[None, :, :] - (latlon_to_unit(lat, lon) * (MOON_RADIUS_M / 1000.0))[:, None, :]
    e = np.einsum("ntk,nk->nt", rel, east)
    n = np.einsum("ntk,nk->nt", rel, north)
    u = np.einsum("ntk,nk->nt", rel, up)
    return np.degrees(np.arctan2(e, n)) % 360.0, np.degrees(np.arctan2(u, np.hypot(e, n)))


def _horizon_at(hz, grid_angle):
    """Linear interpolation of the proxy horizon (n, 64) at grid angles (n, T)."""
    pos = (grid_angle % 360.0) / (360.0 / N_DIR)
    i0 = np.floor(pos).astype(np.int64) % N_DIR
    w = pos - np.floor(pos)
    return (1 - w) * np.take_along_axis(hz, i0, axis=1) + w * np.take_along_axis(hz, (i0 + 1) % N_DIR, axis=1)


def _body(az, el, lon, hz, earth):
    """Path summary + proxy physics for one body: (n, 41)."""
    T = az.shape[1]
    grid_angle = az + lon[:, None]
    sec = np.floor((grid_angle % 360.0) / (360.0 / N_SECTORS)).astype(np.int64) % N_SECTORS
    frac, mean_el = [], []
    for k in range(N_SECTORS):
        m = sec == k
        cnt = m.sum(axis=1)
        frac.append(cnt / T)
        mean_el.append(np.where(cnt > 0, (el * m).sum(axis=1) / np.maximum(cnt, 1), -10.0))

    h = _horizon_at(hz, grid_angle)
    floor = 0.0 if earth else -np.inf  # DTE also needs Earth above 0 deg (dte_visible semantics)

    def visible(shift):
        hh = np.maximum(h + shift, floor)
        return el > hh if earth else el >= hh  # Sun: lit_fraction >= 0.5  <=>  el >= horizon

    vis, vis_hi, vis_lo = visible(0.0), visible(SHIFT_DEG), visible(-SHIFT_DEG)
    margin = el - np.maximum(h, floor)
    return np.column_stack(frac + mean_el + [
        (el > 0).mean(axis=1), vis.mean(axis=1), vis_hi.mean(axis=1), vis_lo.mean(axis=1),
        longest_run_rows(~vis) / T, longest_run_rows(~vis_hi) / T,
        count_runs_rows(~vis) / (T * STEP_MIN / 1440.0),
        (np.abs(margin) < NEAR_DEG).mean(axis=1), np.clip(margin, -3.0, 3.0).mean(axis=1) / 3.0,
    ])


def compute_features(dem, x, y, sun_vec, earth_vec, chunk=1024):
    """(N,) grid coords (m) + one window's ephemeris (T, 3) km -> (N, len(FEATURE_NAMES))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = []
    for s in range(0, len(x), chunk):  # bounded memory for large maps
        xs, ys = x[s : s + chunk], y[s : s + chunk]
        lat, lon = polar_stereo_inverse(xs, ys, MOON_RADIUS_M)
        tf, hz = _terrain(dem, xs, ys)
        s_az, s_el = _azel(sun_vec, lat, lon)
        e_az, e_el = _azel(earth_vec, lat, lon)
        out.append(np.column_stack([tf, _body(s_az, s_el, lon, hz, False), _body(e_az, e_el, lon, hz, True)]))
    return np.concatenate(out)


def proxy_horizon_deg(dem, lat_deg, lon_deg, az_deg):
    """Proxy horizon of one site at local azimuths (for validating the proxy against the exact engine)."""
    x, y = polar_stereo_forward(np.atleast_1d(lat_deg), np.atleast_1d(lon_deg))
    _, hz = _terrain(dem, np.asarray(x, float), np.asarray(y, float))
    return _horizon_at(hz, (np.asarray(az_deg) + lon_deg)[None, :])[0]
