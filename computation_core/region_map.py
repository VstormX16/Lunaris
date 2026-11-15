"""Whole-region maps: ensemble surrogate everywhere, exact physics where the ensemble is unsure.

Decision layer: pixels are ranked by ensemble disagreement; within `refine_budget_s` the most
uncertain ones are recomputed with the exact engine (same code path as /analyze). The
returned `source` grid records which engine produced each pixel (0 = network, 1 = exact).
"""
import contextlib
import time

import numpy as np

from data_engine import spice_loader

from .coordinates import polar_stereo_forward, polar_stereo_inverse
from .features import FEATURE_NAMES, STEP_MIN, WINDOW_DAYS, compute_features
from .scratch_nn.landing_scorer import WINDOW_HOURS, Ensemble, suitability_index
from .site_analysis import exact_targets


def region_map(dem, model: Ensemble, lat_deg, lon_deg, start_utc, half_size_km=20.0, n=64, refine_budget_s=0.0, spice_lock=None):
    """n x n grid centered on a site (row 0 = grid +y). Returns per-pixel grids.

    `spice_lock`: held only around the SPICE calls, so the long NumPy part does not block other requests.
    """
    lock = spice_lock if spice_lock is not None else contextlib.nullcontext()
    cx, cy = polar_stereo_forward(lat_deg, lon_deg)
    offs = np.linspace(-half_size_km * 1000.0, half_size_km * 1000.0, n)
    gx, gy = np.meshgrid(cx + offs, cy - offs)
    x, y = gx.ravel(), gy.ravel()
    lat, lon = polar_stereo_inverse(x, y)
    h = dem.height_nearest(x, y).astype(np.float64)
    ok = ~np.isnan(h)
    lat_ok, lon_ok = lat[ok], lon[ok]

    with lock:
        et = spice_loader.time_grid(start_utc, WINDOW_DAYS, STEP_MIN)
        sun, earth = spice_loader.body_vectors_me("SUN", et), spice_loader.body_vectors_me("EARTH", et)
    X = compute_features(dem, x[ok], y[ok], sun, earth)
    pred, std = model.predict(X, return_std=True)
    unc = model.uncertainty(X, std)
    source = np.zeros(len(X))

    if refine_budget_s > 0:
        t0 = time.perf_counter()
        for i in np.argsort(-unc):
            if time.perf_counter() - t0 >= refine_budget_s:
                break
            with lock:
                pred[i] = exact_targets(dem, float(lat_ok[i]), float(lon_ok[i]), et, sun, earth)
            source[i], unc[i] = 1.0, 0.0

    slope = X[:, FEATURE_NAMES.index("slope_deg")]

    def grid(values):
        out = np.full(n * n, np.nan)
        out[ok] = values
        return out.reshape(n, n)

    return {
        "lat": lat.reshape(n, n), "lon": lon.reshape(n, n), "height_m": h.reshape(n, n),
        "sunlit_fraction": grid(pred[:, 0]), "dte_fraction": grid(pred[:, 1]),
        "longest_darkness_hours": grid(pred[:, 2] * WINDOW_HOURS),
        "longest_outage_hours": grid(pred[:, 3] * WINDOW_HOURS),
        "slope_deg": grid(slope),
        "suitability_index": grid(suitability_index(pred, slope)),
        "uncertainty": grid(unc),
        "source": grid(source),
    }
