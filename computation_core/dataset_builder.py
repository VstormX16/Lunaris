"""Training/test data for the surrogate, generated in parallel from the exact physics engine.

Each sample = (real LOLA site, real 30-day SPICE window):
  X = features.compute_features (the same function the maps use)
  y = site_analysis.exact_targets (the same code path as /analyze)

Splits are disjoint in space AND time:
  train: longitudes 60-360 deg, seed 0          -> data/cache/dataset_train.npz
  test:  longitudes  0-60  deg, other seed/dates -> data/cache/dataset_test.npz

Run:  python -m computation_core.dataset_builder --split train
      python -m computation_core.dataset_builder --split test
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from data_engine import config, spice_loader
from data_engine.lola_downloader import ensure_lola

from .coordinates import polar_stereo_inverse
from .dem import LolaDEM
from .features import FEATURE_NAMES, STEP_MIN, TARGET_NAMES, WINDOW_DAYS, compute_features
from .site_analysis import exact_targets

DATE_RANGE = ("2026-01-01T00:00:00", "2030-12-31T00:00:00")
SITE_RADIUS_M = 140_000.0  # inside the 85S circle (~151.7 km)
TEST_SECTOR = (0.0, 60.0)  # longitudes reserved for the final test set (contains Malapert, Leibnitz)
SPLITS = {
    "train": dict(windows=250, sites=160, seed=0, sectors=[(60.0, 360.0)]),
    "test": dict(windows=60, sites=100, seed=987_654, sectors=[TEST_SECTOR]),
}

_W = {}  # per-worker state


def random_sites(rng, n, sectors):
    """Uniform-by-area sites whose longitude (= grid position angle) lies in the given sectors."""
    widths = np.array([b - a for a, b in sectors])
    u = rng.random(n) * widths.sum()
    edges = np.concatenate([[0.0], np.cumsum(widths)])
    k = np.searchsorted(edges, u, side="right") - 1
    ang = np.radians(np.array([sectors[i][0] for i in k]) + (u - edges[k]))
    r = SITE_RADIUS_M * np.sqrt(rng.random(n))
    return r * np.sin(ang), r * np.cos(ang)


def _init_worker():
    p = config.LOLA_PRODUCT
    _W["dem"] = LolaDEM.from_img(ensure_lola(p), p["size"], p["pixel_m"], mmap=True)
    spice_loader.load_kernels()


def _window(task):
    w, et0, x, y = task
    dem = _W["dem"]
    ok = ~np.isnan(dem.height_nearest(x, y))
    x, y = x[ok], y[ok]
    et = et0 + np.arange(0.0, WINDOW_DAYS * 86400.0, STEP_MIN * 60.0)
    sun, earth = spice_loader.body_vectors_me("SUN", et), spice_loader.body_vectors_me("EARTH", et)
    X = compute_features(dem, x, y, sun, earth)
    lat, lon = polar_stereo_inverse(x, y)
    Y = np.array([exact_targets(dem, float(a), float(b), et, sun, earth) for a, b in zip(lat, lon)])
    meta = np.column_stack([x, y, lat, lon, np.full(len(x), et0)])
    return w, X, Y, meta


def build(split, windows=None, sites=None, workers=None, out=None):
    cfg = dict(SPLITS[split])
    cfg["windows"] = windows or cfg["windows"]
    cfg["sites"] = sites or cfg["sites"]
    rng = np.random.default_rng(cfg["seed"])
    et_lo, et_hi = (spice_loader.utc_to_et(s) for s in DATE_RANGE)
    tasks = [(w, et_lo + rng.random() * (et_hi - et_lo), *random_sites(rng, cfg["sites"], cfg["sectors"]))
             for w in range(cfg["windows"])]  # all randomness drawn here -> deterministic output

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # one BLAS thread per worker; parallelism comes from processes
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    parts, t0 = {}, time.perf_counter()
    with Pool(workers, initializer=_init_worker) as pool:
        for w, X, Y, meta in pool.imap_unordered(_window, tasks):
            parts[w] = (X, Y, meta)
            if len(parts) % 10 == 0 or len(parts) == len(tasks):
                print(f"{split}: {len(parts)}/{len(tasks)} windows  {time.perf_counter() - t0:.0f}s", flush=True)

    X, Y, meta = (np.concatenate([parts[w][i] for w in sorted(parts)]) for i in range(3))
    out = out or config.CACHE_DIR / f"dataset_{split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X.astype(np.float32), Y=Y.astype(np.float32), meta=meta,
                        feature_names=np.array(FEATURE_NAMES), target_names=np.array(TARGET_NAMES))
    return out, X, Y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--windows", type=int)
    ap.add_argument("--sites", type=int)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    path, X, Y = build(a.split, a.windows, a.sites, a.workers)
    print(f"saved {path}  X={X.shape}  Y={Y.shape}")
