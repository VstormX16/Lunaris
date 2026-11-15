"""Hybrid surrogate (physics proxy + learned correction, as an ensemble) and the
Landing Suitability Index.

Prediction for every target j:
  members m: residual head -> clip(proxy_j + net_m(x)_j * scale_j, 0, 1); direct head -> net_m(x)_j
  mean   = sum_m w_m * pred_m          (w from greedy ensemble selection on validation)
  spread = std_m pred_m                (disagreement between members -> uncertainty)
  if validation showed the network does not beat the physics proxy on target j, proxy_j is used.
Uncertainty: an error-estimator network (trained on out-of-sample validation errors of the chosen
predictor) ranks samples by expected error; it works even where the physics proxy was chosen.

Built by `python -m computation_core.scratch_nn.automl`.
"""
import json

import numpy as np

from data_engine import config

from ..features import FEATURE_NAMES, TARGET_NAMES, WINDOW_DAYS
from .architecture import build_net, count_params

MODEL_PATH = config.CACHE_DIR / "surrogate.npz"
METRICS_PATH = config.CACHE_DIR / "surrogate_metrics.json"
PROXY_COLUMNS = ["sun_proxy_frac", "earth_proxy_frac", "sun_proxy_longest_off", "earth_proxy_longest_off"]
PROXY_IDX = [FEATURE_NAMES.index(c) for c in PROXY_COLUMNS]
WINDOW_HOURS = WINDOW_DAYS * 24.0

# Landing Suitability Index weights (engineering judgment, documented, tunable).
W_SUN, W_DTE, W_DARK, W_SLOPE = 0.35, 0.25, 0.20, 0.20
MAX_SAFE_SLOPE_DEG = 15.0
SLOPE_VETO_CAP = 15.0  # score ceiling where the mean slope reaches the safe limit


def to_target(cfg, out, proxy, res_scale):
    """Network output -> target space for one member."""
    if cfg["head"] == "residual":
        return np.clip(proxy + out * res_scale, 0.0, 1.0)
    return out


class Ensemble:
    def __init__(self, cfgs, nets, weights, mean, std, res_scale, use_nn, unc_scale, err_cfg=None, err_net=None):
        self.cfgs, self.nets = cfgs, nets
        self.err_cfg, self.err_net = err_cfg, err_net
        self.weights = np.asarray(weights, float)
        self.mean, self.std, self.res_scale = mean, std, res_scale
        self.use_nn = np.asarray(use_nn, bool)
        self.unc_scale = unc_scale

    def member_predictions(self, X):
        X = np.asarray(X, np.float64)
        Z, P = (X - self.mean) / self.std, X[:, PROXY_IDX]
        return np.stack([to_target(c, n.predict(Z), P, self.res_scale) for c, n in zip(self.cfgs, self.nets)])

    def predict(self, X, return_std=False):
        X = np.asarray(X, np.float64)
        preds = self.member_predictions(X)
        mu = np.where(self.use_nn, np.tensordot(self.weights, preds, axes=1), X[:, PROXY_IDX])
        if not return_std:
            return mu
        return mu, np.where(self.use_nn, preds.std(axis=0), 0.0)

    def error_inputs(self, X, std):
        return np.column_stack([(np.asarray(X, np.float64) - self.mean) / self.std, np.asarray(std) / self.unc_scale])

    def uncertainty(self, X, std):
        """One number per sample, higher = more likely wrong (used to decide what to recompute exactly)."""
        if self.err_net is not None:
            return self.err_net.predict(self.error_inputs(X, std))[:, 0]
        return (np.asarray(std) / self.unc_scale).mean(axis=1)  # fallback: member disagreement

    @property
    def n_params(self):
        return sum(count_params(n) for n in self.nets)

    def save(self, path=MODEL_PATH):
        arrays = {f"m{j}__{k}": v for j, n in enumerate(self.nets) for k, v in n.state().items()}
        if self.err_net is not None:
            arrays.update({f"err__{k}": v for k, v in self.err_net.state().items()})
        np.savez(path, configs=np.array(json.dumps(self.cfgs)), err_config=np.array(json.dumps(self.err_cfg)),
                 feature_names=np.array(FEATURE_NAMES),
                 weights=self.weights, mean=self.mean, std=self.std, res_scale=self.res_scale,
                 use_nn=self.use_nn, unc_scale=self.unc_scale, **arrays)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = np.load(path)
        if list(d["feature_names"]) != FEATURE_NAMES:
            raise ValueError("model was trained with different features; re-run the automl training")
        cfgs = json.loads(str(d["configs"]))
        nets = []
        for j, cfg in enumerate(cfgs):
            net = build_net(cfg, len(FEATURE_NAMES), len(TARGET_NAMES), np.random.default_rng(0))
            net.load_state({k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(f"m{j}__")})
            nets.append(net)
        err_cfg = json.loads(str(d["err_config"])) if "err_config" in d.files else None
        err_net = None
        if err_cfg is not None:
            err_net = build_net(err_cfg, len(FEATURE_NAMES) + len(TARGET_NAMES), 1, np.random.default_rng(0))
            err_net.load_state({k.split("__", 1)[1]: d[k] for k in d.files if k.startswith("err__")})
        return cls(cfgs, nets, d["weights"], d["mean"], d["std"], d["res_scale"], d["use_nn"], d["unc_scale"],
                   err_cfg, err_net)


def suitability_index(pred, slope_deg):
    """0-100 score from (predicted or exact) physics + local slope."""
    pred = np.atleast_2d(pred)
    slope = np.asarray(slope_deg, float)
    slope_ok = np.clip(1.0 - slope / MAX_SAFE_SLOPE_DEG, 0.0, 1.0)
    score = 100.0 * (W_SUN * pred[:, 0] + W_DTE * pred[:, 1] + W_DARK * (1.0 - pred[:, 2]) + W_SLOPE * slope_ok)
    # veto: ground steeper than a lander can stand on cannot be rescued by sunlight and a good Earth view
    return np.where(slope >= MAX_SAFE_SLOPE_DEG, np.minimum(score, SLOPE_VETO_CAP), score)


def mae_report(pred, Y):
    """Mean absolute errors in physical units: fractions as percentage points, durations in hours."""
    err = np.abs(np.asarray(pred) - np.asarray(Y)).mean(axis=0)
    return {
        "sunlit_fraction_mae_pp": float(100 * err[0]),
        "dte_fraction_mae_pp": float(100 * err[1]),
        "longest_dark_mae_h": float(WINDOW_HOURS * err[2]),
        "longest_outage_mae_h": float(WINDOW_HOURS * err[3]),
    }

