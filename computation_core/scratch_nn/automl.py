"""AutoML for the surrogate: search -> ensemble -> per-target decision -> retrain -> honest test.

1. Successive halving (Hyperband-style): sample N configs, train all briefly, keep the best
   1/eta, train those eta x longer, ... finally train the survivors to convergence.
   Most compute goes to promising configs, so many architectures are tried cheaply.
2. Greedy ensemble selection (Caruana et al. 2004) on validation picks member weights.
3. Per target, the network is used only if it beats the physics proxy on validation.
4. Selected members are retrained on train+validation with their best epoch count.
5. An error-estimator network learns, from out-of-sample validation errors, where the chosen
   predictor is likely wrong. It drives the decision of which map pixels get exact physics.
6. The test set (unseen longitudes AND unseen dates) is touched once, for the report.

Model selection uses longitude wedges inside the training area (spatially held out).
Run: python -m computation_core.scratch_nn.automl            (full, ~30-40 min)
     python -m computation_core.scratch_nn.automl --quick    (smoke run)
"""
import argparse
import json
import time

import numpy as np

from data_engine import config

from ..features import FEATURE_NAMES, TARGET_NAMES
from .architecture import build_net, count_params, make_loss
from .loss import Huber
from .landing_scorer import METRICS_PATH, MODEL_PATH, PROXY_IDX, Ensemble, mae_report, to_target
from .optimizer import Adam

LOG_PATH = config.CACHE_DIR / "automl_log.json"
VAL_SECTORS = [(120.0, 150.0), (270.0, 300.0)]  # held-out wedges for model selection


def sample_config(rng):
    head = "residual" if rng.random() < 2 / 3 else "direct"  # prior from the proxy experiment
    # 512-wide nets and batch 128 were dropped after timing: ~40x the cost of 128-wide for a small
    # residual signal, so the budget buys more distinct architectures instead.
    depth, width, taper = int(rng.choice([2, 3, 4])), int(rng.choice([64, 128, 256])), rng.random() < 0.5
    return {
        "head": head,
        "loss": str(rng.choice(["mse", "huber"] if head == "residual" else ["bce", "mse"])),
        "hidden": [max(32, width >> i) if taper else width for i in range(depth)],
        "act": str(rng.choice(["relu", "leaky"])),
        "dropout": float(rng.choice([0.0, 0.05, 0.1, 0.2])),
        "batchnorm": bool(rng.random() < 0.6),
        "lr": float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3))),
        "wd": float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        "batch": int(rng.choice([256, 512])),
        "dur_weight": float(rng.choice([1.0, 2.0, 4.0])),
        "seed": int(rng.integers(2 ** 31)),
    }


def nmae(pred, Y, base):
    """Mean over targets of MAE / MAE-of-predicting-the-mean: every target counts equally."""
    return float((np.abs(pred - Y).mean(axis=0) / base).mean())


class Data:
    """A split, normalized with training statistics; residual targets are pre-scaled."""

    def __init__(self, X, Y, mean, std, res_scale):
        self.P = X[:, PROXY_IDX]
        self.Z = (X - mean) / std
        self.Y = Y
        self.R = (Y - self.P) / res_scale

    def targets(self, cfg):
        return self.R if cfg["head"] == "residual" else self.Y


class Trial:
    def __init__(self, cfg, n_in, n_out, res_scale):
        self.cfg, self.res_scale = cfg, res_scale
        self.rng = np.random.default_rng(cfg["seed"])
        self.net = build_net(cfg, n_in, n_out, self.rng)
        self.opt = Adam(lr=cfg["lr"], weight_decay=cfg["wd"])
        self.loss = make_loss(cfg, n_out)
        self.epochs, self.best, self.best_epoch, self.state, self.seconds = 0, np.inf, 0, None, 0.0

    def predict(self, data):
        return to_target(self.cfg, self.net.predict(data.Z), data.P, self.res_scale)

    def train(self, tr, va, epochs, patience, base):
        metric = lambda out, _: nmae(to_target(self.cfg, out, va.P, self.res_scale), va.Y, base)  # noqa: E731
        t0 = time.perf_counter()
        hist = self.net.fit(tr.Z, tr.targets(self.cfg), self.loss, self.opt, epochs=epochs,
                            batch=self.cfg["batch"], X_val=va.Z, Y_val=va.Y, patience=patience,
                            rng=self.rng, verbose=False, val_metric=metric)
        self.seconds += time.perf_counter() - t0
        vals = [h[1] for h in hist]
        i = int(np.argmin(vals))
        if vals[i] < self.best:  # fit() already restored this call's best weights
            self.best, self.best_epoch, self.state = vals[i], self.epochs + i + 1, self.net.state()
        elif self.state is not None:
            self.net.load_state(self.state)
        self.epochs += len(hist)

    def summary(self):
        return {"config": self.cfg, "val_nmae": self.best, "epochs": self.epochs, "best_epoch": self.best_epoch,
                "params": count_params(self.net), "train_seconds": round(self.seconds, 1)}


def successive_halving(tr, va, base, res_scale, n_configs, eta, r0, keep, final_epochs, patience, seed, log):
    rng = np.random.default_rng(seed)
    n_in, n_out = tr.Z.shape[1], tr.Y.shape[1]
    trials = [Trial(sample_config(rng), n_in, n_out, res_scale) for _ in range(n_configs)]
    alive, budget, rung = list(trials), r0, 0
    while len(alive) > keep:
        for t in alive:
            t.train(tr, va, budget, None, base)
        alive.sort(key=lambda t: t.best)
        log(f"rung {rung}: {len(alive)} configs x {budget} epochs, best val nMAE {alive[0].best:.4f} "
            f"({alive[0].cfg['head']}, {alive[0].cfg['hidden']})")
        alive = alive[: max(keep, len(alive) // eta)]
        budget *= eta
        rung += 1
    for t in alive:
        t.train(tr, va, final_epochs, patience, base)
    alive.sort(key=lambda t: t.best)
    log(f"final: {len(alive)} configs trained to convergence, best val nMAE {alive[0].best:.4f}")
    return trials, alive


def greedy_weights(preds, Y, base, max_size=10):
    """Caruana-style forward selection with replacement; returns member weights."""
    chosen, best = [], np.inf
    for _ in range(max_size):
        scores = [nmae(preds[chosen + [m]].mean(axis=0), Y, base) for m in range(len(preds))]
        m = int(np.argmin(scores))
        if scores[m] >= best - 1e-9:
            break
        chosen.append(m)
        best = scores[m]
    return np.bincount(chosen, minlength=len(preds)) / len(chosen), best


def _rank(a):
    """Ranks with ties averaged (half of all samples have exactly zero error)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    _, first, counts = np.unique(a[order], return_index=True, return_counts=True)
    ranks = np.empty(len(a))
    ranks[order] = np.repeat(first + (counts - 1) / 2.0, counts)
    return ranks


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


ERR_CFG = {"head": "linear", "loss": "huber", "hidden": [64, 32], "act": "relu", "dropout": 0.0,
           "batchnorm": False, "lr": 1e-3, "wd": 1e-4, "batch": 256, "dur_weight": 1.0, "seed": 12345}


def train_error_model(model, X_va, mu_va, sd_va, Y_va, base):
    """Regress log(1 + normalized error) of the chosen predictor on validation samples it never
    trained on; early-stopped on ranking quality (Spearman) of an internal 20% holdout."""
    e = (np.abs(mu_va - Y_va) / base).mean(axis=1)
    t = np.log1p(e / (e.mean() + 1e-12))[:, None]
    inp = model.error_inputs(X_va, sd_va)
    rng = np.random.default_rng(ERR_CFG["seed"])
    idx = rng.permutation(len(e))
    a, b = idx[: int(0.8 * len(e))], idx[int(0.8 * len(e)):]
    net = build_net(ERR_CFG, inp.shape[1], 1, rng)
    net.fit(inp[a], t[a], Huber(), Adam(lr=ERR_CFG["lr"], weight_decay=ERR_CFG["wd"]), epochs=300,
            batch=ERR_CFG["batch"], X_val=inp[b], Y_val=t[b], patience=25, rng=rng, verbose=False,
            val_metric=lambda p, y: -_spearman(p[:, 0], y[:, 0]))
    return net, _spearman(net.predict(inp[b])[:, 0], t[b, 0])


def _refinement(mu, Y, scores, q):
    """MAE after replacing the top-q fraction by `scores` with exact values (error 0)."""
    fixed = np.abs(mu - Y)
    fixed[np.argsort(-scores)[: int(round(q * len(Y)))]] = 0.0
    return mae_report(fixed, 0 * fixed)


def run(n_configs=27, eta=3, r0=6, keep=5, final_epochs=200, patience=20, seed=0, log=print):
    t_start = time.perf_counter()
    dtr = np.load(config.CACHE_DIR / "dataset_train.npz")
    dte = np.load(config.CACHE_DIR / "dataset_test.npz")
    for d in (dtr, dte):
        if list(d["feature_names"]) != FEATURE_NAMES:
            raise ValueError("dataset built with different features; rebuild it with dataset_builder")
    X, Y, meta = dtr["X"].astype(np.float64), dtr["Y"].astype(np.float64), dtr["meta"]
    Xt, Yt = dte["X"].astype(np.float64), dte["Y"].astype(np.float64)

    lon = meta[:, 3] % 360.0
    is_val = np.zeros(len(X), bool)
    for a, b in VAL_SECTORS:
        is_val |= (lon >= a) & (lon < b)
    tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    log(f"data: train {len(tr_i)}, validation {len(va_i)} (held-out wedges), test {len(Xt)} (unseen region+dates)")

    # ---- 1. search on train / validation ----
    mean, std = X[tr_i].mean(0), X[tr_i].std(0) + 1e-8
    res_scale = (Y[tr_i] - X[tr_i][:, PROXY_IDX]).std(0) + 1e-6
    tr, va = Data(X[tr_i], Y[tr_i], mean, std, res_scale), Data(X[va_i], Y[va_i], mean, std, res_scale)
    base_va = np.abs(Y[va_i] - Y[tr_i].mean(0)).mean(0)
    trials, finalists = successive_halving(tr, va, base_va, res_scale, n_configs, eta, r0, keep,
                                           final_epochs, patience, seed, log)

    # ---- 2. ensemble weights and 3. per-target network-vs-physics decision (validation only) ----
    preds_va = np.stack([t.predict(va) for t in finalists])
    weights, ens_val = greedy_weights(preds_va, va.Y, base_va)
    ens_mae = np.abs(np.tensordot(weights, preds_va, axes=1) - va.Y).mean(0)
    proxy_mae = np.abs(va.P - va.Y).mean(0)
    use_nn = ens_mae < proxy_mae
    mu_va = np.where(use_nn, np.tensordot(weights, preds_va, axes=1), va.P)
    sd_va = np.where(use_nn, preds_va.std(axis=0), 0.0)
    log(f"ensemble weights {np.round(weights, 2).tolist()}, val nMAE {ens_val:.4f} "
        f"(best single {finalists[0].best:.4f}); network used for targets {use_nn.tolist()}")

    # ---- 4. retrain the selected members on train+validation ----
    mean_all, std_all = X.mean(0), X.std(0) + 1e-8
    res_all = (Y - X[:, PROXY_IDX]).std(0) + 1e-6
    all_data = Data(X, Y, mean_all, std_all, res_all)
    nets, cfgs = [], []
    for t in finalists:
        r = Trial(t.cfg, X.shape[1], Y.shape[1], res_all)
        r.net.fit(all_data.Z, all_data.targets(t.cfg), r.loss, r.opt, epochs=max(1, t.best_epoch),
                  batch=t.cfg["batch"], patience=None, rng=r.rng, verbose=False)
        nets.append(r.net)
        cfgs.append(t.cfg)
    model = Ensemble(cfgs, nets, weights, mean_all, std_all, res_all, use_nn, np.ones(Y.shape[1]))
    _, sd_all = model.predict(X, return_std=True)
    model.unc_scale = np.where(use_nn, sd_all.mean(0), 1.0) + 1e-9
    model.err_cfg = ERR_CFG
    model.err_net, err_holdout_rho = train_error_model(model, X[va_i], mu_va, sd_va, Y[va_i], base_va)
    log(f"error-estimator network: Spearman on its holdout {err_holdout_rho:.3f}")
    model.save(MODEL_PATH)

    # ---- 5. one-time evaluation on the test set ----
    mu, sd = model.predict(Xt, return_std=True)
    members = model.member_predictions(Xt)
    single = np.where(use_nn, members[0], Xt[:, PROXY_IDX])
    base_te = np.abs(Yt - Y.mean(0)).mean(0)
    err = (np.abs(mu - Yt) / base_te).mean(1)
    u_err = model.uncertainty(Xt, sd)
    u_std = (sd / model.unc_scale).mean(1)
    refine = {f"{int(q * 100)}pct_exact": {
        "by_error_network": _refinement(mu, Yt, u_err, q),
        "by_member_disagreement": _refinement(mu, Yt, u_std, q),
        "by_random_expected": mae_report((1 - q) * np.abs(mu - Yt), 0 * mu),
        "oracle_best_possible": _refinement(mu, Yt, err, q),
    } for q in (0.02, 0.05, 0.10, 0.20)}
    big = np.tile(Xt, (int(np.ceil(100_000 / len(Xt))), 1))[:100_000]
    t0 = time.perf_counter()
    model.predict(big, return_std=True)
    infer_s = time.perf_counter() - t0

    metrics = {
        "data": {"train": int(len(tr_i)), "validation": int(len(va_i)), "test_unseen_region_and_dates": int(len(Xt)),
                 "features": len(FEATURE_NAMES), "targets": TARGET_NAMES},
        "search": {"configs_tried": n_configs, "eta": eta, "first_rung_epochs": r0,
                   "finalists": [t.summary() for t in finalists]},
        "ensemble": {"weights": weights.tolist(), "network_used_per_target": dict(zip(TARGET_NAMES, use_nn.tolist())),
                     "total_params": model.n_params, "inference_seconds_per_100k_samples": round(infer_s, 3)},
        "test": {
            "ensemble": mae_report(mu, Yt),
            "best_single_network": mae_report(single, Yt),
            "physics_proxy_only": mae_report(Xt[:, PROXY_IDX], Yt),
            "predict_training_mean": mae_report(np.tile(Y.mean(0), (len(Yt), 1)), Yt),
        },
        "uncertainty": {"spearman_vs_true_error": {"error_network": _spearman(u_err, err),
                                                   "member_disagreement": _spearman(u_std, err)},
                        "error_network_holdout_spearman": err_holdout_rho, "refinement": refine},
        "total_minutes": round((time.perf_counter() - t_start) / 60, 1),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    LOG_PATH.write_text(json.dumps([t.summary() for t in trials], indent=1))
    return model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=int, default=27)
    ap.add_argument("--quick", action="store_true", help="tiny search for a smoke test")
    a = ap.parse_args()
    kw = dict(n_configs=4, r0=2, keep=2, final_epochs=6, patience=3) if a.quick else dict(n_configs=a.configs)
    _, m = run(log=lambda s: print(s, flush=True), **kw)
    print(json.dumps({k: m[k] for k in ("test", "ensemble", "uncertainty")}, indent=2))
