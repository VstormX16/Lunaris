"""Scratch NN correctness: analytic backprop gradients vs. finite differences."""
import numpy as np
import pytest

from computation_core.scratch_nn.activations import LeakyReLU, ReLU, Sigmoid
from computation_core.scratch_nn.layers import BatchNorm, Dense, Dropout
from computation_core.scratch_nn.loss import BCE, MSE, Huber
from computation_core.scratch_nn.model import Sequential
from computation_core.scratch_nn.optimizer import SGD, Adam


def _net(rng, act):
    return Sequential([Dense(5, 7, rng), BatchNorm(7), act(), Dense(7, 3, rng), Sigmoid()])


@pytest.mark.parametrize("act", [ReLU, LeakyReLU])
@pytest.mark.parametrize("loss", [MSE(), BCE()])
def test_gradients_match_finite_differences(act, loss):
    rng = np.random.default_rng(0)
    net = _net(rng, act)
    X, Y = rng.normal(size=(16, 5)), rng.random((16, 3))

    pred = net.forward(X, training=True)
    net.backward(loss.grad(pred, Y))
    analytic = [g.copy() for _, g in net.params()]

    eps = 1e-6
    for (p, _), g in zip(net.params(), analytic):
        num = np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old = p[idx]
            p[idx] = old + eps
            up = loss.value(net.forward(X, training=True), Y)
            p[idx] = old - eps
            down = loss.value(net.forward(X, training=True), Y)
            p[idx] = old
            num[idx] = (up - down) / (2 * eps)
        assert np.allclose(g, num, rtol=1e-4, atol=1e-7)


def test_dropout_is_identity_at_inference_and_unbiased_in_training():
    rng = np.random.default_rng(1)
    d = Dropout(0.5, rng)
    x = np.ones((2000, 50))
    assert np.array_equal(d.forward(x, training=False), x)
    assert abs(d.forward(x, training=True).mean() - 1.0) < 0.02


@pytest.mark.parametrize("opt", [Adam(lr=0.01), SGD(lr=0.05, momentum=0.9)])
def test_network_learns_a_nonlinear_function(opt):
    """Fit y = (sin(3x0) * cos(2x1) + 1) / 2; a linear model cannot do this."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (2000, 2))
    Y = ((np.sin(3 * X[:, 0]) * np.cos(2 * X[:, 1]) + 1) / 2)[:, None]
    net = Sequential([Dense(2, 64, rng), LeakyReLU(), Dense(64, 64, rng), LeakyReLU(), Dense(64, 1, rng), Sigmoid()])
    net.fit(X[:1600], Y[:1600], MSE(), opt, epochs=150, batch=64,
            X_val=X[1600:], Y_val=Y[1600:], patience=30, rng=rng, verbose=False)
    assert MSE().value(net.predict(X[1600:]), Y[1600:]) < 0.003


def test_state_roundtrip():
    rng = np.random.default_rng(3)
    a, b = _net(rng, ReLU), _net(np.random.default_rng(99), ReLU)
    X = rng.normal(size=(4, 5))
    b.load_state(a.state())
    assert np.allclose(a.predict(X), b.predict(X))


@pytest.mark.parametrize("loss", [MSE(weights=[1, 1, 2]), BCE(weights=[1, 3, 1]),
                                  Huber(delta=0.3), Huber(delta=0.3, weights=[2, 1, 1])])
def test_weighted_and_robust_loss_gradients(loss):
    rng = np.random.default_rng(5)
    pred, y = rng.uniform(0.05, 0.95, (8, 3)), rng.random((8, 3))
    g, eps, num = loss.grad(pred, y), 1e-6, np.zeros_like(pred)
    for idx in np.ndindex(pred.shape):
        p1, p2 = pred.copy(), pred.copy()
        p1[idx] += eps
        p2[idx] -= eps
        num[idx] = (loss.value(p1, y) - loss.value(p2, y)) / (2 * eps)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-8)


def test_greedy_ensemble_ignores_bad_members():
    from computation_core.scratch_nn.automl import greedy_weights
    rng = np.random.default_rng(6)
    Y = rng.random((300, 4))
    preds = np.stack([Y + rng.normal(0, 0.3, Y.shape), Y + rng.normal(0, 0.02, Y.shape),
                      Y + rng.normal(0, 0.02, Y.shape)])
    w, _ = greedy_weights(preds, Y, np.ones(4))
    assert w[0] == 0 and abs(w.sum() - 1) < 1e-12


def _synthetic_hybrid_data(n, rng):
    """Features shaped like the real ones; targets = physics proxy + a learnable correction."""
    from computation_core.features import FEATURE_NAMES
    from computation_core.scratch_nn.landing_scorer import PROXY_IDX
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    X[:, PROXY_IDX] = rng.uniform(0.1, 0.9, (n, 4))
    Y = np.clip(X[:, PROXY_IDX] + 0.05 * np.tanh(X[:, :4]), 0, 1)
    return X, Y


def test_ensemble_roundtrip_mixed_heads_and_physics_fallback(tmp_path):
    from computation_core.features import FEATURE_NAMES
    from computation_core.scratch_nn.architecture import build_net
    from computation_core.scratch_nn.landing_scorer import PROXY_IDX, Ensemble
    rng = np.random.default_rng(7)
    X, _ = _synthetic_hybrid_data(50, rng)
    base = dict(hidden=[16], act="relu", dropout=0.0, lr=1e-3, wd=0.0, batch=32, dur_weight=1.0)
    cfgs = [dict(base, head="residual", loss="mse", batchnorm=False, seed=1),
            dict(base, head="direct", loss="bce", batchnorm=True, seed=2)]
    nets = [build_net(c, len(FEATURE_NAMES), 4, np.random.default_rng(c["seed"])) for c in cfgs]
    from computation_core.scratch_nn.automl import ERR_CFG
    err_net = build_net(ERR_CFG, len(FEATURE_NAMES) + 4, 1, np.random.default_rng(3))
    m = Ensemble(cfgs, nets, [0.5, 0.5], X.mean(0), X.std(0) + 1e-8, np.full(4, 0.05),
                 [True, True, False, True], np.ones(4), ERR_CFG, err_net)
    mu, sd = m.predict(X, return_std=True)
    assert mu.shape == (50, 4) and (mu >= 0).all() and (mu <= 1).all()
    assert np.allclose(mu[:, 2], X[:, PROXY_IDX[2]]) and np.allclose(sd[:, 2], 0)  # physics fallback
    u = m.uncertainty(X, sd)
    assert u.shape == (50,)
    m.save(tmp_path / "e.npz")
    m2 = Ensemble.load(tmp_path / "e.npz")
    assert np.allclose(m2.predict(X), mu) and np.allclose(m2.uncertainty(X, sd), u)


def test_successive_halving_learns_the_correction():
    from computation_core.scratch_nn.automl import Data, nmae, successive_halving
    from computation_core.scratch_nn.landing_scorer import PROXY_IDX
    rng = np.random.default_rng(8)
    X, Y = _synthetic_hybrid_data(3000, rng)
    mean, std = X[:2400].mean(0), X[:2400].std(0) + 1e-8
    res = (Y[:2400] - X[:2400, PROXY_IDX]).std(0) + 1e-6
    tr, va = Data(X[:2400], Y[:2400], mean, std, res), Data(X[2400:], Y[2400:], mean, std, res)
    base = np.abs(va.Y - tr.Y.mean(0)).mean(0)
    trials, finalists = successive_halving(tr, va, base, res, n_configs=4, eta=2, r0=2, keep=2,
                                           final_epochs=10, patience=5, seed=0, log=lambda s: None)
    assert len(trials) == 4 and len(finalists) == 2 and finalists[0].best <= finalists[1].best
    assert finalists[0].best < nmae(va.P, va.Y, base)  # beats the raw physics proxy
