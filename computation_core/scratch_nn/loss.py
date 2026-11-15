"""Loss functions: value and gradient w.r.t. predictions.

Optional per-target `weights` (one per output column) rescale each column's contribution;
the loss is normalized by the mean weight so its magnitude stays comparable.
"""
import numpy as np


class _Weighted:
    def __init__(self, weights=None):
        self.weights = None if weights is None else np.asarray(weights, float)

    def _w(self, pred):
        if self.weights is None:
            return 1.0, 1.0
        return self.weights, float(self.weights.mean())


class MSE(_Weighted):
    def value(self, pred, y):
        w, wm = self._w(pred)
        return float(np.mean(w * (pred - y) ** 2) / wm)

    def grad(self, pred, y):
        w, wm = self._w(pred)
        return 2.0 * w * (pred - y) / (pred.size * wm)


class Huber(_Weighted):
    """Quadratic near zero, linear beyond `delta`: robust to rare large residuals."""

    def __init__(self, delta=1.0, weights=None):
        super().__init__(weights)
        self.delta = delta

    def value(self, pred, y):
        w, wm = self._w(pred)
        a = np.abs(pred - y)
        per = np.where(a <= self.delta, 0.5 * a ** 2, self.delta * (a - 0.5 * self.delta))
        return float(np.mean(w * per) / wm)

    def grad(self, pred, y):
        w, wm = self._w(pred)
        return w * np.clip(pred - y, -self.delta, self.delta) / (pred.size * wm)


class BCE(_Weighted):
    """Binary cross-entropy on probabilities; works with soft targets in [0, 1]."""

    eps = 1e-7

    def value(self, pred, y):
        w, wm = self._w(pred)
        p = np.clip(pred, self.eps, 1 - self.eps)
        return float(-np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / wm)

    def grad(self, pred, y):
        w, wm = self._w(pred)
        p = np.clip(pred, self.eps, 1 - self.eps)
        return w * (p - y) / (p * (1 - p)) / (pred.size * wm)
