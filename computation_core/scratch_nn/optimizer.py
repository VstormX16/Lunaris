"""Optimizers updating (value, gradient) pairs in place."""
import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum, self.v = lr, momentum, {}

    def step(self, params):
        for i, (p, g) in enumerate(params):
            v = self.v.setdefault(i, np.zeros_like(p))
            v[...] = self.momentum * v - self.lr * g
            p += v


class Adam:
    """Adam with bias correction and optional decoupled weight decay (AdamW)."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(params):
            m = self.m.setdefault(i, np.zeros_like(p))
            v = self.v.setdefault(i, np.zeros_like(p))
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            if self.wd and p.ndim > 1:  # decay weights, not biases / norm params
                p -= self.lr * self.wd * p
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
