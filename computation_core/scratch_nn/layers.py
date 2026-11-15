"""Trainable layers, written from scratch in NumPy.

Every layer implements forward(x, training) and backward(grad_out) -> grad_in,
and exposes params(): list of (value, gradient) array pairs for the optimizer.
"""
import numpy as np


class Layer:
    def forward(self, x, training=False):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []


class Dense(Layer):
    """y = x W + b, He-initialised (suits ReLU-family activations)."""

    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=False):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW[...] = self.x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Dropout(Layer):
    """Inverted dropout: scales at train time so inference needs no change."""

    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, training=False):
        if not training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


class BatchNorm(Layer):
    """Per-feature batch normalisation with running statistics for inference."""

    def __init__(self, n, momentum=0.9, eps=1e-5):
        self.gamma, self.beta = np.ones(n), np.zeros(n)
        self.dgamma, self.dbeta = np.zeros(n), np.zeros(n)
        self.run_mean, self.run_var = np.zeros(n), np.ones(n)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, training=False):
        if training:
            mean, var = x.mean(axis=0), x.var(axis=0)
            self.run_mean = self.momentum * self.run_mean + (1 - self.momentum) * mean
            self.run_var = self.momentum * self.run_var + (1 - self.momentum) * var
        else:
            mean, var = self.run_mean, self.run_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, grad):
        n = grad.shape[0]
        self.dgamma[...] = (grad * self.xhat).sum(axis=0)
        self.dbeta[...] = grad.sum(axis=0)
        dxhat = grad * self.gamma
        # closed-form gradient through mean and variance
        return (self.inv_std / n) * (n * dxhat - dxhat.sum(axis=0) - self.xhat * (dxhat * self.xhat).sum(axis=0))

    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]
