"""Activation layers with hand-derived derivatives."""
import numpy as np

from .layers import Layer


class ReLU(Layer):
    def forward(self, x, training=False):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=False):
        self.slope = np.where(x > 0, 1.0, self.alpha)
        return x * self.slope

    def backward(self, grad):
        return grad * self.slope


class Sigmoid(Layer):
    def forward(self, x, training=False):
        # numerically stable: never exponentiates a large positive number
        e = np.exp(-np.abs(x))
        self.y = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1.0 - self.y)
