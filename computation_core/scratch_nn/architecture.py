"""Network construction from a plain-dict config (so models can be saved/rebuilt/searched).

head = "residual": linear output predicting (target - physics proxy) / scale
head = "direct":   sigmoid output predicting the target in [0, 1]
"""
import numpy as np

from .activations import LeakyReLU, ReLU, Sigmoid
from .layers import BatchNorm, Dense, Dropout
from .loss import BCE, MSE, Huber
from .model import Sequential

ACTIVATIONS = {"relu": ReLU, "leaky": LeakyReLU}


def build_net(cfg, n_in, n_out, rng):
    layers, prev = [], n_in
    for h in cfg["hidden"]:
        layers.append(Dense(prev, h, rng))
        if cfg["batchnorm"]:
            layers.append(BatchNorm(h))
        layers.append(ACTIVATIONS[cfg["act"]]())
        if cfg["dropout"] > 0:
            layers.append(Dropout(cfg["dropout"], rng))
        prev = h
    layers.append(Dense(prev, n_out, rng))
    if cfg["head"] == "direct":
        layers.append(Sigmoid())
    else:  # residual corrections start near zero, i.e. at the physics proxy
        layers[-1].W *= 0.1
    return Sequential(layers)


def make_loss(cfg, n_out):
    w = np.ones(n_out)
    w[2:] = cfg["dur_weight"]  # columns 2, 3 are the duration targets
    return {"bce": BCE, "mse": MSE, "huber": Huber}[cfg["loss"]](weights=w)


def count_params(net):
    return int(sum(p.size for p, _ in net.params()))
