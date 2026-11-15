"""Intent + topic classifier written from scratch in NumPy.

hashed word/character n-gram embedding bag -> shared hidden layer -> two softmax heads:
  * intent: what the user wants (explain, why, improve, compare dates, ...)
  * topic : which quantity the sentence is about (sun angle, sunlight, Earth link, slope, night, battery, ...)
Splitting the two lets the model combine them ('improve' + 'sun_angle') for sentences it has never seen, instead of
needing a ready-made class for every pairing.

Reuses the project's own Dense / ReLU / Dropout / Adam; adds what a text model needs on top: an embedding bag with a
sparse (lazy) Adam update, and softmax cross-entropy.
"""
from pathlib import Path

import numpy as np

from ..scratch_nn.activations import ReLU
from ..scratch_nn.layers import Dense, Dropout
from ..scratch_nn.optimizer import Adam
from .entities import extract
from .text import N_BUCKETS, features

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "assistant_nlu.npz"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class IntentModel:
    def __init__(self, labels, topics=("none",), dim=64, hidden=128, dropout=0.3, seed=0):
        self.labels, self.topics = list(labels), list(topics)
        self.rng = np.random.default_rng(seed)
        self.dim, self.hidden = dim, hidden
        self.E = self.rng.normal(0.0, 0.1, (N_BUCKETS, dim))
        self.trunk = [Dense(dim, hidden, self.rng), ReLU(), Dropout(dropout, self.rng)]
        self.head_i = Dense(hidden, len(self.labels), self.rng)
        self.head_t = Dense(hidden, len(self.topics), self.rng)
        self.temperature, self.t_temperature = 1.0, 1.0
        self._m, self._v, self._t = None, None, 0

    # ---- forward / backward ---------------------------------------------------------------------------------
    def _bag(self, id_lists):
        return np.stack([self.E[ids].mean(axis=0) for ids in id_lists])

    def _forward(self, bag, training=False):
        h = bag
        for layer in self.trunk:
            h = layer.forward(h, training)
        return self.head_i.forward(h, training), self.head_t.forward(h, training)

    def logits(self, id_lists):
        return self._forward(self._bag(id_lists))

    def _params(self):
        return [pg for layer in (*self.trunk, self.head_i, self.head_t) for pg in layer.params()]

    def _sparse_adam(self, id_lists, g_bag, lr, b1=0.9, b2=0.999, eps=1e-8):
        """Lazy Adam on only the embedding rows this batch touched."""
        if self._m is None:
            self._m, self._v = np.zeros_like(self.E), np.zeros_like(self.E)
        grad = {}
        for ids, g in zip(id_lists, g_bag):
            share = g / len(ids)
            for i in ids:
                grad[i] = grad.get(i, 0.0) + share
        rows = np.fromiter(grad.keys(), dtype=np.int64)
        G = np.stack([grad[i] for i in rows])
        self._t += 1
        self._m[rows] = b1 * self._m[rows] + (1 - b1) * G
        self._v[rows] = b2 * self._v[rows] + (1 - b2) * G * G
        mh, vh = self._m[rows] / (1 - b1 ** self._t), self._v[rows] / (1 - b2 ** self._t)
        self.E[rows] -= lr * mh / (np.sqrt(vh) + eps)

    @staticmethod
    def _targets(p, idx, smoothing, C):
        t = np.full(p.shape, smoothing / max(C - 1, 1))
        t[np.arange(len(idx)), idx] = 1.0 - smoothing
        return t

    # ---- training -------------------------------------------------------------------------------------------
    def fit(self, texts, y, y_topic=None, epochs=14, batch=64, lr=3e-3, smoothing=0.05, verbose=True):
        ids = [features(t) for t in texts]
        yi = np.array([self.labels.index(c) for c in y])
        yt = np.array([self.topics.index(c) for c in (y_topic if y_topic is not None else ["none"] * len(y))])
        opt = Adam(lr=lr, weight_decay=1e-4)
        n = len(ids)
        for ep in range(epochs):
            order = self.rng.permutation(n)
            loss = 0.0
            for s in range(0, n, batch):
                b = order[s:s + batch]
                zi, zt = self._forward(self._bag([ids[i] for i in b]), training=True)
                pi, pt = softmax(zi), softmax(zt)
                ti = self._targets(pi, yi[b], smoothing, len(self.labels))
                tt = self._targets(pt, yt[b], smoothing, len(self.topics))
                loss += float(-(ti * np.log(pi + 1e-12)).sum() - (tt * np.log(pt + 1e-12)).sum())
                g = self.head_i.backward((pi - ti) / len(b)) + self.head_t.backward((pt - tt) / len(b))
                for layer in reversed(self.trunk):  # the pooled-embedding gradient comes out of the last step
                    g = layer.backward(g)
                opt.step(self._params())
                self._sparse_adam([ids[i] for i in b], g, lr)
            if verbose:
                print(f"epoch {ep + 1:2d}  loss {loss / n:.4f}", flush=True)

    # ---- inference ------------------------------------------------------------------------------------------
    def _proba(self, texts):
        zi, zt = self.logits([features(extract(t).masked) for t in texts])
        return softmax(zi / self.temperature), softmax(zt / self.t_temperature)

    def proba(self, texts):
        return self._proba(texts)[0]

    def proba_topic(self, texts):
        return self._proba(texts)[1]

    def predict(self, text):
        p = self.proba([text])[0]
        order = np.argsort(-p)
        return self.labels[order[0]], float(p[order[0]]), [(self.labels[i], float(p[i])) for i in order[:3]]

    def predict_topic(self, text):
        p = self.proba_topic([text])[0]
        i = int(p.argmax())
        return self.topics[i], float(p[i])

    def accuracy(self, texts, y):
        p = self.proba(texts)
        return float(np.mean([self.labels[i] == c for i, c in zip(p.argmax(axis=1), y)]))

    def topic_accuracy(self, texts, y_topic):
        p = self.proba_topic(texts)
        return float(np.mean([self.topics[i] == c for i, c in zip(p.argmax(axis=1), y_topic)]))

    def calibrate(self, texts, y, y_topic=None):
        """Temperature scaling on held-out data so the confidences mean what they say (texts are already masked)."""
        zi, zt = self.logits([features(t) for t in texts])
        grid = np.arange(0.4, 4.01, 0.05)
        nll = lambda z, idx, T: -np.log(softmax(z / T)[np.arange(len(idx)), idx] + 1e-12).mean()  # noqa: E731
        yi = np.array([self.labels.index(c) for c in y])
        self.temperature = float(min(grid, key=lambda T: nll(zi, yi, T)))
        if y_topic is not None:
            yt = np.array([self.topics.index(c) for c in y_topic])
            self.t_temperature = float(min(grid, key=lambda T: nll(zt, yt, T)))
        return self.temperature

    # ---- persistence ----------------------------------------------------------------------------------------
    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, E=self.E.astype(np.float32), labels=np.array(self.labels), topics=np.array(self.topics),
                            temperature=self.temperature, t_temperature=self.t_temperature,
                            W0=self.trunk[0].W, b0=self.trunk[0].b, Wi=self.head_i.W, bi=self.head_i.b, Wt=self.head_t.W, bt=self.head_t.b)

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = np.load(path)
        m = cls(list(d["labels"]), list(d["topics"]), dim=d["E"].shape[1], hidden=d["W0"].shape[1])
        m.E = d["E"].astype(np.float64)
        m.trunk[0].W, m.trunk[0].b = d["W0"], d["b0"]
        m.head_i.W, m.head_i.b, m.head_t.W, m.head_t.b = d["Wi"], d["bi"], d["Wt"], d["bt"]
        m.temperature, m.t_temperature = float(d["temperature"]), float(d["t_temperature"])
        return m


class IntentEnsemble:
    """Several independently trained models (different data and initialisation); their probabilities are averaged.

    Averaging removes most of the run-to-run luck of a single small network, which matters because the training sentences
    are synthetic and any single model latches onto some of their quirks."""

    def __init__(self, models):
        self.models = models
        self.labels, self.topics = models[0].labels, models[0].topics

    def proba(self, texts):
        return np.mean([m.proba(texts) for m in self.models], axis=0)

    def proba_topic(self, texts):
        return np.mean([m.proba_topic(texts) for m in self.models], axis=0)

    def predict(self, text):
        p = self.proba([text])[0]
        order = np.argsort(-p)
        return self.labels[order[0]], float(p[order[0]]), [(self.labels[i], float(p[i])) for i in order[:3]]

    def predict_topic(self, text):
        p = self.proba_topic([text])[0]
        i = int(p.argmax())
        return self.topics[i], float(p[i])

    def agreement(self, text, intent):
        """Share of the members whose own best intent is `intent`."""
        return float(np.mean([self.labels[int(m.proba([text])[0].argmax())] == intent for m in self.models]))

    def accuracy(self, texts, y):
        p = self.proba(texts)
        return float(np.mean([self.labels[i] == c for i, c in zip(p.argmax(axis=1), y)]))

    def topic_accuracy(self, texts, y_topic):
        p = self.proba_topic(texts)
        return float(np.mean([self.topics[i] == c for i, c in zip(p.argmax(axis=1), y_topic)]))

    @staticmethod
    def paths(base=MODEL_PATH):
        return sorted(Path(base).parent.glob(Path(base).stem + "_*.npz"))

    def save(self, base=MODEL_PATH):
        for old in self.paths(base):
            old.unlink()
        for i, m in enumerate(self.models):
            m.save(Path(base).with_name(f"{Path(base).stem}_{i}.npz"))

    @classmethod
    def load(cls, base=MODEL_PATH):
        return cls([IntentModel.load(p) for p in cls.paths(base)])
