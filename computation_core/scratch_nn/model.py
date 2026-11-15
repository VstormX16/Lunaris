"""Sequential network: training loop, inference, save/load."""
import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        return [pg for layer in self.layers for pg in layer.params()]

    def predict(self, x, batch=8192):
        return np.concatenate([self.forward(x[i : i + batch]) for i in range(0, len(x), batch)])

    def fit(self, X, Y, loss, optimizer, epochs=200, batch=256, X_val=None, Y_val=None,
            patience=20, rng=None, verbose=True, val_metric=None, train_eval=4096):
        """Mini-batch training with early stopping; the best weights are restored at the end.

        val_metric(pred, Y_val) -> float (lower is better) replaces the loss as the
        stopping/selection criterion when given. patience=None never stops early.
        The train loss is reported on a fixed random subset of `train_eval` samples.
        """
        rng = rng or np.random.default_rng(0)
        sub = rng.choice(len(X), size=min(train_eval, len(X)), replace=False)
        history, best, best_state, wait = [], np.inf, None, 0
        for epoch in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                idx = order[i : i + batch]
                if idx.size < 2:  # BatchNorm needs >1 sample
                    continue
                pred = self.forward(X[idx], training=True)
                self.backward(loss.grad(pred, Y[idx]))
                optimizer.step(self.params())
            tr = loss.value(self.predict(X[sub]), Y[sub])
            if X_val is None:
                va = tr
            else:
                pv = self.predict(X_val)
                va = val_metric(pv, Y_val) if val_metric else loss.value(pv, Y_val)
            history.append((tr, va))
            if va < best - 1e-9:
                best, wait = va, 0
                best_state = self.state()
            else:
                wait += 1
            if verbose and (epoch % 10 == 0 or wait == 0):
                print(f"epoch {epoch:3d}  train {tr:.5f}  val {va:.5f}", flush=True)
            if patience is not None and wait >= patience:
                break
        if best_state is not None:
            self.load_state(best_state)
        return history

    # --- persistence: every array a layer owns, in layer order ---
    def state(self):
        out = {}
        for i, layer in enumerate(self.layers):
            for k, v in vars(layer).items():
                if isinstance(v, np.ndarray) and k in ("W", "b", "gamma", "beta", "run_mean", "run_var"):
                    out[f"{i}.{k}"] = v.copy()
        return out

    def load_state(self, state):
        for key, v in state.items():
            i, k = key.split(".")
            getattr(self.layers[int(i)], k)[...] = v
