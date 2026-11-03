"""
Machine-learned starting values for the fit.

Finding a good start (the data-driven initial guess, escaping local minima, building up
overlapping components) takes most of the time of a fit; the final least-squares step is
quick. An initializer is a small neural network, trained on spectra simulated with the model
itself, that predicts the non-linear parameters (widths, shifts, profile parameters, continuum
slope, sigma*) of a spectrum in about a millisecond. It only provides the start: the fit itself,
the linear amplitudes (solved exactly) and the uncertainties are those of the physical model,
so a poor prediction costs time, never accuracy.

    r = fa.fit(spec, model, guess="ml")              # uses the initializer of the model, if there is one

Inference needs NumPy only. Training (``train``) uses scikit-learn; ``simulate`` builds the
training set from existing fits (realistic parameters, real noise and masks).
"""
import glob
import json
import os
import warnings

import numpy as np

__all__ = ("Initializer", "available", "for_model", "simulate", "train")

FOLDERS = [os.path.join(os.path.dirname(__file__), "data", "ml"),
           os.path.join(os.path.expanduser("~"), ".fantasy_agn", "ml")]
LOG_PARAMS = ("fwhm", "sigma")            # predicted as logarithms (positive, wide range)


def target_names(model):
    """Non-linear free parameters of a model: what an initializer predicts."""
    return [p.name for p in model.params if not p.linear and not p.fixed and not p.tie]


def _signature(model):
    return sorted(target_names(model)), (list(model.wave_range) if model.wave_range else None)


def features(spectrum, wave_range, step=2.0):
    """Spectrum on a fixed rest-frame grid, normalised by its median; gaps interpolated."""
    lo, hi = wave_range
    grid = np.arange(lo, hi + step / 2, step)
    good = spectrum.good & np.isfinite(spectrum.flux)
    w, f = spectrum.wave[good], spectrum.flux[good]
    if w.size < 20:
        raise ValueError("too few good pixels for the initializer")
    y = np.interp(grid, w, f, left=np.nan, right=np.nan)
    norm = np.nanmedian(y)
    if not np.isfinite(norm) or norm <= 0:
        norm = np.nanmedian(np.abs(y)) or 1.0
    y = y / norm
    cover = np.isfinite(y)
    y = np.where(cover, y, 0.0)
    return np.clip(np.r_[y, cover.astype(float)], -5, 20)


class Initializer:
    """A trained network: features -> PCA -> MLP (ReLU) -> standardised parameters."""

    def __init__(self, arrays, meta):
        self.a, self.meta = arrays, meta
        self.names = meta["names"]
        self.wave_range = tuple(meta["wave_range"])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        return cls(arrays, meta)

    def save(self, path):
        np.savez_compressed(path, meta=json.dumps(self.meta), **self.a)

    def predict_vector(self, x):
        a = self.a
        h = (x - a["pca_mean"]) @ a["pca_components"].T
        h = (h - a["x_mean"]) / a["x_scale"]
        n = int(self.meta["n_layers"])
        for i in range(n):
            h = h @ a[f"W{i}"] + a[f"b{i}"]
            if i < n - 1:
                h = np.maximum(h, 0.0)
        y = h * a["y_scale"] + a["y_mean"]
        out = {}
        for name, v in zip(self.names, y):
            out[name] = float(np.exp(v)) if name.endswith(LOG_PARAMS) else float(v)
        return out

    def predict(self, spectrum):
        """{parameter: value} for a spectrum (rest frame)."""
        return self.predict_vector(features(spectrum, self.wave_range))

    def matches(self, model):
        names, wr = _signature(model)
        return names == sorted(self.names) and (wr is None or tuple(wr) == self.wave_range)


def available():
    """Paths of the installed initializers."""
    out = []
    for folder in FOLDERS:
        out += sorted(glob.glob(os.path.join(folder, "*.npz")))
    return out


def for_model(model):
    """The initializer whose parameters and window match ``model`` (None if there is none)."""
    for path in available():
        try:
            ini = Initializer.load(path)
        except Exception:
            continue
        if ini.matches(model):
            return ini
    return None


def apply(model, spectrum, initializer=None):
    """Set the non-linear parameters of ``model`` (a copy is modified in place) to the prediction,
    kept inside the limits. Returns True if an initializer was applied."""
    ini = initializer or for_model(model)
    if ini is None:
        return False
    pred = ini.predict(spectrum)
    for name, v in pred.items():
        p = model.params[name]
        lo, hi = p.min, p.max
        span = (hi - lo) if np.isfinite(hi - lo) else max(abs(v), 1.0)
        p.value = float(np.clip(v, lo + 1e-3 * span, hi - 1e-3 * span))
    return True


# ------------------------------------------------------------------ training
def simulate(results, n_aug=6, jitter=0.25, seed=0):
    """
    Training set from fits: for every FitResult (e.g. the epochs of joint fits), the model is
    evaluated at its fitted parameters and at ``n_aug`` perturbed versions (non-linear parameters
    jittered within their limits by ``jitter`` of the range or relative width), with Gaussian
    noise from the real error spectrum and the real mask. Returns (X, Y, names, wave_range).
    """
    rng = np.random.default_rng(seed)
    names = target_names(results[0].model)
    wave_range = tuple(results[0].model.wave_range)
    X, Y = [], []
    for r in results:
        m = r.model
        if sorted(target_names(m)) != sorted(names):
            continue
        base = {n: m.params[n].value for n in names}
        s = r.spectrum
        for k in range(n_aug + 1):
            mm = m.copy()
            vals = {}
            for n in names:
                p = mm.params[n]
                v = base[n]
                if k > 0:
                    if n.endswith(LOG_PARAMS):
                        v = v * np.exp(rng.normal(0, jitter))
                    else:
                        rngw = (p.max - p.min) if np.isfinite(p.max - p.min) else 1.0
                        v = v + rng.normal(0, jitter * 0.25 * rngw)
                    v = float(np.clip(v, p.min, p.max))
                p.value = v
                vals[n] = v
            flux = mm.evaluate(s.wave, s.context)
            noisy = flux + rng.normal(0, 1, s.wave.size) * np.where(np.isfinite(s.err), s.err, 0)
            sim = s.copy(flux=noisy)
            try:
                X.append(features(sim, wave_range))
            except ValueError:
                continue
            Y.append([np.log(vals[n]) if n.endswith(LOG_PARAMS) else vals[n] for n in names])
    return np.array(X), np.array(Y), names, wave_range


def train(X, Y, names, wave_range, n_pca=80, hidden=(256, 256), seed=0, max_iter=400):
    """Fit PCA + MLP (scikit-learn) and return an Initializer (NumPy-only at prediction)."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0] - 1), random_state=seed).fit(X)
    H = pca.transform(X)
    xm, xs = H.mean(0), H.std(0) + 1e-12
    ym, ys = Y.mean(0), Y.std(0) + 1e-12
    net = MLPRegressor(hidden_layer_sizes=hidden, alpha=1e-4, learning_rate_init=1e-3, max_iter=max_iter,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.fit((H - xm) / xs, (Y - ym) / ys)
    arrays = {"pca_mean": pca.mean_, "pca_components": pca.components_, "x_mean": xm, "x_scale": xs,
              "y_mean": ym, "y_scale": ys}
    for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    meta = {"names": list(names), "wave_range": list(wave_range), "n_layers": len(net.coefs_),
            "n_train": int(len(X)), "validation_score": float(net.best_validation_score_ or np.nan)}
    return Initializer(arrays, meta)
