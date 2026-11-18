"""
Joint fitting of several epochs of the same object.

Parameters matching the ``shared`` patterns (by default every narrow-line
parameter) take one value for all epochs; all others are fitted per epoch.
With ``intercalibrate=True`` each epoch after the first also gets a flux scale
factor, so that the constant narrow lines calibrate the relative flux of the
epochs (as in reverberation-mapping campaigns). Variable quantities (broad
lines, continuum) are then reported in the flux scale of the first epoch.
"""
import fnmatch
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve

from .components import C_KMS, FWHM_PER_SIGMA, instrumental_fwhm
from .fitting import (Problem, _at_limit, _fix_unconstrained, _interior, _prepare,
                      _thread_map, fit_many)
from .fitting import fit as fit_single
from .parameters import Parameter, ParameterSet
from .progress import report
from .result import FitResult

__all__ = ("fit_epochs", "MultiEpochResult")


def _is_shared(name, patterns):
    glob = [p.replace("[", "[[]") for p in patterns]
    return any(name == p or fnmatch.fnmatchcase(name, g) for p, g in zip(patterns, glob))


def _level_ratios(data):
    """Median flux level of each epoch relative to the first, on common wavelengths."""
    x0, y0 = data[0][0], data[0][1]
    out = [1.0]
    for x, y, *_ in data[1:]:
        common = (x >= max(x0.min(), x.min())) & (x <= min(x0.max(), x.max()))
        ref = np.interp(x[common], x0, y0)
        r = np.median(y[common]) / np.median(ref) if common.sum() > 10 and np.median(ref) > 0 else 1.0
        out.append(float(r) if np.isfinite(r) and r > 0 else 1.0)
    return np.array(out)


def _legendre_basis(x, wave_range, degree):
    """Legendre polynomials P_1..P_degree of u in [-1, 1] over wave_range (u clipped outside)."""
    from numpy.polynomial import legendre
    a, b = wave_range
    u = np.clip(2 * (np.asarray(x, float) - a) / (b - a) - 1, -1, 1)
    return np.stack([legendre.legval(u, np.eye(degree + 1)[j]) for j in range(1, degree + 1)], axis=1)


class _JointProblem:
    def __init__(self, model, spectra, masks, shared, intercalibrate, calibration_degree=0, wave_range=None,
                 epoch_scaled=(), calibration_limit=0.3, calibration_groups=None, offset_limit=0.0,
                 broadening_limit=0.0):
        self.model = model
        self.n = len(spectra)
        self.data = [(s.wave[m], s.flux[m], s.err[m], s.context) for s, m in zip(spectra, masks)]
        base = model.params
        joint = ParameterSet()
        self.index = []  # per epoch: joint index of each base parameter
        names = {}
        for p in base:
            if _is_shared(p.name, shared):
                joint.add(Parameter(**{**vars(p), "tie": None}))
                names[p.name] = [p.name] * self.n
            else:
                names[p.name] = [f"{p.name}@{k}" for k in range(self.n)]
                for k in range(self.n):
                    joint.add(Parameter(**{**vars(p), "name": names[p.name][k], "tie": None}))
        for p in base:
            if p.tie is None:
                continue
            target, factor, offset = p.tie
            if names[p.name][0] == p.name and names[target][0] != target:
                # shared on purpose (e.g. feii.shift, tied to broad.shift by the hbeta
                # preset): one free value for all epochs instead of the tie
                warnings.warn(f"{p.name} is shared, so it is fitted freely (one value for all "
                              f"epochs) instead of being tied to {target}, which varies")
                joint[p.name].tie = None
                continue
            for k in range(self.n):
                joint[names[p.name][k]].tie = (names[target][k], factor, offset)
        self.scale_names = []
        self.scale_guess = _level_ratios(self.data) if intercalibrate else None
        if intercalibrate:
            for k in range(self.n):
                # limits around the flux-level ratio: uncalibrated spectra (e.g. OzDES,
                # in counts) can differ between nights by much more than calibrated ones
                g = self.scale_guess[k]
                joint.add(Parameter(f"scale@{k}", g, g / 3.0, g * 3.0, fixed=(k == 0),
                                    description="flux calibration factor"))
                self.scale_names.append(f"scale@{k}")
        # wavelength-dependent calibration: s_k(lambda) = scale_k (1 + sum_j c_kj P_j(u)),
        # anchored by the shared narrow lines (and host) over the whole fitted range
        self.cal_degree = int(calibration_degree) if intercalibrate else 0
        self.cal_range = tuple(wave_range) if wave_range is not None else (
            min(d[0].min() for d in self.data), max(d[0].max() for d in self.data))
        if self.cal_degree:
            for k in range(self.n):
                for j in range(1, self.cal_degree + 1):
                    joint.add(Parameter(f"cal{j}@{k}", 0.0, -calibration_limit, calibration_limit,
                                        fixed=(k == 0), description=f"calibration Legendre coefficient {j}"))
        # components with a shared shape and a free amplitude per epoch (e.g. the host galaxy
        # seen through different apertures): their linear parameters are multiplied by amp@k
        self.scaled = []
        slices = dict(zip([c.name for c in model.components], model._slices()))
        for comp in epoch_scaled:
            if comp not in slices:
                raise ValueError(f"epoch_scaled: no component {comp!r} in the model")
            plist = list(base)
            pos = [i for i in range(slices[comp].start, slices[comp].stop) if plist[i].linear]
            for k in range(self.n):
                joint.add(Parameter(f"{comp}.amp@{k}", 1.0, 0.05, 20.0, fixed=(k == 0),
                                    description=f"amplitude of {comp} in this epoch (relative to the reference)"))
            self.scaled.append((comp, np.array(pos)))
        # a wavelength offset per epoch, as a velocity (km/s) added to every shift of the model in that
        # epoch (line shifts, the stellar velocity; an absorption trough moves by wave * v / c). The
        # wavelength calibration of each spectrum and the redshift it was moved to the rest frame with
        # differ slightly between epochs; the shared narrow lines measure the difference, as they measure
        # the flux scale. The reference epoch defines the frame (offset 0).
        self.offset_limit = float(offset_limit or 0.0)
        self.shift_pos, self.shift_weight = np.zeros(0, int), np.zeros(0)
        if self.offset_limit > 0:
            pos, weight = [], []
            for i, p in enumerate(base):
                last = p.name.rsplit(".", 1)[-1]
                if last in ("shift", "velocity"):
                    pos.append(i)
                    weight.append(1.0)
                elif last == "wave" and p.unit == "A":
                    pos.append(i)
                    weight.append(p.value / C_KMS)
            if not any(_is_shared(base_p.name, shared) and not base_p.fixed for base_p in (list(base)[i] for i in pos)):
                raise ValueError("wavelength offsets per epoch need a shared velocity to be measured against: share the "
                                 "shift of the narrow lines (shared='narrow.*')")
            self.shift_pos, self.shift_weight = np.array(pos, int), np.array(weight, float)
            for k in range(self.n):
                joint.add(Parameter(f"offset@{k}", 0.0, -self.offset_limit, self.offset_limit, fixed=(k == 0),
                                    unit="km/s", description="wavelength offset of this epoch (velocity)"))
        # an extra broadening per epoch, as a FWHM (km/s) added in quadrature to every line width (and to
        # the stellar dispersion): the instrumental resolution differs between epochs (slit width,
        # seeing, a different spectrograph), and the shared narrow lines measure the difference. It is
        # the parameter b2 = signed FWHM^2 in units of (100 km/s)^2, so that the derivative does not
        # vanish at zero and an epoch can also be sharper than the reference (down to 80% of its own
        # resolution in quadrature). Exact for Gaussian lines: a line of FWHM w seen at resolution R
        # has (w^2 + R^2), so w_eff^2 = w^2 + b2 is the same as a resolution sqrt(R^2 + b2).
        self.broadening_limit = float(broadening_limit or 0.0)
        self.width_pos, self.width_weight = np.zeros(0, int), np.zeros(0)
        self.res_min = [float(np.min(instrumental_fwhm(d[0], d[3]))) for d in self.data]
        if self.broadening_limit > 0:
            hosts = {c.name for c in model.components if c.kind == "host"}
            pos, weight = [], []
            for i, p in enumerate(base):
                comp, last = p.name.split(".")[0], p.name.rsplit(".", 1)[-1]
                if last == "fwhm" and p.unit == "km/s":
                    pos.append(i)
                    weight.append(1.0)
                elif last == "sigma" and comp in hosts and p.unit == "km/s":
                    pos.append(i)
                    weight.append(1.0 / FWHM_PER_SIGMA ** 2)
            if not any(_is_shared(base_p.name, shared) and not base_p.fixed for base_p in (list(base)[i] for i in pos)):
                raise ValueError("a broadening per epoch needs a shared line width to be measured against: share the "
                                 "width of the narrow lines (shared='narrow.*')")
            self.width_pos, self.width_weight = np.array(pos, int), np.array(weight, float)
            hi = self.broadening_limit ** 2 / 1e4
            for k in range(self.n):
                lo = -min(self.broadening_limit ** 2, 0.8 * self.res_min[k] ** 2) / 1e4
                joint.add(Parameter(f"broadening@{k}", 0.0, lo, hi, fixed=(k == 0), unit="(100 km/s)^2",
                                    description="extra broadening of this epoch: signed FWHM^2 added to every width"))
        self._basis = [_legendre_basis(d[0], self.cal_range, self.cal_degree) if self.cal_degree else None
                       for d in self.data]
        # epochs from one instrument share its colour (Legendre) calibration: tie to the first
        # epoch of each group; the overall scale stays free per epoch (slit losses differ)
        if self.cal_degree and calibration_groups:
            for group in calibration_groups:
                lead = min(group, key=lambda k: (k != 0, k))      # the reference epoch leads its group
                for k in group:
                    if k != lead:
                        for j in range(1, self.cal_degree + 1):
                            joint[f"cal{j}@{k}"].tie = (f"cal{j}@{lead}", 1.0, 0.0)
        self.joint = joint
        jn = joint.names
        self.index = [np.array([jn.index(names[p.name][k]) for p in base]) for k in range(self.n)]
        self.scale_index = [jn.index(n) for n in self.scale_names]
        self.cal_index = [[jn.index(f"cal{j}@{k}") for j in range(1, self.cal_degree + 1)]
                          for k in range(self.n)]
        self.amp_index = [[jn.index(f"{comp}.amp@{k}") for comp, _ in self.scaled] for k in range(self.n)]
        self.offset_index = [[jn.index(f"offset@{k}")] if self.offset_limit > 0 else [] for k in range(self.n)]
        self.broad_index = [[jn.index(f"broadening@{k}")] if self.broadening_limit > 0 else [] for k in range(self.n)]
        T, c, self.free = joint.affine()
        self.T, self.c = sparse.csr_matrix(T), c
        self.lb = np.array([p.min for p in self.free])
        self.ub = np.array([p.max for p in self.free])
        self.linear = np.array([p.linear for p in self.free], bool)
        self.rows = np.cumsum([0] + [d[0].size for d in self.data])
        # epoch of each free parameter (-1: shared); the normal matrix couples an epoch's
        # parameters only with themselves and the shared ones (block-arrowhead structure)
        self.group = np.array([int(p.name.rsplit("@", 1)[1]) if "@" in p.name else -1
                               for p in self.free])
        # per epoch: the free parameters its model depends on, and the map from its
        # full parameters (+ scale) to them, so that its Jacobian is a small dense block
        Tcsr = self.T.tocsr()
        self._epoch_map = []
        for k in range(self.n):
            cols = (list(self.index[k]) + ([self.scale_index[k]] if self.scale_names else []) + self.cal_index[k]
                    + self.amp_index[k] + self.offset_index[k] + self.broad_index[k])
            Tk = Tcsr[cols]
            fc = np.unique(Tk.indices)
            self._epoch_map.append((fc, Tk[:, fc].toarray()))

    def full(self, theta):
        return self.T @ theta + self.c

    def __setstate__(self, state):
        # fits saved before the wavelength-dependent calibration existed
        self.__dict__.update(state)
        self.__dict__.setdefault("cal_degree", 0)
        self.__dict__.setdefault("cal_index", [[] for _ in range(self.n)])
        self.__dict__.setdefault("_basis", [None] * self.n)
        self.__dict__.setdefault("scaled", [])
        self.__dict__.setdefault("amp_index", [[] for _ in range(self.n)])
        self.__dict__.setdefault("offset_limit", 0.0)
        self.__dict__.setdefault("offset_index", [[] for _ in range(self.n)])
        self.__dict__.setdefault("shift_pos", np.zeros(0, int))
        self.__dict__.setdefault("shift_weight", np.zeros(0))
        self.__dict__.setdefault("broadening_limit", 0.0)
        self.__dict__.setdefault("broad_index", [[] for _ in range(self.n)])
        self.__dict__.setdefault("width_pos", np.zeros(0, int))
        self.__dict__.setdefault("width_weight", np.zeros(0))

    def factor(self, k, full, x=None):
        """Calibration factor of epoch k: a number, or s(lambda) at x (default: its fitted pixels)."""
        sc = full[self.scale_index[k]] if self.scale_names else 1.0
        if not self.cal_degree:
            return sc
        B = self._basis[k] if x is None else _legendre_basis(x, self.cal_range, self.cal_degree)
        return sc * (1.0 + B @ full[self.cal_index[k]])

    def epoch_full(self, k, full, broadened=True):
        """
        The full parameters of epoch k as in its own model (scaled components multiplied, shifts
        moved by the offset). ``broadened``: the widths include the epoch's extra broadening (as the
        joint fit evaluates them); else they are the intrinsic widths, to be seen at the resolution
        of :meth:`epoch_resolution`.
        """
        fk = np.array(full[self.index[k]], float)
        for (comp, pos), ai in zip(self.scaled, self.amp_index[k]):
            fk[pos] = fk[pos] * full[ai]
        if self.offset_index[k]:
            fk[self.shift_pos] += self.shift_weight * full[self.offset_index[k][0]]
        if broadened and self.broad_index[k]:
            fk[self.width_pos] = self._widths(fk[self.width_pos], full[self.broad_index[k][0]])
        return fk

    def _widths(self, w, b2):
        return np.sqrt(np.maximum(w ** 2 + 1e4 * b2 * self.width_weight, 1.0))

    def epoch_broadening(self, k, full):
        """Extra broadening of epoch k as a signed FWHM in km/s (negative: sharper than the reference)."""
        if not self.broad_index[k]:
            return 0.0
        b2 = 1e4 * full[self.broad_index[k][0]]
        return float(np.sign(b2) * np.sqrt(abs(b2)))

    def epoch_resolution(self, spectrum, k, full):
        """The spectrum of epoch k with its resolution broadened as fitted (unchanged without broadening)."""
        if not self.broad_index[k]:
            return spectrum
        b2 = 1e4 * full[self.broad_index[k][0]]
        res = float(np.sqrt(max((spectrum.resolution or 0.0) ** 2 + b2, 0.0)))
        curve = getattr(spectrum, "resolution_curve", None)
        if curve is not None:
            curve = (curve[0], np.sqrt(np.maximum(np.asarray(curve[1], float) ** 2 + b2, 0.0)))
        out = spectrum.copy(resolution=res, resolution_curve=curve)
        out.meta["broadening"] = self.epoch_broadening(k, full)
        out.meta["resolution_file"] = spectrum.resolution            # before the broadening
        out.meta["resolution_curve_file"] = getattr(spectrum, "resolution_curve", None)
        return out

    def _epoch(self, k, full):
        """Model of epoch k; Jb is with respect to the joint (shared) parameters of the epoch,
        followed by d model / d amp for the scaled components."""
        x, y, err, ctx = self.data[k]
        fk = self.epoch_full(k, full)
        f, Jb = self.model.evaluate_with_jacobian(x, ctx, fk)
        if self.scaled:
            extra = []
            for (comp, pos), ai in zip(self.scaled, self.amp_index[k]):
                a = full[ai]
                # the component is linear in its weights: d f / d amp = sum_i w_i dF/dw_i
                extra.append(Jb[:, pos] @ (fk[pos] / a if a != 0 else full[self.index[k]][pos]))
                Jb[:, pos] = Jb[:, pos] * a
            Jb = np.hstack([Jb, np.stack(extra, axis=1)])
        nb = len(self.index[k])
        if self.offset_index[k]:
            # every shift moves by the offset: its derivative is the sum of theirs (exact)
            Jb = np.hstack([Jb, (Jb[:, :nb][:, self.shift_pos] @ self.shift_weight)[:, None]])
        if self.broad_index[k]:
            # w_eff = sqrt(w^2 + 1e4 b2 c): d w_eff / d w = w / w_eff, d w_eff / d b2 = 1e4 c / (2 w_eff)
            pos = self.width_pos
            intrinsic = self.epoch_full(k, full, broadened=False)[pos]
            eff = fk[pos]
            col = Jb[:, pos] @ (1e4 * self.width_weight / (2 * eff))
            Jb[:, pos] = Jb[:, pos] * (intrinsic / eff)
            Jb = np.hstack([Jb, col[:, None]])
        return x, y, err, f, Jb, self.factor(k, full)

    def _calib_columns(self, k, full, f, s, err):
        """Derivatives of the residuals with respect to scale_k and the c_kj."""
        cols = []
        if self.scale_names:
            sc = full[self.scale_index[k]]
            cols.append(np.broadcast_to(s / sc * f / err, f.shape))
            if self.cal_degree:
                cols += [sc * f * self._basis[k][:, j] / err for j in range(self.cal_degree)]
        return cols

    def residuals(self, theta):
        return self._evaluate(theta)[0]

    def jacobian(self, theta):
        return self._evaluate(theta)[1]

    def normal(self, theta):
        """
        Residuals, gradient J^T r and the blocks of J^T J, assembled per epoch from
        dense Jacobians over only the free parameters that epoch depends on.
        Returns (r, g, diag(J^T J), [(free indices, J_k^T J_k) per epoch]).
        """
        full = self.full(theta)
        nfree = len(self.free)

        def one(k):
            x, y, err, f, Jb, s = self._epoch(k, full)
            r = (s * f - y) / err
            Je = np.asarray(s)[..., None] * Jb / err[:, None] if np.ndim(s) else s * Jb / err[:, None]
            nb = len(self.index[k])
            Je, Jamp = Je[:, :nb], Je[:, nb:]
            extra = self._calib_columns(k, full, f, s, err)
            if extra:
                Je = np.hstack([Je, np.stack(extra, axis=1)])
            if Jamp.shape[1]:
                Je = np.hstack([Je, Jamp])
            fc, Tk = self._epoch_map[k]
            Jf = Je @ Tk
            return r, fc, Jf.T @ r, Jf.T @ Jf
        parts = _thread_map(one, range(self.n))
        r = np.concatenate([p[0] for p in parts])
        g = np.zeros(nfree)
        d = np.zeros(nfree)
        items = []
        for _, fc, gk, Mk in parts:
            np.add.at(g, fc, gk)
            np.add.at(d, fc, np.diag(Mk))
            items.append((fc, Mk))
        return r, g, d, items

    def _evaluate(self, theta):
        key = theta.tobytes()
        if getattr(self, "_last", (None,))[0] == key:
            return self._last[1]
        full = self.full(theta)
        resid = []
        blocks_r, blocks_c, blocks_v = [], [], []
        # epochs are independent: evaluated in threads
        for k, (x, y, err, f, Jb, s) in enumerate(_thread_map(lambda k: self._epoch(k, full),
                                                               range(self.n))):
            resid.append((s * f - y) / err)
            rows = np.arange(self.rows[k], self.rows[k + 1])
            vals = (np.asarray(s)[..., None] if np.ndim(s) else s) * Jb / err[:, None]
            cols = np.broadcast_to(np.concatenate([self.index[k], self.amp_index[k], self.offset_index[k],
                                                   self.broad_index[k]]).astype(int),
                                   vals.shape)
            blocks_r.append(np.repeat(rows, vals.shape[1]))
            blocks_c.append(cols.ravel())
            blocks_v.append(vals.ravel())
            for ci, col in zip(([self.scale_index[k]] if self.scale_names else []) + self.cal_index[k],
                               self._calib_columns(k, full, f, s, err)):
                blocks_r.append(rows)
                blocks_c.append(np.full(rows.size, ci))
                blocks_v.append(col)
        J = sparse.csr_matrix((np.concatenate(blocks_v),
                               (np.concatenate(blocks_r), np.concatenate(blocks_c))),
                              shape=(self.rows[-1], len(self.c)))
        out = (np.concatenate(resid), (J @ self.T).tocsr())
        self._last = (key, out)
        return out


def fit_epochs(spectra, model, shared=None, intercalibrate=True, wave_range=None,
               errors="covariance", scale_errors=True, reference="snr", calibration_degree=0,
               epoch_scaled=(), rescale_errors=False, calibration_limit=0.3, calibration_groups=None,
               wavelength_offsets=False, epoch_broadening=False, n_jobs=-1, verbose=True):
    """
    Fit several epochs of one object jointly.

    :param spectra: list of Spectrum (rest frame); ``mjd`` is used for light curves.
    :param model: Model to fit to every epoch.
    :param shared: parameter name patterns that are common to all epochs.
        Default: all narrow-line parameters and, if the model has a stellar
        host, the host (its starlight does not vary: one set of template
        weights, sigma* and v* for all epochs, which also makes the fit much
        faster). Add e.g. 'feii.fwhm', 'feii.shift' to also share the FeII
        kinematics.
    :param intercalibrate: fit a flux scale factor per epoch (reference epoch
        = 1). Needs shared narrow lines to be meaningful.
    :param reference: epoch that defines the flux scale: 'snr' (default, the
        epoch with the highest median S/N in the fitted range) or an index into
        ``spectra``. The uncertainty of the reference enters every other
        epoch's calibration, so a high-S/N reference gives smaller errors.
    :param calibration_degree: with ``intercalibrate``, also a smooth
        wavelength-dependent calibration per epoch, s_k(lambda) = scale_k (1 +
        sum_j c_kj P_j(u)) with Legendre polynomials up to this degree over the
        fitted range (1: a tilt, 2: also curvature). SDSS flux-calibration errors
        depend on wavelength (fibre losses, differential refraction); a single
        scale per epoch (PrepSpec, Shen et al. 2016) leaves them in the light
        curves. The coefficients are anchored by the shared narrow lines (and
        host) across the range, so the model needs narrow lines at both ends
        (e.g. [OII] or Hbeta/[OIII] and Halpha/[NII]/[SII]).
    :param epoch_scaled: components (e.g. ``("host",)``) whose shape is shared but whose
        amplitude is free in each epoch (``<comp>.amp@k``, 1 for the reference epoch): the host
        galaxy seen through different slits, fibres or seeing, e.g. spectra of a transient
        from several instruments. The component must also be shared (default for the host).
    :param rescale_errors: multiply the errors of each epoch by sqrt(reduced chi2) of its own
        independent fit (if > 1) before the joint fit, so that epochs with underestimated errors
        (e.g. smoothed or averaged spectra) do not dominate it.
    :param calibration_groups: lists of epoch indices (input order) observed with the same
        instrument and setting: they share the colour terms of ``calibration_degree`` (the
        overall flux scale stays free per epoch). Default: every epoch its own.
    :param calibration_limit: bound on the Legendre coefficients of ``calibration_degree``
        (spectra from different instruments may differ in colour by more than 30%).
    :param wavelength_offsets: fit a wavelength offset per epoch, as a velocity (km/s; reference epoch
        = 0) added to every shift of the model in that epoch (True: within +-150 km/s, or the limit in
        km/s). The epochs are never on exactly the same wavelength scale: the calibration of each
        spectrum differs by a few km/s (tens for slit spectra), and pipeline redshifts that differ from
        file to file (SDSS: ~20 km/s rms) move the spectra against each other when each is taken to the
        rest frame with its own. The shared narrow lines measure the offset of each epoch, as they
        measure its flux scale, so it needs a shared velocity (the default ``narrow.*``).
        ``result.offsets`` lists them, and ``result.redshift()`` gives the redshift of the narrow lines.
    :param epoch_broadening: fit an extra broadening per epoch, a FWHM (km/s) added in quadrature to
        every line width and to the stellar dispersion (reference epoch = 0; True: up to 300 km/s, or
        the limit in km/s; an epoch may also be sharper than the reference). The resolution differs
        between epochs (slit width, seeing, another spectrograph) and is often not known well; the
        shared narrow lines measure it. Without it a difference of resolution changes the shared
        narrow lines' fit in every epoch and so its flux scale. The widths in the epochs' results are
        intrinsic: each epoch's spectrum carries its broadened resolution. ``result.broadening`` lists them.
    :param wave_range: fitted window (default: the model's).
    :param errors: 'covariance' or None.
    :param scale_errors: inflate errors by sqrt(reduced chi2) if > 1.
    :param n_jobs: processes for the independent fits that give the starting
        values (-1: all cores but one; 1: serial); see :func:`fantasy_agn.fit_many`.
    :return: MultiEpochResult
    """
    t0 = time.time()
    if errors not in ("covariance", None):
        raise ValueError("multi-epoch errors must be 'covariance' or None")
    model = model.copy()
    if shared is None:
        shared = ("narrow.*",) + tuple(f"{c.name}.*" for c in model.components if c.kind == "host")
    elif isinstance(shared, str):
        shared = (shared,)
    wave_range = wave_range or model.wave_range
    usable = []
    for s in spectra:
        try:
            _prepare(s, wave_range)
            usable.append(s)
        except ValueError:
            warnings.warn(f"skipping {s.name}: no usable pixels in the fitted range")
    if len(usable) < 2:
        raise ValueError("need at least two spectra with data in the fitted range")
    if reference == "snr":
        def _snr(sp):
            g = _prepare(sp, wave_range)
            return float(np.median(sp.flux[g] / sp.err[g]))
        ref = int(np.argmax([_snr(sp) for sp in usable]))
    else:
        ref = usable.index(spectra[reference])
    # The reference epoch is fitted first internally; results keep the input order.
    order = [ref] + [i for i in range(len(usable)) if i != ref]
    spectra = [usable[i] for i in order]
    s0, m0 = spectra[0], _prepare(spectra[0], wave_range)
    _fix_unconstrained(Problem(model, s0.wave[m0], s0.flux[m0], s0.err[m0], s0.context), model)

    # 1) independent fits give starting values for every epoch
    singles = fit_many(spectra, model, n_jobs=n_jobs, verbose=verbose, wave_range=wave_range,
                       errors=None)
    bad = [s.name for s, r in zip(spectra, singles) if isinstance(r, Exception)]
    if bad:
        raise RuntimeError(f"the independent fit failed for {', '.join(bad)}")
    masks = [r.mask for r in singles]
    if rescale_errors:
        factors = [float(np.sqrt(max(r.rchi2, 1.0))) for r in singles]
        spectra = [sp.copy(err=sp.err * f) for sp, f in zip(spectra, factors)]
        for sp, f in zip(spectra, factors):
            sp.meta["error_rescale"] = f
    epoch_scaled = tuple(epoch_scaled or ())
    for comp in epoch_scaled:
        if not any(_is_shared(p.name, shared) for p in model.params if p.name.startswith(comp + ".")):
            raise ValueError(f"epoch_scaled: share {comp!r} between epochs (shared=... '{comp}.*')")
    groups = None
    if calibration_groups:
        internal = {orig: k for k, orig in enumerate(order)}
        groups = [[internal[g] for g in grp if g in internal] for grp in calibration_groups]
    offset_limit = 150.0 if wavelength_offsets is True else float(wavelength_offsets or 0.0)
    if offset_limit < 0:
        raise ValueError("wavelength_offsets: True, or the limit of the offsets in km/s")
    broadening_limit = 300.0 if epoch_broadening is True else float(epoch_broadening or 0.0)
    if broadening_limit < 0:
        raise ValueError("epoch_broadening: True, or the limit of the broadening in km/s")
    prob = _JointProblem(model, spectra, masks, shared, intercalibrate, calibration_degree, wave_range,
                         epoch_scaled=epoch_scaled, calibration_limit=calibration_limit,
                         calibration_groups=groups, offset_limit=offset_limit,
                         broadening_limit=broadening_limit)

    full0 = np.zeros(len(prob.c))
    per_epoch = np.array([r.full for r in singles])
    ref = per_epoch[0]
    if intercalibrate:
        # Starting scale: median ratio of the shared amplitudes to the first epoch.
        lin = [i for i, p in enumerate(model.params) if p.linear and
               _is_shared(p.name, shared) and ref[i] > 0]
        for k, si in enumerate(prob.scale_index):
            # from the shared (narrow-line) amplitudes; if those are unusable (e.g. not
            # detected in this epoch's own fit), from the flux levels
            s0 = np.median(per_epoch[k, lin] / ref[lin]) if lin else np.nan
            g = prob.scale_guess[k]
            full0[si] = s0 if np.isfinite(s0) and g / 3 < s0 < g * 3 else g
    scales = np.array([full0[si] for si in prob.scale_index]) if intercalibrate else np.ones(len(spectra))
    # starting offsets: how far the shared velocity of each epoch's own fit is from the reference's
    dv0 = np.zeros(len(spectra))
    if offset_limit > 0:
        plist = list(model.params)
        anchor = next((i for i in prob.shift_pos if _is_shared(plist[i].name, shared) and not plist[i].fixed
                       and plist[i].name.startswith("narrow.")),
                      next(i for i in prob.shift_pos if _is_shared(plist[i].name, shared) and not plist[i].fixed))
        w = prob.shift_weight[list(prob.shift_pos).index(anchor)]
        dv0 = np.clip((per_epoch[:, anchor] - per_epoch[0, anchor]) / w, -0.9 * offset_limit, 0.9 * offset_limit)
        for k in range(len(spectra)):
            full0[prob.offset_index[k][0]] = dv0[k]
    shift_w = dict(zip(prob.shift_pos.tolist(), prob.shift_weight.tolist()))
    for i, p in enumerate(model.params):
        vals = per_epoch[:, i] / (scales if p.linear else 1.0)
        if i in shift_w:                               # in the frame of the reference epoch
            vals = vals - shift_w[i] * dv0
        for k in range(len(spectra)):
            full0[prob.index[k][i]] = vals[k]
        if _is_shared(p.name, shared):
            full0[prob.index[0][i]] = np.median(vals)
    for j, (comp, pos) in enumerate(prob.scaled):
        # starting amplitudes: the component's total weight in each independent fit, relative to
        # the shared (median) weights
        tot0 = np.sum(full0[prob.index[0][pos]])
        for k in range(len(spectra)):
            tot = np.sum(per_epoch[k, pos]) / scales[k]
            a = tot / tot0 if tot0 > 0 and tot > 0 else 1.0
            full0[prob.amp_index[k][j]] = 1.0 if k == 0 else float(np.clip(a, 0.06, 19.0))
    jn = prob.joint.names
    theta0 = np.array([full0[jn.index(p.name)] for p in prob.free])

    # 2) joint fit with a sparse Jacobian
    report("joint fit", 0, None, f"{len(theta0)} parameters, {len(spectra)} epochs")
    res = _levenberg_marquardt(prob, _interior(theta0, prob.lb, prob.ub, prob.linear))
    theta = res.x
    full = prob.full(theta)
    chi2 = float(res.fun @ res.fun)
    dof = max(res.fun.size - theta.size, 1)
    scale = max(chi2 / dof, 1.0) if scale_errors else 1.0
    limit = _at_limit(theta, prob.lb, prob.ub)

    cov_full = None
    offset_var, broad_var = {}, {}
    if errors == "covariance":
        blocks = _Blocks.from_epochs(len(theta), prob.group, ~limit, res.jac)
        per_epoch, shared_cov = blocks.covariance()
        T = prob.T.tocsr()
        # covariance of each epoch's full parameters (and of the shared ones) from
        # the blocks only: the full matrix over all epochs is never formed
        epoch_cov = {}
        for cols, cov in per_epoch:
            k = int(prob.group[cols[0]])
            idx = prob.index[k]
            rows = list(idx) + ([prob.scale_index[k]] if intercalibrate else []) + prob.cal_index[k]
            Tk = T[rows][:, cols].toarray()
            ck = Tk @ (cov * scale) @ Tk.T
            if prob.scaled or prob.offset_index[k] or prob.broad_index[k]:
                # the epoch's own parameters are functions of the joint ones: a scaled weight is a w
                # (d(a w_i)/dw_i = a, d(a w_i)/da = w_i), a shift is shift + weight * offset (the
                # widths of the epoch's result are the intrinsic ones: the broadening is its resolution)
                extra_rows = prob.amp_index[k] + prob.offset_index[k] + prob.broad_index[k]
                Ta = T[extra_rows][:, cols].toarray()
                Tall = np.vstack([Tk, Ta])
                call = Tall @ (cov * scale) @ Tall.T
                M = np.zeros((len(rows), len(rows) + len(extra_rows)))
                M[:, :len(rows)] = np.eye(len(rows))
                for j, ((comp, pos), ai) in enumerate(zip(prob.scaled, prob.amp_index[k])):
                    a = full[ai]
                    for i in pos:
                        M[i, i] = a
                        M[i, len(rows) + j] = full[idx[i]]
                if prob.offset_index[k]:
                    j = len(rows) + len(prob.amp_index[k])
                    M[prob.shift_pos, j] = prob.shift_weight
                    offset_var[k] = float(call[j, j])
                if prob.broad_index[k]:
                    j = len(rows) + len(prob.amp_index[k]) + len(prob.offset_index[k])
                    broad_var[k] = float(call[j, j])
                ck = M @ call @ M.T
            epoch_cov[k] = (rows, ck)
        cov_full = _EpochCovariance(epoch_cov, blocks.shared, shared_cov * scale, prob)

    epochs = []
    for k, s in enumerate(spectra):
        idx = prob.index[k]
        sk = prob.factor(k, full, s.wave) if intercalibrate else 1.0
        m = model.copy()
        full_k = prob.epoch_full(k, full, broadened=False)
        m.params.set_full_values(full_k)
        spec = prob.epoch_resolution(s.copy(flux=s.flux / sk, err=s.err / sk), k, full)
        r = res.fun[prob.rows[k]:prob.rows[k + 1]]
        free_k = [p.name for p in m.params if p.free]
        epochs.append(FitResult(
            spectrum=spec, model=m, mask=masks[k], full=full_k,
            cov=None if cov_full is None else cov_full.epoch(k, idx), samples=None,
            chi2=float(r @ r), dof=max(r.size - len(free_k), 1), nfev=res.nfev,
            success=res.success, message=res.message,
            # names without the epoch suffix, as in the epoch's own model
            at_limit=[p.name.rsplit("@", 1)[0] for p, a in zip(prob.free, limit) if a and not p.linear
                      and (p.name.endswith(f"@{k}") or "@" not in p.name)],
            at_zero=[p.name.rsplit("@", 1)[0] for p, a in zip(prob.free, limit) if a and p.linear
                     and (p.name.endswith(f"@{k}") or "@" not in p.name)],
            error_scale=np.sqrt(scale), time=0.0, error_method=errors))
        epochs[-1].calibration = np.array([full[i] for i in prob.cal_index[k]])
        epochs[-1].amplitudes = {comp: float(full[ai]) for (comp, _), ai in zip(prob.scaled, prob.amp_index[k])}
        if prob.offset_index[k]:
            oi = prob.offset_index[k][0]
            epochs[-1].offset = float(full[oi])
            epochs[-1].offset_err = (0.0 if k == 0 else float(np.sqrt(max(offset_var[k], 0))) if k in offset_var
                                     else np.nan)
        if prob.broad_index[k]:
            b = prob.epoch_broadening(k, full)
            var = 1e8 * broad_var.get(k, np.nan)          # of the signed FWHM^2, (km/s)^4
            sd = np.sqrt(max(var, 0.0)) if np.isfinite(var) else np.nan
            epochs[-1].broadening = b
            # error of the signed FWHM: d|b| = d(b^2) / 2|b|, kept finite near zero
            epochs[-1].broadening_err = 0.0 if k == 0 else float(sd / (2 * max(abs(b), np.sqrt(sd)))) if np.isfinite(sd) and sd > 0 else np.nan
        if intercalibrate:
            si = prob.scale_index[k]
            epochs[-1].scale = float(full[si])
            epochs[-1].scale_err = (float(np.sqrt(max(cov_full.variance(si, k), 0)))
                                    if cov_full is not None else np.nan)
        else:
            epochs[-1].scale, epochs[-1].scale_err = 1.0, 0.0
    in_order = [None] * len(epochs)
    for internal, original in enumerate(order):
        in_order[original] = epochs[internal]
    result = MultiEpochResult(prob, full, cov_full, in_order, chi2, dof, res, shared,
                              np.sqrt(scale), time.time() - t0)
    result.reference = order[0]
    result._order = order
    # how it was fitted, to repeat it (simulation_check)
    result.options = {"shared": tuple(shared), "intercalibrate": bool(intercalibrate), "wave_range": wave_range,
                      "errors": errors, "scale_errors": bool(scale_errors), "reference": order[0],
                      "calibration_degree": int(calibration_degree or 0), "epoch_scaled": tuple(epoch_scaled),
                      "rescale_errors": bool(rescale_errors), "calibration_limit": calibration_limit,
                      "calibration_groups": calibration_groups, "wavelength_offsets": offset_limit or False,
                      "epoch_broadening": broadening_limit or False}
    if intercalibrate:
        errs = [e.scale_err for k, e in enumerate(in_order) if k != result.reference and np.isfinite(e.scale_err)]
        result.calibration_error = float(np.median(errs)) if errs else np.nan
        if errs and result.calibration_error > 0.02:
            warnings.warn(
                f"the intercalibration is uncertain: median flux-scale error "
                f"{100 * result.calibration_error:.1f}% per epoch (the shared narrow lines are weak). "
                "This noise can hide real variability; compare with intercalibrate=False (the "
                "pipeline flux calibration), e.g. for SDSS spectra")
    if verbose:
        print(result.summary())
    return result


class _LMResult:
    pass


class _Blocks:
    """
    The normal matrix N = J^T J of the joint problem as blocks: one per epoch
    (its own parameters), the shared block and the epoch-shared couplings.
    Solving or inverting it block by block (Schur complement on the shared
    parameters) costs n_epochs x (block size)^3 instead of (all parameters)^3.
    """

    def __init__(self, N, group, keep):
        N = N.tocsr()
        self.n = N.shape[0]
        idx = np.arange(self.n)
        self.shared = idx[(group < 0) & keep]
        self.epochs = [idx[(group == k) & keep] for k in np.unique(group[group >= 0])]
        self.epochs = [e for e in self.epochs if e.size]
        self.C = N[self.shared][:, self.shared].toarray()
        self.A = [N[e][:, e].toarray() for e in self.epochs]
        self.B = [N[e][:, self.shared].toarray() for e in self.epochs]

    @classmethod
    def from_epochs(cls, n, group, keep, items):
        """The same blocks from per-epoch (free indices, J_k^T J_k) without a sparse N."""
        self = cls.__new__(cls)
        self.n = n
        idx = np.arange(n)
        self.shared = idx[(group < 0) & keep]
        spos = {j: i for i, j in enumerate(self.shared)}
        self.C = np.zeros((self.shared.size, self.shared.size))
        self.epochs, self.A, self.B = [], [], []
        for fc, M in items:
            kept = keep[fc]
            own = np.nonzero(kept & (group[fc] >= 0))[0]
            sh = np.nonzero(kept & (group[fc] < 0))[0]
            if sh.size:
                si = np.array([spos[j] for j in fc[sh]])
                self.C[np.ix_(si, si)] += M[np.ix_(sh, sh)]
            if own.size:
                self.epochs.append(fc[own])
                self.A.append(M[np.ix_(own, own)])
                B = np.zeros((own.size, self.shared.size))
                if sh.size:
                    B[:, si] = M[np.ix_(own, sh)]
                self.B.append(B)
        return self

    def solve(self, g, damp):
        """x with (N + diag(damp)) x = g on the kept parameters (0 elsewhere)."""
        x = np.zeros(self.n)
        s = self.shared
        S = self.C + np.diag(damp[s])
        rhs = g[s].copy()
        sol = []
        for e, A, B in zip(self.epochs, self.A, self.B):
            c = cho_factor(A + np.diag(damp[e]))
            AiB = cho_solve(c, B) if s.size else None
            Aig = cho_solve(c, g[e])
            if s.size:
                S -= B.T @ AiB
                rhs -= B.T @ Aig
            sol.append((e, AiB, Aig))
        xs = cho_solve(cho_factor(S), rhs) if s.size else np.zeros(0)
        x[s] = xs
        for e, AiB, Aig in sol:
            x[e] = Aig - (AiB @ xs if s.size else 0.0)
        return x

    @staticmethod
    def _pinv(M):
        d = np.sqrt(np.clip(np.diag(M), 1e-300, None))
        w, V = np.linalg.eigh(M / np.outer(d, d))
        good = w > w.max() * 1e-12 if w.size else w
        return ((V[:, good] / w[good]) @ V[:, good].T) / np.outer(d, d)

    def _inverses(self):
        Ainv = [self._pinv(A) for A in self.A]
        if self.shared.size:
            S = self.C - sum(B.T @ Ai @ B for Ai, B in zip(Ainv, self.B))
            Sss = self._pinv(S)
        else:
            Sss = np.zeros((0, 0))
        return Ainv, Sss

    def cross_terms(self):
        """
        Pieces of N^-1 between epochs: cov(theta_k, theta_j) = X_k Sss X_j^T
        (k != j) and cov(theta_k, shared) = -X_k Sss. Returns (own indices per
        epoch, X per epoch, Sss): the epochs are coupled only through the
        shared parameters, so the coupling has the rank of the shared block.
        """
        Ainv, Sss = self._inverses()
        return self.epochs, [Ai @ B for Ai, B in zip(Ainv, self.B)], Sss

    def covariance(self):
        """
        The blocks of N^-1 needed per epoch: {epoch position: (indices, cov)} for
        the epoch's parameters plus the shared ones, and the shared block. Uses
        a scaled eigen-decomposition (pseudo-inverse) for robustness.
        """
        s = self.shared
        Ainv, Sss = self._inverses()
        out = []
        for e, Ai, B in zip(self.epochs, Ainv, self.B):
            if s.size:
                X = Ai @ B
                Sks = -X @ Sss
                Skk = Ai + X @ Sss @ X.T
                cov = np.block([[Skk, Sks], [Sks.T, Sss]])
            else:
                cov = Ai
            out.append((np.concatenate([e, s]), cov))
        return out, Sss


def _levenberg_marquardt(prob, theta, max_iter=1000, ftol=1e-10, window=20, min_gain=0.1):
    """
    Bounded Levenberg-Marquardt using the normal equations J^T J, which stay
    small (free parameters squared) however many epochs there are. Parameters
    on a bound with the gradient pushing outward are held fixed for the step.
    Converged when a step no longer improves chi2, or when chi2 improved by less
    than ``min_gain`` over the last ``window`` iterations (statistically
    irrelevant; with many epochs the last digits can take hundreds of steps).
    """
    lb, ub = prob.lb, prob.ub
    n = len(theta)
    r, g, d, items = prob.normal(theta)
    cost = float(r @ r)
    lam, nfev, converged = 1e-3, 1, False
    costs = [cost]
    for _ in range(max_iter):
        d = np.where(d > 0, d, 1.0)
        active = ((theta <= lb) & (g > 0)) | ((theta >= ub) & (g < 0))
        blocks = _Blocks.from_epochs(n, prob.group, ~active, items)
        while True:
            try:
                step = -blocks.solve(g, lam * d)
            except np.linalg.LinAlgError:
                lam *= 10
                if lam > 1e14:
                    step = np.zeros_like(theta)
                    break
                continue
            trial = np.clip(theta + step, lb, ub)
            rt, gt, dt, it_t = prob.normal(trial)
            nfev += 1
            ct = float(rt @ rt)
            if ct < cost:
                break
            lam *= 4
            if lam > 1e14:
                break
        if ct >= cost:  # no improving step: at a minimum
            converged = True
            break
        improvement = cost - ct
        theta, r, g, d, items, cost = trial, rt, gt, dt, it_t, ct
        report("joint fit", nfev, None, f"chi2 = {cost:.1f}")
        lam = max(lam / 3, 1e-12)
        costs.append(cost)
        if improvement <= ftol * cost or (len(costs) > window and costs[-window - 1] - cost < min_gain):
            converged = True
            break
    res = _LMResult()
    res.x, res.fun, res.jac, res.nfev = theta, r, items, nfev
    res.success, res.message = converged, "converged" if converged else "maximum iterations"
    return res


class _EpochCovariance:
    """Covariance of the joint fit, kept per epoch (the full matrix would not fit in memory)."""

    def __init__(self, epoch_cov, shared_free, shared_cov, prob):
        self._epoch = epoch_cov
        self._prob = prob
        self._shared = {}
        T = prob.T.tocsr()
        if len(shared_free):
            # full joint parameters that depend only on shared free parameters
            sub = T[:, shared_free]
            full_cov_diag = np.einsum("ij,jk,ik->i", sub.toarray(), shared_cov, sub.toarray())
            for i in range(T.shape[0]):
                if sub[i].nnz and T[i].nnz == sub[i].nnz:
                    self._shared[i] = full_cov_diag[i]

    def epoch(self, k, idx):
        rows, cov = self._epoch.get(k, (None, None))
        if rows is None:
            return None
        pos = {r: j for j, r in enumerate(rows)}
        sel = [pos[i] for i in idx]
        return cov[np.ix_(sel, sel)]

    def variance(self, i, k=None):
        if i in self._shared:
            return self._shared[i]
        for kk, (rows, cov) in self._epoch.items():
            if k is not None and kk != k:
                continue
            if i in rows:
                j = rows.index(i)
                return cov[j, j]
        return np.nan

    def __getitem__(self, ij):
        i, j = ij
        if i != j:
            raise IndexError("only variances of the joint fit are kept (per-epoch covariances: me.epochs[k].cov)")
        return self.variance(i)


def _cov_from_normal(JtJ, exclude):
    n = JtJ.shape[0]
    cov = np.zeros((n, n))
    keep = ~exclude
    A = JtJ[np.ix_(keep, keep)]
    d = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    w, V = np.linalg.eigh(A / np.outer(d, d))
    good = w > w.max() * 1e-12
    inv = (V[:, good] / w[good]) @ V[:, good].T
    cov[np.ix_(keep, keep)] = inv / np.outer(d, d)
    return cov


def _line_continuum_corr(result, line, components, at):
    """Correlation coefficient of a line flux and the continuum flux density in one fit."""
    flux = result._line_flux_func(line, components)
    _, cov = result.propagate(lambda full: np.array([flux(full),
                                                     result._agn_continuum(at, full)[0]]), covariance=True,
                              depends=result._depends(components))
    d = np.sqrt(np.diag(cov))
    return float(np.clip(cov[0, 1] / (d[0] * d[1]), -1, 1)) if np.all(d > 0) else 0.0


# Rest-frame continuum windows free of strong lines, as in Shen et al. (2011, ApJS 194, 45): the blue
# H-beta window lies on the Fe II 4570 blend, so the fitted Fe II is subtracted before averaging
CONTINUUM_WINDOWS = {"Hb": ((4435.0, 4700.0), (5100.0, 5535.0)), "Ha": ((6000.0, 6250.0), (6800.0, 7000.0))}


class MultiEpochResult:
    """
    Result of :func:`fit_epochs`:

    - ``epochs``: list of per-epoch FitResult (intercalibrated)
    - ``shared``: table of the shared parameters
    - ``scales``: flux calibration factors per epoch
    - ``light_curve()``: broad-line fluxes/widths and continuum per epoch
    """

    def __init__(self, prob, full, cov, epochs, chi2, dof, res, shared, error_scale, time):
        self._prob, self.full, self.cov = prob, full, cov
        self.epochs, self.chi2, self.dof, self.rchi2 = epochs, chi2, dof, chi2 / dof
        self.success, self.nfev, self.shared_patterns = res.success, res.nfev, shared
        self.error_scale, self.time = error_scale, time

    def _err(self, name):
        if self.cov is None:
            return np.nan
        i = self._prob.joint.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0)))

    @property
    def shared(self):
        rows = []
        for p in self._prob.joint:
            if "@" not in p.name:
                i = self._prob.joint.names.index(p.name)
                rows.append((p.name, self.full[i], self._err(p.name), p.unit))
        return pd.DataFrame(rows, columns=["parameter", "value", "error", "unit"])

    @property
    def scales(self):
        """Flux calibration factor of each epoch relative to the reference epoch."""
        rows = [(k, r.spectrum.mjd, r.spectrum.name, r.scale, r.scale_err, k == self.reference)
                for k, r in enumerate(self.epochs)]
        return pd.DataFrame(rows, columns=["epoch", "mjd", "name", "scale", "error", "reference"])

    @property
    def offsets(self):
        """Wavelength offset of each epoch (km/s, with ``wavelength_offsets``) relative to the reference epoch;
        None if they were not fitted."""
        if not any(hasattr(r, "offset") for r in self.epochs):
            return None
        rows = [(k, r.spectrum.mjd, r.spectrum.name, getattr(r, "offset", np.nan), getattr(r, "offset_err", np.nan),
                 k == self.reference) for k, r in enumerate(self.epochs)]
        return pd.DataFrame(rows, columns=["epoch", "mjd", "name", "offset", "error", "reference"])

    @property
    def broadening(self):
        """Extra broadening of each epoch (FWHM in km/s added in quadrature, negative: sharper; with
        ``epoch_broadening``) relative to the reference epoch, and the resolution it gives; None if not fitted."""
        if not any(hasattr(r, "broadening") for r in self.epochs):
            return None
        rows = [(k, r.spectrum.mjd, r.spectrum.name, getattr(r, "broadening", np.nan), getattr(r, "broadening_err", np.nan),
                 float(r.spectrum.resolution or 0.0), k == self.reference) for k, r in enumerate(self.epochs)]
        return pd.DataFrame(rows, columns=["epoch", "mjd", "name", "broadening", "error", "resolution", "reference"])

    def redshift(self, parameter="narrow.shift"):
        """
        Redshift of the object from the joint fit: the redshift the reference epoch was moved to the
        rest frame with, corrected by the shared velocity ``parameter`` (default: the shift of the
        narrow lines, which all epochs measure together). Returns (z, error); the error is that of the
        fit only, without the wavelength calibration of the reference spectrum (a few km/s for SDSS,
        1e-5 in z). Use it to prepare the spectra again with one redshift, or as the systemic redshift.
        """
        ref = self.epochs[getattr(self, "reference", 0)]
        z0 = ref.spectrum.z
        if z0 is None:
            raise ValueError("the reference spectrum has no redshift")
        names = self._prob.joint.names
        if parameter not in names:
            raise ValueError(f"{parameter!r} is not shared by the epochs: give a shared velocity "
                             f"(one of {[n for n in names if '@' not in n and n.endswith(('shift', 'velocity'))]})")
        v = float(self.full[names.index(parameter)])
        return (1 + z0) * (1 + v / C_KMS) - 1, (1 + z0) * self._err(parameter) / C_KMS

    def _epoch_select(self, mjd_range):
        if mjd_range is None:
            return list(range(len(self.epochs)))
        lo, hi = mjd_range
        sel = [k for k, e in enumerate(self.epochs) if e.spectrum.mjd is not None and lo <= e.spectrum.mjd <= hi]
        if len(sel) < 5:
            raise ValueError(f"only {len(sel)} epochs in MJD {lo}-{hi}")
        return sel

    def window_continuum(self, windows=None, subtract_lines=True, min_coverage=0.8):
        """
        Continuum light curves measured in rest-frame windows: per epoch, the mean
        flux density of the intercalibrated spectrum in each window, minus the
        fitted emission lines and Fe II (``subtract_lines``), with the error of the
        mean from the pixel errors. Unlike :meth:`light_curve`, which evaluates the
        fitted AGN continuum at one wavelength, this is measured on the data, as in
        reverberation-mapping campaigns; it includes the host starlight (constant:
        it dilutes the variability, not the lag).

        :param windows: list of (lo, hi) in Å; default: the windows of
            :data:`CONTINUUM_WINDOWS` (Shen et al. 2011) inside the fitted range
        :param min_coverage: share of a window's pixels an epoch must have, else NaN
        :returns: DataFrame with ``mjd`` and ``win{lo}_{hi}``, ``win{lo}_{hi}_err``
        """
        lo_fit = min(float(e.spectrum.wave[e.mask].min()) for e in self.epochs)
        hi_fit = max(float(e.spectrum.wave[e.mask].max()) for e in self.epochs)
        if windows is None:
            windows = [w for ws in CONTINUUM_WINDOWS.values() for w in ws if w[0] >= lo_fit - 1 and w[1] <= hi_fit + 1]
        rows = []
        for e in self.epochs:
            s = e.spectrum
            wave = s.wave if s.rest_frame or not s.z else s.wave / (1 + s.z)
            sub = 0.0
            if subtract_lines:
                parts = e.model_flux(s.wave, components=True)
                kinds = {c.name: c.kind for c in e.model.components}
                sub = sum((v for k, v in parts.items() if kinds.get(k) in ("line", "template")), np.zeros_like(wave))
            row = {"mjd": s.mjd if s.mjd is not None else np.nan}
            for lo, hi in windows:
                inw = (wave >= lo) & (wave <= hi)
                ok = inw & e.mask & np.isfinite(s.flux) & np.isfinite(s.err) & (s.err > 0)
                key = f"win{lo:g}_{hi:g}"
                if inw.sum() == 0 or ok.sum() < max(3, min_coverage * inw.sum()):
                    row[key], row[key + "_err"] = np.nan, np.nan
                    continue
                f = (s.flux - sub)[ok]
                row[key] = float(np.mean(f))
                row[key + "_err"] = float(np.sqrt(np.sum(s.err[ok] ** 2)) / ok.sum())
            rows.append(row)
        return pd.DataFrame(rows)

    def rms_spectrum(self, line="Hb", mjd_range=None, **kw):
        """Mean and rms spectrum of a broad line (see :func:`fantasy_agn.rm.rms_spectrum`)."""
        from .rm import rms_spectrum
        return rms_spectrum([self.epochs[k] for k in self._epoch_select(mjd_range)], line, **kw)

    def reverberation(self, line="Hb", continuum=None, lag_range=None, dt=None, n_mc=1000,
                      f_virial=4.47, mjd_range=None, detrend=False, methods=("iccf", "vn", "drw"),
                      primary="iccf", width=0.0, seed=0):
        """
        Lag of the broad ``line`` behind the continuum, its width in the rms
        spectrum and the reverberation mass; returns a
        :class:`fantasy_agn.rm.Reverberation` (print it, ``.table()``, ``.plot()``).

        Lag methods (``methods``; ``primary`` gives the adopted lag): 'iccf'
        (interpolated cross-correlation, FR/RSS errors), 'vn' (von Neumann
        estimator), 'zdcf' (z-transformed discrete correlation function,
        Alexander 1997) and 'drw' (damped-random-walk model with the correlated
        line and continuum errors of each epoch's fit, see
        :func:`fantasy_agn.rm.gp_lag`). All keep the primary peak of the lag
        distribution weighted for the light-curve overlap (seasonal aliases).

        :param continuum: None: the fitted AGN continuum light curve of these
            epochs; or (mjd, flux, error) arrays, e.g. photometry (better sampled).
        :param mjd_range: (min, max): only epochs (and continuum points) in this
            window, e.g. one observing season; lags need dense sampling.
        :param lag_range: (min, max) observed-frame lag to search, days (default:
            +-5x the lag expected from L5100 and the R-L relation, >= 60 d).
        :param detrend: subtract a linear trend from both light curves (Welsh
            1999): slow variations otherwise dominate the cross-correlation.
        """
        from .rm import Reverberation, ccf_lag, gp_lag, vn_lag, zdcf_lag
        lc = self.light_curve(lines=[line]).iloc[self._epoch_select(mjd_range)]
        if lc["mjd"].isna().any():
            raise ValueError("the epochs need MJDs for a lag")
        if f"{line}_flux" not in lc:
            raise ValueError(f"no broad {line} in the model")
        t2, f2, e2 = lc["mjd"].to_numpy(float), lc[f"{line}_flux"].to_numpy(float), lc[f"{line}_flux_err"].to_numpy(float)
        if continuum is None:
            cc = [c for c in lc.columns if c.startswith("cont") and not c.endswith("_err")]
            if not cc:
                raise ValueError("no continuum light curve in the fit; give continuum=(mjd, flux, error)")
            t1, f1, e1 = t2, lc[cc[0]].to_numpy(float), lc[cc[0] + "_err"].to_numpy(float)
            label = f"continuum {cc[0][4:]} A"
        else:
            t1, f1, e1 = (np.asarray(a, float) for a in continuum)
            label = "continuum (given)"
        if mjd_range is not None and continuum is not None:
            keep = (t1 >= mjd_range[0]) & (t1 <= mjd_range[1])
            t1, f1, e1 = t1[keep], f1[keep], e1[keep]
        if detrend:
            def flat(t, f):
                ok = np.isfinite(f)
                a, b = np.polyfit(t[ok], f[ok], 1)
                return f - (a * t + b) + np.nanmean(f)
            f1, f2 = flat(t1, f1), flat(t2, f2)
            label += ", detrended"
        z = self.epochs[0].spectrum.z
        if lag_range is None:
            # +-5x the lag expected from the R-L relation (Bentz et al. 2013), at least
            # 60 d and at most half the overlap: long baselines with seasonal gaps
            # otherwise produce spurious lags of hundreds of days
            span = min(t1.max(), t2.max()) - max(t1.min(), t2.min())
            lmax = 0.5 * span
            lcol = [c for c in lc.columns if c.startswith("logL") and not c.endswith("_err")]
            if lcol and np.isfinite(lc[lcol[0]]).any():
                tau = 10 ** (1.527 + 0.533 * (np.nanmedian(lc[lcol[0]]) - 44.0)) * (1 + (z or 0))
                lmax = min(lmax, max(5 * tau, 60.0))
            lag_range = (-lmax, lmax)
        results = {}
        if "iccf" in methods or primary == "iccf":
            results["iccf"] = ccf_lag(t1, f1, e1, t2, f2, e2, lag_range=lag_range, dt=dt, n_mc=n_mc, seed=seed)
        if "vn" in methods or primary == "vn":
            results["vn"] = vn_lag(t1, f1, e1, t2, f2, e2, lag_range=lag_range, dt=max(dt or 1.0, 0.5),
                                   n_mc=min(n_mc, 300), seed=seed)
        if "zdcf" in methods or primary == "zdcf":
            results["zdcf"] = zdcf_lag(t1, f1, e1, t2, f2, e2, lag_range=lag_range, n_mc=min(n_mc, 200), seed=seed)
        if "drw" in methods or primary == "drw":
            cc = [c for c in lc.columns if c.startswith("cont") and not c.endswith("_err")]
            spec = (t2, lc[cc[0]].to_numpy(float), lc[cc[0] + "_err"].to_numpy(float)) if cc else None
            corr = lc[f"{line}_cont_corr"].to_numpy(float) if f"{line}_cont_corr" in lc else None
            results["drw"] = gp_lag((t2, f2, e2), continuum=None if continuum is None else (t1, f1, e1),
                                    spec_continuum=spec, corr=corr, lag_range=lag_range,
                                    dt=max(dt or 1.0, 0.5), width=width, seed=seed)
        rms = self.rms_spectrum(line, mjd_range=mjd_range)
        return Reverberation(line, results[primary], rms, label, z=z, f_virial=f_virial,
                             lc=((t1, f1, e1), (t2, f2, e2)), methods=results)

    def quantity_covariance(self, func):
        """
        Values of ``func(epoch_result, full) -> array (m,)`` in every epoch and
        their joint covariance over all epochs, ``(values (n, m), cov (n*m, n*m))``
        with rows epoch-major in input order. Within an epoch the covariance is
        that of the epoch's fit (decomposition, flux calibration); between
        epochs it comes from the shared parameters (e.g. a shared line shape
        or narrow-line fluxes): linear propagation, finite-difference gradients.
        """
        from .fitting import _at_limit
        prob = self._prob
        n = len(self.epochs)
        order = getattr(self, "_order", None) or [self.reference] + [i for i in range(n) if i != self.reference]
        jn = prob.joint.names
        theta = np.array([self.full[jn.index(p.name)] for p in prob.free])
        items = prob.normal(theta)[3]
        limit = _at_limit(theta, prob.lb, prob.ub)
        blocks = _Blocks.from_epochs(len(theta), prob.group, ~limit, items)
        own, X, Sss = blocks.cross_terms()
        Sss = Sss * self.error_scale ** 2
        T = prob.T.tocsr()
        values, U, D = [None] * n, [None] * n, [None] * n
        for e, cols in enumerate(own):
            k = int(prob.group[cols[0]])
            r = self.epochs[order[k]]
            idx = prob.index[k]
            full_k = np.asarray(r.full, float)
            v = np.atleast_1d(np.asarray(func(r, full_k), float))
            sig = r._sigma()
            lo = np.array([p.min for p in r.model.params], float)
            hi = np.array([p.max for p in r.model.params], float)
            grad = np.zeros((v.size, full_k.size))
            for i in np.nonzero(np.nan_to_num(sig) > 0)[0]:
                a, b = full_k.copy(), full_k.copy()
                a[i] = min(a[i] + 0.1 * sig[i], hi[i])
                b[i] = max(b[i] - 0.1 * sig[i], lo[i])
                if a[i] > b[i]:
                    grad[:, i] = (np.asarray(func(r, a), float) - np.asarray(func(r, b), float)) / (a[i] - b[i])
            G = grad @ T[idx].toarray()
            values[order[k]] = v
            U[order[k]] = G[:, cols] @ X[e] - G[:, blocks.shared]
            D[order[k]] = grad @ np.nan_to_num(r.cov) @ grad.T if r.cov is not None else np.zeros((v.size, v.size))
        m = values[0].size
        cov = np.zeros((n * m, n * m))
        Ust = np.vstack(U) if Sss.size else None
        if Ust is not None:
            cov += Ust @ Sss @ Ust.T
        for k in range(n):
            sl = slice(k * m, (k + 1) * m)
            cov[sl, sl] = D[k]
        return np.array(values), cov

    def light_curve(self, lines=None, continuum_at=None):
        """
        Table with, per epoch, the total broad flux and FWHM of each broad line
        (summed over broad components) and the continuum flux density.
        """
        # computed once per choice (it propagates errors of every line in every epoch)
        key = (None if lines is None else tuple(lines), continuum_at)
        cache = self.__dict__.setdefault("_lc_cache", {})
        if key not in cache:
            cache[key] = self._light_curve(lines, continuum_at)
        return cache[key].copy()

    def has_light_curve(self):
        """True when the default light curve was already computed (cheap to show)."""
        return (None, None) in self.__dict__.get("_lc_cache", {})

    def _light_curve(self, lines, continuum_at):
        return light_curve_table(self.epochs, lines, continuum_at)

    def plot_baldwin(self, line="Hb", continuum_at=None, exclude=(), save=None, **kw):
        """
        Intrinsic Baldwin effect: equivalent width of the broad ``line`` against
        the continuum luminosity of each epoch, with a power-law fit (see
        :func:`fantasy_agn.plotting.plot_baldwin`). ``exclude`` lists epoch
        indices to leave out. Returns (figure, fit result).
        """
        from .plotting import plot_baldwin, savefig
        table = self.light_curve(lines=[line], continuum_at=continuum_at)
        table = table[~table.epoch.isin(list(exclude))]
        fig, fit = plot_baldwin(table, line, **kw)
        if save:
            savefig(fig, save)
        return fig, fit

    def plot_light_curves(self, columns=None, fig=None, width="double", save=None):
        """
        Light curves (broad-line fluxes and continuum by default), one panel
        each; ``save="lc.pdf"`` also writes the figure (pdf, eps, svg, png).
        """
        from .plotting import plot_light_curves, savefig
        table = self.light_curve()
        if columns is None:
            columns = [c for c in table.columns if c.endswith(("_flux",)) or c.startswith("cont")]
            columns = [c for c in columns if not c.endswith("_err")]
        out = plot_light_curves(table, columns, fig=fig, width=width,
                                flux_unit=self.epochs[0].spectrum.flux_unit)
        if save:
            savefig(out, save)
        return out

    def plot(self, epoch=0, **kw):
        return self.epochs[epoch].plot(**kw)

    def summary(self):
        lines = [f"Joint fit of {len(self.epochs)} epochs: chi2 = {self.chi2:.1f}, dof = {self.dof}, "
                 f"reduced chi2 = {self.rchi2:.3f}, {self.nfev} evaluations, {self.time:.1f} s",
                 f"  flux reference: epoch {getattr(self, 'reference', 0)} "
                 f"({self.epochs[getattr(self, 'reference', 0)].spectrum.name})",
                 f"  shared: {', '.join(self.shared_patterns)}"]
        if not self.success:
            lines.append("  WARNING: optimizer did not converge")
        if self.error_scale > 1:
            lines.append(f"  errors scaled by sqrt(reduced chi2) = {self.error_scale:.2f}")
        off = self.offsets
        if off is not None:
            v = off.offset[~off.reference].to_numpy(float)
            lines.append(f"  wavelength offsets per epoch: rms {np.sqrt(np.mean(v ** 2)):.1f} km/s, from {round(v.min()) + 0.0:+.0f} to "
                         f"{round(v.max()) + 0.0:+.0f} km/s (median error {np.nanmedian(off.error[~off.reference]):.1f} km/s)")
            limit = getattr(self._prob, "offset_limit", 0.0)
            at = int(np.sum(np.abs(v) > 0.98 * limit)) if limit else 0
            if at:
                lines.append(f"  WARNING: {at} offsets are at the limit of {limit:g} km/s: the wavelength scale of those "
                             "epochs is far off (or their narrow lines are not detected)")
        br = self.broadening
        if br is not None:
            v = br.broadening[~br.reference].to_numpy(float)
            lines.append(f"  extra broadening per epoch (FWHM in quadrature): from {round(v.min()) + 0.0:+.0f} to "
                         f"{round(v.max()) + 0.0:+.0f} km/s, median |b| {np.median(np.abs(v)):.0f} km/s "
                         f"(median error {np.nanmedian(br.error[~br.reference]):.0f} km/s)")
        try:
            z, ze = self.redshift()
            lines.append(f"  redshift of the narrow lines: {z:.6f} +- {ze:.6f}")
        except Exception:                                         # no redshift, or no shared narrow shift
            pass
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def light_curve_table(epochs, lines=None, continuum_at=None):
    """
    Per epoch (a list of FitResults with the same model), the total broad flux, FWHM and EW of each broad line
    (summed over broad components), the continuum flux density at the continuum pivot and, with a redshift and a
    flux unit, log lambda L there; used by MultiEpochResult.light_curve and for epochs fitted one by one.
    """
    first = epochs[0]
    broad = [c for c in first._line_components() if c.role == "broad"]
    if lines is None:
        lines = sorted({ln for c in broad for ln in c.line_names},
                       key=lambda ln: [c.line_names for c in broad if ln in c.line_names][0].index(ln))
    if continuum_at is None:
        pivots = [getattr(c, "pivot", None) for c in first.model.components
                  if c.kind == "continuum"]
        continuum_at = next((p for p in pivots if p), None)
    rows = []
    for k, r in enumerate(epochs):
        row = {"epoch": k, "mjd": r.spectrum.mjd, "name": r.spectrum.name}
        for ln in lines:
            comps = [c.name for c in broad if ln in c.line_names]
            m = r.measure(ln, comps)
            row[f"{ln}_flux"], row[f"{ln}_flux_err"] = m["flux"], m["flux_err"]
            row[f"{ln}_fwhm"], row[f"{ln}_fwhm_err"] = m["fwhm"], m["fwhm_err"]
            row[f"{ln}_ew"], row[f"{ln}_ew_err"] = m["ew"], m["ew_err"]
        if continuum_at:
            row[f"cont{continuum_at:g}"], row[f"cont{continuum_at:g}_err"] = r.continuum(continuum_at)
            for ln in lines:
                # correlation of the line flux with the continuum in this epoch's fit
                # (decomposition and flux calibration): used by the lag methods
                row[f"{ln}_cont_corr"] = _line_continuum_corr(r, ln, [c.name for c in broad
                                                                      if ln in c.line_names], continuum_at)
            if r.spectrum.z and r.spectrum.flux_unit:
                row[f"logL{continuum_at:g}"], row[f"logL{continuum_at:g}_err"] = \
                    r.luminosity(continuum_at)
        rows.append(row)
    return pd.DataFrame(rows)


class EpochSet(MultiEpochResult):
    """
    Epochs fitted one by one (``fa.fit_many``) seen together, with the tools of a joint fit that need only the
    epochs: ``light_curve``, ``rms_spectrum``, ``reverberation`` (lags, rms widths, mass) and ``plot_baldwin``::

        fits = fa.fit_many(spectra, model)
        me = fa.EpochSet(fits)
        rm = me.reverberation("Hb")

    Unlike ``fa.fit_epochs`` nothing is shared and the epochs are not intercalibrated on the narrow lines: the
    flux calibration of each spectrum is taken as it is. Epochs whose fit failed are left out.
    """

    def __init__(self, fits):
        from .result import FitResult
        self.epochs = [f for f in fits if isinstance(f, FitResult)]
        if len(self.epochs) < 2:
            raise ValueError("an EpochSet needs at least two fitted epochs")
        self._prob, self.full, self.cov = None, None, None
        self.chi2 = float(sum(e.chi2 for e in self.epochs))
        self.dof = int(sum(e.dof for e in self.epochs))
        self.rchi2 = self.chi2 / self.dof
        self.success = all(e.success for e in self.epochs)
        self.nfev = int(sum(e.nfev for e in self.epochs))
        self.shared_patterns, self.error_scale = (), 1.0
        self.time = float(sum(e.time for e in self.epochs))
        self.reference = 0

    def summary(self, short=False):
        return (f"{len(self.epochs)} epochs fitted one by one (not intercalibrated): "
                f"reduced chi2 {min(e.rchi2 for e in self.epochs):.3f}-{max(e.rchi2 for e in self.epochs):.3f}")
