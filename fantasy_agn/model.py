"""The Model: a sum of components with a shared, tie-aware parameter set."""
import copy

import numpy as np

from .components import Component
from .parameters import ParameterSet

__all__ = ("Model", "load_model")

MODEL_FORMAT = 1         # version of the saved-model format (Model.to_dict)


class Model:
    """
    Sum of components. Build it from components and combine with ``+``::

        model = Model(PowerLaw(), FeII(), Lines("narrow", [...]), Lines("broad", [...]))
        model["broad.fwhm"].set(value=4000, min=1500, max=12000)
        model.tie("broad.Hg.fwhm", "broad.Hb.fwhm")
        model.fix("feii.shift", 0)

    Parameter names are ``component.parameter``; ``set``, ``fix``, ``free`` and
    ``tie`` also accept shell-style patterns such as ``"broad.*.flux"``.
    """

    def __init__(self, *components, wave_range=None):
        self.components = []
        self.params = ParameterSet()
        self.wave_range = wave_range  # default fitting window (used by fit)
        for c in components:
            self.add(c)

    # ------------------------------------------------------------ building
    def add(self, component):
        if isinstance(component, Model):
            # its ties as they are (adding a component applies its default ties again, also ones
            # that were removed with untie)
            ties = {p.name: p.tie for p in component.params}
            for c in component.components:
                self.add(c)
            for name, tie in ties.items():
                self.params[name].tie = tie
            return self
        if not isinstance(component, Component):
            raise TypeError(f"cannot add {type(component).__name__} to a Model")
        if any(c.name == component.name for c in self.components):
            raise ValueError(f"a component named {component.name!r} already exists")
        self.components.append(component)
        for p in component.params:
            self.params.add(p)
        for name, target, ratio in getattr(component, "default_ties", lambda: [])():
            self.params.tie(name, target, factor=ratio)
        return self

    # ------------------------------------------------------- save, load
    def to_dict(self):
        """
        The model as a recipe of plain data (JSON-compatible): for each component its class and
        the arguments it was made with, then every parameter's value, limits, fixed flag and tie.
        :meth:`from_dict` rebuilds the same model; :meth:`save` writes it to a JSON file.
        """
        comps = []
        for c in self.components:
            recipe = c.__dict__.get("_recipe")
            if recipe is None:
                raise ValueError(f"component {c.name!r} cannot be saved (made by an older version or by hand)")
            kind, args, kwargs = recipe
            comps.append({"type": kind, "args": _plain(args), "kwargs": _plain(kwargs)})
        params = [{"name": p.name, "value": _plain(p.value), "min": _plain(p.min), "max": _plain(p.max),
                   "fixed": bool(p.fixed),
                   "tie": None if p.tie is None else [p.tie[0], float(p.tie[1]), float(p.tie[2])]}
                  for p in self.params]
        return {"fantasy_model": MODEL_FORMAT, "wave_range": _plain(self.wave_range),
                "components": comps, "params": params}

    @classmethod
    def from_dict(cls, d):
        """The model described by :meth:`to_dict` (only FANTASY components are made)."""
        from . import components as C
        from . import host
        if not isinstance(d, dict) or "components" not in d:
            raise ValueError("not a FANTASY model")
        if int(d.get("fantasy_model", 1)) > MODEL_FORMAT:
            raise ValueError("this model was saved by a newer FANTASY version")
        known = {k: v for mod in (C, host) for k, v in vars(mod).items()
                 if isinstance(v, type) and issubclass(v, Component) and not k.startswith("_")}
        m = cls(wave_range=_tuples(d.get("wave_range")))
        for c in d["components"]:
            kind = c.get("type")
            if kind not in known:
                raise ValueError(f"unknown component type {kind!r}")
            m.add(known[kind](*_tuples(c.get("args") or []), **_tuples(c.get("kwargs") or {})))
        inf = float("inf")
        for q in d.get("params") or []:
            if q["name"] not in m.params.names:
                raise ValueError(f"parameter {q['name']!r} does not belong to the model")
            p = m.params[q["name"]]
            p.min = -inf if q.get("min") is None else float(q["min"])     # float("-inf") also reads "-inf"
            p.max = inf if q.get("max") is None else float(q["max"])
            p.value = float(q["value"])
            p.fixed = bool(q.get("fixed"))
            p.tie = None
        for q in d.get("params") or []:
            if q.get("tie"):
                to, factor, offset = q["tie"]
                m.params.tie(q["name"], to, factor=factor, offset=offset)
        m.params._resolve()
        return m

    def save(self, path):
        """Write the model to a JSON file (see :meth:`to_dict`); read it back with :func:`load_model`."""
        import json
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    def __add__(self, other):
        new = self.copy()
        return new.add(copy.deepcopy(other))

    def __radd__(self, other):
        if other == 0:  # allows sum([...])
            return self.copy()
        return NotImplemented

    def __getitem__(self, name):
        if any(c.name == name for c in self.components):
            return self.component(name)
        return self.params[name]

    def component(self, name):
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"no component {name!r}; components: {[c.name for c in self.components]}")

    def remove(self, name):
        """Remove a component (and any ties pointing to it)."""
        comp = self.component(name)
        self.components.remove(comp)
        kept = [p for p in self.params if not p.name.startswith(name + ".")]
        self.params = ParameterSet(kept)
        for p in self.params:
            if p.tie and p.tie[0].startswith(name + "."):
                p.tie = None
        return self

    # parameter editing shortcuts
    def set(self, pattern, value=None, min=None, max=None):
        self.params.set(pattern, value=value, min=min, max=max)
        return self

    def fix(self, pattern, value=None):
        self.params.fix(pattern, value)
        return self

    def free(self, pattern):
        self.params.free(pattern)
        return self

    def tie(self, pattern, to, factor=1.0, offset=0.0):
        self.params.tie(pattern, to, factor, offset)
        return self

    def untie(self, pattern):
        """Free tied parameter(s) again (they keep their current value)."""
        self.params.untie(pattern)
        return self

    def set_profile(self, name, profile, **options):
        """
        Change the line profile of the Lines component ``name`` ('gaussian', 'lorentzian', 'voigt',
        'gauss-hermite' or 'stratified'), keeping its lines, flux ratios and role. Parameters
        present in both profiles (fluxes, shift, fwhm) keep their values, limits and ties; ``options``
        are passed to :class:`Lines` (e.g. ``h3=(0, -0.2, 0.2)``)::

            model.set_profile("broad", "lorentzian")
        """
        from .components import Lines
        old = self.component(name)
        args = getattr(old, "_args", None)
        if not isinstance(old, Lines) or args is None:
            raise ValueError(f"{name!r} is not a Lines component made by this FANTASY version: "
                             "remove it and add it again")
        new = Lines(name, **{**args, "profile": profile, **options})
        if "role" not in options:
            new.role = old.role
        before = {p.name: p for p in self.params if p.name.startswith(name + ".")}
        for p in new.params:
            q = before.get(p.name)
            if q is not None and q.linear == p.linear:
                p.value, p.min, p.max, p.fixed, p.tie = q.value, q.min, q.max, q.fixed, q.tie
        self.components[self.components.index(old)] = new
        self.params = ParameterSet([p for c in self.components for p in c.params])
        for p in self.params:                      # ties to parameters the new profile has not
            if p.tie and p.tie[0] not in self.params:
                p.tie = None
        self.params._resolve()
        return self

    def split_lines(self, name, lines, new_name):
        """
        Move ``lines`` of the Lines component ``name`` into a new component ``new_name`` (same profile
        and role) with its own velocity shift and width, which start at the values and limits of
        ``name`` and are free (also when those of ``name`` are tied or fixed). Their fluxes, flux
        ratios, limits and ties go with them, and ties to them follow.
        For a line whose velocity or width differs from the others of its group, e.g. [O III] that is
        blueshifted against the other narrow lines::

            model.split_lines("narrow", ["[OIII]4959", "[OIII]5007"], "narrow_oiii")
        """
        from .components import Lines
        old = self.component(name)
        args = getattr(old, "_args", None)
        if not isinstance(old, Lines) or args is None:
            raise ValueError(f"{name!r} is not a Lines component made by this FANTASY version")
        if any(c.name == new_name for c in self.components):
            raise ValueError(f"a component named {new_name!r} already exists")
        want = set(lines)
        unknown = sorted(want - set(old.line_names))
        if unknown:
            raise ValueError(f"{name!r} has no line {', '.join(unknown)}")
        move = [n for n in old.line_names if n in want]
        keep = [n for n in old.line_names if n not in want]
        if not move or not keep:
            raise ValueError("move some of the lines, not none or all (rename the component instead)")
        nm = lambda item: item if isinstance(item, str) else str(item[0])

        def part(d, names):
            out = {}
            for a, v in (d or {}).items():
                if (a in want) != (v[0] in want):
                    raise ValueError(f"{a} and {v[0]} have a flux ratio: move both or neither")
                if a in names:
                    out[a] = v
            return out or None

        def make(cname, names):
            a = {**args, "lines": [it for it in args["lines"] if nm(it) in names]}
            for key in ("ratios", "line_ratios", "max_ratios"):
                a[key] = part(args.get(key), names)
            c = Lines(cname, profile=old.profile, **a)
            c.role = old.role
            return c

        kept, moved = make(name, keep), make(new_name, move)
        line_of = lambda pname: next((n for n in move if pname.startswith(f"{name}.{n}.")), None)
        target = lambda pname: new_name + pname[len(name):] if line_of(pname) else pname
        before = {p.name: p for p in self.params}
        own = {f"{new_name}.shift", f"{new_name}.fwhm"}     # the point of separating: free, even if name's are tied
        for c, src in ((kept, lambda n: n), (moved, lambda n: name + n[len(new_name):])):
            for p in c.params:
                q = before.get(src(p.name))
                if q is not None and q.linear == p.linear:
                    p.value, p.min, p.max = q.value, q.min, q.max
                    p.fixed, p.tie = (False, None) if p.name in own else (q.fixed, q.tie)
        i = self.components.index(old)
        self.components[i:i + 1] = [kept, moved]
        self.params = ParameterSet([p for c in self.components for p in c.params])
        for p in self.params:
            if p.tie:
                to = target(p.tie[0])
                p.tie = (to,) + tuple(p.tie[1:]) if to in self.params and to != p.name else None
        self.params._resolve()
        return self

    def starting_from(self, result):
        """
        A copy of the model whose free parameters start at the values of a fit (``result``, a
        FitResult, or a fitted Model), wherever the two have a parameter of the same name; the
        limits, fixed values and ties stay those of this model. A fit of the copy tries these values
        as a start besides the data-driven guess and keeps the better solution, so it never ends
        worse than ``result``. Use it to carry a model worked out on one epoch to the others::

            r = fa.fit(spectra[3], model)                       # agree on the model for one epoch
            me = fa.fit_epochs(spectra, model.starting_from(r)) # then fit them all from there
        """
        source = getattr(result, "model", result)
        if not isinstance(source, Model):
            raise TypeError("starting_from needs a FitResult or a Model")
        new = self.copy()
        used = 0
        for p in new.params:
            if p.free and p.name in source.params:
                p.value = float(min(max(source.params[p.name].value, p.min), p.max))
                used += 1
        if not used:
            raise ValueError("the fit has no parameter in common with this model")
        new._fitted = True                     # fit() also starts from these values
        return new

    def copy(self):
        return copy.deepcopy(self)

    # ---------------------------------------------------------- evaluation
    def _slices(self):
        out, i = [], 0
        for c in self.components:
            out.append(slice(i, i + len(c.params)))
            i += len(c.params)
        return out

    def _attenuation(self):
        idx = [i for i, c in enumerate(self.components) if c.kind == "attenuation"]
        if len(idx) > 1:
            raise ValueError("only one Extinction component is supported")
        return idx[0] if idx else None

    # Multiplicative components: an Extinction dims everything but the host; an
    # Absorption (troughs of absorbers in front of the AGN) and a Calibration
    # (smooth flux-calibration correction) act on everything.
    _MULTIPLY = ("attenuation", "absorption", "calibration")

    @staticmethod
    def _applies(mult, comp):
        return comp.kind != "host" if mult.kind == "attenuation" else True

    def _multipliers(self):
        self._attenuation()                       # only one Extinction
        return [i for i, c in enumerate(self.components) if c.kind in self._MULTIPLY]

    def evaluate(self, x, ctx=None, full=None, components=False):
        """
        Model flux at wavelengths x. With components=True return a dict of the
        (observed, i.e. reddened if there is an Extinction) component spectra.
        """
        x = np.asarray(x, float)
        ctx = ctx or {}
        full = self.params.full_values() if full is None else np.asarray(full, float)
        slices = self._slices()
        mult = self._multipliers()
        T = {i: self.components[i].value(x, full[slices[i]], ctx) for i in mult}
        parts = {}
        for i, (c, s) in enumerate(zip(self.components, slices)):
            if i in T:
                continue
            fc = c.value(x, full[s], ctx)
            for m in mult:
                if self._applies(self.components[m], c):
                    fc = T[m] * fc
            parts[c.name] = fc
        return parts if components else sum(parts.values(), np.zeros_like(x))

    __call__ = evaluate

    def combine(self, x, results):
        """
        Combine per-component (f, J) results (in component order) into the model
        and its Jacobian with respect to the full parameter vector, applying an
        Extinction component to all non-host components.
        """
        f = np.zeros_like(x)
        J = np.empty((x.size, len(self.params)))
        slices = self._slices()
        mult = self._multipliers()
        if not mult:
            for (fc, Jc), s in zip(results, slices):
                f += fc
                J[:, s] = Jc
            return f, J
        # each additive component times the transmissions that apply to it; the
        # derivative of a transmission collects the components it multiplies
        under = {m: np.zeros_like(x) for m in mult}
        for i, (c, (fc, Jc), s) in enumerate(zip(self.components, results, slices)):
            if i in under:
                continue
            Tc = np.ones_like(x)
            acting = [m for m in mult if self._applies(self.components[m], c)]
            for m in acting:
                Tc = Tc * results[m][0]
            f += Tc * fc
            J[:, s] = Tc[:, None] * Jc
            for m in acting:
                others = np.ones_like(x)
                for m2 in acting:
                    if m2 != m:
                        others = others * results[m2][0]
                under[m] += others * fc
        for m in mult:
            J[:, slices[m]] = results[m][1] * under[m][:, None]
        return f, J

    def evaluate_with_jacobian(self, x, ctx, full):
        """Return the model and df/d(full parameters), shape (len(x), n_full)."""
        results = [c.evaluate_with_derivatives(x, full[s], ctx)
                   for c, s in zip(self.components, self._slices())]
        return self.combine(x, results)

    # --------------------------------------------------------------- info
    def table(self):
        return self.params.table()

    def __repr__(self):
        lines = [f"Model with {len(self.components)} components, "
                 f"{len(self.params.free_params())} free parameters:"]
        for c in self.components:
            lines.append(f"  {c.name:14s} {type(c).__name__}")
        return "\n".join(lines)


def load_model(path):
    """A model saved with :meth:`Model.save` (a JSON recipe; no code is run)."""
    import json
    with open(path) as fh:
        return Model.from_dict(json.load(fh))


def _plain(v):
    """Arguments and values as JSON data: tuples and arrays as lists, numpy numbers as Python numbers,
    infinities as the strings 'inf' and '-inf' (standard JSON has no infinity)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in list(v)]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else (None if np.isnan(v) else ("inf" if v > 0 else "-inf"))
    if v is None or isinstance(v, str):
        return v
    raise ValueError(f"cannot save {type(v).__name__} values in a model file")


def _tuples(v):
    """Lists back to tuples (limits such as (start, min, max) and line entries such as (name, wavelength)),
    'inf' and '-inf' back to infinities."""
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if v in ("inf", "-inf"):
        return float(v)
    return v
