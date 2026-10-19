"""The Model: a sum of components with a shared, tie-aware parameter set."""
import copy

import numpy as np

from .components import Component
from .parameters import ParameterSet

__all__ = ("Model",)


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
            for c in component.components:
                self.add(c)
            for p in component.params:
                if p.tie is not None:
                    self.params[p.name].tie = p.tie
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
    # Absorption (troughs of absorbers in front of the AGN) acts on everything.
    _MULTIPLY = ("attenuation", "absorption")

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
