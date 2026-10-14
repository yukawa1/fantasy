"""
Model parameters: values, limits, fixing and linear ties.

Every parameter has a full name ``"component.parameter"`` (e.g. ``"broad.fwhm"``
or ``"narrow.[OIII]5007.flux"`` for per-line parameters). A parameter is either
free, fixed, or tied to another parameter as ``value = factor * other + offset``.
Because ties are linear, the full parameter vector is an affine function of the
free parameters, ``p = T @ theta + c``, which keeps Jacobians exact and cheap.
"""
import fnmatch
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ("Parameter", "ParameterSet")


@dataclass
class Parameter:
    name: str
    value: float
    min: float = -np.inf
    max: float = np.inf
    fixed: bool = False
    linear: bool = False
    unit: str = ""
    description: str = ""
    tie: tuple = field(default=None, repr=False)  # (target name, factor, offset)

    @property
    def free(self):
        return not self.fixed and self.tie is None

    def set(self, value=None, min=None, max=None, fixed=None):
        if min is not None:
            self.min = float(min)
        if max is not None:
            self.max = float(max)
        if value is not None:
            self.value = float(value)
        if fixed is not None:
            self.fixed = bool(fixed)
        if self.free and not (self.min <= self.value <= self.max):
            raise ValueError(
                f"{self.name}: value {self.value} outside limits [{self.min}, {self.max}]")
        return self


class ParameterSet:
    """Ordered collection of parameters with name lookup, wildcards and ties."""

    def __init__(self, params=()):
        self._params = {}
        for p in params:
            self.add(p)

    def add(self, param):
        if param.name in self._params:
            raise ValueError(f"duplicate parameter name {param.name!r}")
        self._params[param.name] = param

    def __getitem__(self, name):
        try:
            return self._params[name]
        except KeyError:
            matches = self.match(name, required=False)
            if len(matches) == 1:
                return matches[0]
            close = [n for n in self._params if name.split(".")[-1] in n][:8]
            raise KeyError(f"no parameter {name!r}. Similar: {close}") from None

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self):
        return len(self._params)

    def __contains__(self, name):
        return name in self._params

    @property
    def names(self):
        return list(self._params)

    def index(self, name):
        return self.names.index(name)

    def match(self, pattern, required=True):
        """
        Parameters whose name matches a pattern with '*' and '?' wildcards
        ('broad.*'). Square brackets are literal, as in '[OIII]5007'.
        """
        if pattern in self._params:
            return [self._params[pattern]]
        glob = pattern.replace("[", "[[]")
        found = [p for n, p in self._params.items() if fnmatch.fnmatchcase(n, glob)]
        if required and not found:
            raise KeyError(f"no parameter matches {pattern!r}")
        return found

    # ----------------------------------------------------------------- editing
    def set(self, pattern, value=None, min=None, max=None):
        for p in self.match(pattern):
            p.set(value=value, min=min, max=max)

    def fix(self, pattern, value=None):
        for p in self.match(pattern):
            p.set(value=value, fixed=True)

    def free(self, pattern):
        for p in self.match(pattern):
            p.fixed = False
            p.tie = None

    def tie(self, pattern, to, factor=1.0, offset=0.0):
        """Tie parameter(s): value = factor * value(to) + offset."""
        target = self[to]
        for p in self.match(pattern):
            if p.name == target.name:
                continue
            if p.linear != target.linear:
                raise ValueError(
                    f"cannot tie {p.name} to {target.name}: only amplitude-type "
                    "parameters can be tied to each other, and likewise for "
                    "widths/shifts")
            if p.linear and offset != 0:
                raise ValueError("ties between amplitudes must have offset=0")
            p.tie = (target.name, float(factor), float(offset))
        self._resolve()  # raises on cycles

    def untie(self, pattern):
        for p in self.match(pattern):
            p.tie = None

    # -------------------------------------------------------------- vectorize
    def _resolve(self):
        """Map every parameter to (free root or None, factor, offset)."""
        out = {}

        def walk(name, seen):
            if name in out:
                return out[name]
            if name in seen:
                raise ValueError(f"circular tie involving {name!r}")
            p = self._params[name]
            if p.tie is None:
                res = (name, 1.0, 0.0) if not p.fixed else (None, 0.0, p.value)
            else:
                root, f, o = walk(p.tie[0], seen | {name})
                res = (root, p.tie[1] * f, p.tie[1] * o + p.tie[2])
            out[name] = res
            return res

        for name in self._params:
            walk(name, frozenset())
        return out

    def free_params(self):
        return [p for p in self if p.free]

    def affine(self):
        """
        Return (T, c, free) such that full = T @ theta + c, where theta are the
        values of the free parameters listed in ``free``.
        """
        free = self.free_params()
        col = {p.name: k for k, p in enumerate(free)}
        res = self._resolve()
        T = np.zeros((len(self), len(free)))
        c = np.zeros(len(self))
        for i, name in enumerate(self.names):
            root, f, o = res[name]
            if root is not None:
                T[i, col[root]] = f
            c[i] = o
        return T, c, free

    def full_values(self):
        T, c, free = self.affine()
        return T @ np.array([p.value for p in free]) + c

    def set_full_values(self, values):
        for p, v in zip(self, values):
            p.value = float(v)

    def copy(self):
        return ParameterSet(Parameter(**vars(p)) for p in self)

    def table(self):
        rows = []
        for p in self:
            status = "free" if p.free else (
                f"= {p.tie[1]:g}*{p.tie[0]}" + (f" + {p.tie[2]:g}" if p.tie[2] else "")
                if p.tie else "fixed")
            rows.append((p.name, p.value, p.min, p.max, p.unit, status))
        return pd.DataFrame(rows, columns=["parameter", "value", "min", "max", "unit", "status"])

    def __repr__(self):
        return self.table().to_string(index=False)
