"""
Model step: the model gallery (candidate models fitted on the spectrum and ranked by BIC) and
the model builder (a workspace where every component and parameter can be edited).

A model is edited as a *state*, plain JSON::

    {"comps": [{"type": "Lines", "kw": {"name": "broad", "lines": ["Hb"], ...}, "on": true}, ...],
     "params": {"broad.fwhm": {"min": 1000, "max": 7000}, "x.shift": {"tie": ["broad.shift", 1]}},
     "wave_range": [3700, 6900]}

the components with their constructor arguments (as in ``Model.to_dict``), plus parameter
overrides on top of what the components set themselves. :func:`code_of` turns a state into
FANTASY code; the code is run to get the model, so what is shown is exactly what is fitted.
"""
import inspect
import math
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .. import presets
from .._threads import SINGLE_THREADED, environment

TYPES = ["PowerLaw", "BrokenPowerLaw", "Blackbody", "Polynomial", "BalmerContinuum",
         "StellarPopulation", "HostGalaxy", "FeII", "HighOrderBalmer", "Lines", "Absorption", "Extinction"]
POSITIONAL = {"Lines": ["name", "lines"], "Absorption": ["troughs"], "Calibration": ["wave_range"]}
CHOICES = {"profile": ["gaussian", "lorentzian", "voigt", "gauss-hermite", "stratified"],
           "role": ["narrow", "broad", "outflow"], "kind": ["optical", "uv", "uv_extended"],
           "law": ["F99", "calzetti"], "templates": ["emiles", "galaxev", "miles"]}
# preset options the gallery switches one at a time: (option, title when on, title when off).
# Options that change the fitted wavelength range (red, wave_range) are left out: BIC values
# are comparable only on the same pixels.
FLIPS = [("host", "+ host galaxy", "no host galaxy"), ("outflow", "+ outflow wing", "no outflow wing"),
         ("coronal", "+ coronal lines", "no coronal lines"), ("oi6046", "+ O I 6046", "no O I 6046"),
         ("feii", "+ Fe II", "no Fe II"), ("balmer", "+ Balmer continuum", "no Balmer continuum"),
         ("balmer_continuum", "+ Balmer continuum", "no Balmer continuum"), ("extinction", "+ dust", "no dust"),
         ("featureless", "+ featureless continuum", "no featureless continuum"),
         ("two_systems", "two narrow systems", "one narrow system"), ("narrow", "+ narrow lines", "no narrow lines"),
         ("heii", "+ He II", "no He II"), ("helium", "+ helium", "no helium"), ("bowen", "+ Bowen lines", "no Bowen lines"),
         ("wolf_rayet", "+ Wolf-Rayet bumps", "no Wolf-Rayet bumps"), ("physical", "physical decomposition", "empirical decomposition")]
MAX_CANDIDATES = 10


# ---------------------------------------------------------------- state <-> code
def _plain(v):
    from ..model import _plain as p
    return p(v)


def _tuples(v):
    from ..model import _tuples as t
    return t(v)


def _cls(name):
    import fantasy_agn as fa
    if name not in TYPES + ["Calibration"]:
        raise ValueError(f"unknown component type {name!r}")
    return getattr(fa, name)


def _as_kwargs(kind, args, kwargs):
    """Constructor arguments as one keyword dict (positional ones by their parameter names)."""
    sig = inspect.signature(_cls(kind).__init__)
    kw = dict(sig.bind_partial(None, *args, **kwargs).arguments)
    kw.pop("self", None)
    if "name" not in kw and "name" in sig.parameters:
        kw["name"] = sig.parameters["name"].default
    return kw


def lit(v):
    """A Python literal for code: tuples for lists, infinity as float('inf'), floats exact."""
    if isinstance(v, (list, tuple)):
        inner = ", ".join(lit(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {lit(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        if math.isinf(v):
            return "float('inf')" if v > 0 else "-float('inf')"
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    return repr(v)


def _same(a, b):
    try:
        return _plain(a) == _plain(b)
    except Exception:
        return False


def _ctor(comp):
    """fa.Type(...) code for one component of a state (defaults left out)."""
    kind, kw = comp["type"], _tuples(comp["kw"])
    sig = inspect.signature(_cls(kind).__init__).parameters
    pos = POSITIONAL.get(kind, ["name"])
    parts = [lit(kw[k]) for k in pos if k in kw]
    for k, v in kw.items():
        if k in pos:
            continue
        if k not in sig:
            raise ValueError(f"{kind} has no argument {k!r}")
        if sig[k].default is not inspect.Parameter.empty and _same(sig[k].default, v):
            continue
        parts.append(f"{k}={lit(v)}")
    return f"fa.{kind}({', '.join(parts)})"


def _fresh(comps, wave_range):
    """The model made by the components alone (their own defaults and ties)."""
    import fantasy_agn as fa
    m = fa.Model(wave_range=tuple(wave_range) if wave_range else None)
    for c in comps:
        kw = _tuples(c["kw"])
        pos = POSITIONAL.get(c["type"], [])
        m.add(_cls(c["type"])(*[kw.pop(k) for k in pos if k in kw], **kw))
    return m


def _override_code(m, overrides):
    """model.set/fix/free/tie/untie lines for the parameter overrides; unknown names are dropped."""
    lines, dropped = [], []
    for name, o in (overrides or {}).items():
        if name not in m.params.names:
            dropped.append(name)
            continue
        p = m.params[name]
        tie = o.get("tie")
        if tie:
            to, factor = tie[0], float(tie[1] if len(tie) > 1 else 1.0)
            if to not in m.params.names:
                dropped.append(name)
                continue
            if not (p.tie and p.tie[0] == to and abs(p.tie[1] - factor) < 1e-12):
                lines.append(f"model.tie({name!r}, {to!r}" + (f", factor={lit(factor)}" if factor != 1 else "") + ")")
            continue
        if p.tie and "tie" in o:
            lines.append(f"model.untie({name!r})")
        new = {}
        for k in ("value", "min", "max"):
            if o.get(k) is not None:
                v = float(_tuples(o[k]))
                if v != getattr(p, k):
                    new[k] = v
        fixed = o.get("fixed")
        if fixed is False and p.fixed:
            lines.append(f"model.free({name!r})")
        lim = {k: v for k, v in new.items() if k != "value"}
        if fixed and not p.fixed:
            if lim:
                lines.append(f"model.set({name!r}, " + ", ".join(f"{k}={lit(v)}" for k, v in lim.items()) + ")")
            lines.append(f"model.fix({name!r}" + (f", {lit(new['value'])}" if "value" in new else "") + ")")
        elif new:
            lines.append(f"model.set({name!r}, " + ", ".join(f"{k}={lit(v)}" for k, v in new.items()) + ")")
    return lines, dropped


def code_of(state):
    """FANTASY code that makes the model of a state, and the parameter overrides it had to drop."""
    comps = [c for c in state.get("comps") or [] if c.get("on", True)]
    if not comps:
        raise ValueError("the model has no components")
    names = [_tuples(c["kw"]).get("name") for c in comps]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"two components are called {', '.join(dup)}: rename one")
    bad = [n for n in names if not (isinstance(n, str) and n and n.replace("_", "a").isalnum() and not n[0].isdigit())]
    if bad:
        raise ValueError(f"component names may contain letters, digits and _ only: {bad}")
    wr = state.get("wave_range")
    body = "".join(f"    {_ctor(c)},\n" for c in comps)
    code = "model = fa.Model(\n" + body + (f"    wave_range=({lit(float(wr[0]))}, {lit(float(wr[1]))}),\n" if wr else "") + ")"
    extra, dropped = _override_code(_fresh(comps, wr), state.get("params"))
    if extra:
        code += "\n" + "\n".join(extra)
    return code, dropped


def run_code(code):
    import fantasy_agn as fa
    ns = {"fa": fa}
    exec(code, ns)                              # code made by code_of: constructors and parameter edits
    return ns["model"]


def decompose(model):
    """State of an existing model: its components' recipes, plus overrides where it differs from them."""
    d = model.to_dict()
    comps = [{"type": c["type"], "kw": _plain(_as_kwargs(c["type"], _tuples(c["args"]), _tuples(c["kwargs"]))), "on": True}
             for c in d["components"]]
    base = _fresh(comps, model.wave_range)
    params = {}
    for p in model.params:
        q = base.params[p.name]
        o = {k: _plain(getattr(p, k)) for k in ("value", "min", "max") if getattr(p, k) != getattr(q, k)}
        if p.fixed != q.fixed:
            o["fixed"] = bool(p.fixed)
        if (p.tie and p.tie[:2]) != (q.tie and q.tie[:2]):
            o["tie"] = None if p.tie is None else [p.tie[0], float(p.tie[1])]
        if o:
            params[p.name] = o
    state = {"comps": comps, "params": params, "wave_range": _plain(model.wave_range)}
    rebuilt = run_code(code_of(state)[0])
    if rebuilt.to_dict()["params"] != d["params"]:
        raise ValueError("this model cannot be taken apart into components exactly")
    return state


def group_of(c):
    kind = c.kind
    if kind == "continuum":
        return "continuum"
    if kind == "host":
        return "host"
    if kind == "template":
        return "feii" if type(c).__name__ == "FeII" else "broad"
    if kind == "line":
        return "absorption" if getattr(c, "absorption", False) else (getattr(c, "role", None) or "narrow")
    return "absorption" if type(c).__name__ in ("Absorption", "Extinction") else "other"


def describe(m):
    """Components and their parameters, for the builder."""
    comps = []
    for c in m.components:
        own = [p for p in m.params if p.name.startswith(c.name + ".")]
        lines = []
        if hasattr(c, "line_names") and hasattr(c, "_wave"):
            lines = [{"name": n, "wave": round(float(w), 2)} for n, w in zip(c.line_names, np.ravel(c._wave))]
        comps.append({"name": c.name, "type": type(c).__name__, "kind": c.kind, "group": group_of(c),
                      "role": getattr(c, "role", None), "profile": getattr(c, "profile", None), "lines": lines,
                      "nfree": int(sum(p.free for p in own)),
                      "params": [{"name": p.name, "value": _plain(p.value), "min": _plain(p.min), "max": _plain(p.max),
                                  "fixed": bool(p.fixed), "linear": bool(p.linear), "unit": p.unit,
                                  "description": p.description, "free": bool(p.free),
                                  "tie": None if p.tie is None else [p.tie[0], float(p.tie[1])]} for p in own]})
    return {"components": comps, "nfree": int(sum(p.free for p in m.params)), "nparams": len(m.params.names),
            "wave_range": _plain(m.wave_range)}


def types():
    """Component types and their constructor arguments with defaults (for the inspector)."""
    out = []
    for t in TYPES:
        k = _cls(t)
        fields = [{"key": name, "default": None if p.default is inspect.Parameter.empty else _plain(p.default),
                   "choices": CHOICES.get(name)}
                  for name, p in inspect.signature(k.__init__).parameters.items() if name != "self"]
        doc = " ".join((k.__doc__ or "").strip().split("\n\n")[0].split())
        doc = doc.replace("``", "")
        out.append({"type": t, "kind": getattr(k, "kind", "other"), "doc": doc, "fields": fields})
    return out


def _bins(x, n):
    edges = np.linspace(x.min(), x.max(), n + 1)
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)


def _binned(v, idx, mask, n):
    out = [None] * n
    for j in np.unique(idx[mask]):
        s = mask & (idx == j)
        val = float(np.median(v[s]))
        out[int(j)] = round(val, 6) if np.isfinite(val) else None
    return out


def preview(spec, model, n=420):
    """Initial guess of a model on the spectrum, per component, binned (the whole spectrum is drawn)."""
    from ..fitting import guess
    g = guess(spec, model)
    x = spec.wave
    w0, w1 = model.wave_range or (x.min(), x.max())
    k = (x >= w0) & (x <= w1)
    parts = g.evaluate(x[k], spec.context, components=True)
    idx, everywhere = _bins(x, n), np.isfinite(spec.flux)

    def full(v):
        f = np.full(x.size, np.nan)
        f[k] = v
        return f
    return {"w0": float(w0), "w1": float(w1), "x0": float(x.min()), "x1": float(x.max()),
            "wave": _binned(x, idx, np.ones(x.size, bool), n), "flux": _binned(spec.flux, idx, everywhere, n),
            "model": _binned(full(sum(parts.values())), idx, k, n),
            "comps": {name: _binned(full(v), idx, k, n) for name, v in parts.items()}}


# ------------------------------------------------------------------- gallery
def card(r, n=420):
    """A fit as a gallery card: fit statistics, the binned spectrum, model and residuals, per group."""
    s, m = r.spectrum, r.mask
    x = s.wave[m]
    parts = r.model_flux(s.wave, components=True)
    groups = {}
    for c in r.model.components:
        if c.name in parts:
            g = group_of(c)
            groups[g] = groups.get(g, 0) + parts[c.name]
    model = r.model_flux(s.wave)
    res = (s.flux - model) / s.err
    run = np.convolve(res[m], np.ones(40) / 40, mode="same")
    nfree = int(sum(p.free for p in r.model.params))
    npix = int(m.sum())
    try:
        kr = {q.quantity: [float(q.value), float(q.error) if np.isfinite(q.error) else None]
              for q in r.key_results().itertuples() if np.isfinite(q.value)}
    except Exception:
        kr = {}
    idx, ok = _bins(x, n), np.ones(x.size, bool)
    b = lambda v: _binned(np.asarray(v, float), idx, ok & np.isfinite(v), n)    # noqa: E731
    return {"rchi2": float(r.rchi2), "chi2": float(r.chi2), "dof": int(r.dof), "nfree": nfree, "npix": npix,
            "bic": float(r.chi2 + nfree * np.log(npix)), "limits": list(r.at_limit),
            "struct": float(np.nanmax(np.abs(run)) * np.sqrt(40)), "w0": float(x.min()), "w1": float(x.max()),
            "wave": b(x), "flux": b(s.flux[m]), "model": b(model[m]), "resid": b(res[m]),
            "groups": {g: b(v[m]) for g, v in groups.items()},
            "components": [{"name": c.name, "kind": c.kind, "group": group_of(c), "role": getattr(c, "role", None),
                            "lines": list(getattr(c, "line_names", [])),
                            "nfree": int(sum(p.free for p in r.model.params if p.name.startswith(c.name + ".")))}
                           for c in r.model.components],
            "results": kr}


def ladder(name, kw):
    """
    Candidate models around preset ``name(**kw)``: the preset itself, one broad component more
    and fewer, and each on/off option of the preset switched (one at a time).
    """
    func = getattr(presets, name)
    params = inspect.signature(func).parameters
    base = dict(kw)
    out = [("base", f"{name} preset", "the preset", base)]
    if "broad_components" in params:
        n = int(base.get("broad_components", params["broad_components"].default))
        for k in (n - 1, n + 1):
            if 1 <= k <= 3:
                out.append((f"b{k}", f"{k} broad component{'s' if k > 1 else ''}", f"broad_components={k}",
                            {**base, "broad_components": k}))
    for key, on, off in FLIPS:
        if key not in params or len(out) >= MAX_CANDIDATES - 1 or any(o[0] == key for o in out):
            continue
        cur = base.get(key, params[key].default)
        if key == "host":
            if name in ("type2", "galaxy", "seyfert2", "composite", "liner", "starforming", "passive") or base.get("physical"):
                continue                                  # the host is the point of these presets
            new = not bool(cur)
        elif isinstance(cur, bool):
            new = not cur
        else:
            continue
        out.append((key, on if new else off, f"{key}={new!r}", {**base, key: new}))
    return out


def _preset_code(name, kw):
    return f"model = fa.presets.{name}({', '.join(f'{k}={v!r}' for k, v in kw.items())})"


def _fit_job(job):
    """Fit one candidate (in a worker process); a card, or the error."""
    key, code, spec = job
    import fantasy_agn as fa
    t0 = time.time()
    try:
        r = fa.fit(spec, run_code(code), verbose=False)
        return key, card(r), None, time.time() - t0
    except Exception as exc:
        return key, None, f"{type(exc).__name__}: {exc}", time.time() - t0


def _run_jobs(jobs, done):
    """Run fit jobs in processes (serially when that fails), calling done(result) for each."""
    import os
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1, 4))
    left = list(jobs)
    if workers > 1 and len(jobs) > 1:
        try:
            with environment(SINGLE_THREADED), ProcessPoolExecutor(workers) as ex:
                futures = {ex.submit(_fit_job, j): j for j in jobs}
                for f in as_completed(futures):
                    done(f.result())
                    left.remove(futures[f])
        except Exception as exc:                                      # e.g. no __main__ guard
            warnings.warn(f"parallel fitting failed ({type(exc).__name__}); running serially")
    for j in left:
        done(_fit_job(j))


class BuilderMixin:
    """Model gallery and model builder of the Model step (mixed into Session)."""

    # ------------------------------------------------------------- builder
    def _mb_spec(self):
        if self.spec is None:
            raise ValueError("prepare a spectrum first")
        return self.spec

    def mb_types(self):
        return types()

    def mb_lines(self):
        import fantasy_agn as fa
        spec = self._mb_spec()
        lo, hi = float(spec.wave.min()), float(spec.wave.max())
        out = []
        for name in fa.LINES:
            try:
                info = fa.line_info(name)
                w = float(info["vacuum"])
            except Exception:
                continue
            if lo <= w <= hi:
                out.append({"name": name, "wave": round(w, 1), "ion": info.get("ion", ""), "type": info.get("type", "")})
        sets = {}
        for key, (label, _) in fa.LINE_SETS.items():
            ls = fa.line_set(key, lo, hi)
            if ls:
                sets[key] = {"label": label, "lines": list(ls)}
        return {"lines": sorted(out, key=lambda d: d["wave"]), "sets": sets, "range": [lo, hi]}

    def mb_decompose(self, source):
        """State to start the builder from: {'model': true} (the session model), {'empty': true}, or {'preset', 'kw'}."""
        import fantasy_agn as fa
        source = source or {}
        if source.get("model"):
            if self.model is None:
                raise ValueError("create a model first")
            m = self.model
        elif source.get("preset"):
            m = getattr(presets, source["preset"])(**(source.get("kw") or {}))
        else:
            spec = self._mb_spec()
            m = fa.Model(fa.PowerLaw(), wave_range=(float(np.ceil(spec.wave.min())), float(np.floor(spec.wave.max()))))
        st = decompose(m)
        code, _ = code_of(st)
        return {"state": st, "code": code, **describe(run_code(code))}

    def mb_build(self, state):
        code, dropped = code_of(state)
        m = run_code(code)
        return {"code": code, "warnings": [f"{n}: no longer in the model, its setting is dropped" for n in dropped], **describe(m)}

    def mb_preview(self, state):
        spec = self._mb_spec()
        with environment(SINGLE_THREADED):
            return preview(spec, run_code(code_of(state)[0]))

    def recipe_code(self, state):
        """Code for the session that makes the builder's model (Use this model)."""
        self._mb_spec()
        code, _ = code_of(state)
        self._preset_made = None                   # an edited model: fitted as it is
        return code + "\nmodel"

    def preset_kw_code(self, name, kw):
        """Code for a gallery candidate made from a preset (Use this model)."""
        if not hasattr(presets, name) or name.startswith("_"):
            raise ValueError(f"unknown preset {name!r}")
        params = inspect.signature(getattr(presets, name)).parameters
        bad = [k for k in kw if k not in params]
        if bad:
            raise ValueError(f"{name} has no option {bad[0]!r}")
        self._preset_made = (name, dict(kw))
        return _preset_code(name, kw) + "\nmodel"

    # ------------------------------------------------------------- gallery
    def _gallery_base(self):
        """The preset the candidates vary: the classification's recommendation, else the optical preset."""
        from ..classify import Classification
        c = getattr(self, "classification", None)
        if c is not None and c.label in Classification.RECOMMENDED:
            name, kw = Classification.RECOMMENDED[c.label]
            return name, dict(kw)
        lo, hi = self.spec.wave.min(), self.spec.wave.max()
        if hi < 5200:
            return ("mgii", {}) if hi > 2900 else ("civ", {})
        return "optical", {}

    def wf_gallery(self):
        """The gallery of the current spectrum: candidates fitted so far, and whether more are coming."""
        g = self.__dict__.get("_gallery")
        if g is None or self.spec is None or g["spec"] is not self.spec:
            c = getattr(self, "classification", None)
            return {"cands": [], "running": False, "label": c.label if c is not None else None}
        return {k: v for k, v in g.items() if k != "spec"}

    def gallery_run(self, preset=None, kw=None):
        """Fit the candidate models of the current spectrum in the background."""
        spec = self._mb_spec()
        self._need_rest_frame()
        g = self.__dict__.get("_gallery")
        if g is not None and g["spec"] is spec and g["running"]:
            return
        name, base = (preset, dict(kw or {})) if preset else self._gallery_base()
        cands = ladder(name, base)
        c = getattr(self, "classification", None)
        g = {"spec": spec, "running": True, "preset": name, "base_kw": base, "name": spec.name,
             "label": c.label if c is not None else None, "reason": c.reason if c is not None else None,
             "cands": [], "pending": [t for _, t, _, _ in cands], "total": len(cands), "error": None}
        keep = [x for x in (self.__dict__.get("_gallery") or {}).get("cands", []) if x.get("mine")] \
            if (self.__dict__.get("_gallery") or {}).get("spec") is spec else []
        g["cands"] = keep
        self._gallery = g
        meta = {k: (t, ch, kwk) for k, t, ch, kwk in cands}

        def add(res, round_=1):
            key, cd, err, dt = res
            t, ch, kwk = meta[key]
            entry = {"key": key, "title": t, "change": ch, "time": round(dt, 1), "preset": [name, kwk],
                     "code": _preset_code(name, kwk), "round": round_}
            if err:
                entry["error"] = err
            else:
                entry.update(cd)
            g["cands"].append(entry)
            if t in g["pending"]:
                g["pending"].remove(t)

        def work():
            try:
                _run_jobs([(k, _preset_code(name, kwk), spec) for k, _, _, kwk in cands], add)
                # round 2: the preset with every change that lowered the BIC by more than 10
                ok = {x["key"]: x for x in g["cands"] if "bic" in x and not x.get("mine")}
                if "base" in ok:
                    gains = sorted((x["bic"] - ok["base"]["bic"], k) for k, x in ok.items() if k != "base")
                    combo, used, keys = dict(base), set(), []
                    for d, k in gains:
                        if d >= -10:
                            break
                        opt = "broad_components" if k.startswith("b") and k[1:].isdigit() else k
                        if opt in used:
                            continue
                        used.add(opt)
                        keys.append(k)
                        combo[opt] = meta[k][2][opt]
                    if len(keys) > 1:
                        title = " + ".join(meta[k][0].lstrip("+ ") for k in keys)
                        meta["combo"] = ("combined: " + title, ", ".join(meta[k][1] for k in keys), combo)
                        g["pending"].append(meta["combo"][0])
                        g["total"] += 1
                        _run_jobs([("combo", _preset_code(name, combo), spec)], lambda r: add(r, 2))
            except Exception as exc:
                g["error"] = f"{type(exc).__name__}: {exc}"
            g["running"] = False
        threading.Thread(target=work, daemon=True).start()

    def gallery_fit(self, state, title="", change=""):
        """Fit a model of the builder on the current spectrum (in the background) and add it to the gallery."""
        spec = self._mb_spec()
        self._need_rest_frame()
        code, _ = code_of(state)
        g = self.__dict__.get("_gallery")
        if g is None or g["spec"] is not spec:
            c = getattr(self, "classification", None)
            g = self._gallery = {"spec": spec, "running": False, "preset": None, "base_kw": {}, "name": spec.name,
                                 "label": c.label if c is not None else None, "reason": c.reason if c is not None else None,
                                 "cands": [], "pending": [], "total": 0, "error": None}
        g["mine_n"] = n = g.get("mine_n", 0) + 1
        key, title = f"mine{n}", f"#{n} {title}".strip()
        g["pending"].append(title)
        g["mine_running"] = g.get("mine_running", 0) + 1

        def work():
            res = _fit_job((key, code, spec))
            entry = {"key": key, "title": title, "change": change, "time": round(res[3], 1), "mine": True,
                     "code": code, "state": state}
            if res[2]:
                entry["error"] = res[2]
            else:
                entry.update(res[1])
            g["cands"].append(entry)
            if title in g["pending"]:
                g["pending"].remove(title)
            g["mine_running"] -= 1
        threading.Thread(target=work, daemon=True).start()
        return key
