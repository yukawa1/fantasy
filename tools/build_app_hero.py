"""
Build ``fantasy_agn/app/static/hero.json``: the real decomposition of a bundled SDSS
spectrum that the app replays on its opening screen. The example is a Seyfert 1 galaxy
(spec-0354-51792-0617, z = 0.069) with a strong host: the fit separates the host galaxy
(E-MILES stellar populations), the AGN continuum, Fe II, and the broad and narrow lines.
Run from the repository root after a change of the fit or of the example:

    python tools/build_app_hero.py
"""
import json
import os
import warnings

import numpy as np

import fantasy_agn as fa

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "spec-0354-51792-0617.fits"
SRC = os.path.join(HERE, "..", "fantasy_agn", "data", "examples", NAME)
OUT = os.path.join(HERE, "..", "fantasy_agn", "app", "static", "hero.json")
IMAGE = os.path.join(HERE, "..", "fantasy_agn", "app", "static", "img", "hero_galaxy.jpg")
IMAGE_FOV = 40.0                                     # arcsec across the sky image
OBJECT = "2MASX J17310787+6200260"                   # SIMBAD name (type Sy1)
BIN = 3
# rest vacuum wavelengths of the labelled features: emission above the spectrum, absorption below
EMISSION = [("[O II]", 3728.5), ("Hγ", 4341.7), ("Hβ", 4862.7), ("[O III]", 5008.2), ("He I", 5877.3),
            ("Hα + [N II]", 6564.6), ("[S II]", 6725.0)]
ABSORPTION = [("Ca II H+K", 3952.0), ("G band", 4305.6), ("Mg b", 5176.7), ("Na D", 5895.6)]


def binned(a, keep):
    """Mean of BIN pixels; NaN where any of them is not fitted (flagged by the pipeline or masked)."""
    a = a[keep]
    n = a.size // BIN * BIN
    return a[:n].reshape(-1, BIN).mean(axis=1)


def listed(a, digits):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in a]


def main():
    warnings.filterwarnings("ignore")
    raw = fa.read(SRC)
    spec = raw.deredden().to_rest_frame()
    cls = fa.classify(spec)
    preset, kw = cls.RECOMMENDED[cls.label]
    r = fa.fit(spec, getattr(fa.presets, preset)(**kw), verbose=False)
    parts = r.model_flux(components=True)
    comps = {c.name: c for c in r.model.components}

    def group(test):
        return sum((v for k, v in parts.items() if test(comps[k])), np.zeros_like(spec.wave))

    role = lambda c: getattr(c, "role", None)  # noqa: E731
    groups = [  # stacked from the bottom
        ("host", "host galaxy", group(lambda c: c.kind == "host")),
        ("agn", "AGN continuum", group(lambda c: c.kind == "continuum" or type(c).__name__ == "HighOrderBalmer")),
        ("feii", "Fe II", group(lambda c: type(c).__name__ == "FeII")),
        ("broad", "broad lines", group(lambda c: role(c) == "broad")),
        ("narrow", "narrow lines", group(lambda c: role(c) in ("narrow", "outflow"))),
    ]
    assert np.allclose(sum(g[2] for g in groups), r.model_flux()), "every component belongs to one group"

    fitted = r.mask & spec.good
    keep = (spec.wave >= spec.wave[fitted].min()) & (spec.wave <= spec.wave[fitted].max())
    nan = np.where(fitted, 0.0, np.nan)               # unfitted pixels break the data, not the model
    k = {q: (float(v), float(e)) for q, v, e, _ in r.key_results().itertuples(index=False)}
    host = k["host fraction at 5100 A"]
    lo, hi = float(spec.wave[fitted].min()), float(spec.wave[fitted].max())
    out = {
        "name": raw.name, "file": NAME, "z": round(float(raw.z), 5), "unit": "10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹",
        "wave": listed(binned(spec.wave, keep), 2),
        "data": listed(binned(spec.flux + nan, keep), 3),
        "err": listed(np.sqrt(binned(spec.err ** 2 + nan, keep) / BIN), 3),
        "model": listed(binned(r.model_flux(), keep), 3),
        "groups": [{"key": key, "label": label, "flux": listed(binned(v, keep), 3)} for key, label, v in groups],
        "labels": [[n, w, "emission"] for n, w in EMISSION if lo < w < hi]
                  + [[n, w, "absorption"] for n, w in ABSORPTION if lo < w < hi],
        "stages": {
            "observed": f"observed spectrum · SDSS · z = {raw.z:.4f}",
            "rest": "to the rest frame:  λ_rest = λ_obs / (1 + z)",
            "host": f"host galaxy · stellar populations (E-MILES), σ* = {k['sigma* (host)'][0]:.0f} km/s",
            "agn": f"+ AGN continuum · {100 * (1 - host[0]):.0f}% of the light at 5100 Å",
            "feii": "+ Fe II · thousands of blended iron transitions",
            "broad": f"+ broad lines · gas orbiting the black hole, Hβ FWHM {k['broad Hb FWHM'][0]:.0f} km/s",
            "narrow": "+ narrow lines · ionised gas farther out",
            "model": f"full model · {len(r.model.params.free_params())} free parameters · reduced χ² = {r.rchi2:.2f}",
        },
        "stats": [
            {"label": "broad Hβ FWHM", "value": k["broad Hb FWHM"][0], "err": k["broad Hb FWHM"][1], "dec": 0, "unit": "km/s"},
            {"label": "log M_BH", "value": k["log M_BH"][0], "err": k["log M_BH"][1], "dec": 2, "unit": "M☉"},
            {"label": "host light at 5100 Å", "value": 100 * host[0], "err": 100 * host[1], "dec": 0, "unit": "%"},
            {"label": "stellar σ*", "value": k["sigma* (host)"][0], "err": k["sigma* (host)"][1], "dec": 0, "unit": "km/s"},
        ],
    }
    # the galaxy on the sky, centred on the fiber; SDSS-I/II fibers are 3" across, BOSS fibers 2"
    from fantasy_agn.catalogs import sky_image
    jpeg, survey = sky_image(raw.ra, raw.dec, fov=IMAGE_FOV, size=480)
    if jpeg is None:
        raise SystemExit("no sky image: the opening screen needs one (internet access)")
    with open(IMAGE, "wb") as fh:
        fh.write(jpeg)
    plate = int(raw.meta.get("plate", 0))
    out["image"] = {"src": "/static/img/hero_galaxy.jpg", "fov": IMAGE_FOV, "survey": survey,
                    "fiber": 3.0 if plate < 3509 else 2.0, "object": OBJECT}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"{OUT}: {len(out['wave'])} points, {os.path.getsize(OUT) / 1024:.0f} kB, {cls.label} -> {preset}{kw}")
    for s in out["stats"]:
        print(f"  {s['label']}: {s['value']:.{s['dec']}f} ± {s['err']:.{s['dec'] + 1}f} {s['unit']}")


if __name__ == "__main__":
    main()
