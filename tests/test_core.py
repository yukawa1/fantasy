import os

import numpy as np
from astropy.io import fits
import pytest

import fantasy_agn as fa
import fantasy_agn.plotting  # noqa: F401  (fa.plotting)
import fantasy_agn.correlations  # noqa: F401  (fa.correlations)
from fantasy_agn.wavelengths import air_to_vac, vac_to_air

os.environ.setdefault("FANTASY_AUTOSAVE", "0")       # the app tests must not write to ~/.fantasy_agn/sessions
HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, "..", "examples")


needs_templates = pytest.mark.skipif(not {"emiles", "galaxev"} <= set(fa.templates.available()),
                                     reason="run fa.templates.download('emiles') and ('galaxev') to test the stellar host")

def test_air_vacuum_roundtrip():
    lam = np.linspace(2000, 11000, 50)
    assert np.allclose(air_to_vac(vac_to_air(lam)), lam, atol=1e-8)
    # H-alpha: 6562.79 A in air is 6564.60 A in vacuum (NIST)
    assert abs(air_to_vac(6562.79) - 6564.60) < 0.01


def test_fitzpatrick99_anchor_points():
    # The optical spline passes through the anchor at 5470 A by construction.
    r_v = 3.1
    expected = -5.13540e-02 + 1.00216 * r_v - 7.35778e-05 * r_v ** 2
    assert np.isclose(fa.fitzpatrick99(5470.0, r_v), expected, rtol=1e-6)


def test_fitzpatrick99_matches_pyastronomy():
    pyasl = pytest.importorskip("PyAstronomy.pyasl")
    from fantasy_agn.extinction import deredden
    w = np.linspace(1200, 11000, 500)
    ours = deredden(w, np.ones_like(w), 0.1)
    theirs = pyasl.unred(w, np.ones_like(w), 0.1)
    assert np.max(np.abs(ours / theirs - 1)) < 2e-3  # differ only in spline type


def test_parameter_ties_and_patterns():
    m = fa.Model(fa.Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007"],
                          ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}))
    m.set("narrow.[OIII]5007.flux", 298.0)
    full = m.params.full_values()
    assert np.isclose(full[m.params.index("narrow.[OIII]4959.flux")], 100.0)
    assert len(m.params.match("narrow.*.flux")) == 3
    m.fix("narrow.shift", 0)
    assert not m["narrow.shift"].free
    with pytest.raises(ValueError):
        m.tie("narrow.[OIII]5007.flux", "narrow.[OIII]4959.flux")  # would be circular
    with pytest.raises(ValueError):
        m.tie("narrow.fwhm", "narrow.Hb.flux")  # width tied to a flux


def test_analytic_jacobians():
    x = np.linspace(3000, 7000, 1500)
    m = fa.Model(fa.PowerLaw(), fa.BrokenPowerLaw("bpl"), fa.Polynomial(order=2),
                 fa.BalmerContinuum(fix_shape=False), fa.FeII(), fa.HighOrderBalmer(),
                 fa.HostGalaxy(),
                 fa.Lines("narrow", ["Hb", "[OIII]5007"]),
                 fa.Lines("broad", ["Hb", "Hg"], fwhm=(4000, 1000, 9000), tie_kinematics=False))
    rng = np.random.default_rng(0)
    full = m.params.full_values() * rng.uniform(0.8, 1.2, len(m.params))
    for i, p in enumerate(m.params):
        if p.name.endswith("shift"):
            full[i] = 150.0
    ctx = {"medium": "air", "resolution": 150.0}
    _, J = m.evaluate_with_jacobian(x, ctx, full)
    for i in range(len(full)):
        h = 1e-6 * max(abs(full[i]), 1)
        a, b = full.copy(), full.copy()
        a[i] += h
        b[i] -= h
        num = (m.evaluate(x, ctx, a) - m.evaluate(x, ctx, b)) / (2 * h)
        scale = np.max(np.abs(num)) + 1e-30
        assert np.max(np.abs(num - J[:, i])) / scale < 1e-4, m.params.names[i]


def test_windowed_gaussians_equal_dense(monkeypatch):
    # Gaussian lines are evaluated only near their centres; the model and the Jacobian are the same
    from fantasy_agn import profiles
    x = np.linspace(3000, 7000, 3000)
    m = fa.Model(fa.PowerLaw(), fa.FeII(), fa.HighOrderBalmer(),
                 fa.Lines("narrow", ["Hb", "Hg", "[OIII]4959", "[OIII]5007", "[NII]6583", "Ha"], tie_kinematics=False,
                          line_ratios={"Hg": ("Hb", (0.35, 0.1, 0.5))}, max_ratios={"[NII]6583": ("Ha", 3)}),
                 fa.Lines("broad", ["Hb", "Ha"], fwhm=(4000, 1000, 9000)))
    rng = np.random.default_rng(1)
    full = m.params.full_values() * rng.uniform(0.8, 1.2, len(m.params))
    ctx = {"medium": "vacuum", "resolution": 150.0}
    f1, J1 = m.evaluate_with_jacobian(x, ctx, full)
    v1 = m.evaluate(x, ctx, full)
    unsorted = m.evaluate(x[::-1], ctx, full)[::-1]              # falls back to the dense profiles
    assert profiles.gaussian_windowed(x, np.array([5000.0]), np.zeros(1), np.full(1, 500.0), np.zeros(1)) is not None
    monkeypatch.setattr(profiles, "WINDOW", 0)
    f0, J0 = m.evaluate_with_jacobian(x, ctx, full)
    scale = np.abs(J0).max(axis=0)
    assert np.allclose(f1, f0, rtol=0, atol=1e-12 * np.abs(f0).max())
    assert np.allclose(v1, f0, rtol=0, atol=1e-12 * np.abs(f0).max())
    assert np.allclose(unsorted, f0, rtol=0, atol=1e-12 * np.abs(f0).max())
    assert np.all(np.abs(J1 - J0).max(axis=0) <= 1e-10 * scale.max())
    kin = [i for i, p in enumerate(m.params) if not p.linear]
    assert np.all(np.abs(J1 - J0)[:, kin].max(axis=0) <= 1e-9 * scale[kin])



def test_scanner_equals_linear_solve():
    # the grid searches of the fit get chi2 from Problem.scanner; it must be what solve_linear gives
    from fantasy_agn.fitting import Problem, _groups
    x = np.linspace(4200, 6900, 1500)
    m = fa.Model(fa.PowerLaw(), fa.FeII(), fa.HostGalaxy(n_components=3), fa.Extinction(),
                 fa.Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007", "[NII]6583", "Ha"],
                          ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}, max_ratios={"[NII]6583": ("Ha", 3)}),
                 fa.Lines("broad", ["Hb", "Ha"], fwhm=(4000, 1000, 9000), line_ratios={"Hb": ("Ha", (0.3, 0.1, 0.5))}),
                 fa.Lines("wing", ["Hb"], fwhm=(3000, 1000, 9000), profile="lorentzian", role="broad"))
    m.tie("feii.shift", "broad.shift")                  # one parameter moves two components
    m.fix("narrow.Hb.flux", 40.0)                       # a fixed amplitude is part of the data side
    rng = np.random.default_rng(2)
    ctx = {"medium": "vacuum", "resolution": 150.0}
    y = m.evaluate(x, ctx) * 30 + rng.normal(0, 1, x.size)
    prob = Problem(m, x, y, np.full(x.size, 1.0), ctx)
    theta = prob.solve_linear(prob.theta_from_model())[1]
    seen = set()
    for name, ks in _groups(prob).items():
        scan = prob.scanner(theta, ks)
        if scan is None:                                # a multiplicative component: the general path is used
            assert name.startswith("extinction")
            continue
        seen.add(name)
        for _ in range(3):
            vals = np.array([rng.uniform(prob.lb[k], prob.ub[k]) for k in ks])
            t = theta.copy()
            t[ks] = vals
            assert np.isclose(scan(vals), prob.solve_linear(t)[0], rtol=1e-9, atol=0), name
    assert {"narrow", "broad", "wing", "feii", "continuum.index"} <= seen


def _toy_model():
    return fa.Model(
        fa.PowerLaw("continuum", pivot=5000),
        fa.Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007"], fwhm=(400, 100, 1200),
                 shift=(0, -500, 500), ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}),
        fa.Lines("broad", ["Hb"], fwhm=(3000, 1000, 12000), shift=(0, -3000, 3000)),
        wave_range=(4700, 5100))


def _toy_truth():
    m = _toy_model()
    m.set("continuum.amp", 50).set("continuum.index", -1.2)
    m.set("narrow.Hb.flux", 300).set("narrow.[OIII]5007.flux", 1500)
    m.set("narrow.fwhm", 450).set("narrow.shift", -60)
    m.set("broad.Hb.flux", 6000).set("broad.fwhm", 4200).set("broad.shift", 350)
    return m


def _toy_spectrum(rng, noise=2.0):
    wave = np.arange(4650, 5150, 1.0)
    clean = _toy_truth().evaluate(wave, {"medium": "vacuum"})
    return fa.Spectrum(wave, clean + rng.normal(0, noise, wave.size),
                       np.full(wave.size, noise), rest_frame=True)


def test_recovers_parameters_from_bad_start():
    spec = _toy_spectrum(np.random.default_rng(1))
    model = _toy_model()
    model.set("broad.fwhm", 11000).set("narrow.fwhm", 1100).set("broad.shift", -2500)
    r = fa.fit(spec, model, verbose=False)
    truth = _toy_truth().params.full_values()
    p = r.params.set_index("parameter")
    for name in ["broad.fwhm", "broad.shift", "broad.Hb.flux", "narrow.fwhm",
                 "narrow.[OIII]5007.flux", "continuum.amp"]:
        i = r.model.params.index(name)
        assert abs(p.loc[name, "value"] - truth[i]) < 5 * p.loc[name, "error"], name
    lines = r.lines.set_index("line")
    assert np.isclose(lines.loc["[OIII]4959", "flux"] * 2.98, lines.loc["[OIII]5007", "flux"])
    assert 0.8 < r.rchi2 < 1.25


def test_covariance_errors_are_calibrated():
    rng = np.random.default_rng(2)
    truth = _toy_truth().params.full_values()
    names = ["broad.fwhm", "broad.Hb.flux", "narrow.fwhm", "narrow.[OIII]5007.flux"]
    pulls = []
    for _ in range(40):
        r = fa.fit(_toy_spectrum(rng), _toy_model(), verbose=False, scale_errors=False)
        p = r.params.set_index("parameter")
        pulls.append([(p.loc[n, "value"] - truth[r.model.params.index(n)]) / p.loc[n, "error"]
                      for n in names])
    std = np.std(pulls, axis=0)
    assert np.all((std > 0.65) & (std < 1.35)), std


def test_monte_carlo_errors_and_measure():
    spec = _toy_spectrum(np.random.default_rng(3))
    r = fa.fit(spec, _toy_model(), errors="mc", n_mc=20, seed=1, verbose=False)
    assert r.samples.shape == (20, len(r.model.params))
    m = r.measure("Hb", ["broad"])
    assert abs(m["fwhm"] - r.model["broad.fwhm"].value) < 50  # single Gaussian: same FWHM
    assert m["fwhm_err"] > 0 and np.isfinite(m["ew"])


@pytest.mark.parametrize("fname", ["spec-1059-52618-0453.fits", "spec-2770-54510-0433.fits"])
def test_presets_on_sdss_spectra(fname):
    spec = fa.read(os.path.join(EXAMPLES, fname)).to_rest_frame()
    assert spec.resolution > 100  # SDSS: ~140 km/s from wdisp
    assert "hbeta" in fa.presets.available(spec)
    r = fa.fit(spec, fa.presets.hbeta(), verbose=False)
    assert r.success and r.rchi2 < 3
    mbh = r.bh_mass()
    assert 6 < mbh["log_mbh"] < 11 and mbh["components"] == ["broad", "broad2"]
    assert mbh["recipe_key"] == "vp06"
    t = r.bh_masses()
    assert {"vp06", "db20", "db20_sigma", "db20_l5100", "shen24"} <= set(t.recipe)
    assert np.all(np.isfinite(t.log_mbh)) and np.all((t.log_mbh > 6) & (t.log_mbh < 11))
    # Fe II: a row of the line table and R_FeII
    fe = r.feii()
    assert fe["line"] == "FeII4434-4684" and fe["flux"] > 0 and 0 < fe["r_feii"] < 5
    assert "FeII4434-4684" in set(r.lines.line)
    assert "R_FeII" in set(r.key_results().quantity)
    ms = r.main_sequence()
    assert ms["r_feii"] == pytest.approx(fe["r_feii"]) and ms["population"] == ("A" if ms["fwhm"] < 4000 else "B")
    assert ms["spectral_type"][0] == ms["population"]
    from matplotlib.figure import Figure
    from fantasy_agn.plotting import plot_main_sequence
    ax = plot_main_sequence([r, r], highlight=0, fig=Figure()).axes[0]
    assert ax.get_xlim()[1] >= 3 and ax.get_ylim()[1] >= 16000
    h = fa.fit(spec, fa.presets.halpha(), verbose=False)
    assert h.rchi2 < 3 and not h.at_limit
    gh, rv = h.bh_mass("Ha"), h.bh_mass("Ha", recipe="rv15")
    assert gh["recipe"] == "Greene & Ho 2005" and gh["luminosity"] == "broad line" and 5 < gh["log_mbh"] < 11
    log_l = (gh["log_mbh"] - 6.30 - 2.06 * np.log10(h.measure("Ha", gh["components"])["fwhm"] / 1000)) / 0.55 + 42
    assert rv["log_mbh"] == pytest.approx(gh["log_mbh"] + 0.27 - 0.08 * (log_l - 42), abs=1e-3)


def test_bh_recipes_reference_values():
    from fantasy_agn.result import BH_RECIPES

    def logm(line, key, lum, width):
        rec = BH_RECIPES[line][key]
        return rec["a"] + rec["c"] * np.log10(width / 1000) + rec["b"] * (np.log10(lum) - rec["lnorm"])

    # Greene & Ho (2005): M = 2.0e6 Msun at L(Ha) = 1e42 erg/s, FWHM = 1000 km/s
    assert logm("Ha", "gh05", 1e42, 1000) == pytest.approx(np.log10(2.0e6), abs=0.005)
    # Dalla Bonta et al. (2020) eqs. 38 and 40 with log f = 0.683, for Mrk 335 (their Table A1):
    # log L(Hb) = 42.083, FWHM = 1792 km/s, sigma = 1380 km/s
    lhb = 10 ** 42.083
    assert logm("Hb", "db20", lhb, 1792) == pytest.approx(
        0.683 + 7.015 + 0.784 * 0.083 + 1.387 * (np.log10(1792) - 3.5), abs=1e-3)
    assert logm("Hb", "db20_sigma", lhb, 1380) == pytest.approx(
        0.683 + 7.530 + 0.703 * 0.083 + 2.183 * (np.log10(1380) - 3.5), abs=1e-3)
    # their L5100 form goes through eq. 24: log L5100 = 43.396 + 1.003 (log L(Hb) - 41.746)
    l5100 = 10 ** (43.396 + 1.003 * (42.083 - 41.746))
    assert logm("Hb", "db20_l5100", l5100, 1792) == pytest.approx(logm("Hb", "db20", lhb, 1792), abs=2e-3)
    assert logm("Hb", "db20_sigma_l5100", l5100, 1380) == pytest.approx(logm("Hb", "db20_sigma", lhb, 1380), abs=2e-3)
    # Shen et al. (2024): log M = 0.85 + 0.5 log(L5100/1e44) + 2 log FWHM; Mg II -2.05 + 0.6 log(L3000/1e45) + 3 log FWHM
    assert logm("Hb", "shen24", 1e44, 3000) == pytest.approx(0.85 + 2 * np.log10(3000), abs=1e-6)
    assert logm("MgII2798", "shen24", 1e45, 3000) == pytest.approx(-2.05 + 3 * np.log10(3000), abs=1e-6)


def test_fantasy1_line_sets():
    import pandas as pd
    from fantasy_agn.lines import DATA
    for key, (label, fname) in fa.LINE_SETS.items():
        old = pd.read_csv(os.path.join(DATA, fname + ".csv"), skipinitialspace=True)
        names = fa.line_set(key)
        # every line of the old list is in the library (duplicated positions count once)
        assert len(names) == old.position.round(2).nunique(), key
        assert all(n in fa.LINES for n in names)
    assert {"[FeVII]6087", "[FeX]6375", "[FeXIV]5303", "[FeXI]7892", "[ArXIV]4412"} <= set(fa.line_set("coronal"))
    assert len(fa.line_set("feII_forbidden", 4000, 5600)) > 50
    assert fa.line_info("[FeII]4287")["series"] == "[Fe II] 7F"


def test_multi_epoch_intercalibration():
    rng = np.random.default_rng(4)
    spectra, scales = [], []
    for k in range(4):
        m = _toy_truth()
        m.params["broad.Hb.flux"].value *= 1 + 0.3 * np.sin(k)
        s = 1.0 if k == 0 else rng.uniform(0.85, 1.15)
        wave = np.arange(4650, 5150, 1.0)
        flux = s * m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 2.0, wave.size)
        spectra.append(fa.Spectrum(wave, flux, np.full(wave.size, 2.0), rest_frame=True,
                                   mjd=50000 + k))
        scales.append(s)
    res = fa.fit_epochs(spectra, _toy_model(), reference=0, verbose=False)
    sc = res.scales
    assert np.all(np.abs(sc.scale - scales)[1:] < 4 * sc.error[1:])
    # default reference (highest S/N): scales are relative to that epoch
    res2 = fa.fit_epochs(spectra, _toy_model(), verbose=False)
    sc2 = res2.scales
    expected = np.array(scales) / scales[res2.reference]
    others = sc2.epoch != res2.reference
    assert sc2.reference.sum() == 1 and sc2.scale[res2.reference] == 1.0
    assert np.all(np.abs(sc2.scale - expected)[others] < 4 * sc2.error[others])
    lc = res.light_curve()
    assert list(lc.columns[:3]) == ["epoch", "mjd", "name"] and "Hb_flux" in lc


def test_spectrum_readers_and_preprocessing(tmp_path):
    wave = np.linspace(4000, 6000, 300)
    path = tmp_path / "spec.txt"
    np.savetxt(path, np.column_stack([wave, np.ones_like(wave), 0.1 * np.ones_like(wave)]))
    s = fa.read(str(path), z=0.1)
    assert s.medium == "air" and s.z == 0.1
    r = s.to_rest_frame()
    assert np.isclose(r.wave[0], 4000 / 1.1) and np.isclose(r.flux[0], 1.1)
    c = r.crop(3700, 5000).mask((4000, 4100))
    assert c.wave.max() <= 5000 and not c.good[(c.wave >= 4000) & (c.wave <= 4100)].any()
    d = s.deredden(ebv=0.1)
    assert np.all(d.flux > s.flux)


def test_unconstrained_amplitudes_are_fixed():
    spec = _toy_spectrum(np.random.default_rng(5))
    model = _toy_model() + fa.Lines("extra", ["Ha"], fwhm=(3000, 1000, 9000))
    r = fa.fit(spec, model, verbose=False)  # H-alpha is far outside 4700-5100 A
    assert "extra.Ha.flux" in r.unconstrained
    assert {"extra.fwhm", "extra.shift"} <= set(r.unconstrained)


def test_optical_preset_follows_blue_continuum():
    # A single power law cannot follow the continuum curvature over 3700-6900 A;
    # the default optical model (broken power law + Balmer continuum) must.
    spec = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).to_rest_frame()
    r = fa.fit(spec, fa.presets.optical(), verbose=False)
    resid = (spec.flux - r.model_flux()) / spec.err
    blue = r.mask & (spec.wave < 4800)
    assert r.rchi2 < 1.3
    assert np.mean(resid[blue] ** 2) < 1.2


@needs_templates
def test_negative_host_is_flagged():
    spec = fa.read(os.path.join(EXAMPLES, "spec-2770-54510-0433.fits")).to_rest_frame()
    r = fa.fit(spec, fa.presets.optical(host=True), verbose=False)
    if r.negative_host_fraction() > 0.1:
        assert "host galaxy is negative" in r.summary()


def test_any_number_of_broad_components():
    from fantasy_agn.presets import broad_fwhm_ranges
    for n in (2, 3, 4, 5):
        ranges = broad_fwhm_ranges(n)
        assert len(ranges) == n
        assert all(a[1] < b[1] and a[2] < b[2] for a, b in zip(ranges, ranges[1:]))
    m = fa.presets.hbeta(broad_components=4)
    assert [c.name for c in m.components if getattr(c, "role", None) == "broad" and "Hb" in c.line_names] == \
        ["broad", "broad2", "broad3", "broad4"]


def test_high_order_balmer_does_not_double_count():
    # H-gamma and H-delta are fitted as separate lines; the template must not contain them.
    assert fa.HighOrderBalmer()._wave.max() < 4000


def test_empty_spectrum_is_not_fitted():
    # A file whose pixels all have zero inverse variance must not get invented errors.
    wave = np.arange(4000.0, 6000.0)
    s = fa.Spectrum(wave, np.zeros_like(wave), np.full_like(wave, np.inf), rest_frame=True)
    assert s.good.sum() == 0
    with pytest.raises(ValueError):
        fa.fit(s, _toy_model(), verbose=False)


def test_line_fit_with_errors_and_scatter():
    from fantasy_agn.plotting import fit_line
    slopes = []
    for seed in range(15):
        rng = np.random.default_rng(seed)
        x = rng.uniform(43, 45, 80)
        y = 2 - 0.3 * (x - 44) + rng.normal(0, 0.08, 80) + rng.normal(0, 0.05, 80)
        f = fit_line(x + rng.normal(0, 0.05, 80), y, np.full(80, 0.05), np.full(80, 0.05),
                     n_boot=50, seed=seed)
        slopes.append(f["b"])
    assert abs(np.mean(slopes) + 0.3) < 0.02
    assert abs(f["scatter"] - 0.08) < 0.03


@pytest.mark.parametrize("profile", ["gaussian", "lorentzian", "voigt", "gauss-hermite"])
@pytest.mark.parametrize("resolution", [0.0, 150.0])
def test_line_profiles(profile, resolution):
    from fantasy_agn.profiles import evaluate_profile, PROFILES
    x = np.linspace(4000, 5800, 72001)            # 0.025 A steps
    lam0 = np.array([4862.68, 5008.24])
    values = {"shift": np.array([300.0, -150.0]), "fwhm": np.array([2500.0, 900.0]),
              "fwhm_l": np.array([1800.0, 400.0]), "h3": np.array([0.08, -0.05]),
              "h4": np.array([0.05, -0.03])}
    kin = [values[k] for k in PROFILES[profile]]
    sig_inst = resolution / 2.354820045
    P, derivs = evaluate_profile(profile, x, lam0, kin, sig_inst)
    dx = x[1] - x[0]
    # unit flux (Lorentzian wings beyond the grid hold a known small fraction)
    tol = 2e-2 if profile in ("lorentzian", "voigt") else 1e-6
    assert np.allclose(P.sum(axis=0) * dx, 1.0, atol=tol)
    for k in range(len(kin)):
        h = 1e-4 * max(abs(kin[k][0]), 1.0)
        up = [v.copy() for v in kin]; dn = [v.copy() for v in kin]
        up[k][0] += h; dn[k][0] -= h
        num = (evaluate_profile(profile, x, lam0, up, sig_inst)[0][:, 0]
               - evaluate_profile(profile, x, lam0, dn, sig_inst)[0][:, 0]) / (2 * h)
        assert np.max(np.abs(num - derivs[k][:, 0])) < 1e-5 * np.max(np.abs(num)) + 1e-12, \
            PROFILES[profile][k]


def test_profiles_match_references():
    from scipy.special import voigt_profile
    from fantasy_agn.profiles import evaluate_profile, C_KMS
    x = np.linspace(4700, 5000, 3001)
    lam0, fwhm_g, fwhm_l, inst = np.array([4862.68]), 1500.0, 1200.0, 150.0
    V, _ = evaluate_profile("voigt", x, lam0, [np.zeros(1), np.array([fwhm_g]),
                                                np.array([fwhm_l])], inst / 2.354820045)
    sig = lam0[0] * np.hypot(fwhm_g, inst) / 2.354820045 / C_KMS
    gam = lam0[0] * fwhm_l / 2 / C_KMS
    assert np.allclose(V[:, 0], voigt_profile(x - lam0[0], sig, gam), rtol=1e-6, atol=1e-12)
    L, _ = evaluate_profile("lorentzian", x, lam0, [np.zeros(1), np.array([fwhm_l])], 0.0)
    lorentz = gam / np.pi / ((x - lam0[0]) ** 2 + gam ** 2)
    assert np.allclose(L[:, 0], lorentz, rtol=1e-3)
    # positive h3 puts more flux on the red side of the peak
    GH, _ = evaluate_profile("gauss-hermite", x, lam0, [np.zeros(1), np.array([2000.0]),
                                                         np.array([0.1]), np.zeros(1)], 0.0)
    red = x > lam0[0]
    assert GH[red, 0].sum() > GH[~red, 0].sum()


def test_fit_recovers_non_gaussian_profiles():
    rng = np.random.default_rng(7)
    wave = np.arange(4600.0, 5150.0, 1.0)
    truth = fa.Model(fa.PowerLaw(pivot=5000),
                     fa.Lines("broad", ["Hb"], profile="gauss-hermite",
                              fwhm=(3500, 1000, 9000), shift=(0, -2000, 2000)))
    truth.set("continuum.amp", 40).set("continuum.index", -1.0)
    truth.set("broad.Hb.flux", 4000).set("broad.fwhm", 3500).set("broad.shift", 250)
    truth.set("broad.h3", 0.12).set("broad.h4", 0.06)
    flux = truth.evaluate(wave, {"medium": "vacuum", "resolution": 150.0})
    spec = fa.Spectrum(wave, flux + rng.normal(0, 1.0, wave.size), np.ones_like(wave),
                       rest_frame=True, resolution=150.0)
    model = fa.Model(fa.PowerLaw(pivot=5000),
                     fa.Lines("broad", ["Hb"], profile="gauss-hermite",
                              fwhm=(3000, 1000, 9000), shift=(0, -2000, 2000)))
    r = fa.fit(spec, model, verbose=False)
    p = r.params.set_index("parameter")
    for name, true in [("broad.h3", 0.12), ("broad.h4", 0.06), ("broad.fwhm", 3500),
                       ("broad.Hb.flux", 4000)]:
        assert abs(p.loc[name, "value"] - true) < 4 * p.loc[name, "error"], name




@needs_templates
def test_stellar_host_jacobian_and_copy():
    h = fa.StellarPopulation(reddening=True)
    x = np.arange(4150.0, 5550.0, 1.0)
    ctx = {"medium": "vacuum", "resolution": 150.0}
    p = np.array([q.value for q in h.params], float)
    n = h.n_templates
    p[:n] = np.random.default_rng(0).uniform(0, 1, n)
    p[n:n + 3] = 120.0, 180.0, 0.3
    f, J = h.evaluate(x, p, ctx)
    for i in list(range(0, n, 7)) + [n, n + 1, n + 2]:
        d = 1e-4 * max(abs(p[i]), 1.0)
        a, b = p.copy(), p.copy()
        a[i] += d
        b[i] -= d
        num = (h.evaluate(x, a, ctx)[0] - h.evaluate(x, b, ctx)[0]) / (2 * d)
        assert np.max(np.abs(num - J[:, i])) < 1e-6 * np.max(np.abs(num)) + 1e-12
    import copy
    h2 = copy.deepcopy(h)
    assert h2._T is h._T and h2.params[0] is not h.params[0]
    # the host is the same function of wavelength on any grid
    x2 = np.arange(4000.0, 6000.0, 0.7)
    on_x2 = np.interp(x[100:-100], x2, h.evaluate(x2, p, ctx)[0])
    assert np.max(np.abs(on_x2 - f[100:-100])) < 2e-3 * np.max(f)


@needs_templates
def test_stellar_host_recovers_kinematics():
    from fantasy_agn.templates import load
    from fantasy_agn.wavelengths import air_to_vac
    d = load("emiles")
    lam = air_to_vac(d["lam"])
    k = (np.argmin(abs(d["ages"] - 5.0)), np.argmin(abs(d["metals"])))
    tpl = d["templates"][:, k[0], k[1]]
    wave = np.arange(4300.0, 5500.0, 1.0)
    sigma, vel, res = 200.0, 80.0, 150.0
    # independent truth: direct convolution in wavelength
    fine = np.arange(4200, 5600, 0.05)
    t = np.interp(fine, lam, tpl)
    sig_t = np.interp(wave, lam, d["fwhm"]) / 2.3548 / wave * 299792.458
    sig_tot = np.sqrt(sigma ** 2 + (res / 2.3548) ** 2 - sig_t ** 2)
    gal = np.empty_like(wave)
    for j, xj in enumerate(wave):
        c = xj / (1 + vel / 299792.458)
        s = c * sig_tot[j] / 299792.458
        m = np.abs(fine - c) < 6 * s
        kern = np.exp(-0.5 * ((fine[m] - c) / s) ** 2)
        gal[j] = np.sum(kern * t[m]) / kern.sum()
    gal *= 50 / np.median(gal)
    err = gal / 60
    rng = np.random.default_rng(3)
    spec = fa.Spectrum(wave, gal + rng.normal(0, err), err, rest_frame=True, resolution=res)
    model = fa.Model(fa.StellarPopulation(), wave_range=(4300, 5500))
    r = fa.fit(spec, model, verbose=False)
    hp = r.host_properties()
    assert abs(hp["sigma_star"][0] - sigma) < 4 * hp["sigma_star"][1]
    assert abs(hp["v_star"][0] - vel) < 4 * hp["v_star"][1]
    assert 0.8 < r.rchi2 < 1.25


@needs_templates
def test_stellar_host_is_never_negative():
    spec = fa.read(os.path.join(EXAMPLES, "spec-2770-54510-0433.fits")).to_rest_frame()
    r = fa.fit(spec, fa.presets.hbeta(host=True), verbose=False)
    assert r.negative_host_fraction() == 0
    hp = r.host_properties()
    assert hp["host_fraction_5100"][0] < 0.25
    # a luminous quasar shows no stellar absorption: that must be flagged
    warning = hp.get("warning", "")
    assert "not detected" in warning or "not reliable" in warning


def test_extinction_jacobian_and_recovery():
    model = _toy_model() + fa.Extinction(law="F99")
    model.fix("continuum.index", -1.2)
    x = np.arange(4650.0, 5150.0, 1.0)
    ctx = {"medium": "vacuum", "resolution": 0.0}
    full = model.params.full_values()
    full[model.params.index("extinction.ebv")] = 0.3
    _, J = model.evaluate_with_jacobian(x, ctx, full)
    for i in range(len(full)):
        h = 1e-6 * max(abs(full[i]), 1)
        a, b = full.copy(), full.copy()
        a[i] += h
        b[i] -= h
        num = (model.evaluate(x, ctx, a) - model.evaluate(x, ctx, b)) / (2 * h)
        assert np.max(np.abs(num - J[:, i])) < 1e-4 * (np.max(np.abs(num)) + 1e-30)
    # reddened toy spectrum over a wide range: E(B-V) and intrinsic flux are recovered
    truth = _toy_truth() + fa.Extinction(law="F99")
    truth.set("extinction.ebv", 0.25)
    wave = np.arange(3800.0, 7000.0, 1.0)
    rng = np.random.default_rng(11)
    flux = truth.evaluate(wave, ctx)
    spec = fa.Spectrum(wave, flux + rng.normal(0, 0.5, wave.size), np.full(wave.size, 0.5),
                       rest_frame=True)
    m = _toy_model() + fa.Extinction(law="F99")
    m.wave_range = None
    m.fix("continuum.index", -1.2)
    r = fa.fit(spec, m, verbose=False)
    p = r.params.set_index("parameter")
    assert abs(p.loc["extinction.ebv", "value"] - 0.25) < 4 * p.loc["extinction.ebv", "error"]
    assert abs(p.loc["broad.Hb.flux", "value"] - 6000) < 4 * p.loc["broad.Hb.flux", "error"]


def test_mcmc_matches_covariance_errors():
    spec = _toy_spectrum(np.random.default_rng(1))
    rc = fa.fit(spec, _toy_model(), verbose=False)
    rm = fa.fit(spec, _toy_model(), errors="mcmc", seed=1, verbose=False)
    assert rm.mcmc_info["converged"]
    pc, pm = rc.params.set_index("parameter"), rm.params.set_index("parameter")
    for n in pc[pc.status == "free"].index:
        assert 0.8 < pm.loc[n, "error"] / pc.loc[n, "error"] < 1.2, n


def test_hmc_samples_a_gaussian():
    from fantasy_agn.sampling import hmc
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T + 5 * np.eye(5)
    prec = np.linalg.inv(cov)
    mean = np.arange(5.0)

    def lpg(t):
        d = t - mean
        return -0.5 * d @ prec @ d, -prec @ d

    # precondition with a deliberately wrong (diagonal) covariance
    draws, info = hmc(lpg, mean + 1, np.diag(np.diag(cov)), np.full(5, -np.inf),
                      np.full(5, np.inf), seed=1)
    assert info["converged"]
    n = info["ess"]
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4 * sd / np.sqrt(n))
    # standard error of a sample covariance: sqrt((C_ii C_jj + C_ij^2) / n)
    se = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov ** 2) / n)
    assert np.all(np.abs(np.cov(draws.T) - cov) < 4 * se)


def test_hmc_respects_and_reflects_at_bounds():
    from fantasy_agn.sampling import hmc
    # half-normal: N(0, 1) truncated at 0 has mean sqrt(2/pi) and variance 1 - 2/pi
    lpg = lambda t: (-0.5 * float(t @ t), -t)
    draws, info = hmc(lpg, np.array([0.5, 0.5]), np.eye(2), np.array([0.0, -np.inf]),
                      np.array([np.inf, np.inf]), n_samples=1500, seed=2)
    assert draws[:, 0].min() >= 0
    assert abs(draws[:, 0].mean() - np.sqrt(2 / np.pi)) < 0.06
    assert abs(draws[:, 0].var() - (1 - 2 / np.pi)) < 0.06
    assert abs(draws[:, 1].mean()) < 0.1 and abs(draws[:, 1].var() - 1) < 0.15


def _write_desi_files(tmp_path, rng):
    """Synthetic DESI coadd + redrock files following the DESI data model."""
    from astropy.table import Table
    from scipy.sparse import dia_matrix
    from fantasy_agn.components import FWHM_PER_SIGMA
    # the real DESI arms, all on one 0.8 A grid starting at 3600 A
    arms = {"B": (3600.0, 5800.0, 1.9), "R": (5760.0, 7620.0, 1.7), "Z": (7520.0, 9824.0, 1.5)}
    tids = np.array([39627001, 39627002, 39627003], dtype=np.int64)
    z = 0.2
    truth = _toy_truth()
    truth.wave_range = None
    hdus = [fits.PrimaryHDU()]
    fm = Table({"TARGETID": tids, "TARGET_RA": [150.1, 150.2, 150.3],
                "TARGET_DEC": [2.1, 2.2, 2.3], "MEAN_MJD": [59300.5, 59301.5, 59302.5]})
    hdus.append(fits.BinTableHDU(fm, name="FIBERMAP"))
    offsets = np.arange(5, -6, -1)
    for arm, (lo, hi, fwhm_a) in arms.items():
        wave = np.arange(lo, hi + 0.01, 0.8)
        n = wave.size
        sig_pix = fwhm_a / FWHM_PER_SIGMA / 0.8 * (1 + 0.1 * (wave - wave.mean()) / np.ptp(wave))
        diag = np.exp(-0.5 * (offsets[:, None] / sig_pix[None, :]) ** 2)
        diag /= diag.sum(axis=0, keepdims=True)
        res = dia_matrix((diag, offsets), shape=(n, n))
        flux = np.empty((3, n)); ivar = np.empty((3, n)); mask = np.zeros((3, n), np.int32)
        for k in range(3):
            # intrinsic (no instrument) spectrum in the observed frame, then the LSF
            intrinsic = truth.evaluate(wave / (1 + z), {"medium": "vacuum", "resolution": 0.0})
            intrinsic = intrinsic / (1 + z) * (1 + 0.3 * k)
            clean = res @ intrinsic
            noise = 0.4
            flux[k] = clean + rng.normal(0, noise, n)
            ivar[k] = 1 / noise ** 2
        mask[0, 100:140] = 1                    # a masked stretch in the B arm
        hdus += [fits.ImageHDU(wave, name=f"{arm}_WAVELENGTH"),
                 fits.ImageHDU(flux, name=f"{arm}_FLUX"),
                 fits.ImageHDU(ivar, name=f"{arm}_IVAR"),
                 fits.ImageHDU(mask, name=f"{arm}_MASK"),
                 fits.ImageHDU(np.repeat(diag[None], 3, axis=0), name=f"{arm}_RESOLUTION")]
    coadd = tmp_path / "coadd-main-dark-12345.fits"
    fits.HDUList(hdus).writeto(coadd)
    rr = Table({"TARGETID": tids, "Z": [z, 1.3, 0.5], "ZERR": [1e-4] * 3, "ZWARN": [0, 0, 4],
                "SPECTYPE": ["QSO", "QSO", "GALAXY"], "DELTACHI2": [500.0, 300.0, 20.0]})
    redrock = tmp_path / "redrock-main-dark-12345.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(rr, name="REDSHIFTS")]).writeto(redrock)
    return str(coadd), str(redrock), tids, z


def test_desi_coadd_reader_and_resolution(tmp_path):
    coadd, redrock, tids, z = _write_desi_files(tmp_path, np.random.default_rng(5))
    t = fa.desi.targets(coadd, redrock=redrock)
    assert list(t.TARGETID) == list(tids) and list(t.SPECTYPE) == ["QSO", "QSO", "GALAXY"]
    with pytest.raises(ValueError):
        fa.read(coadd)                          # several targets: must choose one
    s = fa.read(coadd, targetid=tids[0], redrock=redrock)
    assert s.z == z and s.medium == "vacuum" and s.flux_unit == 1e-17 and s.mjd == 59300.5
    assert np.all(np.diff(s.wave) > 0) and s.wave[0] == 3600.0 and s.wave[-1] > 9800
    assert np.allclose(np.diff(s.wave), 0.8)    # arms merged on the common grid
    assert not s.good[100:140].any()            # masked pixels excluded
    # the resolution curve follows the LSF: FWHM 1.9 A in B at 4000 A ~ 142 km/s
    wc, fc = s.resolution_curve
    b = np.arange(3600.0, 5800.01, 0.8)
    expected = 1.9 * (1 + 0.1 * (4000 - b.mean()) / np.ptp(b)) / 4000 * 299792.458
    assert abs(np.interp(4000, wc, fc) - expected) < 3
    assert np.interp(9000, wc, fc) < np.interp(4000, wc, fc)
    # fitting the toy lines recovers the intrinsic narrow width (instrument removed
    # with the resolution at each line's wavelength)
    rest = s.to_rest_frame()
    r = fa.fit(rest, _toy_model(), verbose=False)
    p = r.params.set_index("parameter")
    assert abs(p.loc["narrow.fwhm", "value"] - 450) < 4 * p.loc["narrow.fwhm", "error"] + 10
    assert abs(p.loc["broad.Hb.flux", "value"] - 6000) < 4 * p.loc["broad.Hb.flux", "error"]
    qsos = fa.desi.read_all(coadd, redrock=redrock, spectype="QSO")
    assert [q.meta["targetid"] for q in qsos] == list(tids[:2])


def test_desi_from_sparcl_record():
    wave = np.arange(3600.0, 9824.0, 0.8)
    rec = {"wavelength": wave, "flux": np.ones_like(wave), "ivar": np.full_like(wave, 4.0),
           "mask": np.zeros(wave.size, int), "redshift": 0.3, "targetid": 123,
           "wave_sigma": np.full_like(wave, 0.8), "data_release": "DESI-DR1", "ra": 1.0, "dec": 2.0}
    s = fa.desi.from_sparcl(rec)
    assert s.z == 0.3 and np.allclose(s.err, 0.5) and s.name == "desi-123"
    assert abs(np.interp(4000, *s.resolution_curve) - 2.3548 * 0.8 / 4000 * 299792.458) < 1


def test_build_up_never_worse_and_nested():
    # adding broad components must not increase chi2 (nested models)
    spec = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).to_rest_frame()
    chi2 = [fa.fit(spec, fa.presets.halpha(broad_components=n), verbose=False).chi2
            for n in (1, 2, 3)]
    assert chi2[0] >= chi2[1] - 1e-6 and chi2[1] >= chi2[2] - 1e-6
    direct = fa.fit(spec, fa.presets.halpha(broad_components=3), build_up=False, verbose=False)
    staged = fa.fit(spec, fa.presets.halpha(broad_components=3), verbose=False)
    assert staged.chi2 <= direct.chi2 + 1e-6


def test_compare_models():
    spec = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).to_rest_frame()
    variants = fa.presets.variants("halpha", broad_components=(1, 2))
    assert set(variants) == {"1 broad", "1 broad + outflow", "2 broad", "2 broad + outflow"}
    cmp = fa.compare(spec, variants)
    t = cmp.table
    assert list(t.BIC) == sorted(t.BIC) and t.acceptable.iloc[0]
    s = cmp.summary().set_index("quantity")
    assert {"Ha FWHM", "Ha sigma", "Ha flux"} <= set(s.index)
    row = s.loc["Ha FWHM"]
    assert row.min_all <= row.value <= row.max_all
    if row.n_acceptable >= 2:
        assert row.sys_err > 0 and row.total_err >= row.stat_err
    else:
        assert np.isnan(row.sys_err)


def test_max_ratios():
    comp = fa.Lines("outflow", ["[NII]6548", "Ha", "[NII]6583"], fwhm=(900, 300, 2000),
                    ratios={"[NII]6548": ("[NII]6583", 1 / 3.05)},
                    max_ratios={"[NII]6583": ("Ha", 3.0)})
    assert any(q.name.endswith("Ha.excess") for q in comp.params)
    x = np.arange(6450.0, 6680.0, 0.5)
    ctx = {"medium": "vacuum", "resolution": 100.0}
    p = np.array([10.0, 5.0, 30.0, -100.0, 900.0])
    f, J = comp.evaluate(x, p, ctx)
    assert np.allclose(comp.line_fluxes(p), [10.0, 15.0, 30.0])     # Ha = 30/3 + 5
    for i in range(p.size):
        d = 1e-5 * max(abs(p[i]), 1.0)
        a, b = p.copy(), p.copy()
        a[i] += d
        b[i] -= d
        num = (comp.evaluate(x, a, ctx)[0] - comp.evaluate(x, b, ctx)[0]) / (2 * d)
        assert np.max(np.abs(num - J[:, i])) < 1e-6 * np.max(np.abs(num)) + 1e-12
    # Data with [NII]/Ha = 6 in the wing: the fit stops at the limit of 3.
    m = fa.Model(fa.PowerLaw(), comp)
    truth = fa.Lines("t", ["[NII]6548", "Ha", "[NII]6583"])
    y = 1.0 + truth.evaluate(x, np.array([60 / 3.05, 10.0, 60.0, -100.0, 900.0]), ctx)[0]
    rng = np.random.default_rng(1)
    err = np.full_like(x, 0.02)
    spec = fa.Spectrum(x, y + rng.normal(0, 0.02, x.size), err, medium="vacuum", resolution=100.0)
    r = fa.fit(spec, m, verbose=False)
    t = r.lines.set_index("line")
    assert t.loc["[NII]6583", "flux"] <= 3.0 * t.loc["Ha", "flux"] * (1 + 1e-6)
    assert np.isclose(r.measure("Ha")["flux"], t.loc["Ha", "flux"], rtol=1e-3)


def test_halpha_tied_outflow_and_hei():
    m = fa.presets.halpha(broad_components=2, outflow={"shift": -450.0, "fwhm": 950.0})
    assert m.params["outflow.shift"].fixed and m.params["outflow.shift"].value == -450.0
    assert m.params["outflow.fwhm"].fixed and m.params["outflow.fwhm"].value == 950.0
    assert "narrow.HeI6678.flux" in m.params.names and "broad_hei.HeI6678.flux" in m.params.names
    assert "outflow" not in " ".join(fa.presets.halpha(outflow=False).params.names)
    labels = list(fa.presets.variants("halpha", broad_components=(1,),
                                      outflow=(False, True, {"shift": 0, "fwhm": 900})))
    assert labels == ["1 broad", "1 broad + outflow", "1 broad + [OIII]-tied outflow"]


def test_variants_profiles():
    v = fa.presets.variants("hbeta", broad_components=(2,), outflow=(True,),
                            broad_profile=("gaussian", "gauss-hermite"))
    assert list(v) == ["2 broad + outflow", "2 GH broad + outflow"]
    assert v["2 GH broad + outflow"].components[-2].profile == "gauss-hermite" or any(
        getattr(c, "profile", "") == "gauss-hermite" for c in v["2 GH broad + outflow"].components)


def test_broken_powerlaw_free_break():
    c = fa.BrokenPowerLaw(break_wave=(4600, 4000, 6000))
    x = np.linspace(3800, 6800, 800)
    p = np.array([2.0, -1.2, -2.3, 5030.0])
    f, J = c.evaluate(x, p, {})
    for i in range(p.size):
        d = 1e-6 * max(abs(p[i]), 1)
        a, b = p.copy(), p.copy()
        a[i] += d
        b[i] -= d
        num = (c.evaluate(x, a, {})[0] - c.evaluate(x, b, {})[0]) / (2 * d)
        assert np.max(np.abs(num - J[:, i])) < 1e-5 * np.max(np.abs(num)) + 1e-10
    fixed = fa.BrokenPowerLaw(pivot=5030.0)
    assert np.allclose(fixed.evaluate(x, p[:3], {})[0], f)


def test_line_library():
    t = fa.line_table()
    assert len(t) == len(fa.LINES) and t.index.is_unique and t.vacuum.is_monotonic_increasing
    assert set(t.medium) == {"air", "vacuum"}
    # tabulated vacuum values are used as they are, air values are converted
    assert fa.LINES["MgII2798"] == 2798.75 and fa.LINES["[NeIV]2424"] == 2423.83
    assert fa.LINES["Ha"] == pytest.approx(air_to_vac(6562.82))
    # Mg II blend = gf-weighted (2:1) mean of the vacuum doublet, within 2 km/s
    mean = (2 * fa.LINES["MgII2796"] + fa.LINES["MgII2803"]) / 3
    assert abs(fa.LINES["MgII2798"] / mean - 1) * 299792.458 < 2
    # hydrogen lines agree with the Rydberg formula (within 5 km/s: it ignores the fine structure)
    rh = 109677.583
    for name, n1, n2 in (("Ha", 2, 3), ("Hb", 2, 4), ("H12", 2, 12), ("Paa", 3, 4), ("Pab", 3, 5), ("Pag", 3, 6),
                         ("Pa9", 3, 9), ("Pa14", 3, 14), ("Brg", 4, 7), ("Bra", 4, 5), ("Br13", 4, 13),
                         ("Pfb", 5, 7), ("Pfe", 5, 10), ("Lyb", 1, 3), ("Lya", 1, 2)):
        rydberg = 1e8 / (rh * (1 / n1 ** 2 - 1 / n2 ** 2))
        assert abs(fa.LINES[name] / rydberg - 1) * 299792.458 < 5, name
    assert fa.lines_in(series="Paschen")[-1] == "Paa"
    assert fa.lines_in(12000, 22000, type="recombination", ion="H I") == [
        "Pab", "Br13", "Br12", "Br11", "Br10", "Bre", "Paa", "Brd", "Brg"]
    assert fa.line_wavelength("Pa-alpha") == fa.line_wavelength("Paalpha") == fa.LINES["Paa"]
    assert fa.line_info("Brgamma")["series"] == "Brackett"
    assert all(fa.line_info(n)["type"] == "coronal" for n in fa.lines_in(ion="Fe VII"))
    # every line can be fitted
    m = fa.Model(fa.PowerLaw()).add(fa.Lines("nir", ["Paa", "Brg", "H2-1-0S(1)", "[SiVI]19630"]))
    assert len(m.components[1].line_names) == 4


def test_line_ratio_parameter():
    lines = ["Hd", "Hg", "Hb"]
    c = fa.Lines("broad", lines, fwhm=(3000, 1000, 9000),
                 line_ratios={"Hg": ("Hb", (0.35, 0.1, 0.5)), "Hd": ("Hb", (0.19, 0.05, 0.3))})
    assert [q.name for q in c.params] == ["broad.Hb.flux", "broad.shift", "broad.fwhm", "broad.Hg.ratio",
                                          "broad.Hd.ratio"]
    x = np.linspace(4000, 5000, 600)
    p = np.array([100.0, 150.0, 4000.0, 0.4, 0.2])
    ctx = {"medium": "vacuum", "resolution": 0}
    f, J = c.evaluate(x, p, ctx)
    for k in range(len(p)):                     # analytic derivatives, including the ratios
        q = p.copy(); h = 1e-6 * max(abs(q[k]), 1); q[k] += h
        num = (c.evaluate(x, q, ctx)[0] - f) / h
        assert np.allclose(num, J[:, k], rtol=1e-4, atol=1e-6 * np.abs(J[:, k]).max())
    assert np.allclose(c.line_fluxes(p), [20.0, 40.0, 100.0])
    old = fa.Lines("narrow", ["Hb", "[OIII]5007"])     # a component saved before ratios existed
    del old._ratio_rows, old._ratio_cols
    assert np.all(np.isfinite(old.evaluate(x, np.array([1.0, 1.0, 0.0, 400.0]), ctx)[0]))
    with pytest.raises(ValueError):
        fa.Lines("b", ["Hg", "Hb"], line_ratios={"Hg": ("Ha", 0.4)})
    # a known Balmer decrement is recovered
    truth = fa.Model(fa.PowerLaw()).add(fa.Lines("broad", ["Hg", "Hb"], fwhm=(3000, 1000, 9000),
                                                   line_ratios={"Hg": ("Hb", (0.35, 0.1, 0.5))}))
    truth.set("continuum.amp", 10).set("broad.Hb.flux", 500).set("broad.Hg.ratio", 0.42).set("broad.fwhm", 4000)
    w = np.arange(4200, 5100, 1.0)
    rng = np.random.default_rng(1)
    s = fa.Spectrum(w, truth.evaluate(w, ctx) + rng.normal(0, 0.3, w.size), np.full(w.size, 0.3), rest_frame=True)
    model = truth.copy()
    model.set("broad.Hg.ratio", 0.3).set("broad.Hb.flux", 100)
    r = fa.fit(s, model, verbose=False)
    assert r.model.params["broad.Hg.ratio"].value == pytest.approx(0.42, abs=0.02)
    # presets: broad H-gamma as a ratio, H-delta tied to it; the old behaviour on request
    names = fa.presets.hbeta().params.names
    assert "broad.Hg.ratio" in names and "broad2.Hd.ratio" in names and "broad.Hg.flux" not in names
    assert fa.presets.hbeta().params["broad.Hd.ratio"].tie[0] == "broad.Hg.ratio"
    assert "broad.Hd.ratio" not in fa.presets.hbeta(wave_range=(4300, 5550)).params.names
    assert "broad.Hg.flux" in fa.presets.hbeta(balmer_decrement=False).params.names
    assert "broad.Hg.ratio" in fa.presets.optical().params.names


def test_preset_new_lines():
    m = fa.presets.optical(coronal=True, oi6046=True)
    names = m.params.names
    for ln in ("H8", "He", "[SII]4069", "[SII]4076", "[NI]5198", "[NI]5200", "HeI6678"):
        assert f"narrow.{ln}.flux" in names
    assert m.params["narrow.[NeIII]3968.flux"].tie is not None
    assert "broad_he.HeI4471.flux" in names and "broad_oi.OI6046.flux" in names
    assert "coronal.[FeVII]6087.flux" in names and "coronal_high.[FeX]6375.flux" in names
    assert "coronal" not in [c.name for c in fa.presets.optical().components]
    comps = {c.name: c for c in fa.presets.hbeta(coronal=True).components}
    assert comps["coronal"].line_names == ["[FeVII]5159"]
    assert comps["coronal_high"].line_names == ["[FeXIV]5303"]
    # library wavelengths are vacuum (air converted)
    assert abs(fa.LINES["[FeVII]6087"] - 6088.7) < 0.1


def test_balmer_continuum_broadening():
    x = np.arange(3000.0, 4500.0, 1.0)
    sharp = fa.BalmerContinuum(fwhm=0.0)
    smooth = fa.BalmerContinuum()
    f0 = sharp.evaluate(x, np.array([1.0, 15000.0, 1.0, 0.0]), {})[0]
    f1 = smooth.evaluate(x, np.array([1.0, 15000.0, 1.0, 5000.0]), {})[0]
    assert f0[x > 3650].max() == 0 and f1[(x > 3650) & (x < 3670)].min() > 0.05
    # flux is conserved (to the edge of the grid) and far below the edge nothing changes
    assert abs(f0.sum() - f1.sum()) < 0.01 * f0.sum()
    assert np.allclose(f0[x < 3300], f1[x < 3300], rtol=0.02)
    import copy
    c = copy.deepcopy(smooth)
    assert c._cache == {} and c.params is not smooth.params


def test_optical_physical():
    m = fa.presets.optical(broad_components=3, outflow=True, physical=True, host="pca")
    assert m.wave_range == (3550, 6900)
    assert m.params["continuum.index"].fixed and m.params["continuum.index"].value == -1.7
    assert "host" in [c.name for c in m.components] and "balmer_cont" in [c.name for c in m.components]
    for c in m.components:
        if getattr(c, "role", "") == "broad" and f"{c.name}.fwhm" in m.params.names:
            assert m.params[f"{c.name}.fwhm"].max <= 15000 * 1.3 + 1
    assert m.params["outflow_ha.fwhm"].tie is not None
    free = fa.presets.optical(physical=True, slope=None, host="pca")
    p = free.params["continuum.index"]
    assert not p.fixed and np.isclose(p.min, -7 / 3) and p.max == -1.0
    assert fa.presets.optical().wave_range == (3700, 6900)


def test_balmer_decrement_warning():
    x = np.arange(4700.0, 6800.0, 1.0)
    truth = fa.Model(fa.PowerLaw("continuum", pivot=5100),
                     fa.Lines("broad", ["Hb"], fwhm=(4000, 1000, 9000)),
                     fa.Lines("broad_ha", ["Ha"], fwhm=(4000, 1000, 9000)))
    full = truth.params.full_values()
    full[truth.params.index("broad.Hb.flux")] = 100.0
    full[truth.params.index("broad_ha.Ha.flux")] = 150.0          # ratio 1.5
    ctx = {"medium": "vacuum", "resolution": 0.0}
    y = truth.evaluate(x, ctx, full)
    spec = fa.Spectrum(x, y + np.random.default_rng(2).normal(0, 0.01, x.size), np.full(x.size, 0.01),
                       medium="vacuum")
    r = fa.fit(spec, truth.copy(), verbose=False)
    ratio, err = r.balmer_decrement()
    assert abs(ratio - 1.5) < 0.02 and err > 0
    assert "Halpha/Hbeta" in r.summary(short=True)
    assert fa.fit(spec, fa.Model(fa.PowerLaw("continuum", pivot=5100),
                                 fa.Lines("broad", ["Hb"])), verbose=False).balmer_decrement() is None


def _synthetic_galaxy(n2, o3, s2, o1, ha=2000.0, broad_ha=0.0, seed=0):
    """Old stellar population (sigma* 150 km/s) plus narrow lines with given log ratios."""
    x = np.arange(3750.0, 6950.0, 1.0)
    ctx = {"medium": "vacuum", "resolution": 150.0}
    host = fa.StellarPopulation()
    p = np.zeros(len(host.params))
    p[host.n_templates // 2] = 1.0
    p[host.n_templates:host.n_templates + 2] = 0.0, 150.0
    stars = host.evaluate(x, p, ctx)[0]
    stars *= 100.0 / np.interp(5100, x, stars)
    hb = ha / 2.86
    flux = {"Hb": hb, "[OIII]5007": hb * 10 ** o3, "[OIII]4959": hb * 10 ** o3 / 2.98,
            "Ha": ha, "[NII]6583": ha * 10 ** n2, "[NII]6548": ha * 10 ** n2 / 3.05,
            "[SII]6716": 0.55 * ha * 10 ** s2, "[SII]6731": 0.45 * ha * 10 ** s2,
            "[OI]6300": ha * 10 ** o1, "[OI]6364": ha * 10 ** o1 / 3.0}
    narrow = fa.Lines("n", list(flux), fwhm=(300, 50, 1000))
    q = np.array(list(flux.values()) + [0.0, 300.0])
    y = stars + narrow.evaluate(x, q, ctx)[0]
    if broad_ha:
        b = fa.Lines("b", ["Hb", "Ha"])
        y = y + b.evaluate(x, np.array([broad_ha / 3.0, broad_ha, 0.0, 5000.0]), ctx)[0]
    err = np.full(x.size, 0.8)
    y = y + np.random.default_rng(seed).normal(0, 0.8, x.size)
    return fa.Spectrum(x, y, err, medium="vacuum", resolution=150.0, rest_frame=True, z=0.03,
                       name="synthetic")


@needs_templates
@pytest.mark.parametrize("ratios, label", [
    ((-0.6, -0.3, -0.5, -1.6), "starforming"),
    ((0.0, 1.0, -0.3, -1.0), "seyfert2"),
    ((0.25, 0.2, 0.05, -0.5), "liner"),
])
def test_classify_narrow_line_galaxies(ratios, label):
    c = fa.classify(_synthetic_galaxy(*ratios))
    assert c.label == label, c
    m = c.measurements
    assert abs(m["log_nii_ha"] - ratios[0]) < 0.08 and abs(m["log_oiii_hb"] - ratios[1]) < 0.1
    assert m["host_fraction_5100"] > 0.9 and abs(m["sigma_star"] - 150) < 40
    assert c.model().__class__.__name__ == "Model"


@needs_templates
def test_classify_type1_and_passive():
    c = fa.classify(_synthetic_galaxy(0.0, 0.9, -0.3, -1.0, broad_ha=12000.0))
    assert c.label == "type1_host" and abs(c.measurements["broad_fwhm"] - 5000) < 600
    passive = fa.classify(_synthetic_galaxy(0.0, 0.0, -0.3, -1.0, ha=5.0))
    assert passive.label == "passive"
    fig = c.plot()
    assert len(fig.axes) == 4


@needs_templates
def test_classify_seyfert1_with_broad_wing():
    # a ~2600 km/s broad line with a very broad wing: the Balmer part of the outflow wing must not take the
    # broad core and leave only the wing to the broad-line test (it then failed the FWHM limit)
    s = _synthetic_galaxy(-0.3, 0.2, -0.5, -1.4, ha=1500.0)
    ctx = {"medium": "vacuum", "resolution": 150.0}
    core = fa.Lines("core", ["Hb", "Ha"]).evaluate(s.wave, np.array([3000.0, 9000.0, 0.0, 2600.0]), ctx)[0]
    wing = fa.Lines("wing", ["Hb", "Ha"]).evaluate(s.wave, np.array([1500.0, 4500.0, -800.0, 11000.0]), ctx)[0]
    c = fa.classify(s.copy(flux=s.flux + core + wing))
    assert c.label.startswith("type1"), c


def test_bpt_boundaries():
    from fantasy_agn.classify import kauffmann03, kewley01, kewley06
    assert kauffmann03(-0.5) < kewley01(-0.5)                  # composite region exists
    assert np.isclose(kewley01(-0.5), 0.61 / -0.97 + 1.19)
    assert kewley06(0.0, "sii") == 0.76 and kewley06(0.0, "oi") == 1.30


SDSS_CLASSES = {"spec-0630-52050-0245.fits": "starforming", "spec-2139-53878-0330.fits": "seyfert2",
                "spec-1614-53120-0408.fits": "liner", "spec-0394-51913-0625.fits": "passive",
                "spec-2293-53730-0278.fits": "type1_host", "spec-0615-52347-0108.fits": "type1"}


@needs_templates
@pytest.mark.parametrize("fname, label", SDSS_CLASSES.items())
def test_classify_sdss(fname, label):
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "classes", fname)
    if not os.path.exists(path):
        pytest.skip("SDSS class examples not downloaded")
    assert fa.classify(fa.read(path).to_rest_frame()).label == label



def test_app_prune_feii_step():
    import json
    from fantasy_agn.app.session import Session
    s = Session()
    s.wf["mode"] = "single"
    assert s.wf_feii() is None
    s.run_step(s.wf_read_code(mode="single", paths=[os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")]), "# read")
    _wait(s)
    s.run_step(s.wf_prepare_code(extinction="none", wmin=4150, wmax=5550), "# prepare")
    _wait(s)
    # finding the redshift of the spectrum as read: a cell, the candidates and a figure
    assert s.wf_state()["redshift"] is None and s.wf_redshift_figure() is None
    with pytest.raises(ValueError):
        s.wf_find_redshift_code(5)
    code = s.wf_find_redshift_code(0)
    assert code.startswith("# find redshift\nzf = fa.find_redshift(raw[0])")
    s.run_step(code, "# find redshift")
    _wait(s)
    zs = s.wf_state()["redshift"]
    json.dumps(zs)
    assert s.error is None and zs["k"] == 0 and zs["secure"] and zs["kind"] == "quasar" and len(zs["candidates"]) >= 2
    assert abs(zs["z"] - zs["file_z"]) < 1e-3 and zs["candidates"][0]["dchi2"] == 0 and zs["candidates"][1]["dchi2"] > 0.2
    assert s.wf_redshift_figure()[:4] == b"\x89PNG"
    # the instrumental resolution: from the file by default, or given as FWHM or R
    row = s.loaded_state()["rows"][0]
    assert 100 < row["resolution"] < 200 and row["resolution_varies"] and 60 < row["pixel_kms"] < 80
    assert not s.prepared_state()["resolution_given"] and "with_resolution" not in s.cells[-1].code
    code = s.wf_prepare_code(extinction="none", wmin=4150, wmax=5550, resolution="2000", resolution_unit="R")
    assert ".with_resolution(R=2000)" in code and s.wf["prepare"]["resolution"] == 2000
    assert ".with_resolution(150)" in s.wf_prepare_code(extinction="none", resolution=150)
    for bad in ({"resolution": "abc"}, {"resolution": -1}, {"resolution": 10, "resolution_unit": "R"},
                {"resolution": 9000}, {"resolution": 100, "resolution_unit": "A"}):
        with pytest.raises(ValueError):
            s.wf_prepare_code(extinction="none", **bad)
    s.run_step(code, "# prepare")
    _wait(s)
    assert s.error is None and s.prepared_state()["resolution_given"] and abs(s.prepared_state()["resolution"] - 149.9) < 0.1
    s.run_step(s.wf_prepare_code(extinction="none", wmin=4150, wmax=5550), "# prepare")
    _wait(s)
    assert not s.prepared_state()["resolution_given"] and s.wf["prepare"]["resolution"] == "file"
    code = s.wf_prepare_code(z_source="median", extinction="none")
    assert "z = float(np.median([s.z for s in raw if s.z]))" in code and ".to_rest_frame(z=z)" in code
    with pytest.raises(ValueError):
        s.wf_prepare_code(z_source="nonsense")
    s.run_cell("spec = spectra[0]\n" + s.preset_code("hbeta", broad_components=1, outflow=False))
    _wait(s)
    with pytest.raises(ValueError):
        s.wf_prune_feii_code()                               # nothing fitted yet
    s.run_step(s.wf_fit_code(errors="none"), "# fit")
    _wait(s)
    assert s.wf_feii()["errors"] is False and s.wf_feii()["rows"] == []
    with pytest.raises(ValueError, match="no errors"):
        s.wf_prune_feii_code()
    s.run_step(s.wf_fit_code(errors="covariance"), "# fit")
    _wait(s)
    assert s.wf_state()["fit"]["epochs"][0]["undersampled"] == []      # SDSS: the narrow lines are resolved
    full = s.result
    f = s.wf_state()["feii"]
    json.dumps(f)                                            # the state the browser polls
    assert f["errors"] and not f["pruned"] and f["before"] is None and 0 < f["weak"] < len(f["rows"])
    with pytest.raises(ValueError):
        s.wf_prune_feii_undo_code()
    with pytest.raises(ValueError):
        s.wf_prune_feii_code(nsigma=0)
    s.run_step(s.wf_prune_feii_code(), "# prune Fe II")
    _wait(s)
    assert s.error is None and s.result is not full and s.ns["r_feii_all"] is full
    g = s.wf_state()["feii"]
    json.dumps(g)
    assert g["pruned"] and g["weak"] == f["weak"] and g["before"]["bic"] > g["now"]["bic"] and g["line"]
    assert s.wf_state()["fit"]["kind"] == "single" and s.wf_result(0)["fit"]["rchi2"] > 0
    # another threshold starts again from the full fit and replaces the cell
    code = s.wf_prune_feii_code(nsigma=3)
    assert "nsigma=3" in code
    s.run_step(code, "# prune Fe II")
    _wait(s)
    assert s.error is None and s.ns["r_feii_all"] is full and len(s.result.feii_pruned) >= g["weak"]
    assert sum(c.code.startswith("# prune Fe II") for c in s.cells) == 1
    s.run_cell(s.wf_prune_feii_undo_code())
    _wait(s)
    assert s.result is full and not s.wf_state()["feii"]["pruned"]
    # the simulation check of the fit: a cell, a small state, stale after the next fit
    assert s.wf_state()["simcheck"] == {"id": 0}
    with pytest.raises(ValueError):
        s.wf_simcheck_code(n=3)
    s.run_step(s.wf_simcheck_code(n=10), "# simulation check")
    _wait(s)
    m = s.wf_state()["simcheck"]
    json.dumps(m)
    assert s.error is None and m["n"] + m["failed"] == 10 and m["pruned"] is None
    assert {r["label"] for r in m["rows"]} >= {"broad H\u03b2 FWHM", "Fe II flux"} and all(r["flag"] for r in m["rows"])
    # a new fit makes the kept fit stale: nothing to go back to
    s.run_step(s.wf_prune_feii_code(), "# prune Fe II")
    _wait(s)
    s.run_step(s.wf_fit_code(errors="covariance"), "# fit")
    _wait(s)
    assert not s.wf_state()["feii"]["pruned"] and s.wf_state()["feii"]["before"] is None
    assert s.wf_state()["simcheck"] == {"id": 0}
    # tying the narrow lines' kinematics to another component: judged on this fit, no new fit
    t = s.wf_tie_check("narrow")
    json.dumps(t)
    assert t["available"] and t["rows"] and all(r["verdict"] for r in t["rows"]) and "FWHM and flux" in t["targets"]
    assert not s.wf_tie_check("nothing")["available"]
    # undetected or weak lines get an upper limit in the table
    rows = s._line_rows(s.result)
    assert any(r["limit"] for r in rows if not r["detected"]) or all(r["detected"] for r in rows)
    # the one-file report
    from fantasy_agn.app.report import build
    page = build(s)
    assert page.startswith("<!doctype html>") and "data:image/png;base64," in page and "Settings" in page
    assert "fa.simulation_check" not in page or "Checks" in page
    assert "import fantasy_agn as fa" in page


def _wait(session, timeout=300):
    import time
    t0 = time.time()
    while session.busy and time.time() - t0 < timeout:
        time.sleep(0.05)
    assert not session.busy


def test_app_session_cells():
    from fantasy_agn.app.session import Session
    s = Session()
    s.run_cell("x = 2\nprint('hello')\nx * 3")
    _wait(s)
    c = s.cells[-1]
    assert c.n == 2 and [o[0] for o in c.out] == ["hello", "6"]
    s.run_cell("1 / 0")                                   # errors are shown, not raised
    _wait(s)
    assert s.cells[-1].out[-1][1] == "err" and "ZeroDivisionError" in s.cells[-1].out[-1][0]
    assert "ZeroDivisionError" in s.state()["error"]
    s.run_cell("x = 5\nx * 3", index=1)                  # edit and rerun a cell
    _wait(s)
    assert s.cells[1].out[-1][0] == "15" and s.cells[1].n == 4
    # the buttons generate code that runs in the same namespace
    path = os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")
    s.run_cell(s.read_code(path=path, deredden=False))
    _wait(s)
    assert s.spec is not None and s.done() == ["read"] and s.figure_version == 1
    code = s.fit_code("hbeta", broad_components=1, outflow=False)
    assert code.startswith("model = fa.presets.hbeta(broad_components=1, outflow=False)")
    s.run_cell(code)
    _wait(s)
    st = s.state()
    assert st["done"] == ["read", "model", "fit", "results"] and st["has_result"]
    assert any(r["label"].startswith("broad Hb FWHM") for r in st["results"])
    script = s.script()
    assert "fa.read(" in script and "fa.presets.hbeta(" in script and "1 / 0" in script


def test_app_server_requires_token():
    import json
    import threading
    import urllib.request
    import urllib.error
    from fantasy_agn.app.server import Handler, make_server
    server = make_server(port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert Handler.token in page and "__FANTASY_TOKEN__" not in page
        for req in (urllib.request.Request(base + "/api/state"),
                    urllib.request.Request(base + "/api/exec", data=b'{"code": "1"}', method="POST"),
                    urllib.request.Request(base + "/api/state", headers={"X-Fantasy-Token": "wrong"}),
                    urllib.request.Request(base + "/api/state", headers={"X-Fantasy-Token": Handler.token,
                                                                         "Host": "evil.example.com"})):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 403
        ok = urllib.request.Request(base + "/api/state", headers={"X-Fantasy-Token": Handler.token})
        state = json.loads(urllib.request.urlopen(ok).read())
        assert state["cells"][0]["code"] == "import fantasy_agn as fa"
        # downloads reach their own handlers (the figure routes take only classify/corner/figure)
        for path, msg in (("/api/wf/plots.pdf", "no fit yet"), ("/api/wf/lines.csv", "no fit yet"),
                          ("/api/wf/classify.pdf", "not classified yet")):
            req = urllib.request.Request(base + path, headers={"X-Fantasy-Token": Handler.token})
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(req)
            assert e.value.code == 404 and msg in e.value.read().decode()
    finally:
        server.shutdown()
        server.server_close()


def test_app_static_links_versioned():
    import os
    import re
    from fantasy_agn.app.server import STATIC, _versioned
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    out = _versioned(html)
    assert re.search(r'"/static/wf\.css\?v=\d+"', out) and re.search(r'"/static/wf\.js\?v=\d+"', out)
    # nothing else changes
    assert re.sub(r"\?v=\d+", "", out) == html
    # Results › Anatomy: the view, its elements and its script are all there
    with open(os.path.join(STATIC, "wf.js"), encoding="utf-8") as fh:
        js = fh.read()
    assert 'data-v="anatomy"' in html and 'id="res-anatomy"' in html
    for el in ("an-plot", "an-resid", "an-mini", "an-budget", "an-peelr", "an-donut", "an-legend", "an-zoom"):
        assert f'id="{el}"' in html and f'"{el}"' in js
    assert "function drawAnatomy" in js and 'rv === "anatomy"' in js
    # Results › Degeneracies: the wheel of the correlations
    assert 'data-v="degen"' in html and 'id="res-degen"' in html
    for el in ("dg-wheel", "dg-side", "dg-th", "dg-thv", "dg-src"):
        assert f'id="{el}"' in html and f'"{el}"' in js
    # Results › Atlas: the cards of a sample or of the epochs
    assert 'data-v="atlas"' in html and 'id="res-atlas"' in html
    for el in ("at-grid", "at-dist", "at-chips", "at-sort", "at-dir", "at-common", "at-same", "at-size"):
        assert f'id="{el}"' in html and f'"{el}"' in js
    assert "function drawAtlas" in js and 'rv === "atlas"' in js and "/api/wf/atlas" in js
    # Results › Light curves & lags: driver, responder, the adopted lag method
    for el in ("lx-pa", "lx-pb", "lx-swap", "lx-quick", "lx-seasons", "lx-lc", "lx-ccf", "lx-meth", "lx-tau", "lx-play", "lx-adopt", "lx-readout", "lx-ghost"):
        assert f'id="{el}"' in html and f'"{el}"' in js
    assert "/api/wf/lagpair" in js and "primary:" in js and 'rv === "time"' in js
    # Model › the model gallery and the model builder (its own script, also versioned)
    assert re.search(r'"/static/mb\.js\?v=\d+"', out)
    with open(os.path.join(STATIC, "mb.js"), encoding="utf-8") as fh:
        mbjs = fh.read()
    for el in ('id="galcard"', 'id="gx-body"', 'id="gx-new"', 'id="mb"', 'id="openmb"'):
        assert el in html
    for path in ("/api/wf/gallery", "/api/mb/gallery", "/api/mb/fit", "/api/mb/build", "/api/mb/preview",
                 "/api/mb/decompose", "/api/wf/mbtypes", "/api/wf/mblines", '"recipe"', '"presetkw"'):
        assert path in mbjs, path
    assert "MB.render()" in js and "MB.openCurrent()" in js
    assert "#dg-scope button" in js and "function drawDegen" in js and 'rv === "degen"' in js and "/api/wf/degeneracy" in js


def test_app_model_editor_code():
    from fantasy_agn.app.session import Session
    s = Session()
    s.run_cell(s.preset_code("hbeta", broad_components=1, outflow=False))
    _wait(s)
    assert s.model is not None and "model" in s.done()
    s.run_cell(s.add_code({"type": "Lines", "name": "broad2", "lines": ["Hg", "Hb"], "role": "broad",
                           "fwhm": [8000, 3000, 15000]}))
    _wait(s)
    assert "broad2" in [c.name for c in s.model.components]
    code = s.params_code([{"name": "narrow.fwhm", "value": 500, "min": 100, "max": 900},
                          {"name": "broad2.shift", "tie": "broad.shift"},
                          {"name": "continuum.index", "value": -1.7, "fixed": True}])
    assert "model.set('narrow.fwhm', value=500, min=100, max=900)" in code
    assert "model.tie('broad2.shift', 'broad.shift')" in code and "model.fix('continuum.index', -1.7)" in code
    s.run_cell(code)
    _wait(s)
    p = s.model.params
    assert p["narrow.fwhm"].max == 900 and p["broad2.shift"].tie[0] == "broad.shift" and p["continuum.index"].fixed
    s.run_cell(s.remove_code("heii"))
    _wait(s)
    assert "heii" not in [c.name for c in s.model.components]
    st = s.state()["model"]
    assert st["n_free"] == len(s.model.params.free_params())
    for bad in ({"type": "Lines", "name": "x", "lines": ["NotALine"]},
                {"type": "Lines", "name": "bad name!", "lines": ["Hb"]},
                {"type": "Lines", "name": "broad2", "lines": ["Hb"]},
                {"type": "Lines", "name": "y", "lines": ["Hb"], "fwhm": [100, 500, 900]},
                {"type": "os.system"}):
        with pytest.raises(ValueError):
            s.add_code(bad)
    with pytest.raises(ValueError):
        s.params_code([{"name": "narrow.fwhm", "value": 500}])       # nothing changed


def test_app_docs():
    from fantasy_agn.app.docs import all_pages
    pages = all_pages()
    ids = [p["id"] for p in pages]
    for pid in ("start", "model", "params", "lines", "api-models", "api-results"):
        assert pid in ids
    models = next(p for p in pages if p["id"] == "api-models")["html"]
    assert "fa.presets.optical(" in models and "broad_components" in models
    assert "&lt;" not in models.split("<pre>")[0][:0]          # rendered without raising
def test_model_set_profile_and_untie():
    m = fa.presets.hbeta()
    m.tie("feii.shift", "broad.shift")
    m.set("broad.fwhm", value=4000, min=2000, max=8000)
    m.set_profile("broad", "gauss-hermite")
    c = m.component("broad")
    assert c.profile == "gauss-hermite" and c.role == "broad" and "broad.h3" in m.params
    assert m["broad.fwhm"].value == 4000 and m["broad.fwhm"].max == 8000      # settings kept
    assert m["feii.shift"].tie[0] == "broad.shift" and "broad.Hg.ratio" in m.params
    assert [x.name for x in m.components].index("broad") == 4                  # same place
    m.set_profile("broad", "lorentzian")
    assert "broad.h3" not in m.params
    x = np.linspace(4200, 5500, 500)
    assert np.all(np.isfinite(m.evaluate(x)))
    m.untie("feii.shift")
    assert m["feii.shift"].tie is None
    with pytest.raises(ValueError):
        m.set_profile("feii", "lorentzian")                                     # not a Lines component


def test_app_build_own_model_code():
    from fantasy_agn.app.session import Session
    s = Session()
    s.run_cell(s.empty_model_code())
    _wait(s)
    assert [c.name for c in s.model.components] == ["continuum"]
    code = s.add_code({"type": "Lines", "name": "broad", "lines": ["Hb", "Ha"], "role": "broad", "profile": "lorentzian",
                       "fwhm": [3000, 1000, 15000], "shift": [0, -3000, 3000]})
    assert "profile='lorentzian'" in code
    s.run_cell(code)
    _wait(s)
    s.run_cell(s.add_code({"type": "Lines", "name": "narrow", "lines": ["Hb", "[OIII]5007"], "role": "narrow",
                           "own_kinematics": True}))
    _wait(s)
    assert "narrow.Hb.fwhm" in s.model.params                                   # every line its own width
    code = s.add_code({"type": "FeII", "name": "feii", "tie_to": "broad"})
    assert "model.tie('feii.shift', 'broad.shift')" in code and "model.tie('feii.fwhm', 'broad.fwhm')" in code
    s.run_cell(code + "\n" + s.add_code({"type": "Lines", "name": "abs", "lines": ["NaI5890", "NaI5896"],
                                          "absorption": True}))
    _wait(s)
    st = {c["name"]: c for c in s.model_state()["components"]}
    assert st["feii"]["tied_to"] == "broad" and st["abs"]["absorption"] and st["broad"]["profile"] == "lorentzian"
    s.run_cell(s.profile_code("broad", "voigt") + "\n" + s.untie_code("feii"))
    _wait(s)
    assert s.model.component("broad").profile == "voigt" and s.model["feii.shift"].tie is None
    for bad in (lambda: s.profile_code("feii", "voigt"), lambda: s.profile_code("broad", "box"),
                lambda: s.untie_code("feii"), lambda: s.add_code({"type": "FeII", "name": "fe2", "tie_to": "nothing"})):
        with pytest.raises((ValueError, KeyError)):
            bad()


@needs_templates
def test_app_workflow_tie_absorption_exports():
    from fantasy_agn.app.session import Session
    s = Session()
    s.wf["mode"] = "single"
    s.run_step(s.wf_read_code(mode="single", paths=[os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")]), "# read")
    _wait(s)
    s.run_step(s.wf_prepare_code(extinction="none"), "# prepare")
    _wait(s)
    s.run_cell("c = fa.classify(spectra[0])")                  # 'Suggest from classification'
    _wait(s)
    s.run_cell("spec = spectra[0]\n" + s.preset_code("hbeta", broad_components=1, outflow=False))
    _wait(s)
    code = s.tie_kinematics_code("feii", "narrow")
    assert code.startswith("model.tie('feii.shift', 'narrow.shift')") and "model.tie('feii.fwhm', 'narrow.fwhm')" in code
    with pytest.raises(ValueError):
        s.tie_kinematics_code("feii", "feii")
    s.run_cell(code + "\n" + s.add_code({"type": "Absorption", "troughs": "5175:400"}))
    _wait(s)
    assert s.model.params["feii.fwhm"].tie[0] == "narrow.fwhm" and "absorption" in [c.name for c in s.model.components]
    with pytest.raises(ValueError):
        s.wf_fit_code(errors="bogus")
    s.run_step(s.wf_fit_code(errors="covariance"), "# fit")
    _wait(s)
    st = s.wf_state()
    assert st["fit"]["kind"] == "single" and st["classified"]           # the classification stays after the fit
    # the Results page: the final class from the fitted values, the preliminary one beside it
    assert st["final_class"]
    d = s.classify_state(0)
    assert d["stage"] == "final" and d["label"] == "type1" and "in the fit" in d["reason"]
    assert d["preliminary"]["label"] == "type1" and d["starlight"] is not None
    assert s.classify_state(0, "preliminary")["stage"] == "preliminary"     # the Model step keeps the preliminary one
    import fantasy_agn as fa
    fc = fa.classify_fit(s.result)
    assert fc.fit is s.result and fc.measurements["source"] == "fit" and fc.measurements["broad_snr"] > 5
    assert "recommended" not in repr(fc)
    assert s.classify_figure(0, "diagrams", "pdf")[:4] == b"%PDF"
    # Results › Degeneracies: the correlations of the free parameters, symmetric with a unit diagonal
    dg = s.wf_degeneracy(0)
    n = len(dg["params"])
    assert dg["source"] == "covariance" and n > 5 and len(dg["corr"]) == n
    Cm = np.array(dg["corr"])
    assert np.allclose(np.diag(Cm), 1, atol=1e-3) and np.allclose(Cm, Cm.T, atol=1e-3) and np.all(np.abs(Cm) <= 1)
    assert {p["comp"] for p in dg["params"]} <= {g["name"] for g in dg["groups"]}
    assert {g["family"] for g in dg["groups"]} >= {"broad", "narrow", "feii"}
    assert s.wf_degeneracy(0) is dg                                           # cached
    csv = s.wf_components_csv(0)
    head = csv.splitlines()[1].split(",")
    assert head[:5] == ["wavelength", "flux", "error", "model", "residual"]
    assert {"comp_feii", "host", "agn", "transmission_absorption"} <= set(head)
    assert s.wf_result(0)["corner_params"] == []
    with pytest.raises(ValueError):                                     # no samples with covariance errors
        s.wf_corner(0)
    assert s.wf_custom_figure(0, {"xmin": "4600", "xmax": "5100", "legend": "0", "group": "all"}, "pdf")[:4] == b"%PDF"


IRSA_ANSWER = b"""<?xml version="1.0"?><results status="ok"><result><desc>E(B-V)</desc>
<statistics><refPixelValueSandF> 0.0094 (mag) </refPixelValueSandF>
<refPixelValueSFD> 0.0109 (mag) </refPixelValueSFD></statistics></result>
<result><statistics><refPixelValueSandF> 9.9 (mag) </refPixelValueSandF>
<refPixelValueSFD> 9.9 (mag) </refPixelValueSFD></statistics></result></results>"""


def test_galactic_dereddening(monkeypatch):
    import io
    import urllib.request
    from fantasy_agn.extinction import galactic_ebv
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(IRSA_ANSWER)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("SFD_DIR", raising=False)
    from fantasy_agn import extinction
    extinction._IRSA_CACHE.clear()
    ebv, src = galactic_ebv(214.8, 53.6)                         # no file value, no maps -> IRSA
    assert ebv == 0.0094 and "Schlafly" in src and "214.8" in calls[0]
    assert galactic_ebv(214.8, 53.6, sf11=False)[0] == 0.0109
    ebv, src = galactic_ebv(None, None, file_value=0.05)         # DESI: value in the file
    assert abs(ebv - 0.043) < 1e-9 and "file" in src
    assert len(calls) == 1                                   # the same position is asked only once
    with pytest.raises(RuntimeError):
        galactic_ebv(None, None)                                 # nothing to go on

    x = np.linspace(3800, 9000, 200)
    s = fa.Spectrum(x, np.ones_like(x), np.full_like(x, .1), z=0.1, ra=214.8, dec=53.6)
    assert not s.dereddened and "not corrected" in repr(s)
    d = s.deredden()
    assert d.dereddened and d.meta["ebv"] == 0.0094 and d.flux[0] > d.flux[-1] > 1
    with pytest.warns(UserWarning, match="already corrected"):
        assert np.allclose(d.deredden(ebv=0.1).flux, d.flux)    # never twice
    # dereddening in the observed or the rest frame gives the same spectrum
    a = s.deredden(ebv=0.05).to_rest_frame()
    b = s.to_rest_frame().deredden(ebv=0.05)
    assert np.allclose(a.flux, b.flux) and np.allclose(a.wave, b.wave)


def test_app_read_options():
    from fantasy_agn.app.session import Session
    s = Session()
    code = s.read_code(path="x.fits")
    assert code.startswith("spec = fa.read('x.fits').deredden().to_rest_frame()")
    assert "spec = fa.read('x.fits').deredden(ebv=0.03).to_rest_frame(z=0.1)" in s.read_code(path="x.fits", ebv=0.03, z=0.1)
    plain = s.read_code(path="x.fits", deredden=False, rest_frame=False)
    assert ".deredden" not in plain and "to_rest_frame" not in plain
    s.run_cell(s.read_code(path=os.path.join(EXAMPLES, "spec-1059-52618-0453.fits"), ebv=0.03, rest_frame=False))
    _wait(s)
    assert s.spec.meta["ebv"] == 0.03 and not s.spec.rest_frame
    with pytest.raises(ValueError, match="rest frame"):
        s.fit_code("hbeta")


def test_optical_line_options():
    m = fa.presets.optical(outflow=True, red=True)
    names = m.params.names
    assert m.wave_range == (3700, 7500.0)
    for ln in ("HeII4686", "HeI7065", "[ArIII]7136", "[OII]7320", "[OII]7330"):
        assert f"narrow.{ln}.flux" in names
    assert "outflow.[NeIII]3869.flux" in names and "broad_he.HeI7065.flux" in names
    plain = fa.presets.optical()
    assert plain.wave_range == (3700, 6900) and "narrow.[ArIII]7136.flux" not in plain.params.names
    assert "narrow.HeII4686.flux" in plain.params.names


def test_app_preset_options():
    from fantasy_agn.app.session import Session
    s = Session()
    assert s.preset_code("optical", 2, True, coronal=True, red=True, oi6046=False).startswith(
        "model = fa.presets.optical(broad_components=2, outflow=True, coronal=True, red=True)")
    assert "red" not in s.preset_code("hbeta", 1, False, coronal=True, red=True)      # hbeta has no red side
    assert "coronal" not in s.preset_code("galaxy (simple)", 1, False, coronal=True)
    assert "physical=True" in s.preset_code("optical (physical)", 2, True, coronal=True)


def test_templates_download_automatically(monkeypatch, tmp_path):
    import urllib.request
    from fantasy_agn import templates
    monkeypatch.setenv("FANTASY_DATA", str(tmp_path))
    calls = []

    def fake_retrieve(url, path):
        calls.append(url)
        lam = np.linspace(3000, 9000, 50)
        with open(path, "wb") as fh:                          # np.savez(path) would add .npz
            np.savez(fh, lam=lam, templates=np.ones((50, 2, 1)), ages=np.array([1.0, 10.0]),
                     metals=np.array([0.0]), fwhm=np.full(50, 2.51), masses=np.ones((2, 1)))
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setenv("FANTASY_NO_DOWNLOAD", "1")
    with pytest.raises(FileNotFoundError, match="automatic download is off"):
        templates.load("bc03")
    assert calls == []
    monkeypatch.delenv("FANTASY_NO_DOWNLOAD")
    d = templates.load("BC03")                                    # alias, downloaded on first use
    assert calls[0].endswith("spectra_galaxev_9.0.npz") and d["name"] == "galaxev" and d["medium"] == "air"
    assert templates.available() == ["galaxev"] and not list(tmp_path.rglob("*.part"))
    templates.load("bc03")
    assert len(calls) == 1                                        # cached

    def broken(url, path):
        open(path, "wb").write(b"not a template file")
    monkeypatch.setattr(urllib.request, "urlretrieve", broken)
    with pytest.raises(Exception):
        templates.download("fsps")
    assert not list(tmp_path.rglob("*fsps*"))                     # no half-written file left


def test_class_presets():
    for name in ("seyfert2", "composite", "liner", "starforming", "passive"):
        m = getattr(fa.presets, name)(host="pca")
        assert m.wave_range == (3700, 7500.0) and "narrow" in [c.name for c in m.components]
        assert getattr(fa.presets, name)(host="pca", red=False).wave_range == (3700, 6900)
    sy = fa.presets.seyfert2(host="pca")
    assert "narrow_hi.[OIII]5007.flux" in sy.params.names and "outflow.[SII]6716.flux" in sy.params.names
    assert sy.params["continuum.index"].fixed
    sf = fa.presets.starforming(host="pca", wolf_rayet=True)
    p = sf.params
    assert p["narrow.Ha.flux"].tie[0] == "narrow.Hb.flux" and abs(p["narrow.Ha.flux"].tie[1] - 2.86) < 1e-9
    assert p["narrow.H10.flux"].tie is not None and "extinction.ebv" in p.names
    assert "wolf_rayet.WR4650.flux" in p.names and "continuum.amp" not in p.names
    assert "broad.Ha.flux" in fa.presets.liner(host="pca", broad=True).params.names
    assert fa.presets.passive(host="pca").params["narrow.fwhm"].max == 600
    assert fa.presets.retired is fa.presets.passive
    from fantasy_agn.classify import Classification
    assert Classification.RECOMMENDED["seyfert2"][0] == "seyfert2"
    assert Classification.RECOMMENDED["retired"][0] == "passive"


def test_sdss_resolution_curve():
    s = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits"))
    w, f = s.resolution_curve
    assert f[np.argmin(abs(w - 3900))] > f[np.argmin(abs(w - 8000))] + 20      # blue end is worse
    assert abs(np.median(f) - s.resolution) < 15


def test_fit_many_parallel_equals_serial():
    import glob
    files = sorted(glob.glob(os.path.join(EXAMPLES, "spec-*.fits")))[:2]
    spectra = [fa.read(f).to_rest_frame() for f in files]
    model = fa.presets.hbeta(broad_components=1)
    from fantasy_agn.progress import set_hook
    reports = []
    old = set_hook(reports.append)
    try:
        serial = fa.fit_many(spectra, model, n_jobs=1, verbose=False)
        parallel = fa.fit_many(spectra, model, n_jobs=2, verbose=False)
    finally:
        set_hook(old)
    done = [(r["done"], r["total"]) for r in reports]
    assert done == [(0, 2), (1, 2), (2, 2)] * 2          # serial, then parallel
    for a, b in zip(serial, parallel):
        assert abs(a.chi2 - b.chi2) < 1e-6 * a.chi2
    bad = fa.Spectrum(np.linspace(7000, 8000, 50), np.ones(50), np.ones(50), rest_frame=True, z=0.0)
    with pytest.warns(UserWarning):
        out = fa.fit_many([spectra[0], bad], model, n_jobs=1, verbose=False)
    assert isinstance(out[1], Exception) and out[0].chi2 > 0
@needs_templates
def test_host_thread_safe_on_different_grids():
    """The app plots (all pixels) while a cell computes on the fitted pixels."""
    import threading
    m = fa.Model(fa.StellarPopulation(templates="miles"))
    ctx = {"medium": "vacuum", "resolution": 150.0}
    xa, xb = np.linspace(4000, 6800, 2500), np.linspace(4100, 6500, 1900)
    ref = {xa.size: m.evaluate(xa, ctx), xb.size: m.evaluate(xb, ctx)}
    bad = []

    def work(x):
        for _ in range(15):
            try:
                if not np.allclose(m.evaluate(x, ctx), ref[x.size]):
                    bad.append(x.size)
            except ValueError:
                bad.append(x.size)
    threads = [threading.Thread(target=work, args=(x,)) for x in (xa, xb) * 3]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not bad
def test_cli_info_examples(tmp_path, capsys):
    from fantasy_agn.cli import main
    assert main(["info"]) == 0
    assert "processor cores" in capsys.readouterr().out
    assert main(["examples", str(tmp_path)]) == 0
    files = sorted(os.listdir(tmp_path))
    assert "first_fit.py" in files and len([f for f in files if f.endswith(".fits")]) == 2
    s = fa.read(fa.examples.path("seyfert2"))
    assert s.wave.size > 1000 and s.z > 0


def test_cli_selftest(monkeypatch, capsys):
    """The offline self-test (without the template checks) passes."""
    from fantasy_agn import cli
    monkeypatch.setattr(cli, "_templates_ready", lambda name="emiles": False)
    assert cli.selftest(full=False)
    out = capsys.readouterr().out
    assert "All checks passed (3 skipped)" in out


def _ozdes_files(tmp_path, n_epochs=4):
    """Synthetic OzDES DR2 files: an original multi-epoch file and a public (SSA) one."""
    from astropy.io import fits
    rng = np.random.default_rng(2)
    n, crval, cdelt, crpix = 5000, 6295.027832, 1.037739939825, 2470
    wave = crval + (np.arange(n) + 1 - crpix) * cdelt
    z = 0.3
    line = lambda a: a * np.exp(-0.5 * ((wave - 4862.68 * (1 + z)) / 40.0) ** 2) \
        + 0.8 * np.exp(-0.5 * ((wave - 5008.24 * (1 + z)) / 6.0) ** 2)
    hdr = dict(CRVAL1=crval, CDELT1=cdelt, CRPIX1=crpix, CTYPE1="wavelength", CUNIT1="angstrom")
    h0 = fits.PrimaryHDU(10 + line(3.0) + rng.normal(0, 0.1, n))
    h0.header.update(hdr, Z=z, QOP=4, RA=37.2, DEC=-5.2, VERCOADD="v27", SOURCE="TESTQSO")
    hdus = [h0, fits.ImageHDU(np.full(n, 0.01), name="VARIANCE"),
            fits.ImageHDU(np.zeros(n, np.uint8), name="BADPIX")]
    mjds = [56652.43, 56652.46, 56917.6, 56920.6][:n_epochs]
    for k, mjd in enumerate(mjds):
        sc = 0.5 + 0.1 * k                          # counts differ between exposures
        f = fits.ImageHDU((10 + line(2.0 + k)) / sc + rng.normal(0, 0.2, n), name=f"SPECTRUM_{mjd}")
        f.header.update(hdr, UTMJD=mjd, QC="ok" if k != 2 else "poorConditions", INCOADD=True,
                        EXPOSED=2400.0, FLXSCALE=sc)
        mask = np.zeros(n, np.uint8)
        mask[100:110] = 1
        hdus += [f, fits.ImageHDU(np.full(n, 0.04), name=f"VARIANCE_{mjd}"),
                 fits.ImageHDU(mask, name=f"BADPIX_{mjd}")]
    orig = tmp_path / "TESTQSO.fits"
    fits.HDUList(hdus).writeto(orig)
    pub = fits.PrimaryHDU(10 + line(3.0) + rng.normal(0, 0.1, n))
    pub.header.update(hdr, Z=z, RA=37.2, DEC=-5.2, OBJECT="TESTQSO", SURVEY="ozdes_dr2", TMID=56652.45,
                      TXEL=1, WMIN="3732.85", WMAX="8920.51")
    public = tmp_path / "TESTQSO_56652.4500.fits"
    pub.writeto(public)
    return str(orig), str(public)


def test_ozdes_original_file(tmp_path):
    orig, _ = _ozdes_files(tmp_path)
    s = fa.read(orig)
    assert s.z == pytest.approx(0.3) and s.flux_unit is None and s.mjd is None
    assert s.meta["n_epochs"] == 4 and s.medium == "vacuum"
    assert s.wave[2469] == pytest.approx(6295.027832)          # CRPIX1 is 1-based
    e = fa.read(orig, epoch=1)
    assert e.mjd == pytest.approx(56652.46) and e.meta["qc"] == "ok"
    assert np.all(np.isinf(e.err[100:110])) and np.allclose(e.err[200], 0.2)   # mask and variance
    assert len(fa.ozdes.read_epochs(orig)) == 4
    assert len(fa.ozdes.read_epochs(orig, skip_qc=True)) == 3
    nights = fa.ozdes.read_epochs(orig, combine="night")
    assert [n.meta.get("n_exposures", 1) for n in nights] == [2, 1, 1]
    # the two exposures of the first night are put on one scale before averaging
    a, b = fa.read(orig, epoch=0), fa.read(orig, epoch=1)
    i = 300
    expect = (a.flux[i] * 0.5 / (0.5 * 0.2) ** 2 + b.flux[i] * 0.6 / (0.6 * 0.2) ** 2) / (
        1 / (0.5 * 0.2) ** 2 + 1 / (0.6 * 0.2) ** 2)
    assert nights[0].flux[i] == pytest.approx(expect)
    with pytest.raises(IndexError):
        fa.read(orig, epoch=4)
    with pytest.raises(ValueError):                             # luminosities need calibrated flux
        r = fa.fit(s.to_rest_frame(), fa.presets.hbeta(broad_components=1, feii=False), verbose=False)
        r.luminosity()


def test_ozdes_public_file(tmp_path):
    _, public = _ozdes_files(tmp_path)
    s = fa.read(public)
    assert s.meta["err_estimated"] and s.mjd == pytest.approx(56652.45) and s.flux_unit is None
    good = np.isfinite(s.err)
    assert np.median(s.err[good]) == pytest.approx(0.1, rel=0.25)   # noise estimated from the data
    with pytest.raises(ValueError):
        fa.read(public, epoch=0)
    # GAMA files (also with WMIN) are still recognised as GAMA
    g = fa.read(os.path.join(EXAMPLES, "G09_Y1_GS1_080.fit"))
    assert g.medium == "air"


def test_mask_absorption():
    rng = np.random.default_rng(3)
    w = np.exp(np.linspace(np.log(1400), np.log(1800), 4000))
    flux = 100 + 40 * np.exp(-0.5 * ((w - 1549) / 15) ** 2) + rng.normal(0, 1, w.size)
    clean = fa.Spectrum(w, flux, np.ones(w.size), rest_frame=True)
    assert len(clean.mask_absorption().meta["absorbers"]) <= 1     # noise: ~1 pixel in 10^5
    trough = 30 * np.exp(-0.5 * ((w - 1520.0) / 0.4) ** 2)            # a narrow absorber
    broad_abs = 25 * np.exp(-0.5 * ((w - 1480.0) / 8.0) ** 2)         # BAL-like: too wide
    s = fa.Spectrum(w, flux - trough - broad_abs, np.ones(w.size), rest_frame=True).mask_absorption()
    ranges = s.meta["absorbers"]
    assert any(lo < 1520 < hi for lo, hi in ranges)                  # the narrow trough
    assert not s.good[np.argmin(abs(w - 1520))]
    assert s.good[(w > 1470) & (w < 1490)].all()                     # the broad one is kept
    assert (~s.good).sum() < 60


def test_app_model_defaults_follow_presets():
    from fantasy_agn.app.session import Session, model_defaults
    d = model_defaults()
    assert d["optical"]["outflow"] is True and d["composite"]["outflow"] is False
    assert d["civ"]["broad_components"] == 2 and "outflow" not in d["passive / retired"]
    code = Session().read_code(path="x.fits", mask_absorption=True)
    assert ".mask_absorption()" in code


@needs_templates
def test_app_new_spectrum_drops_previous_model():
    """A model built for one spectrum must not be fitted to the next one."""
    from fantasy_agn.app.session import Session
    s = Session()
    assert "model = None" not in s.read_code(path="a.fits")
    s.ns["model"] = fa.presets.seyfert2()
    assert "model = None" in s.read_code(path="b.fits")


def test_app_borderline_host_compares_both():
    from types import SimpleNamespace
    from fantasy_agn.app.session import Session
    from fantasy_agn.classify import Classification
    s = Session()
    spec = fa.read(fa.examples.path("type1")).to_rest_frame()
    c = Classification.__new__(Classification)
    c.spectrum, c.label = spec, "type1"
    c.measurements = {"host_borderline": True, "host_fraction_5100": 0.17, "sigma_star_snr": 3.8}
    s.ns.update(spec=spec, c=c)
    code = s.fit_code(model="optical")
    assert "model_host = fa.presets.optical(broad_components=2, outflow=True, host=True)" in code
    assert "if r_host.bic < r.bic - 10:" in code
    assert "model_host" not in s.fit_code(model="optical", host="none")       # an explicit choice wins
    c.measurements["host_borderline"] = False
    assert "model_host" not in s.fit_code(model="optical")


def test_app_host_check_after_create_model():
    """Classify -> Create model -> Fit also compares with the host; an edited model does not."""
    from fantasy_agn.app.session import Session
    from fantasy_agn.classify import Classification
    s = Session()
    spec = fa.read(fa.examples.path("type1")).to_rest_frame()
    c = Classification.__new__(Classification)
    c.spectrum, c.label = spec, "type1"
    c.measurements = {"host_borderline": True, "host_fraction_5100": 0.2, "sigma_star_snr": 3.0}
    s.ns.update(spec=spec, c=c)
    exec(s.preset_code("optical", 2, True, host="default").split("\n")[0], s.ns)
    assert "model_host" in s.fit_model_code()
    s.ns["model"].set("narrow.fwhm", max=900)                  # edited: fitted as it is
    assert "model_host" not in s.fit_model_code()


def test_plot_line_and_components():
    import matplotlib
    matplotlib.use("Agg")
    from fantasy_agn.plotting import plain_line, _curves
    spec = fa.read(fa.examples.path("type1")).to_rest_frame()
    r = fa.fit(spec, fa.presets.hbeta(), verbose=False)
    lines = r.fitted_lines()
    assert lines.line.iloc[0] == "Hb" and len(lines.components.iloc[0]) == 3      # narrow + 2 broad first
    fig = r.plot_line()                                                            # the main broad line
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any(l.startswith("Broad 1") for l in labels) and any(l.startswith("Broad 2") for l in labels)
    assert "Sum" in labels and "Data" in labels
    r.plot_line("[OIII]5007", velocity=False, residuals=False, others=True)
    r.plot_lines(["Hb", "[OIII]5007"])
    with pytest.raises(ValueError):
        r.plot_line("Ha")                                                          # not in this model
    names = [c[1] for c in _curves(r, "auto")]
    assert "1st broad comp." in names and "2nd broad comp." in names and "Broad lines" in names
    assert "1st broad comp." not in [c[1] for c in _curves(r, True)]
    assert plain_line("[OIII]5007") == "[O III] λ5007" and plain_line("Hb") == "Hβ"


def test_save_load_fit(tmp_path):
    spec = fa.read(fa.examples.path("type1")).to_rest_frame()
    r = fa.fit(spec, fa.presets.hbeta(broad_components=1), verbose=False)
    path = fa.save(r, str(tmp_path / "q.fantasy"))
    r2 = fa.load(path)
    assert np.allclose(r.model_flux(), r2.model_flux()) and r2.saved_with["type"] == "FitResult"
    (tmp_path / "bad.fantasy").write_bytes(b"not a fantasy file")
    with pytest.raises(Exception):
        fa.load(str(tmp_path / "bad.fantasy"))


@needs_templates
def test_save_load_host_is_small(tmp_path):
    spec = fa.read(fa.examples.path("seyfert2")).to_rest_frame()
    r = fa.fit(spec, fa.presets.passive(), verbose=False)
    path = fa.save(r, str(tmp_path / "g.fantasy"))
    assert os.path.getsize(path) < 2e6                     # templates and caches are not stored
    r2 = fa.load(path)
    assert np.allclose(r.model_flux(), r2.model_flux())
    assert r2.host_properties()["sigma_star"][0] == pytest.approx(r.host_properties()["sigma_star"][0])


def test_app_project_round_trip():
    import time
    from fantasy_agn.app.session import Session
    s = Session()
    for code in (s.read_code(path=fa.examples.path("type1"), ebv=0.03), "model = fa.presets.hbeta(broad_components=1)",
                 "r = fa.fit(spec, model, verbose=False)\nr.key_results()", "f = lambda q: q"):
        s.run_cell(code)
        while s.busy:
            time.sleep(0.1)
    t = Session()
    t.load_project(s.save_project())
    assert len(t.cells) == len(s.cells) and t.result is not None and t.spec.name == s.spec.name
    assert "f" in t.info and t.results


def test_compare_variants_and_uncertainty_code():
    spec = fa.read(fa.examples.path("type1")).to_rest_frame()
    models = fa.presets.variants("hbeta", broad_components=(1, 2), outflow=(False,), host=(False,))
    assert list(models) == ["1 broad", "2 broad"]
    cmp = fa.compare(spec, models, n_jobs=1)
    assert cmp.result is cmp.results[cmp.best]
    assert "Hb FWHM" in cmp.systematic() and "log L5100" in cmp.table
    from fantasy_agn.app.session import Session
    s = Session()
    s.ns["spec"] = spec
    code = s.uncertainty_code(model="optical")
    assert "fa.presets.variants('optical', host=(False, True))" in code and "cmp = fa.compare" in code
    with pytest.raises(ValueError):
        s.uncertainty_code(model="passive / retired")            # no broad lines


def test_diagnostics_physics():
    from fantasy_agn.diagnostics import _ne_sii, _te_oiii
    # [SII] density: low- and high-density limits and a mid value (Sanders et al. 2016)
    assert _ne_sii(1.44) < 10 and _ne_sii(0.45) > 1e4 and 500 < _ne_sii(0.886) < 1100
    # [OIII] temperature: R = 7.90 exp(32900/T) at low density
    T = 12000.0
    assert _te_oiii(7.90 * np.exp(32900 / T), ne=1.0) == pytest.approx(T, rel=0.01)


@needs_templates
def test_diagnostics_seyfert2():
    spec = fa.read(fa.examples.path("seyfert2")).to_rest_frame()
    r = fa.fit(spec, fa.presets.seyfert2(), verbose=False)
    d = r.diagnostics().set_index("quantity")
    assert d.loc["BPT class", "note"] == "Seyfert"
    assert 1.0 < d.loc["log [OIII]5007/Hb", "value"] < 1.3 and d.loc["log [OIII]5007/Hb", "error"] > 0
    assert 100 < d.loc["n_e ([SII])", "value"] < 3000 and 1e4 < d.loc["T_e ([OIII])", "value"] < 2.5e4
    k = r.key_results().set_index("quantity")
    assert np.isfinite(k.loc["log [OIII]/Hb", "error"]) and np.isfinite(k.loc["[OIII]5007 flux", "error"])


def test_fit_sample(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    files = [fa.examples.path("type1"), os.path.join(tmp_path, "missing.fits")]
    s = fa.fit_sample(files, out=str(tmp_path / "out"), model="hbeta", deredden=False, n_jobs=1, verbose=False)
    t = s.table
    assert len(t) == 2 and len(s.failed) == 1 and t.preset.iloc[0] == "hbeta()"
    assert np.isfinite(t["broad Hb FWHM"].iloc[0]) and os.path.exists(tmp_path / "out" / "catalog.csv")
    r = s.load(0)
    assert r.rchi2 == pytest.approx(t.rchi2.iloc[0])
    assert os.path.exists(s.gallery())


def test_iccf_recovers_lag():
    from fantasy_agn.rm import ccf_lag
    rng = np.random.default_rng(3)
    t = np.arange(0, 400, 0.5)
    x = np.zeros(t.size)
    for i in range(1, t.size):                       # damped random walk
        x[i] = x[i - 1] - x[i - 1] * 0.5 / 60 + rng.normal(0, 0.12)
    ts = np.sort(rng.choice(t[t > 30], 70, replace=False))
    c = np.interp(ts, t, 10 + x) + rng.normal(0, 0.05, ts.size)
    l = np.interp(ts, t, 5 + 0.5 * np.interp(t - 20, t, x)) + rng.normal(0, 0.03, ts.size)
    res = ccf_lag(ts, c, np.full(ts.size, 0.05), ts, l, np.full(ts.size, 0.03), lag_range=(-60, 80), n_mc=150)
    assert abs(res["tau_cen"] - 20) < 3 * max(res["tau_err_lo"], res["tau_err_hi"], 1.0)
    assert res["r_max"] > 0.8


def _rm_epochs(lag=10.0, n=40, seed=3):
    """Multi-epoch toy fit: the continuum follows a random walk, the broad H-beta lags it."""
    rng = np.random.default_rng(seed)
    tg = np.arange(49900, 50300, 0.5)
    x = np.cumsum(rng.normal(0, 0.02, tg.size))
    x = (x - x.mean()) / x.std() * 0.15
    t_obs = np.sort(rng.uniform(50040, 50200, n))
    spectra = []
    for t in t_obs:
        m = _toy_truth()
        m.params["continuum.amp"].value *= 1 + np.interp(t, tg, x)
        m.params["broad.Hb.flux"].value *= 1 + 1.2 * np.interp(t - lag, tg, x)
        wave = np.arange(4650, 5150, 1.0)
        flux = m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 1.0, wave.size)
        spectra.append(fa.Spectrum(wave, flux, np.ones(wave.size), rest_frame=True, z=0.0, mjd=float(t)))
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, n_jobs=1, verbose=False)
    me._sim_continuum = (tg, 50.0 * (1 + x))
    return me


def test_rms_spectrum_width():
    rng = np.random.default_rng(8)
    spectra = []
    for k in range(12):
        m = _toy_truth()
        m.params["broad.Hb.flux"].value *= 1 + 0.4 * np.sin(0.7 * k)
        wave = np.arange(4650, 5150, 1.0)
        flux = m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 1.0, wave.size)
        spectra.append(fa.Spectrum(wave, flux, np.ones(wave.size), rest_frame=True, mjd=50000 + 7 * k))
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, n_jobs=1, verbose=False)
    rms = me.rms_spectrum("Hb")
    # only the broad flux varies: the rms profile is the broad Gaussian, sigma = FWHM / 2.355
    assert rms["sigma_rms"] == pytest.approx(4200 / 2.3548, rel=0.12)
    rv = me.reverberation("Hb", continuum=(np.arange(50000, 50084, 3.0), np.ones(28) + np.arange(28) * 0.01,
                                           np.full(28, 0.01)), n_mc=20)
    assert "lag (rest frame)" in rv.table().quantity.values
    assert {"p_side", "significance"} <= set(rv.lag_table().columns)


def test_app_light_curves_and_lag_pair():
    import time
    from fantasy_agn.app.session import Session
    rng = np.random.default_rng(8)
    spectra = []
    for k in range(12):
        m = _toy_truth()
        m.params["broad.Hb.flux"].value *= 1 + 0.4 * np.sin(0.7 * k)
        wave = np.arange(4650, 5150, 1.0)
        flux = m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 1.0, wave.size)
        spectra.append(fa.Spectrum(wave, flux, np.ones(wave.size), rest_frame=True, mjd=50000 + 7 * k))
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, n_jobs=1, verbose=False)
    me.light_curve()
    s = Session()
    s.ns["me"] = me
    for _ in range(100):                                          # the Fe II light curve comes in the background
        L = s.wf_lightcurve()
        if not L.get("feii_pending"):
            break
        time.sleep(0.1)
    names = [d["name"] for d in L["series"]]
    assert "Hb_flux" in names and "Hb_fwhm" in names and any(n.startswith("cont") for n in names)
    lo, hi = L["lag_range"]
    assert lo == -hi and hi >= 10
    cont = next(n for n in names if n.startswith("cont"))
    P = s.wf_lagpair(cont, "Hb_flux")
    assert P["n"] == 12 and P["lag_range"] == [lo, hi]
    keys = [m["key"] for m in P["methods"]]
    assert "iccf" in keys and set(keys) <= {"iccf", "vn", "zdcf", "drw"}
    for m in P["methods"]:
        assert lo <= m["tau"] <= hi and len(m["samples"]) <= 300
        if m["key"] in ("iccf", "vn", "drw"):
            assert len(m["curve"]["lags"]) == len(m["curve"]["y"]) > 10
    assert s.wf_lagpair(cont, "Hb_flux") is P                         # cached
    with pytest.raises(ValueError):
        s.wf_lagpair(cont, "nothing")
    # the continuum measured on the data in windows: data minus the fitted lines, close to the fitted continuum
    w = me.window_continuum([(5040, 5095), (3000, 3100)], min_coverage=0.8)
    assert list(w.columns) == ["mjd", "win5040_5095", "win5040_5095_err", "win3000_3100", "win3000_3100_err"]
    assert w["win3000_3100"].isna().all() and np.isfinite(w["win5040_5095"]).all()
    fitted = np.array([e.model_flux(np.arange(5040, 5096.0), components=True)[next(c.name for c in e.model.components
                       if c.kind == "continuum")].mean() for e in me.epochs])
    assert np.all(np.abs(w["win5040_5095"] - fitted) < 5 * w["win5040_5095_err"] + 0.05 * np.abs(fitted))
    code = s.wf_reverb_code("Hb", primary="drw", window=[5100, 5535])
    assert "w = me.window_continuum([(5100, 5535)])" in code and "continuum=(w.mjd, w.win5100_5535, w.win5100_5535_err)" in code
    # the adopted method of the measurement
    assert "primary='vn'" in s.wf_reverb_code("Hb", methods=("iccf", "vn"), primary="vn")
    assert "primary" not in s.wf_reverb_code("Hb", primary="iccf")
    assert "primary='vn'" in s.wf_reverb_code("Hb", methods=("vn", "drw"), primary="zdcf")


def test_lag_significance():
    from fantasy_agn.rm import lag_significance
    rng = np.random.default_rng(1)
    weak = {"tau_cen": 6.0, "samples_all": rng.normal(6.0, 4.5, 4000)}
    p, sig = lag_significance(weak)
    assert abs(p - 0.909) < 0.02 and abs(sig - 1.33) < 0.1
    clear = {"tau_cen": 20.0, "samples": rng.normal(20.0, 2.0, 500)}      # every sample above zero
    p, sig = lag_significance(clear)
    assert p == 1 and abs(sig - 2.88) < 0.01                             # the limit 500 samples allow
    p, sig = lag_significance({"tau_cen": -8.0, "samples_all": rng.normal(-8.0, 4.0, 2000)})
    assert abs(sig - 2.0) < 0.15                                         # a negative lag: its own side
    assert np.isnan(lag_significance({"tau_cen": 3.0})[1])


def test_absorption_jacobian_with_extinction():
    m = fa.Model(fa.PowerLaw("continuum", pivot=1450), fa.Lines("broad", ["CIV1549"], fwhm=(4000, 1000, 20000)),
                 fa.Absorption([(1510.0, 1200, 0.8), (1548.0, 300, 1.5)]), fa.Extinction())
    m.set("extinction.ebv", 0.2)
    x = np.linspace(1450, 1700, 1500)
    full = m.params.full_values()
    f, J = m.evaluate_with_jacobian(x, {}, full)
    assert np.allclose(f, m.evaluate(x, {}, full))
    num = np.empty_like(J)
    for k in range(full.size):
        h = 1e-6 * max(abs(full[k]), 1)
        a, b = full.copy(), full.copy()
        a[k] += h
        b[k] -= h
        num[:, k] = (m.evaluate(x, {}, a) - m.evaluate(x, {}, b)) / (2 * h)
    assert np.max(np.abs(J - num)) < 1e-4 * np.max(np.abs(num))


def test_fit_absorbed_recovers_trough():
    rng = np.random.default_rng(5)
    truth = fa.Model(fa.PowerLaw("continuum", pivot=1450),
                     fa.Lines("broad", ["CIV1549"], fwhm=(5000, 1000, 20000), shift=(0, -3000, 3000)),
                     fa.Absorption([(1530.0, 800, 1.2)]))
    truth.set("continuum.amp", 20).set("broad.CIV1549.flux", 800)
    w = np.arange(1450, 1720, 0.4)
    spec = fa.Spectrum(w, truth.evaluate(w, {}) + rng.normal(0, 0.3, w.size), np.full(w.size, 0.3),
                       rest_frame=True)
    model = fa.Model(fa.PowerLaw("continuum", pivot=1450),
                     fa.Lines("broad", ["CIV1549"], fwhm=(3000, 1000, 20000), shift=(0, -3000, 3000)),
                     wave_range=(1450, 1715))
    r = fa.fit_absorbed(spec, model)
    assert r.rchi2 < 1.3 < r.no_absorbers.rchi2
    t = r.absorbers.iloc[(r.absorbers.wave - 1530).abs().argsort()].iloc[0]
    assert abs(t.wave - 1530) < 1.0 and t.tau == pytest.approx(1.2, rel=0.25)
    assert r.measure("CIV1549")["flux"] == pytest.approx(800, rel=0.05)
    r.plot_line("CIV1549")


def test_block_solver_matches_dense():
    from scipy import sparse
    from fantasy_agn.multiepoch import _Blocks
    rng = np.random.default_rng(1)
    group = np.array([-1, -1, -1] + [k for k in range(4) for _ in range(5)])
    n = group.size
    J = rng.normal(size=(200, n))
    for k in range(4):                                   # rows of epoch k see only its own + shared parameters
        rows = slice(50 * k, 50 * (k + 1))
        J[rows, (group >= 0) & (group != k)] = 0.0
    N = J.T @ J
    g = rng.normal(size=n)
    keep = np.ones(n, bool)
    keep[5] = False                                      # a parameter held at its bound
    damp = 0.1 * np.diag(N)
    b = _Blocks(sparse.csr_matrix(N), group, keep)
    x = b.solve(g, damp)
    kk = np.nonzero(keep)[0]
    ref = np.linalg.solve(N[np.ix_(kk, kk)] + np.diag(damp[kk]), g[kk])
    assert np.allclose(x[kk], ref) and x[5] == 0
    inv = np.linalg.inv(N[np.ix_(kk, kk)])
    per_epoch, _ = b.covariance()
    for cols, cov in per_epoch:
        pos = [int(np.nonzero(kk == c)[0][0]) for c in cols]
        assert np.allclose(cov, inv[np.ix_(pos, pos)], rtol=1e-6, atol=1e-10)


def test_multi_epoch_reports_limits_per_epoch():
    """A width at its limit in one epoch is reported there (it was shown as 'free', error 0)."""
    rng = np.random.default_rng(4)
    spectra = []
    for k in range(4):
        m = _toy_truth()
        wave = np.arange(4650, 5150, 1.0)
        flux = m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 2.0, wave.size)
        spectra.append(fa.Spectrum(wave, flux, np.full(wave.size, 2.0), rest_frame=True, mjd=50000 + k))
    model = _toy_model()
    model.set("broad.fwhm", value=3000, min=1000, max=3500)      # the true width (4200) is outside
    me = fa.fit_epochs(spectra, model, reference=0, n_jobs=1, verbose=False)
    for e in me.epochs:
        assert "broad.fwhm" in e.at_limit
        row = e.params.set_index("parameter").loc["broad.fwhm"]
        assert row.status == "at limit" and not np.isfinite(row.error)


def test_broad_components_are_physical():
    from fantasy_agn.presets import broad_fwhm_ranges
    assert all(b <= 15000 for n in (1, 2, 3, 4) for _, _, b in broad_fwhm_ranges(n))
    (s1, lo1, hi1), (s2, lo2, hi2) = broad_fwhm_ranges(2)
    assert hi1 == 9000 and lo2 == 3000 and hi2 == 15000          # core and very broad component
    m = fa.presets.optical()
    assert m.params["broad2.fwhm"].max == 15000 and m.params["broad2.shift"].max == 3000


@needs_templates
def test_starforming_h8_not_case_b():
    # H8 is blended with HeI 3889: its flux must not be tied to Case B H-beta
    m = fa.presets.starforming()
    assert m.params["narrow.H8.flux"].tie is None and m.params["narrow.Hg.flux"].tie is not None


def test_mask_sky():
    x = np.arange(5000.0, 7000.0, 1.0)
    s = fa.Spectrum(x, np.ones_like(x), np.full_like(x, 0.1), z=0.0, rest_frame=True)
    m = s.mask_sky()
    assert np.isinf(m.err[np.argmin(abs(x - 5577.3))]) and np.isinf(m.err[np.argmin(abs(x - 6900))])
    assert np.isfinite(m.err[np.argmin(abs(x - 6000))])
    # rest frame: bands are observed-frame
    r = fa.Spectrum(x, np.ones_like(x), np.full_like(x, 0.1), z=0.1, rest_frame=True).mask_sky()
    assert np.isinf(r.err[np.argmin(abs(x * 1.1 - 6900))])


def test_calibration_polynomial_recovers_tilt():
    x = np.arange(4000.0, 7000.0, 2.0)
    m = fa.Model(fa.PowerLaw("continuum", pivot=5100))
    m.add(fa.Calibration((4000, 7000), degree=3, start=2))
    truth = m.params.full_values().copy()
    names = m.params.names
    truth[names.index("calibration.c2")] = 0.05
    truth[names.index("calibration.c3")] = -0.03
    y = m.evaluate(x, {}, truth)
    rng = np.random.default_rng(1)
    s = fa.Spectrum(x, y + rng.normal(0, 0.002 * y.mean(), x.size), np.full(x.size, 0.002 * y.mean()),
                    rest_frame=True, z=0.0)
    r = fa.fit(s, m, verbose=False)
    full = dict(zip(names, r.full))
    assert full["calibration.c2"] == pytest.approx(0.05, abs=0.01)
    assert full["calibration.c3"] == pytest.approx(-0.03, abs=0.01)
    assert "calibration.c1" not in names


@needs_templates
def test_refine_keeps_only_what_the_bic_wants():
    spec = _synthetic_galaxy(0.0, 1.0, -0.3, -1.0)          # Seyfert 2 ratios, E-MILES stars
    r = fa.refine(spec, "seyfert2", min_rchi2=0.0, max_steps=1)
    t = r.refinement
    assert len(t) == 2 and set(t.columns) >= {"extension", "dBIC", "kept"}
    # a single Gaussian per line and no calibration error in the data: nothing to add
    assert not t.kept.any() and r.preset_options == {}
    # chi2/nu below the threshold: no extension tried
    assert len(fa.refine(spec, "seyfert2", result=r).refinement) == 0



def test_prune_feii_drops_undetected_multiplets():
    def model():
        return fa.Model(fa.PowerLaw(pivot=5000), fa.FeII(fwhm=(2500, 1000, 6000), shift=(0, -500, 500)),
                        fa.Lines("broad", ["Hb"], fwhm=(4000, 1500, 9000), shift=(0, -1000, 1000)))
    wave = np.arange(4300, 5500, 1.0)
    rng = np.random.default_rng(4)
    truth = model()
    truth.set("continuum.amp", 30.0).set("broad.Hb.flux", 4000.0)
    amps = [p.name for p in truth.params if p.name.startswith("feii.") and p.linear]

    def spectrum(strength):
        for n in amps:
            truth.set(n, strength)
        clean = truth.evaluate(wave, {"medium": "vacuum"})
        return fa.Spectrum(wave, clean + rng.normal(0, 1.0, wave.size), np.full(wave.size, 1.0), rest_frame=True)

    # no Fe II in the data: the free multiplets pick up noise; pruning removes them
    r = fa.fit(spectrum(0.0), model(), verbose=False)
    pr = fa.prune_feii(r)
    t = pr.feii_pruning
    assert set(t.columns) == {"multiplet", "amplitude", "error", "kept"} and len(t) > 3
    assert pr.feii_pruned == list(t.multiplet[~t.kept]) and len(pr.feii_pruned) > 0.7 * len(t)
    assert all(pr.model.params[n].fixed and pr.model.params[n].value == 0 for n in pr.feii_pruned)
    assert pr.feii()["flux"] <= r.feii()["flux"] and pr.bic < r.bic
    assert pr.error_method == "covariance" and np.array_equal(pr.mask, r.mask)
    # strong Fe II: the multiplets are detected (all but those at the edge of the range) and stay
    r = fa.fit(spectrum(300.0), model(), verbose=False)
    strong = fa.prune_feii(r)
    n = len(strong.feii_pruning)
    assert strong.feii_pruning.kept.sum() >= n - 2
    assert abs(strong.feii()["flux"] / r.feii()["flux"] - 1) < 0.05
    again = fa.prune_feii(strong)                        # nothing left to drop: the fit is returned as it is
    assert again is strong and again.feii_pruned == [] and again.feii_pruning.kept.all()
    assert len(fa.prune_feii(r, nsigma=1e6).feii_pruned) == n       # everything dropped: still a fit
    with pytest.raises(ValueError):
        fa.prune_feii(fa.fit(spectrum(300.0), model(), verbose=False, errors=None))
    with pytest.raises(ValueError):
        fa.prune_feii(fa.fit(spectrum(0.0), fa.Model(fa.PowerLaw(pivot=5000)), verbose=False))
    # a list of fits (epochs fitted one by one): each pruned on its own, failed epochs passed through
    a, b = fa.fit(spectrum(0.0), model(), verbose=False), fa.fit(spectrum(300.0), model(), verbose=False)
    failed = ValueError("this epoch could not be fitted")
    out = fa.prune_feii([a, failed, b], n_jobs=1)
    assert len(out) == 3 and out[1] is failed and out[0] is not a
    assert len(out[0].feii_pruned) > len(out[2].feii_pruned) and out[0].feii_nsigma == 2.0
    with pytest.raises(ValueError):
        fa.prune_feii([failed], n_jobs=1)


def test_fit_sample_can_prune_feii():
    s = fa.fit_sample([os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")], model="hbeta", prune_feii=True, n_jobs=1,
                      deredden=False, verbose=False)               # no extinction lookup: tests stay offline
    row = s.table.iloc[0]
    assert "error" not in s.table or not isinstance(row.get("error"), str)
    assert row.feii_groups > row.feii_dropped > 0
    assert np.isfinite(row["R_FeII all groups"]) and np.isfinite(row["broad Hb FWHM all groups"])
    assert abs(row["R_FeII"] - row["R_FeII all groups"]) < 0.1 and row["R_FeII"] != row["R_FeII all groups"]



def test_simulation_check_recovers_a_sound_fit_and_repeats_the_pruning():
    spec = _toy_spectrum(np.random.default_rng(3))
    r = fa.fit(spec, _toy_model(), verbose=False)
    sim = fa.simulation_check(r, n=24, n_jobs=1, seed=1)
    t = sim.table.set_index("quantity")
    assert sim.n == 24 and sim.failed == 0 and sim.pruned is None and sim.result is r
    assert set(sim.table.columns) >= {"value", "mean", "bias", "bias_sd", "bias_z", "scatter", "error", "cover68", "flag"}
    assert len(sim.values) == 24 and list(sim.values.columns) == list(sim.table.quantity)
    # a well-measured blend: unbiased, and the error bar is the scatter
    row = t.loc["broad Hb FWHM"]
    assert row.flag == "ok" and abs(row.bias_sd) < 0.8 and 0.5 < row.error / row.scatter < 2 and row.cover68 > 0.4
    assert not sim.biased and "broad Hb FWHM" in sim.summary()
    assert abs(np.median(sim.rchi2) - 1) < 0.2
    # the same noise for the same seed; custom results
    again = fa.simulation_check(r, n=4, n_jobs=1, seed=1, quantities={"cont": lambda x: x.continuum(5000)})
    assert list(again.table.quantity) == ["cont"] and again.table.n[0] == 4
    with pytest.raises(ValueError):
        fa.simulation_check(r, n=1)

    # a pruned fit: every simulation is fitted with all multiplets and pruned again
    wave = np.arange(4300, 5500, 1.0)
    rng = np.random.default_rng(4)
    model = fa.Model(fa.PowerLaw(pivot=5000), fa.FeII(fwhm=(2500, 1000, 6000), shift=(0, -500, 500)),
                     fa.Lines("broad", ["Hb"], fwhm=(4000, 1500, 9000), shift=(0, -1000, 1000)))
    truth = model.copy()
    truth.set("continuum.amp", 30.0).set("broad.Hb.flux", 4000.0)
    for p in truth.params:
        if p.name.startswith("feii.") and p.linear:
            truth.set(p.name, 0.0)
    s2 = fa.Spectrum(wave, truth.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 1.0, wave.size),
                     np.full(wave.size, 1.0), rest_frame=True)
    pr = fa.prune_feii(fa.fit(s2, model, verbose=False))
    assert pr.feii_pruned and pr.feii_nsigma == 2.0
    from fantasy_agn.simcheck import _procedure
    m, nsig = _procedure(pr)
    assert nsig == 2.0 and all(m.params[n].free for n in pr.feii_pruned) and m.params["feii.fwhm"].free
    sim2 = fa.simulation_check(pr, n=6, n_jobs=1)
    assert sim2.pruned == 2.0 and sim2.n == 6 and "pruned at 2 sigma" in sim2.summary()



def test_resolution_can_be_set_and_undersampled_lines_are_flagged():
    # a text-like spectrum: 69 km/s pixels, no instrumental resolution, a narrow line of 60 km/s FWHM
    wave = 4800 * np.exp(np.arange(900) * 69.0 / 299792.458)
    rng = np.random.default_rng(5)
    model = fa.Model(fa.PowerLaw(pivot=5000),
                     fa.Lines("narrow", ["Hb", "[OIII]5007"], fwhm=(300, 50, 1200), shift=(0, -300, 300)),
                     fa.Lines("broad", ["Hb"], fwhm=(4000, 1500, 9000), shift=(0, -1000, 1000)))
    truth = model.copy()
    truth.set("continuum.amp", 20.0).set("narrow.fwhm", 60.0).set("narrow.Hb.flux", 150.0)
    truth.set("narrow.[OIII]5007.flux", 600.0).set("broad.Hb.flux", 2000.0)
    spec = fa.Spectrum(wave, truth.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 0.5, wave.size),
                       np.full(wave.size, 0.5), rest_frame=True)
    assert spec.resolution == 0 and abs(spec.pixel_kms() - 69.0) < 0.1
    r = fa.fit(spec, model, verbose=False)
    und = r.undersampled()
    assert [u["parameter"] for u in und] == ["narrow.fwhm"] and und[0]["pixels"] < 1.2 and not und[0]["resolution_known"]
    assert "too narrow to be sampled" in r.summary() and "with_resolution" in r.summary()
    # with the resolution of the instrument the observed line is wider than a pixel
    known = spec.with_resolution(R=2000)
    assert abs(known.resolution - 149.9) < 0.1 and known.meta["resolution_source"] == "given" and spec.resolution == 0
    assert known.with_resolution(120).resolution == 120 and known.context["resolution"] == known.resolution
    r2 = fa.fit(known, model, verbose=False)
    assert r2.undersampled() == [] and "too narrow" not in r2.summary()
    sdss = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits"))
    assert sdss.resolution_curve is not None and sdss.with_resolution(150).resolution_curve is None
    for bad in ({}, {"fwhm": 100, "R": 2000}, {"fwhm": -5}, {"R": 10}):
        with pytest.raises(ValueError):
            spec.with_resolution(**bad)



def test_model_starting_from_a_fit():
    spec = _toy_spectrum(np.random.default_rng(6))
    model = _toy_model()
    r = fa.fit(spec, model, verbose=False)
    start = model.starting_from(r)
    assert start is not model and start._fitted and not getattr(model, "_fitted", False)
    free = [p.name for p in model.params if p.free]
    assert all(start.params[n].value == pytest.approx(np.clip(r.model.params[n].value, model.params[n].min,
                                                              model.params[n].max)) for n in free)
    # the limits, fixed values and ties are those of the model, not of the fit
    tight = _toy_model()
    tight.set("broad.fwhm", min=1000, max=3000).fix("narrow.shift", 12.0)
    s2 = tight.starting_from(r)
    assert s2.params["broad.fwhm"].value == 3000 and s2.params["narrow.shift"].value == 12.0 and s2.params["narrow.shift"].fixed
    # a fit from it is at least as good as the fit it came from, on the same data
    again = fa.fit(spec, start, verbose=False)
    assert again.chi2 <= r.chi2 * (1 + 1e-9)
    assert model.starting_from(r.model).params["broad.fwhm"].value == pytest.approx(r.model.params["broad.fwhm"].value)
    with pytest.raises(TypeError):
        model.starting_from(spec)
    with pytest.raises(ValueError):
        fa.Model(fa.Polynomial("other", order=1)).starting_from(r)



def test_wavelength_offsets_between_epochs_are_fitted():
    # epochs of one object on slightly different wavelength scales: the joint fit measures the offsets
    rng = np.random.default_rng(8)
    wave = np.arange(4650, 5150, 1.0)
    truth = _toy_truth()
    true = np.array([0.0, 45.0, -60.0, 20.0, -25.0])
    spectra = []
    for k, dv in enumerate(true):
        clean = truth.evaluate(wave / (1 + dv / 299792.458), {"medium": "vacuum"})
        spectra.append(fa.Spectrum(wave, clean + rng.normal(0, 2.0, wave.size), np.full(wave.size, 2.0), rest_frame=True,
                                   z=0.1, mjd=50000.0 + 10 * k, name=f"ep{k}"))
    plain = fa.fit_epochs(spectra, _toy_model(), reference=0, n_jobs=1, verbose=False)
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, wavelength_offsets=True, n_jobs=1, verbose=False)
    assert plain.offsets is None and "wavelength offsets" not in plain.summary()
    off = me.offsets
    assert list(off.columns) == ["epoch", "mjd", "name", "offset", "error", "reference"] and off.reference[0]
    assert off.offset[0] == 0 and off.error[0] == 0
    pull = (off.offset[1:] - true[1:]) / off.error[1:]
    assert np.all(np.abs(pull) < 4) and np.all(off.error[1:] < 15) and np.max(np.abs(off.offset[1:] - true[1:])) < 20
    assert me.chi2 < plain.chi2 - 50 and "wavelength offsets per epoch" in me.summary()
    # every epoch's own model carries its offset: the narrow lines of epoch 2 sit 60 km/s to the blue
    shift = [e.model.params["narrow.shift"].value for e in me.epochs]
    assert abs((shift[2] - shift[0]) - off.offset[2]) < 1e-9 and np.isfinite(me.epochs[2].params.set_index("parameter").loc["narrow.shift", "error"])
    # the redshift of the narrow lines, in the frame of the reference epoch
    z, ze = me.redshift()
    expected = 1.1 * (1 - 60.0 / 299792.458) - 1                 # the narrow lines of the truth are at -60 km/s
    assert abs(z - expected) < 5 * ze + 1e-5 and 0 < ze < 1e-4 and "redshift of the narrow lines" in me.summary()
    with pytest.raises(ValueError):
        me.redshift("broad.shift")                               # fitted per epoch: not a redshift of the object
    with pytest.raises(ValueError, match="shared velocity"):
        fa.fit_epochs(spectra, _toy_model(), shared=("narrow.*.flux",), wavelength_offsets=True, n_jobs=1, verbose=False)
    tight = fa.fit_epochs(spectra, _toy_model(), reference=0, wavelength_offsets=30, n_jobs=1, verbose=False)
    assert np.all(np.abs(tight.offsets.offset) <= 30 + 1e-9) and "at the limit" in tight.summary()



def test_broadening_between_epochs_is_fitted():
    # epochs observed at different resolutions: unmodelled, the narrow lines bias the flux scales
    rng = np.random.default_rng(3)
    wave = np.arange(4650, 5150, 1.0)
    extra = np.array([0, 200, -100, 150, 250, 0])                 # signed FWHM in quadrature (km/s)
    scales = np.array([1, 1.1, 0.9, 1.2, 0.95, 1.05])
    spectra = []
    for k, (b, sc) in enumerate(zip(extra, scales)):
        R = np.sqrt(150.0 ** 2 + np.sign(b) * b ** 2)
        m = _toy_truth()
        m.set("broad.Hb.flux", 6000 * (1 + 0.2 * np.sin(k)))
        clean = m.evaluate(wave, {"medium": "vacuum", "resolution": R}) * sc
        spectra.append(fa.Spectrum(wave, clean + rng.normal(0, 2.0, wave.size), np.full(wave.size, 2.0), rest_frame=True,
                                   z=0.1, mjd=50000.0 + 10 * k, name=f"ep{k}", resolution=150.0))
    plain = fa.fit_epochs(spectra, _toy_model(), reference=0, n_jobs=1, verbose=False)
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, epoch_broadening=True, n_jobs=1, verbose=False)
    assert plain.broadening is None and "broadening" not in plain.summary()
    br = me.broadening
    assert list(br.columns) == ["epoch", "mjd", "name", "broadening", "error", "resolution", "reference"]
    assert br.broadening[0] == 0 and br.error[0] == 0
    assert np.all(np.abs(br.broadening[1:] - extra[1:]) < 4 * br.error[1:] + 15)
    s_plain = np.array([e.scale for e in plain.epochs]) / scales - 1
    s_me = np.array([e.scale for e in me.epochs]) / scales - 1
    assert np.max(np.abs(s_me)) < 0.02 < np.max(np.abs(s_plain)) and me.chi2 < plain.chi2 - 100
    # each epoch's result: intrinsic widths, seen at its broadened resolution; it reproduces the joint model
    e1 = me.epochs[1]
    assert abs(e1.model.params["narrow.fwhm"].value - me.epochs[0].model.params["narrow.fwhm"].value) < 1e-9
    assert abs(e1.spectrum.resolution - np.sqrt(150 ** 2 + br.broadening[1] ** 2)) < 1e-6
    assert e1.rchi2 < 1.3 and "extra broadening per epoch" in me.summary()
    with pytest.raises(ValueError, match="shared line width"):
        fa.fit_epochs(spectra, _toy_model(), shared=("narrow.*.flux",), epoch_broadening=True, n_jobs=1, verbose=False)
    # the calibration check sees the resolution differences the plain fit ignores
    bad, good = fa.calibration_check(plain, n_jobs=1), fa.calibration_check(me, n_jobs=1)
    assert list(bad.table.columns) == ["quantity", "mean", "snr", "rchi2", "p", "scatter", "error", "excess", "n", "flag"]
    assert bad.table.quantity[0] == "narrow.[OIII]5007.flux" and not bad.ok and bad.excess > 0.01
    assert good.ok and good.excess < 0.01 and "Calibration check" in good.summary() and len(good.epochs) == 6
    # a simulation check of the joint fit: the error bars of the flux scales and the light curve are right
    sim = fa.simulation_check(me, n=6, n_jobs=1)
    assert isinstance(sim, fa.JointSimulationCheck) and sim.n == 6
    t = sim.table.set_index("quantity")
    assert {"flux scale", "broad.Hb.flux per epoch", "broadening"} <= set(t.index)
    assert abs(t.loc["flux scale", "bias_sd"]) < 0.5 and 0.5 < t.loc["broad.Hb.flux per epoch", "spread"] < 1.6


def test_find_redshift_of_a_quasar_and_a_galaxy():
    c = 299792.458
    # a quasar: found with the quasar templates, refined on the narrow lines, clearly better than the next
    quasar = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits"))
    zf = fa.find_redshift(quasar)
    assert zf.kind == "quasar" and zf.secure and zf.method == "narrow lines"
    assert abs(zf.z - quasar.z) / (1 + quasar.z) * c < 150 and 0 < zf.error < 3e-4
    assert abs(zf.template_z - quasar.z) / (1 + quasar.z) * c < 300 and zf.template_error >= 75 / c
    t = zf.candidates
    assert list(t.columns) == ["z", "error", "kind", "chi2", "dchi2", "rchi2", "peaks", "lines"] and len(t) >= 2
    assert t.dchi2[0] == 0 and np.all(np.diff(t.chi2) >= 0) and zf.dchi2 / zf.n_pixels > 0.2
    assert "[OIII]5007" in t.lines[0] and "Hb" in [n for n, _, _ in zf.lines()] and t.peaks[0].endswith(f"of {len(zf.peaks)}")
    assert "secure" in zf.summary() and "narrow H-beta and [OIII]" in zf.summary()
    model = zf.model(0)
    assert model.shape == (zf.n_pixels,) and np.isfinite(model).all()
    assert zf.plot().axes
    # the file's redshift plays no part, and the templates alone give the same answer to 300 km/s
    blind = quasar.copy(z=None)
    plain = fa.find_redshift(blind, refine=False, kinds=("quasar",))
    assert plain.method == "templates" and abs(plain.z - quasar.z) / (1 + quasar.z) * c < 300
    # a galaxy: the stellar templates; not changed by the narrow-line refinement
    # (the SDSS class examples are downloaded separately; skipped without them)
    classes = os.path.join(EXAMPLES, "classes")
    if os.path.isdir(classes) and os.listdir(classes):
        galaxy = fa.read(os.path.join(classes, sorted(os.listdir(classes))[0]))
        zg = fa.find_redshift(galaxy)
        assert abs(zg.z - galaxy.z) / (1 + galaxy.z) * c < 150 and zg.secure
        assert zg.kind == "quasar" or zg.method == "templates"
    # a restricted range that leaves the truth out: an answer, but a different one
    off = fa.find_redshift(quasar, z_range=(1.0, 3.0), refine=False)
    assert 1.0 <= off.z <= 3.0 and off.candidates.rchi2[0] > 2 * zf.candidates.rchi2[0]
    with pytest.raises(ValueError, match="rest frame"):
        fa.find_redshift(quasar.to_rest_frame())
    with pytest.raises(ValueError):
        fa.find_redshift(quasar, kinds=("star",))
    with pytest.raises(ValueError, match="position"):
        fa.find_redshift(quasar.copy(ra=None, dec=None), catalog=True)


def test_component_reverberation():
    # the broad line follows the continuum with a 10-day lag; FeII does not vary
    me = _rm_epochs(lag=10.0)
    from fantasy_agn.rm import component_reverberation
    t = np.arange(50000, 50200, 1.0)
    cont = me._sim_continuum
    cr = component_reverberation(me, "Hb", continuum=(t, np.interp(t, *cont), np.full(t.size, 0.005)),
                                 lag_range=(-40, 40), feii_band=None)
    row = cr.table.set_index("kind").loc["line"]
    assert abs(row.lag_obs - 10.0) < 4 and row.significance > 3
    assert set(cr.light_curves.columns) >= {"mjd", "broad Hb", "broad Hb_clean"}


def test_fit_epochs_wavelength_dependent_calibration():
    # each epoch has a calibration tilt; the shared narrow lines (Hb, [OIII]) anchor it
    rng = np.random.default_rng(5)
    wave = np.arange(4650, 5150, 1.0)
    spectra, tilts, flux = [], [], []
    for k in range(8):
        m = _toy_truth()
        m.params["broad.Hb.flux"].value = b = 6000 * (1 + 0.3 * np.sin(k))
        tilt = 0.0 if k == 0 else 0.06 * rng.normal()
        f = m.evaluate(wave, {"medium": "vacuum"}) * (1 + tilt * (2 * (wave - 4700) / 400 - 1))
        spectra.append(fa.Spectrum(wave, f + rng.normal(0, 1.0, wave.size), np.ones(wave.size),
                                   rest_frame=True, mjd=50000 + k))
        tilts.append(tilt)
        flux.append(b)
    me = fa.fit_epochs(spectra, _toy_model(), reference=0, calibration_degree=1, n_jobs=1, verbose=False)
    got = np.array([e.calibration[0] for e in me.epochs])
    assert np.allclose(got, tilts, atol=0.04)
    lc = me.light_curve(lines=["Hb"])
    assert np.std(lc.Hb_flux / np.array(flux) - 1) < 0.025


def test_read_with_coordinates(tmp_path):
    from fantasy_agn.spectrum import parse_coordinates
    from astropy.coordinates import SkyCoord
    assert parse_coordinates(188.7362, 12.5822) == pytest.approx((188.7362, 12.5822))
    assert parse_coordinates("12:34:56.7", "+12:34:56") == pytest.approx((188.73625, 12.58222), abs=1e-4)
    assert parse_coordinates("12h34m56.7s", "+12d34m56s") == pytest.approx((188.73625, 12.58222), abs=1e-4)
    assert parse_coordinates("12:34:56.7 +12:34:56") == pytest.approx((188.73625, 12.58222), abs=1e-4)
    assert parse_coordinates("188.7362 12.5822") == pytest.approx((188.7362, 12.5822))
    assert parse_coordinates(SkyCoord(188.7362, 12.5822, unit="deg")) == pytest.approx((188.7362, 12.5822))
    with pytest.raises(ValueError):
        parse_coordinates("12:34:56")
    path = tmp_path / "spec.txt"
    x = np.linspace(4000, 7000, 300)
    np.savetxt(path, np.c_[x, np.ones_like(x), 0.1 * np.ones_like(x)])
    s = fa.read(str(path), z=0.1)
    assert s.ra is None
    with pytest.raises(RuntimeError, match="no coordinates"):
        s.deredden(source="sfd")
    s = fa.read(str(path), z=0.1, ra="12:34:56.7", dec="+12:34:56")
    assert (s.ra, s.dec) == pytest.approx((188.73625, 12.58222), abs=1e-4)
    assert s.with_coordinates("10.0 -5.0").dec == pytest.approx(-5.0)
    assert s.deredden(ebv=0.05).meta["ebv"] == 0.05
    from fantasy_agn.app.session import Session
    code = Session().read_code(path=str(path), ra="12:34:56.7", dec="+12:34:56")
    assert "ra='12:34:56.7', dec='+12:34:56'" in code and ".deredden()" in code
    code = Session().read_code(path=str(path), ra="188.7362", dec="12.5822", deredden=False)
    assert "ra=188.7362, dec=12.5822" in code
    with pytest.raises(ValueError):
        Session().read_code(path=str(path), ra="188.7")


def test_stratified_blr_profile():
    from fantasy_agn.profiles import evaluate_profile
    x = np.linspace(4600, 5100, 2001)
    lam0 = np.array([4862.7])
    kin = [np.array([200.0]), np.array([2500.0]), np.array([1.2]), np.array([2.4])]
    P, d = evaluate_profile("stratified", x, lam0, kin, 50.0)
    assert P[:, 0].sum() * (x[1] - x[0]) == pytest.approx(1.0, abs=1e-4)
    for i, h in enumerate((1.0, 1.0, 1e-4, 1e-4)):
        a = [k.copy() for k in kin]; b = [k.copy() for k in kin]
        a[i] += h; b[i] -= h
        num = (evaluate_profile("stratified", x, lam0, a, 50.0)[0] -
               evaluate_profile("stratified", x, lam0, b, 50.0)[0]) / (2 * h)
        assert np.max(np.abs(num - d[i])) < 1e-5 * np.max(np.abs(num))
    # span = 0: a single Gaussian
    P0, _ = evaluate_profile("stratified", x, lam0, [kin[0], kin[1], np.array([0.0]), kin[3]], 50.0)
    G, _ = evaluate_profile("gaussian", x, lam0, kin[:2], 50.0)
    assert np.allclose(P0, G)
    # a fit recovers a stratified line
    m = fa.Model(fa.PowerLaw("continuum", pivot=5000),
                 fa.Lines("broad", ["Hb"], fwhm=(2000, 500, 9000), shift=(0, -2000, 2000), profile="stratified"),
                 wave_range=(4600, 5100))
    truth = m.params.full_values().copy()
    names = m.params.names
    for n, v in (("continuum.amp", 20.0), ("broad.Hb.flux", 3000.0), ("broad.fwhm", 2500.0),
                 ("broad.span", 1.0), ("broad.gamma", 2.0)):
        truth[names.index(n)] = v
    y = m.evaluate(x, {"medium": "vacuum"}, truth)
    rng = np.random.default_rng(3)
    s = fa.Spectrum(x, y + rng.normal(0, 0.3, x.size), np.full(x.size, 0.3), rest_frame=True, z=0.0)
    r = fa.fit(s, m, verbose=False)
    got = dict(zip(names, r.full))
    assert got["broad.Hb.flux"] == pytest.approx(3000, rel=0.03)
    assert got["broad.fwhm"] == pytest.approx(2500, rel=0.15)
    assert got["broad.span"] == pytest.approx(1.0, abs=0.25)


def test_app_stratified_option():
    from fantasy_agn.app.session import Session, model_defaults
    from fantasy_agn.app.server import _options
    d = model_defaults()
    assert d["optical"]["stratified"] and d["hbeta"]["stratified"] and not d["seyfert2"]["stratified"]
    extra = _options({"stratified": True})
    code = Session().preset_code("hbeta", 2, True, **extra)
    assert "broad_profile='stratified'" in code and "broad_components=1" in code
    with pytest.raises(ValueError, match="no broad lines"):
        Session().preset_code("seyfert2", 1, True, **extra)
    m = fa.presets.hbeta(broad_components=1, broad_profile="stratified")
    assert {"broad.span", "broad.gamma"} <= set(m.params.names) and "broad2.fwhm" not in m.params.names


def test_read_text_double_spaces_and_physical_units(tmp_path):
    # two columns separated by several spaces, fluxes in erg/s/cm2/A, no errors
    x = np.linspace(3400, 7500, 400)
    f = 2e-16 * (1 + 0.1 * np.sin(x / 300)) + np.random.default_rng(1).normal(0, 1e-17, x.size)
    path = tmp_path / "transient.txt"
    path.write_text("".join(f"{a:.6f}  {b:.6E}\n" for a, b in zip(x, f)))
    s = fa.read(str(path))
    assert np.all(np.isfinite(s.flux)) and s.flux_unit == 1e-17
    assert np.median(s.flux) == pytest.approx(20, rel=0.1)
    assert np.median(s.flux / s.err) == pytest.approx(20, rel=0.4)      # local scatter: noise 1e-17
    assert "err_estimated" in s.meta
    # comma-separated with an error column and a header
    path2 = tmp_path / "spec.csv"
    e = 1e-17 * (1 + 0.5 * x / x.max())
    path2.write_text("wave,flux,err\n" + "".join(f"{a},{b},{c}\n" for a, b, c in zip(x, f, e)))
    s2 = fa.read(str(path2))
    assert s2.flux_unit == 1e-17 and np.allclose(s2.err, e * 1e17)
    # a constant error column is a placeholder: errors from the local scatter instead
    path3 = tmp_path / "const.txt"
    path3.write_text("".join(f"{a} {b} 7.071068e-17\n" for a, b in zip(x, f)))
    assert fa.read(str(path3)).meta.get("err_column_constant")
    # an explicit flux_unit switches the automatic scaling off
    s3 = fa.read(str(path), flux_unit=1.0)
    assert s3.flux_unit == 1.0 and np.median(s3.flux) < 1e-10


@needs_templates
def test_tde_preset_and_blackbody():
    from fantasy_agn.components import Blackbody
    b = Blackbody()
    x = np.linspace(3500, 7000, 60)
    f, J = b.evaluate(x, [2.0, 25000.0], {})
    num = (b.evaluate(x, [2.0, 25001.0], {})[0] - b.evaluate(x, [2.0, 24999.0], {})[0]) / 2
    assert np.allclose(J[:, 1], num, rtol=1e-5, atol=1e-12)
    assert b.evaluate(np.array([5100.0]), [2.0, 25000.0], {})[0][0] == pytest.approx(2.0)
    m = fa.presets.tde()
    names = [c.name for c in m.components]
    assert names[:2] == ["continuum", "host"] and "broad_heii" in names
    assert {"NIII4640", "HeII4686"} <= set(m.component("broad_heii").line_names)
    assert "feii" not in names and "coronal" not in names
    b2 = [c.name for c in fa.presets.bff().components]
    assert "feii" in b2 and "coronal" in b2
    assert fa.presets.bff().params["broad.fwhm"].value == 1500
    assert "tde" in fa.presets.PRESETS and "bff" in fa.presets.PRESETS


def test_fit_epochs_host_amplitude_per_epoch():
    # the host has one shape but enters each epoch with a different amplitude (aperture, seeing)
    def model():
        m = fa.Model(fa.PowerLaw("continuum", pivot=5000),
                     fa.PowerLaw("host", pivot=5000, index=(2.0, 1.0, 3.0)),
                     fa.Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007"], fwhm=(400, 100, 1200),
                              shift=(0, -500, 500), ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}),
                     fa.Lines("broad", ["Hb"], fwhm=(3000, 1000, 12000), shift=(0, -3000, 3000)),
                     wave_range=(4700, 5100))
        m.fix("host.index", 2.0)
        return m
    rng = np.random.default_rng(2)
    wave = np.arange(4650, 5150, 1.0)
    amps = [1.0, 0.6, 1.5, 0.8]
    spectra = []
    for k, a in enumerate(amps):
        m = model()
        m.set("continuum.amp", 30 * (1 + 0.3 * np.sin(k))).set("continuum.index", -1.2)
        m.set("host.amp", 40 * a)
        m.set("narrow.Hb.flux", 300).set("narrow.[OIII]5007.flux", 1500).set("narrow.fwhm", 450)
        m.set("broad.Hb.flux", 5000).set("broad.fwhm", 4000)
        y = m.evaluate(wave, {"medium": "vacuum"})
        spectra.append(fa.Spectrum(wave, y + rng.normal(0, 1.0, wave.size), np.full(wave.size, 0.5),
                                   rest_frame=True, mjd=50000 + k))
    # a power-law "host" is degenerate with an AGN slope free per epoch: share the slope here
    # (a real host is identified by its absorption lines)
    me = fa.fit_epochs(spectra, model(), shared=("narrow.*", "host.*", "continuum.index"),
                       epoch_scaled=("host",), reference=0, rescale_errors=True, n_jobs=1, verbose=False)
    got = [e.amplitudes["host"] for e in me.epochs]
    assert np.allclose(got, amps, atol=0.12)
    host = [float(e.full[e.model.params.names.index("host.amp")]) for e in me.epochs]
    # the epoch models carry the scaled host: its level follows the amplitudes
    assert np.allclose(np.array(host) / host[0], amps, atol=0.12)
    # errors were 2x too small: rescaled per epoch
    assert all(e.spectrum.meta.get("error_rescale", 1) > 1.5 for e in me.epochs)
    with pytest.raises(ValueError, match="share"):
        fa.fit_epochs(spectra, model(), shared=("narrow.*",), epoch_scaled=("host",), n_jobs=1, verbose=False)


@needs_templates
def test_app_workflow_steps(tmp_path):
    """Read -> Prepare -> Model -> Fit -> Results of the step-by-step page, as code cells."""
    import json
    import shutil
    from fantasy_agn.app.session import Session
    src = os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")
    files = []
    for date in ("20030110", "20021210"):                 # dates in the names set the epochs' order
        files.append(str(tmp_path / f"obj_{date}.fits"))
        shutil.copy(src, files[-1])
    s = Session()
    with pytest.raises(ValueError, match="Sample or Multi-epoch"):
        s.wf_read_code("single", paths=files)
    s.run_step(s.wf_read_code("multi", paths=files, group="date"), "# read")
    _wait(s)
    rows = s.loaded_state()["rows"]
    assert [r["date"] for r in rows] == ["2002-12-10", "2003-01-10"] and rows[0]["format"] == "SDSS spec FITS"
    assert s.raw[0].mjd == fa.spectrum.mjd_from_name(files[1])
    code = s.wf_prepare_code(z_source="file", extinction="value", ebv=0.03, wmin=4150, wmax=5550,
                             masks=[(5000, 4990)])
    assert ".deredden(ebv=0.03)" in code and ".crop(4150, 5550)" in code and ".mask((4990, 5000))" in code
    s.run_step(code, "# prepare")
    _wait(s)
    s.run_step(s.wf_prepare_code(z_source="file", extinction="value", ebv=0.03, wmin=4150, wmax=5550), "# prepare")
    _wait(s)
    assert sum(c.code.startswith("# prepare") for c in s.cells) == 1      # a new choice replaces the cell
    assert s.prepared_state()["n"] == 2 and s.wf_preview(1, "prepared")["rest_frame"]
    s.run_step(s.preset_code("hbeta", 1, False), "model = fa.presets.")
    _wait(s)
    g = s.wf_guess()
    assert set(g["components"]) >= {"continuum", "broad"} and len(g["x"]) == len(g["model"])
    # what is the same in all epochs: the table of the Model step, its defaults and the fit code
    sh = {r["name"]: r for r in s.wf_state()["sharing"]}
    assert sh["narrow"]["flux"] and sh["narrow"]["shape"] and sh["broad"]["flux"] is False and sh["broad"]["shape"]
    assert sh["continuum"]["flux"] is False and not s.wf_state()["sharing_custom"]
    legacy = ("narrow.*",) + tuple(s.broad_shape_params())
    assert s.sharing_patterns(True) == legacy and s.sharing_patterns(False) is None
    assert f"shared={legacy!r}" in s.wf_fit_code(epochs="joint", shared_shape=True)
    assert "shared=" not in s.wf_fit_code(epochs="joint")
    assert "wavelength_offsets" not in s.wf_fit_code(epochs="joint")
    assert "wavelength_offsets=True" in s.wf_fit_code(epochs="joint", wavelength_offsets=True)
    assert s.prepared_state()["unusable"] == 0 and s.prepared_state()["z_spread_kms"] == 0
    s.set_sharing("broad", "shape", False)
    s.set_sharing("continuum", "flux", True)
    code = s.wf_fit_code(epochs="joint", shared_shape=True)
    assert "'broad.fwhm'" not in code and "'continuum.amp'" in code and s.wf_state()["sharing_custom"]
    assert "shared=('narrow.shift', 'x')" in s.wf_fit_code(epochs="joint", shared="narrow.shift, x").replace("\n", " ")
    s.set_sharing("narrow", "flux", False)
    s.set_sharing("narrow", "shape", False)
    s.set_sharing("continuum", "flux", False)
    for name in sh:
        if sh[name]["shape"]:
            s.set_sharing(name, "shape", False)
    with pytest.raises(ValueError, match="nothing is shared"):
        s.wf_fit_code(epochs="joint", shared_shape=True)
    with pytest.raises(ValueError):
        s.set_sharing("nosuch", "flux", True)
    with pytest.raises(ValueError):
        s.set_sharing("broad", "width", True)
    s.set_sharing(reset=True)
    assert not s.wf_state()["sharing_custom"] and s.sharing_patterns(True) == legacy
    # one epoch first: a single fit with its checks, then all epochs start from it
    assert s.wf_state()["trial"] is None and "starting_from" not in s.wf_fit_code(epochs="own", n_jobs=1)
    code = s.wf_fit_code(trial="auto")
    k = s.wf_reference()
    assert code.startswith(f"# trial fit\nspec = spectra[{k}]") and "r = fa.fit(spec, model)" in code
    with pytest.raises(ValueError):
        s.wf_fit_code(trial=7)
    s.run_step(code, "# trial fit")
    _wait(s)
    t = s.wf_state()["trial"]
    json.dumps(t)
    assert s.error is None and t["k"] == k and t["active"] and t["n"] == 2 and not t["model_changed"]
    assert s.fit_state()["kind"] == "single" and len(s.wf_results()) == 1 and s.wf_state()["feii"]["rows"]
    trial = s.result
    assert s.wf_fit_code(trial=1 - k).startswith(f"# trial fit\nspec = spectra[{1 - k}]")
    assert "fa.fit_epochs(spectra, model," in s.wf_fit_code(epochs="joint", from_trial=False)
    assert "fa.fit_epochs(spectra, model.starting_from(r)," in s.wf_fit_code(epochs="joint")
    own = s.wf_fit_code(epochs="own", n_jobs=1)
    assert "fa.fit_many(spectra, model.starting_from(r), n_jobs=1)" in own
    s.run_step(own, "# fit")
    _wait(s)
    assert s.error is None and s.fit_state()["kind"] == "fits" and len(s.wf_results()) == 2
    t = s.wf_state()["trial"]
    assert t["k"] == k and not t["active"] and s.result is trial
    # the trial keeps its own cell, so the session script still defines r before the fit that starts from it
    codes = [c.code for c in s.cells]
    assert sum(c.startswith("# trial fit") for c in codes) == 1
    assert next(i for i, c in enumerate(codes) if c.startswith("# trial fit")) < next(i for i, c in enumerate(codes) if "starting_from(r)" in c)
    assert s.fits[k].chi2 <= trial.chi2 * (1 + 1e-6)          # never worse than the trial it started from
    # Fe II pruning of the epochs fitted one by one: every epoch, the epochs together again, and back
    fe = s.wf_state()["feii"]
    json.dumps(fe)
    assert fe["kind"] == "fits" and fe["errors"] and not fe["pruned"] and len(fe["rows"]) == 2 and fe["weak"] > 0
    before = s.fits
    with pytest.raises(ValueError):
        s.wf_prune_feii_undo_code()
    code = s.wf_prune_feii_code()
    assert "fits = fa.prune_feii(fits_feii_all)" in code and "fa.EpochSet(fits)" in code
    s.run_step(code, "# prune Fe II")
    _wait(s)
    fe2 = s.wf_state()["feii"]
    assert s.error is None and s.fits is not before and s.ns["fits_feii_all"] is before and s.ns["me"] is not None
    assert fe2["pruned"] and fe2["can_undo"] and fe2["weak"] == fe["weak"] and len(s.wf_results()) == 2
    assert all(f.feii_pruned for f in s.fits)
    s.run_cell(s.wf_prune_feii_undo_code())
    _wait(s)
    assert s.fits is before and not s.wf_state()["feii"]["pruned"]
    s.run_step(s.wf_fit_code(epochs="joint", reference="1", n_jobs=1, calibration_degree=1), "# fit")
    _wait(s)
    assert s.wf_state()["feii"] is None                      # a joint fit has no pruning step
    f = s.fit_state()
    assert s.error is None and f["kind"] == "joint" and f["reference"] == 1 and len(f["epochs"]) == 2
    cal = f["calibration_error"]                             # the flux-scale tile of the Results step
    assert cal is None or (isinstance(cal, float) and 0 <= cal < 1)
    json.dumps(f)
    res = s.wf_result(1)
    assert res["epochs"][1]["reference"] and res["plot"]["labels"] and s.wf_result_more(part="all")["all"]["columns"]
    # the table of all epochs: readable headings, the date and MJD in full, the flux scale of each epoch
    allep = s.wf_result_more(part="all")["all"]
    assert allep["columns"][:4] == ["Epoch", "Spectrum", "Date", "MJD"] and "H\u03b2 flux" in allep["columns"]
    assert not any("_" in c for c in allep["columns"]) and all(len(r) == len(allep["columns"]) for r in allep["rows"])
    assert "e+" not in allep["rows"][0][3] and allep["rows"][0][2].count("-") == 2
    if "Flux scale" in allep["columns"]:
        col = allep["columns"].index("Flux scale")
        assert any("reference" in r[col] for r in allep["rows"])
    assert f["z_best"] is None or (abs(f["z_best"][0] - s.spectra[0].z) < 1e-3)
    # a black-hole mass comes with the scatter of its recipe, not only the error of the fit
    mass = [q for q in s.wf_result_more(1)["key"] if q["name"].startswith("log M_BH")]
    assert mass and all(0.2 < q["scatter"] < 0.7 for q in mass)
    assert s.wf_result_more(1)["key"]
    broad = [r for r in res["lines"] if r["component"] == "broad" and r["line"] == "Hb"][0]
    assert broad["detected"] and res["lines"][0]["role"] == "broad"          # broad lines first
    json.dumps(res, default=float)
    assert s.wf_lines_csv().count("\n") > 4 and s.wf_lines_fits()[:6] == b"SIMPLE"
    assert s.wf_plots_pdf()[:4] == b"%PDF"
    nb = json.loads(s.notebook())
    assert any("fa.fit_epochs(" in "".join(c["source"]) for c in nb["cells"])
    # the checks of a joint fit: the calibration check is a cell with a small state; the report
    J = s.wf_state()["jointcheck"]
    assert J["cal"] == {"id": 0} and J["sim"] == {"id": 0} and J["n_epochs"] == 2
    assert s.wf_calcheck_code().startswith("# calibration check\nchk = fa.calibration_check(me)")
    assert "fa.simulation_check(me, n=4)" in s.wf_simcheck_code(n=4)
    s.run_step(s.wf_calcheck_code(), "# calibration check")
    _wait(s)
    J = s.wf_state()["jointcheck"]
    json.dumps(J)
    assert s.error is None and J["cal"]["id"] and J["cal"]["rows"] and "excess" in J["cal"]["rows"][0]
    from fantasy_agn.app.report import build
    page = build(s)
    assert "joint fit of 2 epochs" in page and "fa.calibration_check" in page and "Flux scale" in page
    # the host amplitude per epoch is a choice of the sharing table, given to fit_epochs as epoch_scaled
    hosts = [r for r in s.sharing_table() if r.get("host")]
    if hosts:
        s.set_sharing(hosts[0]["name"], "flux", False)
        assert f"epoch_scaled=('{hosts[0]['name']}',)" in s.wf_fit_code(epochs="joint", n_jobs=1)
        assert f"{hosts[0]['name']}.*" in s.wf_fit_code(epochs="joint", n_jobs=1)
        s.set_sharing(reset=True)
    assert "epoch_broadening=True" in s.wf_fit_code(epochs="joint", epoch_broadening=True)
    # classification only; a sample is fitted from its files
    s.run_step(s.wf_read_code("classify", paths=files[:1]), "# read")
    _wait(s)
    assert s.wf["fit_kind"] is None and s.multi is None                       # old fits cleared
    s.run_step(s.wf_prepare_code(z_source="file", extinction="value", ebv=0.03), "# prepare")
    _wait(s)
    s.run_step(s.wf_fit_code(), "# fit")
    _wait(s)
    assert s.fit_state()["labels"] == [s.classes[0].label] and s.classify_figure(0)[:4] == b"\x89PNG"
    cs = s.classify_state(0)
    assert cs["starlight"]["x"] and cs["halpha"]["narrow"] and cs["lines_fit"]["lines"] and set(cs["diagrams"]) == {"nii", "sii", "oi", "whan"}
    assert s.classify_figure(0, "diagrams")[:4] == b"\x89PNG"
    s.run_step(s.wf_read_code("sample", paths=files), "# read")
    _wait(s)
    code = s.wf_fit_code(sample_model="hbeta", n_jobs=2, out=str(tmp_path / "fits"))
    assert "fa.fit_sample(files, out=" in code and "model='hbeta'" in code and "n_jobs=2" in code
    assert "prune_feii" not in code
    pruned = s.wf_fit_code(sample_model="hbeta", n_jobs=2, out=str(tmp_path / "fits"), prune_feii=True)
    assert "prune_feii=True, n_jobs=2" in pruned and s.wf["fit_opts"]["prune_feii"] is True


def test_app_page_and_workflow_endpoints():
    import json
    import threading
    import urllib.error
    import urllib.request
    from fantasy_agn.app.server import Handler, make_server
    server = make_server(port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "wf.js" in page and Handler.token in page
        try:                                           # the classic page is gone
            urllib.request.urlopen(base + "/classic")
            raise AssertionError("/classic still served")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        hdr = {"X-Fantasy-Token": Handler.token}
        for path in ("/api/wf/result?k=0", "/api/wf/preview?k=0", "/api/wf/guess"):
            assert json.loads(urllib.request.urlopen(urllib.request.Request(base + path, headers=hdr)).read()) == {}
        nb = urllib.request.urlopen(urllib.request.Request(base + "/api/session.ipynb", headers=hdr)).read()
        assert json.loads(nb)["nbformat"] == 4
        req = urllib.request.Request(base + "/api/wf/mode", data=b'{"mode": "multi"}', headers=hdr, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["wf"]["mode"] == "multi"
    finally:
        server.shutdown()
        server.server_close()


def test_analyze_residuals_suggests_missing_parts():
    """A model without the [OIII] outflow and Fe II: the residuals ask for both, with code."""
    s = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).deredden(ebv=0.03).to_rest_frame()
    m = fa.presets.optical(broad_components=1, outflow=False)
    m.remove("feii")
    r = fa.fit(s, m, verbose=False)
    t = fa.analyze_residuals(r)
    assert len(t) and (t.sigma >= 4).all() and list(t.sigma) == sorted(t.sigma, reverse=True)
    out = t[t.line == "[OIII]5007"].iloc[0]
    assert "outflow" in out.suggestion and out.action["lines"] == ["[OIII]4959", "[OIII]5007"]
    assert (t.code == "model.add(fa.FeII())").any()
    exec(out.code, {"fa": fa, "model": m})                     # the suggested code runs
    assert "outflow_oiii5007" in [c.name for c in m.components]
    # several fits: features are merged and counted per epoch
    t2 = fa.analyze_residuals([r, r])
    assert t2.epochs.map(len).max() == 2


def test_suggest_lines_finds_missing_line():
    """A line the model lacks is found at the kinematics of the narrow lines, in milliseconds; its code
    adds it tied to them; a model that has every line gets no suggestion."""
    import time
    rng = np.random.default_rng(7)
    truth = _toy_truth()
    truth.add(fa.Lines("extra", ["[ArIV]4740"]))
    truth.tie("extra.fwhm", "narrow.fwhm").tie("extra.shift", "narrow.shift").set("extra.[ArIV]4740.flux", 250)
    wave = np.arange(4650, 5150, 1.0)
    spec = fa.Spectrum(wave, truth.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 2.0, wave.size),
                       np.full(wave.size, 2.0), rest_frame=True)
    r = fa.fit(spec, _toy_model(), verbose=False)
    t0 = time.time()
    t = fa.suggest_lines(r)
    assert time.time() - t0 < 1.0
    assert list(t.line) == ["[ArIV]4740"] and t.component[0] == "narrow" and t.dbic[0] < -10
    assert abs(t.flux[0] - 250) < 5 * t.flux_err[0]
    m = _toy_model()
    exec(t.attrs["code"], {"fa": fa, "model": m})
    assert m.params["extra_narrow.fwhm"].tie[0] == "narrow.fwhm"
    r2 = fa.fit(spec, m, verbose=False)
    assert r2.bic < r.bic - 10 and fa.suggest_lines(r2).empty
    # pure noise around a complete model: nothing
    assert fa.suggest_lines(fa.fit(_toy_spectrum(np.random.default_rng(3)), _toy_model(), verbose=False)).empty
    with pytest.raises(TypeError):
        fa.suggest_lines([r])
    # the high-ionisation list is searched, also its lines typed forbidden ([Fe V] 4181, [Fe VI], [Ar V])
    from fantasy_agn.residuals import _library
    assert {"[FeV]4181", "[FeVI]5677", "[ArV]6435", "[FeVII]6087"} <= set(_library("vacuum")[0])
    # the app: the Results state lists it, Add writes the tied code, an added line is marked
    from fantasy_agn.app.session import Session
    s = Session()
    s.ns.update(r=r, model=_toy_model(), fa=fa)
    s.wf["fit_kind"] = "single"
    L = s.wf_linescan()
    assert [x["line"] for x in L["rows"]] == ["[ArIV]4740"] and not L["rows"][0]["added"]
    code = s.wf_addlines_code(["[ArIV]4740"])
    assert code.startswith("# add lines") and "model.tie('extra_narrow.shift', 'narrow.shift')" in code
    s.ns.update(spectra=[spec], spec=spec)
    s.wf.update(mode="single", fit_opts={"errors": "covariance"})
    both = s.wf_addlines_code(["[ArIV]4740"], refit=True)     # Add & refit: one cell, the last fit's options
    assert both.startswith("# add lines and refit") and both.index("model.add(") < both.index("r = fa.fit(spec, model")
    # multi-epoch, after the trial fit of one epoch: Add & refit fits that epoch again, not all of them
    s.ns.update(spectra=[spec, _toy_spectrum(np.random.default_rng(5))])
    s.wf.update(mode="multi")
    trial = s.wf_addlines_code(["[ArIV]4740"], refit=True)
    assert "spec = spectra[0]" in trial and "r = fa.fit(spec, model)" in trial and "fit_epochs" not in trial
    assert s.wf["fit_kind"] == "single"
    exec(code, s.ns)
    assert s.wf_linescan()["rows"][0]["added"]
    with pytest.raises(ValueError, match="already in the model"):
        s.wf_addlines_code(["[ArIV]4740"])


def test_split_lines_and_offset_suggestion():
    """[O III] blueshifted against Hb in the data: suggest_lines and analyze_residuals ask to separate
    it (Model.split_lines), which keeps its flux ratio, limits and ties; refitted, nothing is left."""
    m = _toy_model().set("narrow.shift", 40).set("narrow.fwhm", 300, min=80)
    m.split_lines("narrow", ["[OIII]4959", "[OIII]5007"], "narrow_oiii")
    assert [c.name for c in m.components] == ["continuum", "narrow", "narrow_oiii", "broad"]
    assert m.component("narrow").line_names == ["Hb"]
    assert m["narrow_oiii.shift"].value == 40 and m["narrow_oiii.fwhm"].min == 80
    assert m["narrow_oiii.[OIII]4959.flux"].tie[0] == "narrow_oiii.[OIII]5007.flux"
    assert fa.Model.from_dict(m.to_dict()).to_dict() == m.to_dict()
    t = _toy_model().add(fa.Lines("extra", ["[ArIV]4740", "[ArIV]4711"]))
    t.tie("extra.shift", "narrow.shift").fix("extra.fwhm", 500).split_lines("extra", ["[ArIV]4711"], "extra_ar")
    assert t["extra_ar.shift"].tie is None and not t["extra_ar.fwhm"].fixed and t["extra_ar.fwhm"].value == 500
    assert t["extra.shift"].tie[0] == "narrow.shift" and t["extra.fwhm"].fixed
    for bad, match in ((["[OIII]4959"], "flux ratio"), (["Hb", "[OIII]4959", "[OIII]5007"], "not none or all"),
                       (["Ha"], "no line")):
        with pytest.raises(ValueError, match=match):
            _toy_model().split_lines("narrow", bad, "x")
    truth = _toy_truth().split_lines("narrow", ["[OIII]4959", "[OIII]5007"], "narrow_oiii")
    truth.set("narrow_oiii.shift", -120)                       # 60 km/s bluer than Hb
    wave = np.arange(4650, 5150, 1.0)
    rng = np.random.default_rng(11)
    spec = fa.Spectrum(wave, truth.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 2.0, wave.size),
                       np.full(wave.size, 2.0), rest_frame=True)
    r = fa.fit(spec, _toy_model(), verbose=False)
    t = fa.suggest_lines(r)
    sp = t[t.action == "split"]
    # the stronger [O III] sets the shared velocity: Hb (or [O III]) is the line to separate
    assert len(sp) == 1 and sp.lines.iloc[0] in (["Hb"], ["[OIII]4959", "[OIII]5007"])
    assert abs(sp.dv.iloc[0]) > 20 and sp.dbic.iloc[0] < -10
    code = sp.code.iloc[0]
    assert code.startswith("model.split_lines('narrow', ") and ("'narrow_hb'" in code or "'narrow_oiii'" in code)
    assert fa.analyze_residuals(r).code.str.startswith("model.split_lines('narrow', ").any()   # the residual table too
    m = _toy_model()
    exec(t.attrs["code"], {"fa": fa, "model": m})
    r2 = fa.fit(spec, m, verbose=False)
    assert r2.bic < r.bic - 10 and (fa.suggest_lines(r2).action != "split").all()
    v = {ln: r2.model[c.name + ".shift"].value for c in r2.model.components if c.name.startswith("narrow")
         for ln in c.line_names}
    assert abs(v["[OIII]5007"] - v["Hb"] + 60) < 20
    # the app: the row is marked once separated, and Separate & refit writes the code
    from fantasy_agn.app.session import Session
    s = Session()
    s.ns.update(r=r, model=_toy_model(), fa=fa)
    s.wf["fit_kind"] = "single"
    row = [x for x in s.wf_linescan()["rows"] if x["action"] == "split"][0]
    assert row["lines"] == list(sp.lines.iloc[0]) and not row["added"]
    app = s.wf_addlines_code([row["line"]])
    assert code in app
    exec(app, s.ns)
    assert [x for x in s.wf_linescan()["rows"] if x["action"] == "split"][0]["added"]
    s.ns["model"] = _toy_model()                             # the Apply of the residual table
    assert s.split_code("narrow", ["Hb"], "narrow_hb") == "model.split_lines('narrow', ['Hb'], 'narrow_hb')\nmodel"
    with pytest.raises(ValueError):
        s.split_code("narrow", ["[OIII]4959"], "narrow_x")


def test_app_residuals_step(tmp_path):
    from fantasy_agn.app.session import Session
    s = Session()
    s.run_step(s.wf_read_code("single", paths=[os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")]), "# read")
    _wait(s)
    s.run_step(s.wf_prepare_code(z_source="file", extinction="value", ebv=0.03), "# prepare")
    _wait(s)
    with pytest.raises(ValueError, match="fit first"):
        s.wf_residuals_code()
    s.run_step(s.preset_code("hbeta", 1, False), "model = fa.presets.")
    _wait(s)
    s.run_step(s.wf_fit_code(), "# fit")
    _wait(s)
    code = s.wf_residuals_code(nsigma=3)
    assert "fa.analyze_residuals(r, nsigma=3)" in code
    s.run_step(code, "# residuals")
    _wait(s)
    st = s.wf_residuals()
    assert s.error is None and st["scope"] == "all" and all("suggestion" in row for row in st["rows"])


def test_app_shared_broad_shape(tmp_path):
    """Joint fit with the broad-line shape shared between epochs: only the fluxes vary."""
    import shutil
    from fantasy_agn.app.session import Session
    src = os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")
    files = []
    for date in ("20021210", "20030110"):
        files.append(str(tmp_path / f"obj_{date}.fits"))
        shutil.copy(src, files[-1])
    s = Session()
    s.run_step(s.wf_read_code("multi", paths=files), "# read")
    _wait(s)
    s.run_step(s.wf_prepare_code(z_source="file", extinction="value", ebv=0.03, wmin=4150, wmax=5550), "# prepare")
    _wait(s)
    s.run_step(s.preset_code("hbeta", 2, False), "model = fa.presets.")
    _wait(s)
    shape = s.broad_shape_params()
    assert {"broad.fwhm", "broad.shift", "broad2.fwhm", "feii.fwhm"} <= set(shape)
    assert not any(p.endswith(".flux") for p in shape) and "feii.shift" not in shape     # tied to broad.shift
    assert "'broad.fwhm'" not in s.wf_fit_code(epochs="joint", n_jobs=1)          # off: default sharing
    code = s.wf_fit_code(epochs="joint", n_jobs=1, shared_shape=True)
    assert "'narrow.*'" in code and "'broad.fwhm'" in code and "'broad2.shift'" in code
    s.run_step(code, "# fit")
    _wait(s)
    me = s.multi
    assert s.error is None and me is not None and s.wf["fit_opts"]["shared_shape"]
    widths = {round(e.model.params["broad.fwhm"].value, 6) for e in me.epochs}
    assert len(widths) == 1                                                         # one shape for all epochs


def test_ml_initializer_round_trip(tmp_path, monkeypatch):
    """A small initializer trained on simulated spectra starts the fit; without one, the
    data-driven guess is used with a warning."""
    pytest.importorskip("sklearn")
    from fantasy_agn import ml
    monkeypatch.setattr(ml, "FOLDERS", [str(tmp_path)])
    s = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).deredden(ebv=0.03).to_rest_frame()
    m = fa.presets.hbeta(broad_components=1, host=False)
    with pytest.warns(UserWarning, match="no machine-learned initializer"):
        r = fa.fit(s, m, guess="ml", verbose=False)
    assert r.start != "learned"
    with pytest.raises(ValueError):
        fa.fit(s, m, guess="nonsense", verbose=False)
    X, Y, names, wr = ml.simulate([r], n_aug=40, jitter=0.2)
    assert X.shape[0] == Y.shape[0] == 41 and names == ml.target_names(m)
    ini = ml.train(X, Y, names, wr, n_pca=10, hidden=(16,), max_iter=50)
    ini.save(str(tmp_path / "hbeta.npz"))
    ini = ml.for_model(m)
    assert ini is not None and ini.matches(m)
    pred = ini.predict(s)
    assert set(pred) == set(names) and all(np.isfinite(v) for v in pred.values())
    r2 = fa.fit(s, m, guess="ml", verbose=False)
    assert r2.start == "learned" and r2.chi2 < 1.05 * r.chi2


def test_text_flux_unit_from_header_and_unknown(tmp_path):
    from fantasy_agn.spectrum import flux_unit_from_text
    assert flux_unit_from_text("# flux [1e-16 erg/s/cm2/A]") == 1e-16
    assert flux_unit_from_text("F_lambda (10^-17 erg s^-1 cm^-2 A^-1)") == 1e-17
    assert flux_unit_from_text("x10⁻¹⁶ erg/s/cm²/Å") == 1e-16
    assert flux_unit_from_text("# MJD 55432.1e-3") is None and flux_unit_from_text("erg/s/cm2/A") is None
    w = np.linspace(4000, 6000, 400)
    f = 5 + np.sin(w / 50)
    rows = "\n".join(f"{a:.2f} {b:.5f} 0.1" for a, b in zip(w, f))
    (tmp_path / "scaled.dat").write_text("# wave flux err\n# flux in units of 1e-16 erg/s/cm2/A\n" + rows + "\n")
    (tmp_path / "plain.dat").write_text(rows + "\n")
    s = fa.read(str(tmp_path / "scaled.dat"))
    assert s.flux_unit == 1e-16 and "header" in s.meta["flux_unit_source"] and np.median(s.flux) == pytest.approx(5, abs=0.1)
    s = fa.read(str(tmp_path / "plain.dat"))                    # values not physical, no unit: unknown
    assert s.flux_unit is None and "flux_unit" in s.meta["flux_unit_unknown"]
    assert fa.read(str(tmp_path / "plain.dat"), flux_unit=1e-16).flux_unit == 1e-16
    # luminosities: refused when the unit is unknown or implausible
    model = fa.Model(fa.PowerLaw(pivot=5000.0))
    for unit in (None, 1.0):
        sp = fa.Spectrum(w, f, np.full_like(w, 0.1), z=0.1, rest_frame=True, flux_unit=unit)
        r = fa.fit(sp, model, verbose=False)
        with pytest.raises(ValueError, match="flux_unit"):
            r.luminosity(5000)
    sp = fa.Spectrum(w, f, np.full_like(w, 0.1), z=0.1, rest_frame=True, flux_unit=1e-17)
    assert 40 < fa.fit(sp, model, verbose=False).luminosity(5000)[0] < 46          # log lambda L_lambda


def test_with_flux_unit_and_prepare_option(tmp_path):
    from fantasy_agn.app.session import Session
    w = np.linspace(4000, 6000, 300)
    rows = "\n".join(f"{a:.2f} {5 + np.sin(a / 50):.5f} 0.1" for a in w)
    (tmp_path / "plain.dat").write_text(rows + "\n")
    s = fa.read(str(tmp_path / "plain.dat"))
    t = s.with_flux_unit(1e-16)
    assert s.flux_unit is None and t.flux_unit == 1e-16 and np.array_equal(s.flux, t.flux)
    assert "flux_unit_unknown" not in t.meta and t.meta["flux_unit_source"] == "given"
    with pytest.raises(ValueError):
        s.with_flux_unit(0)
    se = Session()
    se.wf["mode"] = "single"
    se.run_step(se.wf_read_code(mode="single", paths=[str(tmp_path / "plain.dat")]), "# read")
    _wait(se)
    assert se.loaded_state()["rows"][0]["flux_unit"] is None
    code = se.wf_prepare_code(z=0.1, z_source="manual", extinction="none", flux_unit="1e-16")
    assert ".with_flux_unit(1e-16)" in code
    se.run_step(code, "# prepare")
    _wait(se)
    assert se.prepared_state()["flux_unit"] == 1e-16
    for bad in ("abc", "2", "-1e-17"):
        with pytest.raises(ValueError):
            se.wf_prepare_code(z=0.1, z_source="manual", extinction="none", flux_unit=bad)


def test_gama_spectrum_unit_date_and_app_format():
    from fantasy_agn.app.session import Session
    from fantasy_agn.spectrum import _detect_format
    path = os.path.join(EXAMPLES, "G09_Y1_GS1_080.fit")
    assert _detect_format(path) == "gama"
    s = fa.read(path)
    assert s.flux_unit == 1e-17 and "ROW1" in s.meta["flux_unit_source"]       # 'Flux-calibrated ... in 10^-17'
    assert s.mjd == pytest.approx(54530.465, abs=1e-3) and s.z == pytest.approx(0.05488)
    assert s.meta["cataid"] == 3631902 and s.name == "G09_Y1_GS1_080" and s.wave[0] == pytest.approx(3727.79)
    se = Session()
    code = se.wf_read_code(mode="single", paths=[path], format="gama")
    assert "fa.read(f, format='gama')" in code
    with pytest.raises(ValueError):
        se.wf_read_code(mode="single", paths=[path], format="bogus")


def test_zdcf_matches_pyzdcf_and_finds_lag():
    from fantasy_agn.rm import zdcf_lag
    from fantasy_agn.zdcf import zdcf
    rng = np.random.default_rng(7)
    t1 = np.sort(rng.uniform(0, 100, 40))
    t2 = np.sort(rng.uniform(0, 100, 35))
    f1 = np.sin(t1 / 9) + rng.normal(0, .1, 40)
    f2 = np.sin((t2 - 6) / 9) + rng.normal(0, .1, 35)
    z = zdcf((t1, f1, np.full(40, .1)), (t2, f2, np.full(35, .1)), num_mc=20)
    # pyZDCF 1.0.3 (Jankov et al. 2022; numpy 1.26) on the same light curves: rows 0, 28 and 55
    ref = [[-67.2266159058, 10.9791717529, 1.371673584, 0.2394884592, 0.35404571, 0.321538073, 10.0],
           [1.89757967, 0.4619024992, 0.3894264698, 0.7829048309, 0.1381606015, 0.1068289116, 12.0],
           [69.7778489855, 2.2063005235, 14.5637495253, 0.4830226224, 0.3230257976, 0.2630679887, 9.0]]
    assert len(z) == 56
    assert np.allclose(z.iloc[[0, 28, 55]].to_numpy(), ref, rtol=0, atol=1e-8)
    # a damped random walk and its copy delayed by 20 d
    tt = np.arange(0, 400, 0.5)
    x = np.zeros_like(tt)
    for i in range(1, len(tt)):
        x[i] = x[i - 1] * np.exp(-0.5 / 100) + rng.normal(0, 0.1)
    a = np.sort(rng.choice(tt[:-60], 120, replace=False))
    b = np.sort(rng.choice(tt[60:], 100, replace=False))
    fa_, fb_ = np.interp(a, tt, x) + rng.normal(0, .02, a.size), np.interp(b - 20, tt, x) + rng.normal(0, .02, b.size)
    lag = zdcf_lag(a, fa_, np.full(a.size, .02), b, fb_, np.full(b.size, .02), lag_range=(-80, 80), n_mc=50)
    assert abs(lag["tau_cen"] - 20) < 3 * max(lag["tau_err_lo"], lag["tau_err_hi"], 1.0)


def test_uv_feii_extended_model():
    old, new = fa.FeII(kind="uv"), fa.FeII(kind="uv_extended")
    assert len(old.groups) == 6 and len(new.groups) == 14
    assert new._wave.min() < 2250 and new._wave.max() > 3200
    assert {"a6D", "a4D_UV60", "a4D_UV61", "a4D_UV62", "a4D_UV63", "a4P", "IZw1"} <= set(new.groups)
    # within the multiplets the 2015 model shares, the strongest lines are the same
    t = fa.lines.load_template("feii_uv_extended")
    w = dict(zip(t[2], [t[0][t[3] == i] for i in range(len(t[2]))]))
    for g, lam in (("a4D_UV62", 2755.736), ("a4D_UV63", 2739.547), ("a4D_UV60", 2926.585)):
        i = t[2].index(g)
        strongest = t[0][t[3] == i][np.argmax(t[1][t[3] == i])]
        assert abs(strongest - fa.lines._to_vacuum(lam)) < 0.05, g
    assert len(w) == 14
    # the amplitudes of all groups are recovered from a synthetic spectrum
    rng = np.random.default_rng(3)

    def model():
        return fa.presets.mgii(feii="uv_extended", balmer_continuum=False, wave_range=(2200, 3300))
    truth = model()
    truth.set("continuum.amp", 100).set("broad.fwhm", 4000).set("feii_uv.fwhm", 2500).set("feii_uv.shift", 150)
    groups = [f"feii_uv.{g}" for g in new.groups]
    for n, a in zip(groups, rng.uniform(800, 3200, len(groups))):
        truth.set(n, a)
    truth.set("broad.MgII2798.flux", 3000)
    wave = np.arange(2200, 3300, 1.0)
    clean = truth.evaluate(wave, {"medium": "vacuum"})
    noise = 0.02 * np.median(clean)
    spec = fa.Spectrum(wave, clean + rng.normal(0, noise, wave.size), np.full(wave.size, noise), rest_frame=True)
    r = fa.fit(spec, model(), verbose=False)
    p = r.params.set_index("parameter")
    tv = dict(zip(truth.params.names, truth.params.full_values()))
    pulls = [(p.loc[n, "value"] - tv[n]) / p.loc[n, "error"] for n in groups + ["feii_uv.fwhm", "feii_uv.shift"]]
    assert np.all(np.abs(pulls) < 4) and r.rchi2 < 1.2


@pytest.mark.parametrize("preset", ["optical", "mgii", "civ", "type2", "galaxy"])
@needs_templates
def test_model_save_load_roundtrip(preset, tmp_path):
    # a saved model (a JSON recipe) rebuilds the same components, parameters, limits, fixes and ties
    m = getattr(fa.presets, preset)()
    first = next(c for c in m.components if isinstance(c, fa.Lines))
    m.set_profile(first.name, "lorentzian")
    free = [p for p in m.params if not p.fixed and not p.tie and not p.linear and np.isfinite(p.max)]
    m.set(free[0].name, free[0].value, min=free[0].min, max=0.5 * (free[0].value + free[0].max))
    m.fix(free[1].name)
    path = m.save(str(tmp_path / "model.json"))
    m2 = fa.load_model(path)
    state = lambda mm: [(p.name, p.value, p.min, p.max, p.fixed, p.tie) for p in mm.params]
    assert state(m) == state(m2)
    assert [type(c).__name__ for c in m.components] == [type(c).__name__ for c in m2.components]
    lo, hi = m.wave_range or (3600, 7000)
    x = np.linspace(lo, hi, 300)
    assert np.allclose(m.evaluate(x), m2.evaluate(x))


def test_load_model_rejects_unknown_components():
    with pytest.raises(ValueError):
        fa.Model.from_dict({"components": [{"type": "os.system", "args": ["ls"], "kwargs": {}}], "params": []})


def _two_line_spectrum(f1, f2, seed=1):
    # continuum + two lines of the same velocity width; f1, f2 scale their fluxes
    w = np.linspace(4000, 6000, 2001)
    rng = np.random.default_rng(seed)
    g = lambda c, a: a * np.exp(-0.5 * ((w - c) / (8.0 * c / 5000)) ** 2) * 5000 / c
    f = 10 * (w / 5000) ** -1 + g(4500, f1) + g(5500, f2)
    return fa.Spectrum(w, f + rng.normal(0, 0.5, w.size), err=np.full(w.size, 0.5), rest_frame=True)


def _two_line_models():
    base = fa.Model(fa.components.PowerLaw(index=(-1, -3, 3)))
    linked, free = base.copy(), base.copy()
    linked.add(fa.Lines("x", [("A", 4500.0), ("B", 5500.0)], fwhm=(1100, 300, 3000), ratios={"B": ("A", 1.0)}))
    free.add(fa.Lines("x", [("A", 4500.0), ("B", 5500.0)], fwhm=(1100, 300, 3000)))
    return {"base": base, "linked": linked, "free": free}


def test_predictive_check_transfers_only_through_links():
    # each line is withheld in turn: the linked model predicts it from the other line, the model with
    # independent amplitudes cannot (the withheld amplitude is constrained only by the withheld block)
    models = _two_line_models()
    blocks = {"A": [(4460, 4540)], "B": [(5460, 5540)]}
    chk = fa.predictive_check(_two_line_spectrum(5, 5), models, blocks, reference="base", n_jobs=1)
    t = chk.table.set_index(["model", "block"])
    assert t.loc[("linked", "A"), "predictable"] and t.loc[("linked", "B"), "predictable"]
    assert not t.loc[("free", "A"), "predictable"] and "x.A.flux" in t.loc[("free", "A"), "unconstrained"]
    gain, se, n = chk.gain("linked")
    assert n == 2 and gain > 3 * se > 0
    spec = chk.specificity("linked")
    assert spec[2] == "base" and spec[0] == gain           # the free model predicts nothing, so it is no competitor
    assert np.isnan(chk.gain("free")[0])


def test_link_test_and_profile_likelihood():
    models = _two_line_models()
    ok = fa.link_test(_two_line_spectrum(5, 5), models["linked"], models["free"], n_jobs=1)
    bad = fa.link_test(_two_line_spectrum(7, 1), models["linked"], models["free"], n_jobs=1)
    assert ok["dof"] == 1 and ok["p_chi2"] > 0.01
    assert bad["T"] > 100 and bad["p_chi2"] < 1e-10
    prof = fa.profile_likelihood(_two_line_spectrum(5, 5), models["linked"], "x.fwhm",
                                 np.linspace(700, 1500, 9), n_jobs=1)
    lo, hi = prof.attrs["interval"]
    assert lo < 1130 < hi and hi - lo < 200 and not prof.attrs["at_grid_edge"]   # true FWHM 1130 km/s


def _blend_spectrum(narrow_fwhm, noise=0.15, seed=1):
    # broad + narrow Hb and [OIII]; true broad FWHM 3000, [OIII] 450 km/s
    w = np.linspace(4600, 5200, 1201)
    g = lambda c, fwhm, a: a * np.exp(-0.5 * ((w - c) / (fwhm / 2.3548 / 299792.458 * c)) ** 2)
    f = 10 + g(4861.3, 3000, 6) + g(4861.3, narrow_fwhm, 3) + g(5006.8, 450, 12)
    f = f + np.random.default_rng(seed).normal(0, noise, w.size)
    return fa.Spectrum(w, f, err=np.full(w.size, noise), rest_frame=True)


def _blend_model():
    m = fa.Model(fa.components.PowerLaw(index=(-1, -3, 3)))
    m.add(fa.Lines("b", [("Hb", 4861.3)], fwhm=(2500, 1200, 8000)))
    m.add(fa.Lines("n", [("Hb", 4861.3)], fwhm=(500, 100, 1500)))
    m.add(fa.Lines("o", [("O3", 5006.8)], fwhm=(500, 100, 1500)))
    return m


def test_model_confidence_predicts_what_a_tie_does_to_a_result():
    tie = [("n.fwhm", "o.fwhm")]
    good = fa.model_confidence(fa.fit(_blend_spectrum(450), _blend_model(), verbose=False), tie, {"b.fwhm": 60})
    bad = fa.model_confidence(fa.fit(_blend_spectrum(800), _blend_model(), verbose=False), tie, {"b.fwhm": 60})
    assert good.verdict == "safe" and good.probability > 0.95
    assert bad.verdict == "changes the result" and bad.probability < 0.05 and bad.test_p < 1e-6
    real = bad.refit().iloc[0]
    assert abs(real.predicted_shift - real.real_shift) < 0.15 * abs(real.real_shift)


def test_model_confidence_stays_low_when_noisy_data_cannot_decide():
    r = fa.fit(_blend_spectrum(600, noise=1.5), _blend_model(), verbose=False)
    conf = fa.model_confidence(r, [("n.fwhm", "o.fwhm")], {"b.fwhm": 60})
    assert conf.test_p > 0.01 and conf.verdict != "safe"       # not rejected, and still not safe


def test_model_confidence_forms_of_ties_and_targets():
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    i = r.model.params.index("b.fwhm")
    conf = fa.model_confidence(r, [("n.fwhm", "o.fwhm"), ("continuum.index", 0.0)],
                               {"b.fwhm": 60, "double": (lambda full: 2 * full[i], 120)})
    assert len(conf.ties) == 2 and len(conf.targets) == 2
    t = conf.targets.set_index("target")
    assert t.loc["double", "shift"] == pytest.approx(2 * t.loc["b.fwhm", "shift"], rel=1e-5)
    with pytest.raises(ValueError):
        fa.model_confidence(r, [], {"b.fwhm": 60})


def test_model_confidence_does_not_call_a_target_without_an_error_safe():
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    r.cov[r.model.params.index("b.fwhm"), :] = 0     # as for the width of a component with no flux
    r.cov[:, r.model.params.index("b.fwhm")] = 0
    conf = fa.model_confidence(r, [("n.fwhm", "o.fwhm")], {"b.fwhm": 60})
    assert conf.verdict == "cannot assess" and np.isnan(conf.probability)


def test_model_confidence_marks_a_tie_the_model_already_has():
    m = _blend_model()
    m.params.tie("n.shift", "o.shift")
    r = fa.fit(_blend_spectrum(450), m, verbose=False)
    conf = fa.model_confidence(r, [("n.shift", "o.shift"), ("n.fwhm", "o.fwhm")], {"b.fwhm": 60})
    assert "already in the model" in conf.ties.verdict.values and conf.verdict == "safe"


def test_candidate_ties_pairs_widths_and_shifts_but_not_narrow_with_broad():
    m = fa.Model(fa.components.PowerLaw(index=(-1, -3, 3)))
    m.add(fa.Lines("b", [("Hb", 4861.3)], role="broad", fwhm=(2500, 1200, 8000)))
    m.add(fa.Lines("n", [("Hb", 4861.3)], role="narrow", fwhm=(500, 100, 1500)))
    m.add(fa.Lines("o", [("O3", 5006.8)], role="narrow", fwhm=(500, 100, 1500)))
    ties = fa.candidate_ties(fa.fit(_blend_spectrum(450), m, verbose=False))
    assert ("n.fwhm", "o.fwhm") in ties and ("n.shift", "o.shift") in ties
    assert not any("b." in a + b for a, b in ties)


def test_model_confidence_never_calls_a_rejected_tie_safe():
    r = fa.fit(_blend_spectrum(800), _blend_model(), verbose=False)
    conf = fa.model_confidence(r, [("n.fwhm", "o.fwhm")], {"b.fwhm": 1000})     # huge tolerance: harmless on paper
    assert conf.probability > 0.95 and conf.test_p < 1e-3 and conf.verdict == "rejected by the data"


def _duplicate_lines_spectrum():
    w = np.arange(4300.0, 5700.0, 25.0)
    sigma = 5000 * 6000 / 2.3548 / 299792.458
    f = 10 + 500 * np.exp(-0.5 * ((w - 5000) / sigma) ** 2) / (sigma * 2.5066)
    return fa.Spectrum(w, f + np.random.default_rng(1).normal(0, 0.3, w.size), err=np.full(w.size, 0.3), rest_frame=True)


def test_repeated_amplitude_columns_do_not_stop_the_fit_early(monkeypatch):
    spec = _duplicate_lines_spectrum()
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)))
    for i in range(30):
        model.add(fa.Lines(f"c{i}", [("A", 5000.0)], fwhm=(2000, 1000, 12000), shift=(0, -500, 500)))
        if i:
            model.tie(f"c{i}.fwhm", "c0.fwhm")
            model.tie(f"c{i}.shift", "c0.shift")
    projected = fa.fit(spec, model, guess=False, verbose=False)
    monkeypatch.setenv("FANTASY_VARPRO", "0")
    joint = fa.fit(spec, model, guess=False, verbose=False)
    assert projected.chi2 == pytest.approx(joint.chi2, abs=0.2)


def test_broad_line_with_a_masked_centre_is_kept():
    w = np.arange(4300.0, 5700.0)
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)))
    model.add(fa.Lines("b", [("A", 5000.0)], fwhm=(9000, 1000, 20000), shift=(0, -500, 500)))
    full = {"continuum.amp": 10.0, "continuum.index": 0.0, "b.A.flux": 20000.0}
    for name, value in full.items():
        model.params[name].value = value
    spec = fa.Spectrum(w, np.ones_like(w), err=np.full(w.size, 0.3), rest_frame=True)
    flux = model.evaluate(w, spec.context, np.array([p.value for p in model.params]))
    flux = flux + np.random.default_rng(1).normal(0, 0.3, w.size)
    spec = fa.Spectrum(w, flux, err=np.full(w.size, 0.3), rest_frame=True).mask((4800, 5200))
    model.params["b.fwhm"].value = 3000
    r = fa.fit(spec, model, verbose=False)
    assert not r.unconstrained
    assert r.model.params["b.A.flux"].value > 5000


def test_monte_carlo_warns_about_refits_that_did_not_converge(monkeypatch):
    from fantasy_agn import fitting
    real = fitting._least_squares

    def stopped(prob, theta, max_nfev=None, tol=1e-10):
        res = real(prob, theta, max_nfev, tol)
        res.success = False
        return res

    w = np.arange(4300.0, 5700.0, 10.0)
    spec = fa.Spectrum(w, 10 + np.random.default_rng(2).normal(0, 0.3, w.size), err=np.full(w.size, 0.3), rest_frame=True)
    r = fa.fit(spec, fa.Model(fa.PowerLaw(index=(0, -3, 3))), verbose=False)
    monkeypatch.setattr(fitting, "_least_squares", stopped)
    prob = fitting.Problem(r.model, spec.wave, spec.flux, spec.err, spec.context)
    with pytest.warns(UserWarning, match="3 of 3 Monte Carlo refits"):
        fitting._monte_carlo(prob, prob.theta_from_model(), 3, 1.0, 0, 1)


def test_component_without_flux_has_no_measured_width():
    w = np.arange(4300.0, 5700.0, 2.0)
    dip = 3 * np.exp(-0.5 * ((w - 5000) / 20) ** 2)              # a positive line cannot fit this
    flux = 10 - dip + np.random.default_rng(4).normal(0, 0.1, w.size)
    spec = fa.Spectrum(w, flux, err=np.full(w.size, 0.1), rest_frame=True)
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)),
                     fa.Lines("b", [("A", 5000.0)], fwhm=(3000, 1000, 9000), shift=(0, -500, 500)),
                     fa.Lines("c", [("A", 5000.0)], fwhm=(3000, 1000, 9000), shift=(0, -500, 500)))
    model.tie("c.fwhm", "b.fwhm")
    r = fa.fit(spec, model, verbose=False)
    assert r.model.params["b.A.flux"].value == 0 and r.model.params["c.A.flux"].value == 0
    assert {"b.fwhm", "b.shift", "c.fwhm", "c.shift"} == set(r.not_measured)
    t = r.params.set_index("parameter")
    assert t.loc["b.fwhm", "status"] == "not measured" and np.isnan(t.loc["b.fwhm", "error"])
    assert t.loc["c.fwhm", "status"] == "not measured" and np.isnan(t.loc["c.fwhm", "error"])
    assert r.lines["fwhm"].isna().all() and r.lines["shift"].isna().all()
    assert not set(r.at_limit) & set(r.not_measured)


def test_width_shared_with_a_live_component_stays_measured():
    spec = _duplicate_lines_spectrum()
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)),
                     fa.Lines("b", [("A", 5000.0)], fwhm=(3000, 1000, 12000), shift=(0, -500, 500)),
                     fa.Lines("c", [("B", 5300.0)], fwhm=(3000, 1000, 12000), shift=(0, -500, 500)))
    model.tie("c.fwhm", "b.fwhm")
    r = fa.fit(spec, model, verbose=False)
    assert "b.fwhm" not in r.not_measured and "c.fwhm" not in r.not_measured
    assert r.params.set_index("parameter").loc["b.fwhm", "error"] > 0


def test_flux_larger_error_than_value_is_not_measured():
    w = np.arange(4300.0, 5700.0)
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)))
    model.add(fa.Lines("b", [("A", 5000.0)], fwhm=(9000, 1000, 20000), shift=(0, -500, 500)))
    for name, value in {"continuum.amp": 10.0, "continuum.index": 0.0, "b.A.flux": 100000.0}.items():
        model.params[name].value = value
    spec = fa.Spectrum(w, np.ones_like(w), err=np.full(w.size, 0.3), rest_frame=True)
    flux = model.evaluate(w, spec.context, np.array([p.value for p in model.params]))
    flux = flux + np.random.default_rng(1).normal(0, 0.3, w.size)
    spec = fa.Spectrum(w, flux, err=np.full(w.size, 0.3), rest_frame=True).mask((4600, 5400))
    model.params["b.fwhm"].value = 3000
    r = fa.fit(spec, model, verbose=False)
    assert r.unmeasured_fluxes == ["b.A.flux"]
    assert r.params.set_index("parameter").loc["b.A.flux", "status"] == "not measured"
    assert "error larger than the flux" in r.summary(short=True)


def _undetected_line(seed=5):
    w = np.arange(4300.0, 5700.0, 2.0)
    flux = 10 + np.random.default_rng(seed).normal(0, 0.3, w.size)
    spec = fa.Spectrum(w, flux, err=np.full(w.size, 0.3), rest_frame=True)
    model = fa.Model(fa.PowerLaw(index=(0, -3, 3)),
                     fa.Lines("b", [("A", 5000.0)], fwhm=(3000, 1000, 9000), shift=(0, -500, 500)))
    model.params["continuum.amp"].value, model.params["continuum.index"].value = 10, 0
    for name in ("continuum.amp", "continuum.index", "b.fwhm", "b.shift"):
        model.params.fix(name)
    return spec, model


def test_upper_limit_of_an_undetected_line_matches_the_exact_posterior():
    from scipy.stats import norm
    spec, model = _undetected_line()
    r = fa.fit(spec, model, errors="mcmc", seed=1, n_jobs=1, scale_errors=False, mcmc={"n_samples": 1500}, verbose=False)
    assert r.at_zero == ["b.A.flux"] and r.samples[:, 2:].std() > 0
    full = np.array([p.value for p in model.params])
    i = model.params.names.index("b.A.flux")
    full[i] = 0
    empty = model.evaluate(spec.wave, spec.context, full)
    full[i] = 1
    g = (model.evaluate(spec.wave, spec.context, full) - empty) / 0.3
    s = 1 / np.sqrt(g @ g)
    mu = (g @ ((spec.flux - 10) / 0.3)) * s * s
    below = norm.cdf(-mu / s)                                       # normal in the flux, cut at zero
    exact = mu + s * norm.ppf(below + 0.95 * (1 - below))
    assert r.upper_limit("b.A.flux") == pytest.approx(exact, rel=0.08)


def test_upper_limit_needs_posterior_samples():
    spec, model = _undetected_line()
    with pytest.raises(ValueError):
        fa.fit(spec, model, verbose=False).upper_limit("b.A.flux")


def test_mcmc_chains_do_not_depend_on_the_number_of_processes():
    from fantasy_agn.sampling import hmc
    a = hmc(_normal_log_prob, np.zeros(2), np.eye(2), np.full(2, -np.inf), np.full(2, np.inf),
            n_warmup=40, n_samples=40, seed=3, n_jobs=1)[0]
    b = hmc(_normal_log_prob, np.zeros(2), np.eye(2), np.full(2, -np.inf), np.full(2, np.inf),
            n_warmup=40, n_samples=40, seed=3, n_jobs=2)[0]
    assert np.array_equal(a, b)


def _normal_log_prob(t):
    return -0.5 * float(t @ t), -t


def _blend_fwhm(r):
    return r.full[r.model.params.names.index("b.fwhm")]


def test_robustness_of_a_well_measured_blend_matches_the_error_bars():
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    rob = fa.robustness(r, parameters=["b.fwhm", "n.fwhm"], quantities={"b fwhm": _blend_fwhm},
                        n_grid=8, starts="warm", n_jobs=1)
    assert not rob.improved and list(rob.parameters.flag) == ["measured", "measured"]
    for _, p in rob.parameters.iterrows():
        assert p.lo68 < p.value < p.hi68
        assert 0.8 < 0.5 * (p.hi68 - p.lo68) / p.error < 1.25     # a parabola: the covariance error is right
    q = rob.quantities.iloc[0]
    assert q.lo68 <= q.value <= q.hi68 and q.lo95 <= q.lo68 and q.hi95 >= q.hi68


def test_robustness_flags_a_weak_line_that_can_slide_to_its_limit():
    r = fa.fit(_blend_spectrum(450, noise=3.0), _blend_model(), verbose=False)
    rob = fa.robustness(r, parameters=["b.fwhm", "n.fwhm"], n_grid=8, starts="warm", n_jobs=1)
    flags = rob.parameters.set_index("parameter").flag
    assert flags["b.fwhm"] == "measured"
    assert flags["n.fwhm"] == "not excluded up to the lower limit"
    assert "n.fwhm" in rob.summary()


def test_robustness_finds_second_minima_and_interval_ends():
    from fantasy_agn.robustness import _crossing, _minima
    v = np.arange(9.0)
    d = np.array([9, 4, 0, 4, 9, 5, 2.5, 5, 9])        # global minimum at 2, a second one at 6 behind a barrier
    assert _minima(v, d, 2) == [6]
    assert _crossing(v, d, 2, 1.0, -1) == (1.75, False) and _crossing(v, d, 2, 1.0, +1) == (2.25, False)
    flat = np.array([3, 2, 0, 0.5, 0.8])                 # never above 4 towards the lower end
    assert _crossing(v[:5], flat, 2, 4.0, -1) == (0.0, True)


def test_broad_voigt_line_with_a_masked_centre_is_kept():
    from fantasy_agn.fitting import Problem, _fix_unconstrained
    w, c = np.linspace(4600, 5200, 1201), 4861.3
    f = 10 + 8 / (1 + ((w - c) / (c * 3000 / 2 / 299792.458)) ** 2)
    keep = np.abs(w - c) >= 60                          # the centre is masked: only the wings are fitted
    m = fa.Model(fa.components.PowerLaw(index=(0, -3, 3)),
                 fa.Lines("b", [("Hb", c)], fwhm=(3000, 0, 15000), profile="voigt"))
    prob = Problem(m, w[keep], f[keep], np.full(keep.sum(), 0.1), {})
    assert _fix_unconstrained(prob, m) == []


def test_compare_fits_reports_both_fits_on_the_same_pixels():
    s = _blend_spectrum(450, noise=0.3)
    a = fa.fit(s, _blend_model(), verbose=False)
    b = fa.fit(s, _blend_model().fix("o.fwhm", 600.0), verbose=False)
    cmp = fa.compare_fits(a, b, labels=("free", "fixed"), quantities={"b fwhm": lambda r: (_blend_fwhm(r), 1.0)})
    st = cmp.stats
    assert st["n_pixels"] == (a.mask & b.mask).sum() and st["n_free"][0] == st["n_free"][1] + 1
    assert np.isclose(st["chi2"][0], a.chi2) and st["dchi2"] > 25        # the fixed width fits worse
    row = cmp.table.iloc[0]
    assert np.isclose(row["difference"], row["fixed"] - row["free"])
    assert "fixed" in cmp.summary()
    fig = cmp.plot()
    assert len(fig.axes) == 4


def test_compare_fits_refuses_two_different_spectra():
    a = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    b = fa.fit(_two_line_spectrum(5, 5), _two_line_models()["free"], verbose=False)
    with pytest.raises(ValueError):
        fa.compare_fits(a, b)


def test_fwhm_of_a_profile_with_a_shoulder_at_half_maximum_is_fragile():
    from fantasy_agn.fitcompare import fwhm_fragile, half_max_crossings
    v = np.linspace(-20000, 20000, 8001)
    g = lambda c, s, a: a * np.exp(-0.5 * ((v - c) / s) ** 2)
    single = g(0, 2000, 1)
    assert not fwhm_fragile(v, single)[0] and abs(fwhm_fragile(v, single)[1] - 1.156) < 0.01
    lo, hi = half_max_crossings(v, single)
    assert abs((hi - lo) - 2.3548 * 2000) < 5
    shoulder = g(0, 2000, 1) + g(6000, 2500, 0.5)       # a red shelf right at half maximum
    assert fwhm_fragile(v, shoulder)[0]


def test_robustness_finds_a_better_fit_and_starts_from_it():
    s = _blend_spectrum(450)
    good = fa.fit(s, _blend_model(), verbose=False)
    poor = fa.fit(s, _blend_model().fix("b.fwhm", 4000.0), verbose=False)
    poor.model.params["b.fwhm"].fixed = False          # a result that is not at the best fit
    rob = fa.robustness(poor, parameters=["n.fwhm"], quantities={}, n_grid=4, starts="warm", n_jobs=1)
    assert rob.improved and rob.best.chi2 < poor.chi2 - 100
    again = fa.fit(s, rob.best_model(), verbose=False)
    assert again.chi2 <= good.chi2 + 0.01


def test_robustness_of_the_results_alone():
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    rob = fa.robustness(r, parameters=[], quantities={"b fwhm": lambda r: (_blend_fwhm(r), r.measure("Hb", ["b"])["fwhm_err"])},
                        n_jobs=1)
    assert rob.parameters.empty and "b fwhm" in rob.summary()
    q = rob.quantities.iloc[0]
    # a well-measured width: the direct scan of the result agrees with its error bar
    assert q.flag == "ok" and 0.8 < 0.5 * (q.hi68 - q.lo68) / q.error < 1.25


def test_narrow_line_inside_a_masked_band_gets_no_flux():
    # [OIII] is fitted; a second narrow line shares its width and lies inside a masked band. Only a very wide
    # profile would reach the fitted pixels, so the line is kept at first, but at the fitted width it has no effect.
    w = np.linspace(4800, 5200, 801)
    f = 10 + 20 * np.exp(-0.5 * ((w - 5006.8) / 3.0) ** 2) + np.random.default_rng(2).normal(0, 0.2, w.size)
    s = fa.Spectrum(w, f, err=np.full(w.size, 0.2), rest_frame=True).mask((4880, 4960))
    m = fa.Model(fa.components.PowerLaw(index=(0, -3, 3)),
                 fa.Lines("n", [("O3", 5006.8), ("X", 4920.0)], fwhm=(400, 100, 9000)))
    r = fa.fit(s, m, verbose=False)
    assert r.model.params["n.X.flux"].value == 0 and "n.X.flux" in r.unconstrained
    assert np.isfinite(r.model_flux()).all() and r.model_flux().max() < 2 * f.max()


def test_value_and_error_are_rounded_to_the_error():
    from fantasy_agn.plotting import _pm
    assert _pm(9337, 514) == "9340 $\\pm$ 510"
    assert _pm(14600, 848) == "(1.460 $\\pm$ 0.085)$\\times10^{4}$"
    assert _pm(0.1573, 0.061) == "0.157 $\\pm$ 0.061"
    assert _pm(float("nan"), 1) == "–"


def test_profiles_overlay_scales_each_line_to_its_peak():
    import matplotlib
    matplotlib.use("Agg")
    s = _blend_spectrum(450)
    a = fa.fit(s, _blend_model(), verbose=False)
    b = fa.fit(s, _blend_model().fix("b.fwhm", 3500.0), verbose=False)
    fig = fa.plotting.plot_profiles([a, b], "Hb", labels=["free", "fixed"], data=False)
    ax = fig.axes[0]
    curves = [ln for ln in ax.get_lines() if ln.get_label() and not ln.get_label().startswith("_")]
    assert len(curves) == 2 and all(abs(ln.get_ydata().max() - 1) < 1e-9 for ln in curves)
    assert "FWHM" in curves[0].get_label() and curves[1].get_label().startswith("fixed")
    with pytest.raises(ValueError):
        fa.plotting.plot_profiles([a, b], ["Hb", "O3"])


def test_publication_figure_zoom_panels_and_legend_inside():
    import matplotlib
    matplotlib.use("Agg")
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    fig = r.plot(zoom=[(4800, 4920), (4960, 5050)], legend="inside")
    assert len(fig.axes) == 4                         # spectrum, residuals and two zoom panels
    assert fig.axes[0].get_legend() is not None
    lo, hi = fig.axes[2].get_xlim()
    assert (lo, hi) == (4800, 4920)
    fig = r.plot(width="mnras", legend="right")
    assert abs(fig.get_size_inches()[0] - 3.31) < 1e-9


def test_app_writes_the_code_of_the_figure():
    from fantasy_agn.app.session import Session
    s = Session.__new__(Session)
    code = s.figure_code({"width": "aa", "legend": "inside", "zoom": "4650-5150, 6350-6800", "color_broad": "#2a78d6"})
    assert "width='aa'" in code and "legend='inside'" in code and "zoom=[(4650.0, 5150.0), (6350.0, 6800.0)]" in code
    assert "colors" not in code                       # the default colour of the broad lines is left out
    compile(code, "<figure code>", "exec")


def test_zoom_panels_can_subtract_the_continuum():
    import matplotlib
    matplotlib.use("Agg")
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    fig = r.plot(zoom=[(4800, 4920)], zoom_subtract="continuum")
    z = fig.axes[2]
    data = [ln for ln in z.get_lines() if ln.get_drawstyle() == "steps-mid"][0].get_ydata()
    # the continuum (10) is gone: the data far from the line scatter around 0
    assert abs(np.nanmedian(data[:20])) < 1.0 and "continuum" in z.get_ylabel()


def test_figure_font_height_and_line_grid_options():
    import matplotlib
    matplotlib.use("Agg")
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    h0 = r.plot().get_size_inches()[1]
    assert abs(r.plot(height=1.25).get_size_inches()[1] - 1.25 * h0) < 1e-9
    fig = r.plot(style="sans-serif")
    assert "sans" in fig.axes[0].xaxis.label.get_fontfamily()[0]
    fig = fa.plotting.plot_lines(r, ["Hb", "O3"], residuals=True, sharey=True, legend=False)
    mains = fig.axes[0::2]
    assert mains[0].get_ylim() == mains[1].get_ylim() and not fig.legends


def test_legend_inside_works_in_the_app_figures():
    from matplotlib.figure import Figure
    from fantasy_agn.app.session import Session
    r = fa.fit(_blend_spectrum(450), _blend_model(), verbose=False)
    fig = r.plot(fig=Figure(), legend="inside")              # a bare Figure has no renderer of its own
    assert fig.axes[0].get_legend() is not None
    s = Session.__new__(Session)
    for fmt in ("png", "pdf", "svg"):
        assert len(s.custom_figure({"legend": "inside", "label_lines": "1"}, fmt=fmt, r=r)) > 1000


def test_profiles_of_many_epochs_follow_a_sequential_ramp():
    import matplotlib
    matplotlib.use("Agg")
    fits = [fa.fit(_blend_spectrum(450, seed=k), _blend_model(), verbose=False) for k in range(8)]
    fig = fa.plotting.plot_profiles(fits, "Hb", labels=[f"epoch {k}" for k in range(8)], data=False)
    curves = [ln for ln in fig.axes[0].get_lines() if not ln.get_label().startswith("_")]
    assert len(curves) == 8 and len({tuple(ln.get_color()) for ln in curves}) == 8


def test_epoch_table_and_correlations_of_epochs_fitted_one_by_one():
    import matplotlib
    matplotlib.use("Agg")
    fits = []
    for k, scale in enumerate((0.6, 0.8, 1.0, 1.2, 1.4)):          # the broad line grows with the epochs
        w = np.linspace(4600, 5200, 1201)
        g = lambda c, fwhm, a: a * np.exp(-0.5 * ((w - c) / (fwhm / 2.3548 / 299792.458 * c)) ** 2)
        f = 10 + g(4861.3, 3000, 6 * scale) + g(4861.3, 450, 3) + g(5006.8, 450, 12)
        f = f + np.random.default_rng(k).normal(0, 0.15, w.size)
        s = fa.Spectrum(w, f, err=np.full(w.size, 0.15), rest_frame=True)
        s.mjd = 55000 + 30 * k
        m = _blend_model()
        m.components[1].role = "broad"
        fits.append(fa.fit(s, m, verbose=False))
    t = fa.epoch_table(fits)
    assert len(t) == 5 and {"Hb_flux", "Hb_fwhm", "Hb_sigma", "Hb_shift"} <= set(t.columns)
    c = fa.epoch_correlations(t, ["Hb_flux", "Hb_ew", "Hb_fwhm"])
    row = c[(c.x == "Hb_flux") & (c.y == "Hb_ew")].iloc[0]
    assert row.rho > 0.9 and row.p < 0.05 and not row.trivial
    fig = fa.plotting.plot_correlations(t, ["Hb_flux", "Hb_ew", "Hb_fwhm"])
    assert len(fig.axes) == 6                                     # three on the diagonal, three pairs
    with pytest.raises(ValueError):
        fa.epoch_table(fits[:1])
    # the app: the table goes to the browser once; saved figures and their code honour the lag and the left-out epochs
    from fantasy_agn.app.session import Session
    s = Session()
    s.ns["fits"] = fits
    s.wf.update(mode="multi", fit_kind="fits")
    assert s.wf_correlations() == {"id": 0, "n_epochs": 5}
    assert "epoch_table(fits)" in s.wf_correlations_code()
    s.ns["ep"] = t
    d = s.wf_correlations_data()
    hb = next(c for c in d["columns"] if c["name"] == "Hb_flux")
    assert len(d["t"]) == 5 and hb["log"] and len(hb["v"]) == 5 and "Hb_flux" in d["map"]
    view = s._epoch_view(lag=45, off=[4])
    assert len(view) == 4
    assert s.wf_relation_figure("Hb_flux", "Hb_ew", "png", 80, 0, [4])[:4] == b"\x89PNG"
    code = s.wf_relation_code("Hb_flux", "Hb_ew", 45, [4])
    assert "shift_continuum(ep, 45)" in code and "drop(index=[4])" in code
    s.ns["fits"] = list(fits)                                     # a new fit: the old table does not count any more
    assert s.wf_correlations()["id"] == 0
    # Results › Atlas: one card per epoch, binned over the fitted range, with the broad-line flux to order by
    at = s.wf_atlas()
    assert at["kind"] == "epochs" and [c["k"] for c in at["cards"]] == list(range(5))
    c0, c4 = at["cards"][0], at["cards"][4]
    assert c0["mjd"] == 55000 and len(c0["flux"]) == len(c0["model"]) == len(c0["resid"]) == 200
    assert c4["bflux"] > c0["bflux"] and c0["rchi2"] < 2 and c0["struct"] < 10
    assert s.wf_atlas() is at                                    # cached


def test_relation_over_epochs_recovers_a_power_law():
    import matplotlib
    import pandas as pd
    matplotlib.use("Agg")
    rng = np.random.default_rng(3)
    logl = np.linspace(43, 45, 30)
    ew = 10 ** (2 - 0.5 * (logl - 44) + rng.normal(0, 0.05, logl.size))      # slope -0.5, scatter 0.05 dex
    t = pd.DataFrame({"mjd": 55000 + np.arange(30.0), "logL5100": logl, "logL5100_err": np.full(30, 0.01),
                      "Hb_ew": ew, "Hb_ew_err": 0.01 * ew})
    fit = fa.correlations.fit_relation(t, "logL5100", "Hb_ew", n_boot=200)
    assert fit["log_y"] and not fit["log_x"]
    assert abs(fit["b"] + 0.5) < 3 * fit["b_err"] and abs(fit["scatter"] - 0.05) < 0.03
    fig, res = fa.plotting.plot_relation(t, "logL5100", "Hb_ew", n_boot=100)
    assert res["n"] == 30 and "log" in fig.axes[0].get_ylabel()
    t2 = t.copy()
    t2.loc[3, "Hb_ew_err"] = 1e4 * t2.loc[3, "Hb_ew"]                         # one huge error bar
    fig, _ = fa.plotting.plot_relation(t2, "logL5100", "Hb_ew", fit=False)
    assert np.ptp(fig.axes[0].get_ylim()) < 2                                 # the range follows the points


def test_app_sample_results_say_what_to_check_and_why_spectra_failed():
    import pandas as pd
    from fantasy_agn.app.session import Session
    from fantasy_agn.sample import SampleResult
    t = pd.DataFrame({"name": list("abcd"), "z": [0.1, 0.2, 1.2, 0.3], "cls": ["type1", "type1", np.nan, "passive"],
                      "preset": ["optical()", "optical()", np.nan, "passive()"], "rchi2": [1.1, 1.0, np.nan, 31.0],
                      "broad Hb FWHM": [3000.0, 80000.0, np.nan, np.nan],
                      "error": [np.nan, np.nan, "ValueError: fewer than 10 usable pixels in the fitted range", np.nan]})
    s = Session()
    s.ns["sample"] = SampleResult(t)
    st = s.sample_state()
    assert st["status"] == ["ok", "check", "failed", "check"] and (st["fitted"], st["check"], st["failed"]) == (3, 2, 1)
    assert "80000" in st["notes"][1] and "31.0" in st["notes"][3]
    assert st["failures"][0]["n"] == 1 and st["failures"][0]["fixable"] and st["median_rchi2"] == pytest.approx(1.1)
    assert [c["label"] for c in st["columns"]][:2] == ["Spectrum", "z"] and st["rows"][2][2] is None
    # Results › Atlas: rows without a kept fit, or that failed, become cards that say why
    s.wf["fit_kind"] = "sample"
    at = s.wf_atlas()
    assert at["kind"] == "sample" and len(at["cards"]) == 4
    assert "fewer than 10 usable pixels" in at["cards"][2]["error"] and "out=folder" in at["cards"][0]["error"]


def test_app_atlas_card_of_a_kept_sample_fit(tmp_path):
    import pandas as pd
    from fantasy_agn.app.session import Session
    from fantasy_agn.sample import SampleResult
    r = fa.fit(_blend_spectrum(450, seed=1), _blend_model(), verbose=False)
    path = str(tmp_path / "a.fantasy")
    fa.save(r, path)
    t = pd.DataFrame({"name": ["a"], "cls": ["type1"], "preset": ["optical()"], "rchi2": [float(r.rchi2)],
                      "broad Hb FWHM": [80000.0], "saved": [path]})
    s = Session()
    s.ns["sample"] = SampleResult(t)
    s.wf["fit_kind"] = "sample"
    c = s.wf_atlas()["cards"][0]
    assert "error" not in c and c["cls"] == "type1" and c["fwhm"] == 80000.0      # the numbers come from the catalog
    assert c["w0"] < c["w1"] and len(c["flux"]) == 200 and c["scale"] > 0


def _rest_spectrum(lo, hi, **kw):
    wave = np.arange(float(lo), float(hi), 2.0)
    return fa.Spectrum(wave, np.ones_like(wave), 0.1 * np.ones_like(wave), z=1.0, rest_frame=True, **kw)


def test_classify_uses_a_uv_model_when_the_spectrum_ends_too_blue():
    from fantasy_agn.classify import _uv_choice
    from matplotlib.figure import Figure
    mg = fa.classify(_rest_spectrum(2100, 5050))
    assert mg.label == "uv_mgii" and mg.RECOMMENDED[mg.label] == ("mgii", {}) and not mg.uncertain
    both = fa.classify(_rest_spectrum(1300, 3300))
    assert both.label == "uv_mgii" and "C IV is covered too" in both.reason     # Mg II is preferred
    civ = fa.classify(_rest_spectrum(1300, 2400))
    assert civ.label == "uv_civ" and civ.model().components
    assert civ.plot_steps(fig=Figure()) is not None
    assert _uv_choice(_rest_spectrum(3700, 7000)) is None                       # the optical classifier
    with pytest.raises(ValueError, match="no line model"):
        fa.classify(_rest_spectrum(3000, 3600))
    hole = _rest_spectrum(2100, 5050)
    hole.err[(hole.wave > 2650) & (hole.wave < 2900)] = np.nan                  # Mg II missing
    assert _uv_choice(hole) is None


def _fake_catalogs(monkeypatch, down=()):
    """Replies of the catalog services: a quasar at (10, 20), a star at (30, 40), nothing at (50, 60)."""
    import fantasy_agn.catalogs as cat
    calls = []

    def tap(service, query, timeout):
        calls.append(query)
        if service in down:
            raise OSError(f"{service}: down")
        rows = []
        if "CIRCLE('ICRS',10.000000,20.000000" in query:
            if " basic " in query:
                rows += [{"main_id": "QSO A", "otype": "QSO", "rvz_redshift": 0.5, "nbref": 9, "ra": 10.0, "dec": 20.0},
                         {"main_id": "side", "otype": "X", "rvz_redshift": None, "nbref": 1, "ra": 10.0, "dec": 20.0001}]
            if "objdir" in query:
                rows.append({"prefname": "A", "prefphytype": "QSO", "z": 0.5, "zflag": "SUN", "n_crosref": 3,
                             "ra": 10.0, "dec": 20.0})
            if " first " in query:
                rows.append({"name": "FIRST A", "int_flux_20_cm": 12.0, "ra": 10.0, "dec": 20.0})
        if "CIRCLE('ICRS',30.000000,40.000000" in query and " basic " in query:
            rows.append({"main_id": "* B", "otype": "WD*", "rvz_redshift": None, "nbref": 2, "ra": 30.0, "dec": 40.0})
        return rows

    def wise(k, o, radius, timeout):
        if o["ra"] != 10:
            return {}
        return {k: {"designation": "W", "w1mpro": 13.0, "w2mpro": 12.0, "w3mpro": 9.0, "w4mpro": 7.0, "ph_qual": "AAAU",
                    "sep": 0.1}}
    monkeypatch.setattr(cat, "_tap", tap)
    monkeypatch.setattr(cat, "_query_wise", wise)
    return calls


def test_lookup_catalogs_flags_what_disagrees_with_the_fit(monkeypatch):
    import pandas as pd
    calls = _fake_catalogs(monkeypatch)
    t = pd.DataFrame({"name": ["a", "b", "c", "d", "e"], "ra": [10.0, 30.0, 50.0, np.nan, 10.0],
                      "dec": [20.0, 40.0, 60.0, np.nan, 20.0], "z": [0.5, 0.0, 0.3, 0.1, 0.9],
                      "cls": ["passive", "type1", "type1", "type1", "type1"]})
    r = fa.lookup_catalogs(t)
    assert list(r["name"]) == list("abcde") and len(calls) == 8           # one query per catalog for the whole chunk
    a = r.iloc[0]
    assert a.catalog_name == "QSO A" and a.catalog_class == "broad-line AGN" and a.radio_mjy == 12.0
    assert a.wise_w1_w2 == pytest.approx(1.0) and pd.isna(a.wise_w4) and a.n_disagreements == 2     # W4 has quality U
    assert "classified here as passive" in a.disagreements and "W1\u2212W2" in a.disagreements
    assert "Listed as a star" in r.iloc[1].disagreements and r.iloc[2].n_disagreements == 0 and pd.isna(r.iloc[2].catalog_name)
    assert r.iloc[3].note == "no coordinates"
    assert "Redshift differs" in r.iloc[4].disagreements
    one = fa.lookup_catalogs(_rest_spectrum(2100, 5050, ra=10.0, dec=20.0), cls="uv_mgii")
    assert one.iloc[0].n_disagreements == 1 and "Redshift differs" in one.iloc[0].disagreements     # the helper has z = 1


def test_lookup_catalogs_names_a_service_it_cannot_reach(monkeypatch):
    _fake_catalogs(monkeypatch, down=("simbad",))
    r = fa.lookup_catalogs((10.0, 20.0)).iloc[0]
    assert r.unreachable == "simbad" and r.ned_name == "A" and "simbad_name" not in r


def test_lookup_catalogs_uses_the_activity_type_of_ned(monkeypatch):
    import fantasy_agn.catalogs as cat
    asked = []

    def tap(service, query, timeout):
        rows = []
        if service == "ned":
            for ra, name in ((10.0, "Sy1 galaxy"), (30.0, "Sy2 galaxy"), (50.0, "plain")):
                if f"CIRCLE('ICRS',{ra:.6f}," in query:
                    rows.append({"prefname": name, "prefphytype": "G", "z": 0.01, "zflag": "SUN", "n_crosref": 1,
                                 "ra": ra, "dec": 20.0})
        return rows

    def activity(k, name, timeout):
        asked.append(name)
        return {"Sy1 galaxy": {k: "Sy 1.5"}, "Sy2 galaxy": {k: "Sy 2"}}.get(name, {})
    monkeypatch.setattr(cat, "_tap", tap)
    monkeypatch.setattr(cat, "_query_wise", lambda k, o, radius, timeout: {})
    monkeypatch.setattr(cat, "_query_ned_activity", activity)
    import pandas as pd
    t = pd.DataFrame({"name": list("abc"), "ra": [10.0, 30.0, 50.0], "dec": [20.0] * 3, "z": [0.01] * 3,
                      "cls": ["passive", "type1", "type1"]})
    r = fa.lookup_catalogs(t)
    assert sorted(asked) == ["Sy1 galaxy", "Sy2 galaxy", "plain"]
    assert r.iloc[0].catalog_family == "broad" and "NED: Sy 1.5" in r.iloc[0].disagreements
    assert r.iloc[1].catalog_family == "narrow" and "NED: Sy 2" in r.iloc[1].disagreements
    assert r.iloc[2].catalog_family == "galaxy"
    assert cat._activity_family("Flat-Spectrum Radio Source") is None


def test_app_catalog_lookup_of_a_sample_and_of_one_object(monkeypatch):
    import pandas as pd
    from fantasy_agn.app.session import Session
    from fantasy_agn.sample import SampleResult
    _fake_catalogs(monkeypatch)
    t = pd.DataFrame({"name": ["a", "b"], "z": [0.5, 1.0], "cls": ["passive", "uv_mgii"], "ra": [10.0, 50.0],
                      "dec": [20.0, 60.0], "rchi2": [1.0, 1.0], "at_limit": ["", "broad.fwhm"]})
    s = Session()
    s.ns["sample"] = SampleResult(t)
    assert s.sample_state()["catalog"] is None and "limit of the model" in s.sample_state()["notes"][1]
    s.ns["catalogs"] = fa.lookup_catalogs(s.ns["sample"])
    st = s.sample_state()
    labels = [c["label"] for c in st["columns"]]
    assert st["catalog"]["flagged"] == 1 and st["catalog"]["flags"] == [2, 0] and labels[labels.index("Class") + 1] == "Catalogs say"
    assert "classified here as passive" in st["notes"][0]
    assert "catalog_class" in s.sample_csv().splitlines()[0]
    s.ns["catalogs"] = s.ns["catalogs"].iloc[:1]                 # made for other spectra: not shown
    assert s.sample_state()["catalog"] is None
    o = Session()
    assert o.wf_catalog() is None
    o.ns["raw"] = [_rest_spectrum(2100, 5050)]
    assert o.wf_catalog()["has_coords"] is False
    with pytest.raises(ValueError, match="no sky position"):
        o.wf_catalogs_code()
    o.ns["raw"] = [_rest_spectrum(2100, 5050, ra=10.0, dec=20.0)]
    assert "fa.lookup_catalogs(raw[0])" in o.wf_catalogs_code()
    assert o.wf_catalog()["done"] is False
    o.ns["catalogs"] = fa.lookup_catalogs(o.ns["raw"][0])
    state = o.wf_catalog()
    assert state["done"] and state["found"] == 1 and o.wf_catalog_row(0)["catalog_name"] == "QSO A"
    assert "classified" not in (state.get("text") or "")          # no class is known before the fit
    v = fa.catalogs.with_fit_classes(o.ns["catalogs"], cls=["passive"])
    assert "classified here as passive" in v["disagreements"].iloc[0] and v["n_disagreements"].iloc[0] >= 1
    o.ns["raw"] = [_rest_spectrum(2100, 5050, ra=10.0, dec=20.0, name="other")]    # other spectra: not shown
    assert o.wf_catalog()["done"] is False


def test_sky_image_tries_the_next_survey(monkeypatch):
    import io
    from PIL import Image
    import fantasy_agn.catalogs as cat

    def jpeg(noise):
        a = np.random.default_rng(0).integers(0, 255, (40, 40, 3)).astype("uint8") if noise else np.zeros((40, 40, 3), "uint8")
        b = io.BytesIO()
        Image.fromarray(a).save(b, "JPEG")
        return b.getvalue()

    def down(*a):
        raise OSError("down")
    monkeypatch.setattr(cat, "_IMAGE_SURVEYS", (("A", down), ("B", lambda *a: jpeg(False)), ("C", lambda *a: jpeg(True))))
    body, survey = cat.sky_image(10.0, 20.0)
    assert survey == "C" and body[:2] == b"\xff\xd8"
    monkeypatch.setattr(cat, "_IMAGE_SURVEYS", (("A", down),))
    assert cat.sky_image(10.0, 20.0) == (None, None)


def test_iccf_does_not_depend_on_the_order_of_the_epochs():
    from fantasy_agn.rm import iccf
    t = np.arange(0.0, 300.0, 5.0)
    f1 = np.sin(t / 25.0) + 0.5 * np.sin(t / 9.0)
    f2 = np.sin((t - 20.0) / 25.0) + 0.5 * np.sin((t - 20.0) / 9.0)
    lags = np.arange(-60.0, 61.0, 2.0)
    r = iccf(t, f1, t, f2, lags)
    assert abs(lags[np.nanargmax(r)] - 20.0) <= 2.0
    o = np.random.default_rng(0).permutation(t.size)
    assert np.allclose(iccf(t[o], f1[o], t[o], f2[o], lags), r, equal_nan=True)


def test_continuum_shifted_by_a_lag_between_the_epochs():
    import pandas as pd
    t = pd.DataFrame({"mjd": [0.0, 10.0, 20.0, 30.0], "cont5100": [1.0, 2.0, 3.0, 4.0], "cont5100_err": [0.1] * 4,
                      "Hb_flux": [5.0, 6.0, 7.0, 8.0], "Hb_flux_err": [0.1] * 4})
    s = fa.correlations.shift_continuum(t, 5.0)
    assert np.isnan(s.cont5100[0]) and np.allclose(s.cont5100[1:], [1.5, 2.5, 3.5])
    assert np.allclose(s.Hb_flux, t.Hb_flux)                     # the line is not shifted
    assert fa.correlations.shift_continuum(t, 0).equals(t)


def test_epochs_fitted_one_by_one_give_a_light_curve_and_rms_spectrum():
    fits = []
    for k, scale in enumerate((0.6, 1.0, 1.4, 1.1)):
        w = np.linspace(4600, 5200, 1201)
        g = lambda c, fwhm, a: a * np.exp(-0.5 * ((w - c) / (fwhm / 2.3548 / 299792.458 * c)) ** 2)
        f = 10 + g(4861.3, 3000, 6 * scale) + g(4861.3, 450, 3) + g(5006.8, 450, 12)
        s = fa.Spectrum(w, f + np.random.default_rng(k).normal(0, 0.15, w.size), err=np.full(w.size, 0.15), rest_frame=True)
        s.mjd = 55000 + 10 * k
        fits.append(fa.fit(s, _blend_model(), verbose=False))
    me = fa.EpochSet(fits + [RuntimeError("failed")])                 # a failed epoch is left out
    lc = me.light_curve()
    assert len(me.epochs) == 4 and np.all(np.diff(lc.Hb_flux.to_numpy()[[0, 1, 2]]) > 0)
    assert "one by one" in me.summary()
    with pytest.raises(ValueError):
        fa.EpochSet(fits[:1])


def test_app_session_is_autosaved_restored_and_can_be_stopped(tmp_path, monkeypatch):
    import time
    from fantasy_agn.app.session import Session
    monkeypatch.setenv("FANTASY_AUTOSAVE", str(tmp_path))

    def wait(s):
        for _ in range(400):
            if not s.busy:
                break
            time.sleep(0.05)
        for _ in range(100):             # the auto-save runs in a thread after the cell
            if not s._saving.locked():
                break
            time.sleep(0.05)

    s = Session()
    assert s.state()["autosaved"] == {}
    s.run_cell("y = 42")
    wait(s)
    time.sleep(0.2)
    wait(s)
    meta = Session.autosaved()
    assert meta["last"]["cells"] == 1 and "previous" not in meta

    # a long cell is stopped, the namespace keeps what ran before
    s.run_cell("import time\nfor i in range(400):\n    time.sleep(0.02)\nz = 1")
    time.sleep(0.3)
    assert s.stop()
    wait(s)
    assert s.error.startswith("Stopped") and "z" not in s.ns and s.cells[-1].out[-1] == ("Stopped.", "err")
    assert not s.stop()

    # a new app restores it; its first save keeps the restored one as 'previous'
    s2 = Session()
    assert s2.state()["autosaved"]["last"]["cells"] >= 1
    s2.restore("last")
    assert s2.ns["y"] == 42 and s2.state()["autosaved"] is None
    s2.run_cell("w = y + 1")
    wait(s2)
    time.sleep(0.2)
    wait(s2)
    assert set(Session.autosaved()) == {"last", "previous"}
    monkeypatch.setenv("FANTASY_AUTOSAVE", "0")
    assert Session.autosaved() == {}


def test_line_limit_of_an_undetected_line():
    rng = np.random.default_rng(2)
    wave = np.arange(4650, 5150, 1.0)
    m = _toy_truth()
    m.set("narrow.Hb.flux", 0.0)
    spec = fa.Spectrum(wave, m.evaluate(wave, {"medium": "vacuum"}) + rng.normal(0, 2, wave.size),
                       np.full(wave.size, 2.0), rest_frame=True)
    r = fa.fit(spec, _toy_model(), verbose=False)
    lim = r.line_limit("Hb", "narrow")
    # the flux error of a Gaussian of the fitted width from the noise alone
    p = r.model.params
    sl = 4862.68 * (1 + p["narrow.shift"].value / 299792.458) * p["narrow.fwhm"].value / 2.3548 / 299792.458
    g = np.exp(-0.5 * ((wave - 4862.68 * (1 + p["narrow.shift"].value / 299792.458)) / sl) ** 2) / (np.sqrt(2 * np.pi) * sl)
    expected = 2.0 * r.error_scale / np.sqrt(np.sum(g ** 2))
    assert lim["method"] == "noise" and abs(lim["sigma"] / expected - 1) < 0.1
    assert lim["limit"] == pytest.approx(max(lim["flux"], 0) + 3 * lim["sigma"])
    with pytest.raises(ValueError):
        r.line_limit("Ha")
    # a strong line: the limit is above its flux
    assert r.line_limit("[OIII]5007")["limit"] > r.line_limit("[OIII]5007")["flux"]


@needs_templates
def test_model_builder_recipes():
    """The model builder: any preset taken apart into components and rebuilt exactly, edits as code."""
    from fantasy_agn.app import builder as mb
    for name, kw in [("optical", {}), ("optical", {"host": True, "broad_components": 3}), ("optical", {"physical": True}),
                     ("mgii", {}), ("civ", {}), ("seyfert2", {}), ("tde", {}), ("hbeta", {"broad_profile": "gauss-hermite"})]:
        m = getattr(fa.presets, name)(**kw)
        st = mb.decompose(m)
        code, dropped = mb.code_of(st)
        assert not dropped and mb.run_code(code).to_dict()["params"] == m.to_dict()["params"], name
    # an edited state: one more component, a tie, limits, a fixed parameter, your own line
    st = mb.decompose(_toy_model())
    st["comps"].append({"type": "Lines", "kw": {"name": "wing", "lines": ["[OIII]5007", ["mystery", 4990.0]],
                                                "role": "outflow", "fwhm": [900, 300, 2500]}, "on": True})
    st["params"].update({"wing.shift": {"tie": ["narrow.shift", 1]}, "broad.fwhm": {"min": 2000, "max": 7000},
                         "continuum.index": {"fixed": True, "value": -1.5}, "gone.fwhm": {"max": 1}})
    code, dropped = mb.code_of(st)
    assert dropped == ["gone.fwhm"]
    assert "fa.Lines('wing', ('[OIII]5007', ('mystery', 4990)), " in code and "fwhm=(900, 300, 2500)" in code
    assert "model.tie('wing.shift', 'narrow.shift')" in code and "model.set('broad.fwhm', min=2000, max=7000)" in code
    m = mb.run_code(code)
    assert m.params["wing.shift"].tie[0] == "narrow.shift" and m.params["continuum.index"].fixed
    assert m.params["continuum.index"].value == -1.5 and m.params["broad.fwhm"].max == 7000
    # a component that is off is left out; bad names and duplicates are refused
    st["comps"][-1]["on"] = False
    assert "wing" not in mb.code_of(st)[0]
    st["comps"][-1].update(on=True, kw={**st["comps"][-1]["kw"], "name": "broad"})
    with pytest.raises(ValueError, match="two components"):
        mb.code_of(st)
    assert mb.lit(0.33557046979865773) == "0.33557046979865773" and mb.lit(float("inf")) == "float('inf')"
    d = mb.describe(mb.run_code(mb.code_of(mb.decompose(_toy_model()))[0]))
    assert [c["group"] for c in d["components"]] == ["continuum", "narrow", "broad"] and d["nfree"] > 0
    assert {t["type"] for t in mb.types()} >= {"Lines", "FeII", "StellarPopulation", "Absorption"}
    # candidate ladders: switches of the preset's own options, never one that changes the fitted range
    lad = mb.ladder("optical", {})
    assert [k for k, *_ in lad][:3] == ["base", "b1", "b3"] and len(lad) <= mb.MAX_CANDIDATES
    assert all("red" not in kw and "wave_range" not in kw for *_, kw in lad)
    assert [k for k, *_ in mb.ladder("civ", {})] == ["base", "b1", "b3"]


def test_app_model_gallery_and_builder():
    from fantasy_agn.app.session import Session
    import time
    s = Session()
    s.ns["spec"] = _toy_spectrum(np.random.default_rng(4))
    s.ns["model"] = _toy_model()
    out = s.mb_decompose({"model": True})
    assert out["code"].startswith("model = fa.Model(") and len(out["components"]) == 3
    assert {"lines", "sets", "range"} <= set(s.mb_lines()) and any(l["name"] == "Hb" for l in s.mb_lines()["lines"])
    p = s.mb_preview(out["state"])
    assert set(p["comps"]) == {"continuum", "narrow", "broad"} and len(p["wave"]) == 420
    assert s.mb_build(out["state"])["nfree"] == out["nfree"]
    code = s.recipe_code(out["state"])
    assert code.endswith("\nmodel") and s._preset_made is None
    assert s.preset_kw_code("optical", {"host": True}) == "model = fa.presets.optical(host=True)\nmodel"
    with pytest.raises(ValueError):
        s.preset_kw_code("optical", {"nothing": 1})
    # a model of the builder fitted in the background joins the gallery
    st = out["state"]
    st["params"]["broad.fwhm"] = {"max": 9000}
    key = s.gallery_fit(st, "tighter", "broad.fwhm max")
    t0 = time.time()
    while s.wf_gallery()["pending"] and time.time() - t0 < 120:
        time.sleep(0.1)
    g = s.wf_gallery()
    c = next(x for x in g["cands"] if x["key"] == key)
    assert c["mine"] and c["title"] == "#1 tighter" and "error" not in c and c["state"] is st
    assert c["nfree"] == out["nfree"] and np.isfinite(c["bic"]) and len(c["wave"]) == 420
    s.ns["spec"] = _toy_spectrum(np.random.default_rng(5))      # another spectrum: a new gallery
    assert s.wf_gallery()["cands"] == []


def test_fit_helper_process_same_result(monkeypatch):
    """The staged search runs in a helper process while the direct one runs here; the fit is the same
    as with both run in turn (FANTASY_HELPER=0), and a broken helper falls back to running it here."""
    from fantasy_agn import fitting
    m = _toy_model()
    m.add(fa.Lines("outflow", ["[OIII]4959", "[OIII]5007"], role="outflow", fwhm=(1000, 400, 3000),
                   shift=(-300, -1500, 500), ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}))
    assert fitting._extra_components(m)
    spec = _toy_spectrum(np.random.default_rng(11))
    monkeypatch.setenv("FANTASY_THREADS", "4")
    monkeypatch.setenv("FANTASY_HELPER", "0")
    serial = fa.fit(spec, m.copy(), verbose=False)
    monkeypatch.setenv("FANTASY_HELPER", "1")
    used = []
    submit = fitting._HELPER.submit
    monkeypatch.setattr(fitting._HELPER, "submit", lambda *a: used.append(a[0]) or submit(*a))
    helped = fa.fit(spec, m.copy(), verbose=False)
    assert used == ["_staged_fit"] and fitting._HELPER.proc is not None
    assert helped.chi2 == pytest.approx(serial.chi2, rel=1e-9)
    assert np.allclose(helped.full, serial.full, rtol=1e-6, atol=1e-9)
    # a helper that cannot run the call: the search runs here, with the same result
    monkeypatch.setattr(fitting._HELPER, "submit", lambda *a: submit("not_allowed"))
    fallback = fa.fit(spec, m.copy(), verbose=False)
    assert fallback.chi2 == pytest.approx(serial.chi2, rel=1e-9)
    fitting._HELPER.failures = 0
    fitting._HELPER.stop()


def test_pixel_mean_profiles(monkeypatch):
    """Line profiles are means over the pixels: equal to the integral, with exact derivatives, for every
    profile; a narrow line of a pixel-averaged spectrum gets its true width back (point-sampled profiles
    overestimate it); FANTASY_PIXEL=0 and contexts without a pixel grid keep the point values."""
    from scipy.integrate import quad
    from fantasy_agn.profiles import evaluate_profile, gaussian_windowed
    x = np.arange(4990, 5030, 1.15)
    pix = np.full(x.size, 1.15)
    lam0, sig = np.array([5008.24]), np.array([60.0])
    kins = {"gaussian": [80., 150.], "voigt": [80., 150., 100.], "gauss-hermite": [80., 150., 0.1, 0.05],
            "stratified": [80., 400., 1.0, 2.5], "lorentzian": [80., 150.]}
    for name, k in kins.items():
        kin = [np.array([v]) for v in k]
        P, D = evaluate_profile(name, x, lam0, kin, sig, pix)
        point = lambda t: evaluate_profile(name, np.array([t]), lam0, kin, sig)[0][0, 0]
        exact = np.array([quad(point, a - 0.575, a + 0.575)[0] / 1.15 for a in x])
        assert np.abs(P[:, 0] - exact).max() < (1e-10 if name == "gaussian" else 2e-3) * P.max()
        for j in range(len(k)):
            h = 1e-4 * max(abs(k[j]), 1)
            up = [v.copy() for v in kin]
            dn = [v.copy() for v in kin]
            up[j] += h
            dn[j] -= h
            fd = (evaluate_profile(name, x, lam0, up, sig, pix)[0] - evaluate_profile(name, x, lam0, dn, sig, pix)[0])
            assert np.abs(fd / (2 * h) - D[j]).max() < 1e-6 * np.abs(D[j]).max()
    x2, kin2 = np.arange(4800, 5200, 1.15), [np.array([80., -50.]), np.array([150., 3000.])]
    l2, s2, p2 = np.array([5008.24, 4960.3]), np.array([60., 60.]), np.full(x2.size, 1.15)
    P, D = evaluate_profile("gaussian", x2, l2, kin2, s2, p2)
    W = gaussian_windowed(x2, l2, kin2[0], kin2[1], s2, pix=p2)
    for a, b in zip(W, [P] + D):
        assert np.abs(a.toarray() - b).max() < 1e-12 * np.abs(b).max()
    # a narrow line on a log grid of 69 km/s pixels (SDSS), the truth averaged over each pixel
    lw = np.arange(np.log10(4700), np.log10(5100), 1e-4)
    wave = 10 ** lw
    m = fa.Model(fa.PowerLaw("continuum", pivot=5000),
                 fa.Lines("narrow", ["Hb", "[OIII]4959", "[OIII]5007"], fwhm=(300, 50, 1500), shift=(0, -500, 500),
                          ratios={"[OIII]4959": ("[OIII]5007", 1 / 2.98)}), wave_range=(4700, 5100))
    t = m.copy()
    t.set("continuum.amp", 50).set("narrow.Hb.flux", 300).set("narrow.[OIII]5007.flux", 1500)
    t.set("narrow.fwhm", 120).set("narrow.shift", 20)
    fine = 10 ** (lw[:, None] + (np.arange(15) - 7)[None] * 1e-4 / 15)
    truth = t.evaluate(fine.ravel(), {"medium": "vacuum", "resolution": 150.0}).reshape(fine.shape).mean(1)
    spec = fa.Spectrum(wave, truth + np.random.default_rng(0).normal(0, 1.0, wave.size), np.ones(wave.size),
                       rest_frame=True, resolution=150.0)
    assert "pixel" in spec.context
    fwhm = lambda r: r.params.set_index("parameter").loc["narrow.fwhm", "value"]
    r = fa.fit(spec, m.copy(), verbose=False)
    assert abs(fwhm(r) - 120) < 3
    monkeypatch.setenv("FANTASY_PIXEL", "0")
    assert "pixel" not in spec.context
    assert fwhm(fa.fit(spec, m.copy(), verbose=False)) > 125           # point sampling: too wide


def test_app_read_path_forgiving(tmp_path):
    """The Read step's path box: a folder means its spectra; quotes, shell escapes, file:// URLs and
    names with [ ] work; nothing matching gives a message that says why."""
    import shutil
    from fantasy_agn.app.steps import resolve_pattern
    from fantasy_agn.app.session import Session
    src = os.path.join(os.path.dirname(fa.__file__), "data", "examples", "spec-1059-52618-0453.fits")
    d = tmp_path / "My Data"
    d.mkdir()
    shutil.copy(src, d / "a.fits")
    shutil.copy(src, d / "obs[1].fits")
    (d / "notes.md").write_text("x")
    names = lambda r: [os.path.basename(f) for f in r[0]]
    assert names(resolve_pattern(str(d))) == ["a.fits", "obs[1].fits"]
    assert names(resolve_pattern(f"'{d}/a.fits'")) == ["a.fits"]
    assert names(resolve_pattern(str(d).replace(" ", "\\ ") + "/*.fits")) == ["a.fits", "obs[1].fits"]
    assert names(resolve_pattern(f"file://{str(d).replace(' ', '%20')}/obs[1].fits")) == ["obs[1].fits"]
    assert resolve_pattern(str(d / "*.fits"))[1] == str(d / "*.fits")
    with pytest.raises(ValueError, match="does not exist"):
        resolve_pattern(str(tmp_path / "nope" / "*.txt"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no spectra"):
        resolve_pattern(str(tmp_path / "empty"))
    s = Session()
    code = s.wf_read_code("sample", pattern=str(d))
    assert "obs[1].fits" in code and "a.fits" in code


def test_update_check_and_install(monkeypatch, tmp_path, capsys):
    """A newer release on PyPI (faked here: no network) is found, kept for a day, shown by the CLI and
    the app, and installed with the command that fits the installation; pre-releases are ignored."""
    import io
    import json
    import subprocess
    import threading
    import urllib.request
    from fantasy_agn import update, cli
    monkeypatch.setenv("FANTASY_DATA", str(tmp_path))
    monkeypatch.delenv("FANTASY_NO_UPDATE_CHECK", raising=False)
    assert update.newer("2.0.1", "2.0.0") and update.newer("2.1", "2.0.9") and not update.newer("2.0.0", "2.0")
    assert not update.newer("2.1.0rc1", "2.0.0") and not update.newer("1.0.0", "2.0.0")
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return io.BytesIO(json.dumps({"info": {"version": "9.9.0"}}).encode())
    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)
    u = update.check()
    assert u["latest"] == "9.9.0" and u["newer"] and u["current"] == fa.__version__
    assert update.check()["latest"] == "9.9.0" and len(calls) == 1          # cached for a day
    update.check(force=True)
    assert len(calls) == 2 and calls[0] == update.PYPI_URL
    # the command fits the installation
    assert update.update_command("9.9.0", "pip")[-1] == "fantasy_agn==9.9.0"
    monkeypatch.setattr(update, "_uv", lambda: "/usr/bin/uv")
    assert update.update_command("9.9.0", "uv-tool")[:4] == ["/usr/bin/uv", "tool", "install", "--force"]
    assert update.update_command("9.9.0", "source") is None
    ran = []
    monkeypatch.setattr(update, "install_kind", lambda: "pip")
    monkeypatch.setattr(update.subprocess, "run",
                        lambda cmd, **kw: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0, "Successfully installed", ""))
    ok, out = update.install("9.9.0")
    assert ok and ran[0][-1] == "fantasy_agn==9.9.0" and "Successfully installed" in out
    assert update.install(fa.__version__) == (True, f"FANTASY {fa.__version__} is already the latest version.")
    # caches bypassed (a cached index misses a fresh release), colour codes removed, a hint when not listed yet
    assert {"--no-cache-dir", "--refresh-package"} & set(update.update_command("9.9.0", "pip"))   # pip, or uv without pip
    assert "--refresh-package" in update.update_command("9.9.0", "uv-tool")
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "\x1b[31mERROR: No matching distribution found for fantasy_agn==9.9.0\x1b[0m"))
    ok, out = update.install("9.9.0")
    assert not ok and "\x1b" not in out and "try again in a few minutes" in out
    monkeypatch.setattr(update.subprocess, "run",
                        lambda cmd, **kw: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0, "Successfully installed", ""))
    # the CLI
    assert cli.main(["update", "--check"]) == 0
    assert "9.9.0 is published" in capsys.readouterr().out
    # switched off: no request at all
    monkeypatch.setenv("FANTASY_NO_UPDATE_CHECK", "1")
    assert update.check()["latest"] is None and len(calls) == 3            # the CLI asked again; now nothing
    monkeypatch.delenv("FANTASY_NO_UPDATE_CHECK")
    # PyPI unreachable: no error, latest unknown
    monkeypatch.setattr(update.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert update.latest_version(force=True) is None
    # the app: GET /api/update, POST /api/update installs
    monkeypatch.setattr(update, "latest_version", lambda timeout=4.0, force=False: "9.9.0")
    from fantasy_agn.app.server import Handler, make_server
    server = make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base, h = f"http://127.0.0.1:{server.server_address[1]}", {"X-Fantasy-Token": Handler.token}
        import urllib.request as ur
        real = ur.build_opener().open              # urlopen itself is faked above
        got = json.loads(real(ur.Request(base + "/api/update", headers=h)).read())
        assert got["newer"] and got["latest"] == "9.9.0"
        res = json.loads(real(ur.Request(base + "/api/update", data=b'{"version": "9.9.0"}', method="POST",
                                         headers=dict(h, **{"Content-Type": "application/json"}))).read())
        assert res["ok"] and ran[-1][-1] == "fantasy_agn==9.9.0"
    finally:
        server.shutdown()
        server.server_close()


def test_https_falls_back_to_system_certificates(monkeypatch, tmp_path):
    # a server whose root Python's certificate file lacks (IRSA on macOS): tried again with the system's roots
    import io
    import ssl
    import urllib.error
    import urllib.request
    from fantasy_agn import _net
    refused = urllib.error.URLError(ssl.SSLCertVerificationError(1, "unable to get local issuer certificate"))
    sentinel = ssl.create_default_context()
    monkeypatch.setattr(_net, "fallback_context", lambda: sentinel)
    seen = []

    def fake_urlopen(url, timeout=None, context=None):
        seen.append(context)
        if context is None:
            raise refused
        return io.BytesIO(b"ok")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert _net.urlopen("https://example.org", timeout=5).read() == b"ok" and seen == [None, sentinel]
    # other failures are not retried
    seen.clear()
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=None, context=None: seen.append(context) or (_ for _ in ()).throw(
                            urllib.error.URLError(OSError("offline"))))
    with pytest.raises(urllib.error.URLError, match="offline"):
        _net.urlopen("https://example.org")
    assert seen == [None]
    # downloads (the stellar templates) too
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, path: (_ for _ in ()).throw(refused))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _net.download("https://example.org/file", str(tmp_path / "f"))
    assert (tmp_path / "f").read_bytes() == b"ok"
    # the services go through it
    from fantasy_agn import extinction
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None, context=None: (
        (_ for _ in ()).throw(refused) if context is None else io.BytesIO(
            b'<results status="ok"><result><statistics><refPixelValueSFD>0.0306 (mag)</refPixelValueSFD>'
            b'<refPixelValueSandF>0.0263 (mag)</refPixelValueSandF></statistics></result></results>')))
    extinction._IRSA_CACHE.clear()
    try:
        sfd, sf = extinction.ebv_irsa(117.2942, 28.70405)
        assert sfd == pytest.approx(0.0306) and sf == pytest.approx(0.0263)
    finally:
        extinction._IRSA_CACHE.clear()
    # the real fallback context still verifies certificates
    monkeypatch.undo()
    ctx = _net.fallback_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
