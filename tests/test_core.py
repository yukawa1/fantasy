import os

import numpy as np
from astropy.io import fits
import pytest

import fantasy_agn as fa
from fantasy_agn.wavelengths import air_to_vac, vac_to_air

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, "..", "examples")


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
    h = fa.fit(spec, fa.presets.halpha(), verbose=False)
    assert h.rchi2 < 3 and not h.at_limit


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


def test_gui_builds():
    pytest.importorskip("ipywidgets")
    classic = fa.gui(classic=True)
    assert classic.preset.value in fa.presets.PRESETS
    guided = fa.gui()
    from fantasy_agn.workflow import MODELS
    assert guided.model_dd.value in MODELS and "import fantasy_agn" not in guided.code.value


def test_optical_preset_follows_blue_continuum():
    # A single power law cannot follow the continuum curvature over 3700-6900 A;
    # the default optical model (broken power law + Balmer continuum) must.
    spec = fa.read(os.path.join(EXAMPLES, "spec-1059-52618-0453.fits")).to_rest_frame()
    r = fa.fit(spec, fa.presets.optical(), verbose=False)
    resid = (spec.flux - r.model_flux()) / spec.err
    blue = r.mask & (spec.wave < 4800)
    assert r.rchi2 < 1.3
    assert np.mean(resid[blue] ** 2) < 1.2


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


needs_templates = pytest.mark.skipif("emiles" not in fa.templates.available(),
                                     reason="run fa.templates.download('emiles') to test the stellar host")


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
    assert "not detected" in hp.get("warning", "")


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
    assert np.allclose(draws.mean(axis=0), mean, atol=0.35)
    assert np.allclose(np.cov(draws.T), cov, rtol=0.25, atol=0.5)


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
    assert s.figure_bytes("pdf")[:4] == b"%PDF"


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
    finally:
        server.shutdown()
        server.server_close()


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


def test_app_plot_data_and_tables():
    from fantasy_agn.app.session import Session, _plain_label
    s = Session()
    assert s.plot_data() is None and s.data_tables() == []
    s.run_cell(s.read_code(path=os.path.join(EXAMPLES, "spec-1059-52618-0453.fits"), deredden=False))
    _wait(s)
    d = s.plot_data()
    assert len(d["x"]) == len(d["flux"]) and d["resid"] is None and d["series"] == []
    s.run_cell(s.fit_code("hbeta", broad_components=1, outflow=False))
    _wait(s)
    d = s.plot_data()
    names = [x["name"] for x in d["series"]]
    assert names[0] == "model" and "broad lines" in names and "narrow lines" in names
    assert len(d["resid"]) == len(d["x"]) and 4000 < d["x"][0] < d["x"][-1] < 5600
    assert [t["title"] for t in s.data_tables()] == ["Spectrum", "Fitted lines"]
    assert _plain_label("$F_\\lambda$ ($10^{-17}\\,\\mathrm{erg\\,s^{-1}}$)") == "Fλ (10⁻¹⁷ erg s⁻¹)"


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
    ebv, src = galactic_ebv(214.8, 53.6)                         # no file value, no maps -> IRSA
    assert ebv == 0.0094 and "Schlafly" in src and "214.8" in calls[0]
    assert galactic_ebv(214.8, 53.6, sf11=False)[0] == 0.0109
    ebv, src = galactic_ebv(None, None, file_value=0.05)         # DESI: value in the file
    assert abs(ebv - 0.043) < 1e-9 and "file" in src and len(calls) == 2
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
