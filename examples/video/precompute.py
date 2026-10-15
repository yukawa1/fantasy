"""Run the classification and the fit shown in the video; save what it draws."""
import json, time, warnings
import numpy as np
import fantasy_agn as fa
from fantasy_agn.classify import _line_model, DIAG_LINES, _MASK_BROAD, C_KMS
from fantasy_agn.lines import line_wavelength

warnings.filterwarnings("ignore")
FILE = "examples/classes/spec-2293-53730-0278.fits"
spec = fa.read(FILE).to_rest_frame()
t0 = time.time()
c = fa.classify(spec)
t_cls = time.time() - t0
s = c.spectrum
stellar = c.stellar_fit.model_flux(s.wave)
resid = s.copy(flux=s.flux - stellar)
lines = [ln for ln in DIAG_LINES if s.wave.min() < line_wavelength(ln) < s.wave.max()]
r_n = fa.fit(resid, _line_model(lines, False), wave_range=(4750, 6800), verbose=False)
r_b = fa.fit(resid, _line_model(lines, True), wave_range=(4750, 6800), verbose=False)
masks = [(line_wavelength(ln) * (1 - dv / C_KMS), line_wavelength(ln) * (1 + dv / C_KMS))
         for ln, dv in _MASK_BROAD.items()]
t0 = time.time()
model = c.model(broad_components=2, outflow=True)
r = fa.fit(s, model, verbose=False)
t_fit = time.time() - t0
parts = r.model_flux(components=True)
groups = {"AGN power law": ["continuum"], "host galaxy": ["host"], "Fe II": ["feii"],
          "Balmer continuum + lines": ["balmer_cont", "balmer_lines"]}
narrow = [k for k in parts if k.startswith(("narrow", "outflow", "coronal"))]
broad = [k for k in parts if k.startswith("broad")]
out = {k: np.sum([parts[n] for n in v if n in parts], axis=0) for k, v in groups.items()}
out["narrow lines"] = np.sum([parts[n] for n in narrow], axis=0)
out["broad lines"] = np.sum([parts[n] for n in broad], axis=0)
mb = r.measure("Hb", [c_.name for c_ in r._line_components() if c_.role == "broad" and "Hb" in c_.line_names])
bh = r.bh_mass("Hb")
hp = r.host_properties()
np.savez("examples/video/data.npz", wave=s.wave, flux=s.flux, err=s.err, stellar=stellar,
         line_n=r_n.model_flux(s.wave), line_b=r_b.model_flux(s.wave), model=r.model_flux(),
         used=r.mask, masks=np.array(masks), **{f"part_{k}": v for k, v in out.items()})
info = {"name": "NGC 2885", "file": FILE.split("/")[-1], "z": float(s.z), "label": c.label,
        "reason": c.reason, "uncertain": c.uncertain, "t_classify": t_cls, "t_fit": t_fit,
        "meas": {k: (float(v) if np.isscalar(v) else None) for k, v in c.measurements.items()},
        "rchi2": float(r.rchi2), "hb_fwhm": mb["fwhm"], "hb_fwhm_err": mb["fwhm_err"],
        "log_mbh": bh["log_mbh"], "log_mbh_err": bh["log_mbh_err"],
        "sigma_star": hp["sigma_star"][0], "sigma_star_err": hp["sigma_star"][1],
        "host_5100": hp["host_fraction_5100"][0], "decrement": r.balmer_decrement()[0],
        "fit_range": list(r.model.wave_range)}
json.dump(info, open("examples/video/info.json", "w"), indent=1, default=float)
print(json.dumps(info, indent=1, default=float))
