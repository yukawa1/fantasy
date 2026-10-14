"""
Joint multi-epoch fit of a real SDSS reverberation-mapping quasar.

SDSS J141419.84+533815.3 (SDSS objID 1237661417139011660, z = 0.1645) was
observed 77 times: once by SDSS in 2003 and 76 times by the SDSS-RM programme
(BOSS, 2014-2020). This script downloads the spectra from the SDSS DR17 archive
(about 17 MB) into examples/sdss_rm/ and fits the H-beta region of all epochs
together, with the narrow lines shared and used to intercalibrate the flux.

    python examples/multi_epoch_sdss_rm.py
"""
import os
import urllib.request

import pandas as pd

import fantasy_agn as fa

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sdss_rm")
Z = 0.16454  # median of the SDSS pipeline redshifts; one value for all epochs


def download():
    table = pd.read_csv(os.path.join(HERE, "sdss_rm_spectra.csv"))
    os.makedirs(OUT, exist_ok=True)
    paths = []
    for row in table.itertuples():
        name = f"spec-{row.plate:04d}-{row.mjd}-{row.fiberID:04d}.fits"
        survey = "sdss" if str(row.run2d) == "26" else "eboss"
        url = (f"https://data.sdss.org/sas/dr17/{survey}/spectro/redux/{row.run2d}/"
               f"spectra/lite/{row.plate:04d}/{name}")
        path = os.path.join(OUT, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    spectra = [fa.read(p).to_rest_frame(z=Z) for p in download()]
    result = fa.fit_epochs(spectra, fa.presets.hbeta())   # empty epochs are skipped
    light_curve = result.light_curve()
    light_curve.to_csv("sdss_rm_light_curve.csv", index=False)
    print(result.scales.head())
    print(light_curve[["mjd", "Hb_flux", "Hb_flux_err", "cont5100", "cont5100_err"]].head())
    result.plot_light_curves(columns=["Hb_flux", "Hb_fwhm", "cont5100"],
                             save="sdss_rm_light_curves.pdf")
    result.epochs[result.reference].savefig("sdss_rm_reference_epoch.pdf", label_lines=True)
