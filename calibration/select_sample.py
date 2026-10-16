"""
Select the calibration sample from SDSS DR17 (SkyServer SQL): random objects
per group, median S/N >= 8, writes sample.csv (one row per spectrum).

    python calibration/select_sample.py [--scale 1.0] [--out ~/.fantasy_agn/samples]

Groups (and the presets they calibrate):
  t1_low   type 1, z < 0.35: H-alpha and H-beta   (optical, halpha, hbeta)
  t1_hb    type 1, 0.35 < z < 0.75: H-beta        (hbeta)
  t1_mgii  type 1, 0.5 < z < 1.9: MgII            (mgii)
  t1_civ   type 1, 1.7 < z < 2.5: CIV             (civ)
  sy2      MPA-JHU AGN above Schawinski+07 line   (seyfert2)
  liner    MPA-JHU AGN below it                   (liner)
  comp     MPA-JHU composite                      (composite)
  sf       MPA-JHU star-forming                   (starforming)
  passive  EW(H-alpha) < 0.5 A, no BPT class      (passive)
"""
import argparse
import io
import os
import urllib.parse
import urllib.request

import pandas as pd

URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
SPEC = "s.plate, s.mjd, s.fiberID, s.run2d, s.z, s.snMedian, s.class, s.subclass, s.survey"
TYPE1 = (f"SELECT TOP {{n}} {SPEC} FROM SpecObj s WHERE s.zWarning = 0 AND s.snMedian >= 8 "
         "AND s.class = 'QSO' AND s.z BETWEEN {lo} AND {hi} "
         "ORDER BY NEWID()")
MPA = (f"SELECT TOP {{n}} {SPEC}, e.bptclass, l.h_alpha_eqw, l.h_alpha_flux, l.h_beta_flux, "
       "l.oiii_5007_flux, l.nii_6584_flux FROM SpecObj s "
       "JOIN galSpecExtra e ON e.specObjID = s.specObjID JOIN galSpecLine l ON l.specObjID = s.specObjID "
       "WHERE s.zWarning = 0 AND s.snMedian >= 8 AND s.class = 'GALAXY' AND s.z BETWEEN 0.01 AND 0.3 "
       "AND {cond} ORDER BY NEWID()")
# Schawinski et al. 2007: Seyfert above log([OIII]/Hb) = 1.05 log([NII]/Ha) + 0.45
POS = ("l.h_beta_flux > 3 * l.h_beta_flux_err AND l.h_alpha_flux > 3 * l.h_alpha_flux_err "
       "AND l.oiii_5007_flux > 0 AND l.nii_6584_flux > 0")
SEY = "LOG10(l.oiii_5007_flux / l.h_beta_flux) > 1.05 * LOG10(l.nii_6584_flux / l.h_alpha_flux) + 0.45"
GROUPS = {
    "t1_low": (TYPE1, dict(lo=0.02, hi=0.35), 300),
    "t1_hb": (TYPE1, dict(lo=0.35, hi=0.75), 200),
    "t1_mgii": (TYPE1, dict(lo=0.5, hi=1.9), 200),
    "t1_civ": (TYPE1, dict(lo=1.7, hi=2.5), 150),
    "sy2": (MPA, dict(cond=f"e.bptclass = 4 AND {POS} AND {SEY}"), 150),
    "liner": (MPA, dict(cond=f"e.bptclass = 4 AND {POS} AND NOT ({SEY})"), 150),
    "comp": (MPA, dict(cond="e.bptclass = 3"), 100),
    "sf": (MPA, dict(cond="e.bptclass = 1"), 150),
    "passive": (MPA, dict(cond="e.bptclass = -1 AND l.h_alpha_eqw > -0.5 AND s.snMedian >= 10"), 100),
}


def query(sql):
    url = URL + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
    with urllib.request.urlopen(url, timeout=600) as r:
        text = r.read().decode()
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if not lines or "error" in lines[0].lower():
        raise RuntimeError(text[:500])
    return pd.read_csv(io.StringIO("\n".join(lines)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the group sizes")
    ap.add_argument("--out", default="~/.fantasy_agn/samples")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    parts = []
    for group, (sql, kw, n) in GROUPS.items():
        df = query(sql.format(n=max(1, int(n * a.scale)), **kw))
        df.insert(0, "group", group)
        print(f"{group:8s} {len(df):4d} objects, z {df.z.min():.3f}-{df.z.max():.3f}, "
              f"median S/N {df.snMedian.median():.1f}", flush=True)
        parts.append(df)
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.drop_duplicates(["plate", "mjd", "fiberID"])
    path = os.path.join(out, "sample.csv")
    sample.to_csv(path, index=False)
    print(f"{len(sample)} spectra -> {path}")


if __name__ == "__main__":
    main()
