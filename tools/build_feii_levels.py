"""
Build ``fantasy_agn/data/feii_levels.csv``: the atomic levels of every line of the optical Fe II template
(``feII_model.csv``, which is only read), for the Grotrian diagram of the app's Results › Fe II tab.

Each template line is matched to the NIST transition from the same lower term within 0.6 A (air): nist = 1.
A line with no NIST transition (NIST lists no transition probability for it) is identified from the NIST
levels: a J level of its lower term plus hc/lambda (vacuum) must equal a level of opposite parity within
0.002 eV, with Delta J = 0, +-1 (not 0 -> 0); the closest wins: nist = 2. If none fits, the line keeps the
median energy of its lower term and E_k = E_i + hc/lambda, with no upper term: nist = 0. The NIST line
list is a tab-separated export of the NIST Atomic Spectra Database (Kramida, Ralchenko, Reader & NIST ASD Team), Fe II lines with
configurations, terms, J and energies in eV, for instance from

    https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Fe+II&low_w=3500&upp_w=11000&unit=0&format=3
        &line_out=0&en_unit=1&output=0&show_obs_wl=1&show_calc_wl=1&order_out=0&show_av=2&A_out=0
        &f_out=on&conf_out=on&term_out=on&enrg_out=on&J_out=on&g_out=on&submit=Retrieve+Data

Run from the repository root:

    python tools/build_feii_levels.py path/to/feii_nist_lines.tsv
"""
import os
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "fantasy_agn", "data")
HC = 12398.41984                     # eV A
TOL = 0.6                            # A


def nist(path):
    n = pd.read_csv(path, sep="\t", dtype=str)
    n.columns = [c.strip() for c in n.columns]
    for c in n.columns:
        n[c] = n[c].fillna("").str.strip('"').str.strip()
    wcol = [c for c in n.columns if c.startswith("ritz_wl")][0]
    ocol = [c for c in n.columns if c.startswith("obs_wl")][0]
    n["w"] = pd.to_numeric(n[wcol], errors="coerce").fillna(pd.to_numeric(n[ocol], errors="coerce"))
    for c in ("Ei(eV)", "Ek(eV)"):
        n[c] = pd.to_numeric(n[c].str.strip("[]()? +x"), errors="coerce")
    n["ti"] = n.term_i.str.replace(" ", "").str.replace("*", "")
    return n.dropna(subset=["w", "Ei(eV)", "Ek(eV)"])


def levels(n):
    """Every NIST level of the list: energy, term (no spaces, '*' for odd parity), J and its value."""
    a = n[["Ei(eV)", "term_i", "J_i"]].set_axis(["E", "term", "J"], axis=1)
    b = n[["Ek(eV)", "term_k", "J_k"]].set_axis(["E", "term", "J"], axis=1)
    lv = pd.concat([a, b]).drop_duplicates(["E", "term", "J"]).reset_index(drop=True)
    lv["term"] = lv.term.str.replace(" ", "")
    lv["odd"] = lv.term.str.endswith("*")
    lv["j"] = lv.J.map(lambda s: float(Fraction(s)) if s else np.nan)
    return lv


def from_levels(lv, group, w_air):
    """The (lower, upper) level pair of a line from the level energies, or None."""
    from fantasy_agn.wavelengths import air_to_vac
    hv = HC / float(air_to_vac(w_air))
    best = None
    for _, lo in lv[lv.term.str.rstrip("*") == group].iterrows():
        up = lv[(lv.odd != lo.odd) & ((lv.E - lo.E - hv).abs() < 0.002) & ((lv.j - lo.j).abs() <= 1)
                & ~((lv.j == 0) & (lo.j == 0))]
        for _, u in up.iterrows():
            d = abs(u.E - lo.E - hv)
            if best is None or d < best[0]:
                best = (d, lo, u)
    return best and best[1:]


def main(path):
    n = nist(path)
    lv = levels(n)
    t = pd.read_csv(os.path.join(DATA, "feII_model.csv"))
    rows = []
    for g, w in zip(t.ime, t.wav):
        c = n[(n.ti == g) & ((n.w - w).abs() < TOL)]
        if len(c):
            b = c.loc[(c.w - w).abs().idxmin()]
            rows.append([g, w, b.term_i.replace(" ", ""), b.J_i, b.term_k.replace(" ", ""), b.J_k,
                         round(b["Ei(eV)"], 5), round(b["Ek(eV)"], 5), 1])
            continue
        pair = from_levels(lv, g, w)
        if pair:
            lo, up = pair
            rows.append([g, w, lo.term, lo.J, up.term, up.J, round(lo.E, 5), round(up.E, 5), 2])
            continue
        own = n.loc[n.ti == g, "Ei(eV)"]
        if own.empty:
            raise SystemExit(f"no NIST level of the term {g}")
        ei = float(own.median())
        from fantasy_agn.wavelengths import air_to_vac
        rows.append([g, w, g, "", "", "", round(ei, 5), round(ei + HC / float(air_to_vac(w)), 5), 0])
    out = pd.DataFrame(rows, columns=["ime", "wav", "term_i", "J_i", "term_k", "J_k", "Ei_eV", "Ek_eV", "nist"])
    dest = os.path.join(DATA, "feii_levels.csv")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("# Atomic levels of the lines of the optical Fe II template (feII_model.csv, same order), for the\n"
                 "# Grotrian diagram of the app. nist = 1: the NIST transition from the same lower term within 0.6 A;\n"
                 "# nist = 2: identified from the NIST levels (E_i + hc/lambda = E_k within 0.002 eV, opposite parity,\n"
                 "# Delta J = 0, +-1); nist = 0: unidentified, Ei is the median energy of the lower term, Ek = Ei + hc/lambda.\n"
                 "# Atomic data: NIST Atomic Spectra Database (Kramida, Ralchenko, Reader & NIST ASD Team), Fe II lines.\n"
                 "# Built by tools/build_feii_levels.py.\n")
        out.to_csv(fh, index=False)
    print(f"{dest}: {len(out)} lines, {(out.nist == 1).sum()} NIST transitions, {(out.nist == 2).sum()} from the levels, "
          f"{(out.nist == 0).sum()} unidentified")


if __name__ == "__main__":
    main(sys.argv[1])
