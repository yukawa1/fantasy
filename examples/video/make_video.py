"""
Render the FANTASY workflow video (read -> classify -> fit -> results) from the
results saved by precompute.py:

    python examples/video/precompute.py
    python examples/video/make_video.py        # -> examples/video/fantasy_workflow.mp4
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from fantasy_agn.classify import kauffmann03, kewley01
from fantasy_agn.plotting import COLORS

D = np.load("examples/video/data.npz")
I = json.load(open("examples/video/info.json"))
M = I["meas"]
wave, flux = D["wave"], D["flux"]
FPS, DURATION = 25, 42.0
OUT = sys.argv[1] if len(sys.argv) > 1 else "examples/video/fantasy_workflow.mp4"

BG, CODE_BG, ACCENT = "white", "#1e1e24", "#D55E00"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in"})

# Jupyter-like cells: lines typed between (t0, t1); the cell runs between
# run=(start, end); outputs appear at their times.
CELLS = [
    dict(lines=[(3.0, 3.8, "import fantasy_agn as fa"),
                (4.0, 5.6, 'spec = fa.read("spec-2293-53730-0278.fits")'),
                (5.7, 6.3, "spec = spec.to_rest_frame()"),
                (6.4, 6.8, "spec")],
         run=(6.9, 7.6),
         out=[(7.6, f"<Spectrum: {wave.size} px, rest frame, z={I['z']:.4f}>")]),
    dict(lines=[(9.5, 10.3, "c = fa.classify(spec)"), (10.3, 10.6, "c")],
         run=(10.7, 21.2),
         out=[(13.2, f"[1] host fraction at 5100 Å = {M['host_fraction_5100']:.2f}, "
                     f"σ* = {M['sigma_star']:.0f} km/s"),
              (17.4, f"[2] broad Hα: ΔBIC = {M['broad_dbic']:.0f}, "
                     f"FWHM = {M['broad_fwhm']:.0f}, S/N = {M['broad_snr']:.0f}"),
              (20.0, f"[3] log [NII]/Hα = {M['log_nii_ha']:+.2f}, "
                     f"log [OIII]/Hβ = {M['log_oiii_hb']:+.2f}"),
              (21.2, f"→ {I['label']}   ({I['t_classify']:.0f} s)"),
              (21.2, "recommended: fa.presets.optical(physical=True)")]),
    dict(lines=[(22.6, 24.0, "model = c.model(broad_components=2,"),
                (24.0, 24.7, "                outflow=True)"),
                (24.9, 25.8, "r = fa.fit(spec, model)")],
         run=(26.0, 33.4),
         out=[(33.4, f"χ²/ν = {I['rchi2']:.3f}   ({I['t_fit']:.0f} s)")]),
    dict(lines=[(34.6, 35.3, 'r.measure("Hb")["fwhm"]')], run=(35.4, 35.6),
         out=[(35.6, f"{I['hb_fwhm']:.1f}")]),
    dict(lines=[(36.0, 36.7, 'r.bh_mass("Hb")["log_mbh"]')], run=(36.8, 37.0),
         out=[(37.0, f"{I['log_mbh']:.3f}")]),
    dict(lines=[(37.4, 38.3, 'r.host_properties()["sigma_star"]')], run=(38.4, 38.6),
         out=[(38.6, f"({I['sigma_star']:.1f}, {I['sigma_star_err']:.1f})")]),
]


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def frac(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))


lo_fit, hi_fit = max(I["fit_range"][0], wave.min()), I["fit_range"][1]


def draw_code(ax, t):
    ax.set_facecolor(CODE_BG)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.04, 0.975, "Jupyter  ·  FANTASY 2.0", color="#8a8a99", fontsize=10,
            family="monospace", va="top")
    rows = []                     # (kind, text, extra)
    for n, cell in enumerate(CELLS, 1):
        if t < cell["lines"][0][0]:
            break
        r0, r1 = cell["run"]
        state = "idle" if t < r0 else ("run" if t < r1 else "done")
        label = {"idle": "In [ ]:", "run": "In [*]:", "done": f"In [{n}]:"}[state]
        for i, (t0, t1, text) in enumerate(cell["lines"]):
            if t < t0:
                break
            k = int(round(len(text) * frac(t, t0, t1)))
            cursor = "█" if t0 <= t < t1 and int(t * 3) % 2 else ""
            rows.append(("code", text[:k] + cursor, label if i == 0 else ""))
        if state == "run":
            el = t - r0
            if n == 3:                       # the fit: show the real elapsed time
                el = (t - r0) / (r1 - r0) * I["t_fit"]
            spin = "|/-\\"[int(t * 8) % 4]
            rows.append(("run", f"{spin} running … {el:4.0f} s", ""))
        for to, text in cell["out"]:
            if t >= to:
                rows.append(("out", text, ""))
        rows.append(("gap", "", ""))
    y, dy = 0.92, 0.042
    rows = rows[-21:]
    for kind, text, label in rows:
        if kind == "gap":
            y -= dy * 0.5
            continue
        if kind == "code":
            if label:
                ax.text(0.04, y, label, color="#6a9955" if "*" not in label else ACCENT,
                        fontsize=9.5, family="monospace", va="top")
            ax.text(0.20, y, text, color="#d4d4d4", fontsize=10.5, family="monospace", va="top")
        elif kind == "run":
            ax.text(0.20, y, text, color=ACCENT, fontsize=10, family="monospace", va="top")
        else:
            ax.text(0.20, y, text, color="#9cdcfe", fontsize=9, family="monospace", va="top")
        y -= dy


def caption(fig, t, text, t0, step=None):
    a = ease(frac(t, t0, t0 + 0.6))
    if a <= 0:
        return
    s = (f"{step}  " if step else "") + text
    fig.text(0.73, 0.055, s, ha="center", va="center", fontsize=11.5, alpha=a, color="#222222")


def spectrum_axes(fig):
    ax = fig.add_axes([0.49, 0.17, 0.485, 0.70])
    ax.set_xlabel(r"Rest-frame wavelength ($\mathrm{\AA}$)", fontsize=12)
    ax.set_ylabel(r"$F_\lambda$ ($10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$)", fontsize=12)
    ax.tick_params(labelsize=10, top=True, right=True)
    return ax


def scene_title(fig, t):
    a = 1 - ease(frac(t, 2.3, 3.0))
    fig.text(0.73, 0.60, "FANTASY 2.0", ha="center", fontsize=40, weight="bold", alpha=a,
             color="#222222")
    fig.text(0.73, 0.50, "from a spectrum to a physical decomposition", ha="center",
             fontsize=17, alpha=a, color="#444444")
    fig.text(0.73, 0.42, "read  →  classify  →  fit", ha="center", fontsize=15,
             alpha=a, color=ACCENT)


def scene_read(fig, t):
    ax = spectrum_axes(fig)
    p = ease(frac(t, 6.0, 8.5))
    k = max(2, int(p * wave.size))
    ax.plot(wave[:k], flux[:k], color="0.25", lw=0.6)
    ax.set_xlim(wave.min(), wave.max())
    ax.set_ylim(0, np.nanpercentile(flux, 99.8) * 1.1)
    caption(fig, t, f"{I['name']}  ·  SDSS {I['file'][5:-5]}  ·  z = {I['z']:.4f}", 6.0)


def scene_classify(fig, t):
    ax = spectrum_axes(fig)
    v = (wave > 3750) & (wave < 6900)
    ax.plot(wave[v], flux[v], color="0.55", lw=0.6)
    ax.set_xlim(3750, 6900)
    ax.set_ylim(0, np.nanpercentile(flux[v], 99.8) * 1.1)
    # step 1: masks and stellar continuum
    a = ease(frac(t, 10.6, 11.4))
    for lo, hi in D["masks"]:
        ax.axvspan(lo, hi, color="#56B4E9", alpha=0.15 * a, lw=0)
    p = ease(frac(t, 11.0, 13.0))
    k = int(p * v.sum())
    ax.plot(wave[v][:k], D["stellar"][v][:k], color=COLORS.get("host", "#999933"), lw=1.2)
    if t < 14.0:
        caption(fig, t, f"starlight: stellar populations + dust, lines masked  →  "
                f"host {M['host_fraction_5100']:.0%} at 5100 Å, σ★ = "
                f"{M['sigma_star']:.0f} km/s", 11.0, "1 ·")
    # step 2: broad lines in the starlight-subtracted spectrum (inset)
    if t >= 14.0:
        a2 = ease(frac(t, 14.0, 14.6))
        ins = fig.add_axes([0.515, 0.50, 0.235, 0.33])
        ins.patch.set_alpha(0.95 * a2)
        w = (wave > 6420) & (wave < 6720)
        res = flux - D["stellar"]
        ins.plot(wave[w], res[w], color="0.3", lw=0.7, alpha=a2)
        if t >= 15.0:
            ins.plot(wave[w], D["line_n"][w], color="#0072B2", lw=1.1, alpha=ease(frac(t, 15.0, 15.5)),
                     label="narrow only")
        if t >= 16.5:
            ins.plot(wave[w], D["line_b"][w], color=ACCENT, lw=1.3, alpha=ease(frac(t, 16.5, 17.0)),
                     label="+ broad Hα")
        ins.set_title("starlight removed: Hα + [N II]", fontsize=10)
        ins.tick_params(labelsize=8)
        if t >= 15.0:
            ins.legend(fontsize=8, loc="upper left", frameon=False)
        if t < 18.5:
            caption(fig, t, f"broad lines?  ΔBIC = {M['broad_dbic']:.0f},  S/N = "
                    f"{M['broad_snr']:.0f},  FWHM = {M['broad_fwhm']:.0f} km/s  →  yes",
                    15.0, "2 ·")
    # step 3: BPT
    if t >= 18.5:
        a3 = ease(frac(t, 18.5, 19.1))
        bpt = fig.add_axes([0.79, 0.50, 0.17, 0.33])
        bpt.patch.set_alpha(0.95 * a3)
        x = np.linspace(-1.6, 0.46, 200)
        bpt.plot(x, kewley01(x), "k", lw=0.8, alpha=a3)
        xk = np.linspace(-1.6, 0.04, 200)
        bpt.plot(xk, kauffmann03(xk), "k--", lw=0.8, alpha=a3)
        for xx, yy, s in ((-1.2, -0.6, "SF"), (-0.25, -0.75, "comp."), (-0.4, 1.2, "Seyfert"),
                          (0.15, -0.6, "LINER")):
            bpt.text(xx, yy, s, fontsize=8, alpha=a3, color="0.4")
        if t >= 19.3:
            y0 = M["log_oiii_hb"] + 1.5 * (1 - ease(frac(t, 19.3, 19.9)))
            bpt.errorbar(M["log_nii_ha"], y0, xerr=M["log_nii_ha_err"], yerr=M["log_oiii_hb_err"],
                         fmt="*", ms=11, color=ACCENT)
        bpt.set_xlim(-1.6, 0.6); bpt.set_ylim(-1.0, 1.5)
        bpt.set_xlabel("log [N II]/Hα", fontsize=9); bpt.set_ylabel("log [O III]/Hβ", fontsize=9)
        bpt.tick_params(labelsize=8)
        bpt.set_title("narrow lines (BPT)", fontsize=10)
        if t >= 20.4:
            a4 = ease(frac(t, 20.4, 21.0))
            ax.text(0.97, 0.06, "type 1 AGN, host-dominated\nrecommended: optical(physical=True)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=13, alpha=a4,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ACCENT, alpha=a4))
        caption(fig, t, "narrow-line ratios: Seyfert-like (BPT, [S II], [O I], WHAN)", 18.5, "3 ·")


ORDER = [("host galaxy", "#999933", "-"), ("AGN power law", "k", "--"),
         ("Balmer continuum + lines", "#CC79A7", "-."), ("Fe II", "#E69F00", "-"),
         ("broad lines", "#0072B2", "-"), ("narrow lines", "#009E73", "-")]


def scene_fit(fig, t, legend=True, captions=True):
    ax = fig.add_axes([0.49, 0.33, 0.485, 0.54])
    rax = fig.add_axes([0.49, 0.17, 0.485, 0.14], sharex=ax)
    v = (wave >= lo_fit) & (wave <= hi_fit)
    ax.plot(wave[v], flux[v], color="0.6", lw=0.6, label="data")
    t0 = 26.3
    for i, (name, col, ls) in enumerate(ORDER):
        a = ease(frac(t, t0 + 0.9 * i, t0 + 0.9 * i + 0.6))
        if a > 0:
            ax.plot(wave[v], D[f"part_{name}"][v], color=col, ls=ls, lw=1.0, alpha=a, label=name)
    tm = t0 + 0.9 * len(ORDER) + 0.2
    am = ease(frac(t, tm, tm + 0.6))
    if am > 0:
        ax.plot(wave[v], D["model"][v], color=ACCENT, lw=1.1, alpha=am, label="model")
        res = (flux - D["model"]) / D["err"]
        rax.plot(wave[v], res[v], color="0.3", lw=0.5, alpha=am)
        rax.axhline(0, color=ACCENT, lw=0.8, alpha=am)
    ax.set_xlim(lo_fit, hi_fit)
    ax.set_ylim(0, np.nanmax(D["model"][v]) * 1.08)
    rax.set_ylim(-6, 6)
    if legend:
        ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)
    ax.set_ylabel(r"$F_\lambda$", fontsize=12)
    rax.set_ylabel("resid. (σ)", fontsize=10)
    rax.set_xlabel(r"Rest-frame wavelength ($\mathrm{\AA}$)", fontsize=12)
    plt.setp(ax.get_xticklabels(), visible=False)
    for a_ in (ax, rax):
        a_.tick_params(labelsize=10, top=True, right=True)
    if not captions:
        pass
    elif t < tm:
        caption(fig, t, f"simultaneous fit, exact derivatives, {len(D['wave'][v])} pixels  "
                f"({I['t_fit']:.0f} s, sped up)", 26.0)
    else:
        caption(fig, t, f"χ²/ν = {I['rchi2']:.2f}", tm)


def scene_results(fig, t):
    scene_fit(fig, 40.0, legend=False, captions=False)   # the final fit in the background
    a = ease(frac(t, 35.0, 35.6))
    lines = [f"broad Hβ FWHM   {I['hb_fwhm']:.0f} ± {I['hb_fwhm_err']:.0f} km/s",
             f"log M_BH (Hβ)    {I['log_mbh']:.2f} ± {I['log_mbh_err']:.2f}",
             f"σ★ (host)          {I['sigma_star']:.0f} ± {I['sigma_star_err']:.0f} km/s",
             f"host at 5100 Å     {I['host_5100']:.0%}",
             f"broad Hα/Hβ       {I['decrement']:.2f}"]
    shown = [ln for i, ln in enumerate(lines) if t >= 35.0 + 0.6 * i]
    if shown:
        fig.text(0.51, 0.83, "\n".join(shown), va="top", fontsize=12, family="monospace",
                 alpha=a, bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=ACCENT, alpha=0.95 * a))
    if t >= 39.5:
        a2 = ease(frac(t, 39.5, 40.3))
        fig.text(0.73, 0.055, "pip install fantasy_agn   ·   github.com/yukawa1/fantasy",
                 ha="center", fontsize=14, alpha=a2, color=ACCENT)


def frame(fig, t):
    fig.clf()
    fig.patch.set_facecolor(BG)
    draw_code(fig.add_axes([0.0, 0.0, 0.43, 1.0]), t)
    if t < 3.0:
        scene_title(fig, t)
    elif t < 9.5:
        scene_read(fig, t)
    elif t < 22.0:
        scene_classify(fig, t)
    elif t < 34.5:
        if t >= 26.0:
            scene_fit(fig, t)
        else:
            scene_classify(fig, 22.0)
    else:
        scene_results(fig, t)
    labels = [(0.0, "read"), (9.5, "classify"), (22.0, "fit"), (34.5, "results")]
    cur = max(i for i, (s, _) in enumerate(labels) if t >= s)
    for i, (s, name) in enumerate(labels):
        fig.text(0.53 + 0.11 * i, 0.94, name, ha="center", fontsize=13,
                 color=ACCENT if i == cur and t >= 3 else "#bbbbbb",
                 weight="bold" if i == cur and t >= 3 else "normal")


fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264",
                      extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
n = int(DURATION * FPS)
with writer.saving(fig, OUT, dpi=100):
    for i in range(n):
        frame(fig, i / FPS)
        writer.grab_frame(facecolor=BG)
        if i % 100 == 0:
            print(f"frame {i}/{n}", flush=True)
print("wrote", OUT)
