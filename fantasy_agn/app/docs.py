"""
Documentation shown in the app: hand-written guides and an API reference
generated from the docstrings (so it always matches the installed version).
"""
import html
import inspect
import re

GUIDES = [
    ("start", "Getting started", """
<p>FANTASY fits optical and UV spectra of active galaxies: quasars and Seyfert 1s,
type 2 AGN, LINERs and star-forming galaxies. The app guides you through five steps:</p>
<ol>
<li><b>Read</b> a spectrum: drop a file, give a path, or enter a DESI TARGETID. It is
corrected for Galactic extinction and moved to the rest frame (both can be switched off).</li>
<li><b>Classify</b> it: <code>fa.classify(spec)</code> measures the starlight, looks for broad
lines and places the narrow lines on the BPT diagrams. It recommends a model.</li>
<li><b>Model</b>: start from a ready-made model (preset), then add or remove components
and change parameters in the <i>Model</i> tab.</li>
<li><b>Fit</b> the model; the fit uses exact derivatives and a data-driven starting point.</li>
<li><b>Results</b>: the main numbers, the figure (PDF/EPS/PNG), the line table and the whole
session as a Python script.</li>
</ol>
<h3>The code panel</h3>
<p>Every button writes ordinary FANTASY code into a cell and runs it. You can edit any
cell and run it again (<kbd>Shift</kbd>+<kbd>Enter</kbd> or ▶), or add your own cells with
<b>+ cell</b>. All cells share one namespace:</p>
<table class="doc"><tr><td><code>fa</code></td><td>the FANTASY package</td></tr>
<tr><td><code>spec</code></td><td>the spectrum (rest frame)</td></tr>
<tr><td><code>c</code></td><td>the classification</td></tr>
<tr><td><code>model</code></td><td>the model</td></tr>
<tr><td><code>r</code></td><td>the fit result</td></tr></table>
<p>When a cell creates a new <code>spec</code>, <code>c</code>, <code>model</code> or
<code>r</code>, the app updates the steps, the figure and the results, so editing the code
and using the buttons are equivalent. The downloaded script repeats the analysis
without the app.</p>
<h3>Starting the app</h3>
<pre>launch/FANTASY.command        # macOS (double-click)
launch/fantasy-app.sh         # Linux / macOS
launch\\fantasy-app.bat        # Windows
fantasy-app                   # if FANTASY is installed</pre>
<p>The launchers use <a href="https://docs.astral.sh/uv/">uv</a> to run the app with
Python 3.12 and the tested package versions, in an environment of its own.</p>
"""),
    ("read", "Reading spectra", """
<p><code>fa.read(path)</code> recognises SDSS <code>spec-*.fits</code>, DESI coadd files,
SPARCL downloads, 1D FITS spectra and text files with columns wavelength, flux and
(optionally) error. It reads the redshift, the wavelength medium (air/vacuum), the
instrumental resolution and the flux unit when the file has them.</p>
<pre>spec = fa.read("spec-0615-52347-0108.fits").to_rest_frame()
spec = fa.read("my_spectrum.txt", z=0.123).to_rest_frame()
spec = fa.desi.fetch(39627963565609635).to_rest_frame()   # DESI DR1 via Data Lab</pre>
<h3>Galactic extinction</h3>
<p>Correct the Milky Way extinction before measuring fluxes and luminosities. Without it
fluxes and L5100 are too low by 5&ndash;20% for E(B-V) = 0.015&ndash;0.06 (typical at high
Galactic latitude) and the Balmer decrement is 2&ndash;6% too high; ratios of nearby lines
(BPT) change by less than 0.003 dex.</p>
<pre>spec = fa.read("spec-0615-52347-0108.fits").deredden().to_rest_frame()
spec.meta["ebv"], spec.meta["ebv_source"]      # 0.0122, IRSA dust service, SF11
spec = fa.read("my.txt").deredden(ebv=0.03).to_rest_frame(z=0.12)</pre>
<p><code>deredden()</code> uses the Fitzpatrick (1999) law with R<sub>V</sub> = 3.1. Without
<code>ebv</code> it looks up E(B-V) at the position of the spectrum: the value in the file
(DESI), local SFD98 maps (<code>pip install sfdmap2</code> and <code>SFD_DIR</code>), or the
IRSA dust service online (this sends the coordinates to NASA/IPAC). The SFD98 values are
recalibrated by 0.86 (Schlafly &amp; Finkbeiner 2011) unless <code>sf11=False</code>. It
works in the observed or the rest frame and is never applied twice. The fit summary and the
classification note when a spectrum was not corrected.</p>
<p>In the app, <b>Correct Galactic extinction</b> and <b>Move to rest frame</b> in step 1 do
this; leave E(B-V) empty for the automatic value or type your own.</p>
<h3>Useful methods</h3>
<ul>
<li><code>spec.crop(4000, 7000)</code>, <code>spec.mask((5570, 5585))</code>: restrict the
range or exclude bad regions (e.g. sky lines).</li>
<li><code>spec.plot()</code>: the spectrum in the publication style.</li>
</ul>
<p>Fits are done in the rest frame; if the file has no redshift, give it with
<code>to_rest_frame(z=...)</code> or in the <i>z</i> field of the app.</p>
"""),
    ("classify", "Classification", """
<p><code>c = fa.classify(spec)</code> decides what kind of object the spectrum shows, before a
model is chosen. It runs three quick fits:</p>
<ol>
<li><b>Starlight</b>: stellar populations (E-MILES) with dust, plus a weak AGN power law,
with the emission lines masked. This gives the host fraction at 5100 Å and σ★, and
removes the stellar Balmer absorption under Hβ before line ratios are measured.</li>
<li><b>Broad lines</b>: the starlight-subtracted spectrum is fitted with narrow lines (and an
outflow wing) only, and with broad Hβ/Hα added. A broad-line region is detected when
ΔBIC &gt; 25, the broad flux has S/N &gt; 20 and the FWHM is between 1200 km/s and the
limit. Weaker candidates are reported as a possible type 1.8/1.9.</li>
<li><b>Narrow-line diagnostics</b>: the BPT diagram [O III]/Hβ vs [N II]/Hα (Kauffmann et al.
2003; Kewley et al. 2001), the [S II] and [O I] diagrams (Kewley et al. 2006) and the WHAN
diagram (Cid Fernandes et al. 2011).</li>
</ol>
<table class="doc">
<tr><th>class</th><th>meaning</th><th>recommended model</th></tr>
<tr><td>type1</td><td>broad lines, AGN-dominated</td><td>optical</td></tr>
<tr><td>type1_host</td><td>broad lines, host ≥ 50% at 5100 Å</td><td>optical (physical)</td></tr>
<tr><td>seyfert2, liner, composite</td><td>narrow-line AGN</td><td>seyfert2, liner, composite</td></tr>
<tr><td>starforming</td><td>below the Kauffmann line</td><td>starforming</td></tr>
<tr><td>retired, passive</td><td>EW(Hα) &lt; 3 Å / &lt; 0.5 Å</td><td>passive</td></tr>
</table>
<p><code>c.uncertain</code> is set when a ratio is within its error of a boundary or a
diagnostic is missing. <code>c.plot()</code> shows the BPT and WHAN diagrams,
<code>c.plot_steps()</code> the three steps. The diagnostics need Hα, so they work up to
z ≈ 0.4 for SDSS and DESI.</p>
"""),
    ("model", "Building a model", """
<p>A model is a sum of components. Start from a preset and change it:</p>
<pre>model = fa.presets.optical(physical=True, broad_components=2, outflow=True)
model.add(fa.Lines("broad_hei", ["HeI5876"], role="broad", fwhm=(4000, 1000, 15000)))
model.remove("feii")
model</pre>
<h3>Presets</h3>
<table class="doc">
<tr><td><code>hbeta</code></td><td>4150–5550 Å: power law, FeII, narrow and broad Hβ/Hγ, [O III], HeII</td></tr>
<tr><td><code>halpha</code></td><td>6200–6900 Å: Hα, [N II], [S II], [O I], HeI; outflow can follow [O III]</td></tr>
<tr><td><code>optical</code></td><td>3700–6900 Å: everything above, Balmer continuum, high-order Balmer lines;
<code>physical=True</code>: one power law (α<sub>λ</sub> = −1.7), stellar host, broad FWHM ≤ 15000 km/s</td></tr>
<tr><td>options</td><td><code>coronal=True</code>: [Fe VII] 3759, 5159, 5721, 6087, [Fe X] 6375, [Fe XIV] 5303 (also
<code>hbeta</code>, <code>halpha</code>, <code>type2</code>); <code>oi6046=True</code>: broad O I 6046;
<code>red=True</code>: fit to 7500 Å with He I 7065, [Ar III] 7136, [O II] 7320, 7330. In the app these are the
<i>Also include</i> boxes of step 3. Look at the residuals: strong positive residuals at 3759, 5721, 6087 Å mean
coronal lines.</td></tr>
<tr><td><code>mgii</code>, <code>civ</code></td><td>UV lines with UV FeII</td></tr>
<tr><td><code>seyfert2</code>, <code>composite</code></td><td>stellar populations with dust, weak featureless
continuum (scattered light), narrow lines [O II] 3727&ndash;[O II] 7330 in two kinematic systems (high ionisation:
[Ne III], [O III], He II, [Ar III]; low: Balmer, [N II], [S II], [O I]&hellip;), outflow wing incl. [S II]</td></tr>
<tr><td><code>liner</code></td><td>old stellar population with dust, one (possibly broad) narrow-line system;
<code>broad=True</code> for a LINER 1.9</td></tr>
<tr><td><code>starforming</code></td><td>BC03 young + old populations with dust, high-order Balmer series, He I,
auroral lines ([O III] 4363, [N II] 5755, [S III] 6312, [O II] 7320), Case B Balmer ratios + gas E(B-V), wing on
the strong lines; <code>wolf_rayet=True</code> for WR galaxies</td></tr>
<tr><td><code>passive</code> (<code>retired</code>)</td><td>stellar populations with dust; weak lines with
limited widths</td></tr>
<tr><td><code>type2</code>, <code>galaxy</code></td><td>the simpler earlier versions</td></tr>
</table>
<h3>Components</h3>
<table class="doc">
<tr><td><code>fa.Lines(name, lines, role=, fwhm=, shift=, profile=)</code></td><td>a group of emission
lines with shared kinematics; <code>role</code> is narrow, broad or outflow; profiles: gaussian,
lorentzian, voigt, gauss-hermite</td></tr>
<tr><td><code>fa.PowerLaw</code>, <code>fa.BrokenPowerLaw</code>, <code>fa.Polynomial</code></td><td>continua</td></tr>
<tr><td><code>fa.FeII</code></td><td>optical (or UV) iron template</td></tr>
<tr><td><code>fa.BalmerContinuum</code>, <code>fa.HighOrderBalmer</code></td><td>Balmer pseudo-continuum</td></tr>
<tr><td><code>fa.StellarPopulation</code>, <code>fa.HostGalaxy</code></td><td>host galaxy: stellar templates (σ★, age,
metallicity) or PCA eigenspectra. Template sets: <code>"emiles"</code> (E-MILES, Vazdekis et al. 2016; default),
<code>"bc03"</code> (Bruzual &amp; Charlot 2003), <code>"fsps"</code>, <code>"xsl"</code>; in the presets
<code>host="bc03"</code>. A set is downloaded automatically the first time it is used (13&ndash;94 MB, once, from
github.com/micappe/ppxf_data); <code>FANTASY_NO_DOWNLOAD=1</code> switches this off.</td></tr>
<tr><td><code>fa.Extinction</code></td><td>intrinsic reddening of the AGN components</td></tr>
</table>
<p>Limits are given as <code>(start, min, max)</code>, e.g. <code>fwhm=(3000, 1000, 9000)</code> in
km/s. Line names come from the line list (see <i>Line list</i>); add your own as
<code>("name", wavelength_in_vacuum)</code>.</p>
<p>In the <b>Model</b> tab you can do all of this without typing: choose a preset, add
components from a form, remove them, and edit the parameter table. Each change is
written as code into a cell.</p>
"""),
    ("classes", "Models for each class", """
<p>The presets for narrow-line galaxies were chosen by fitting SDSS examples of every class with
models that differ in one ingredient at a time, and keeping what the data support (BIC) and what
gives physical results (σ★ against the SDSS value, Balmer decrement, line ratios against MPA-JHU).</p>
<table class="doc">
<tr><th>ingredient</th><th>result</th><th>used in</th></tr>
<tr><td>dust screen on the starlight</td><td>χ²/ν 4.6 → 1.5 (composite), 4.2 → 2.6 (Seyfert 2); σ★ close
to SDSS, ages no longer at the 15 Gyr limit</td><td>all</td></tr>
<tr><td>young populations (BC03 from 1 Myr)</td><td>ΔBIC ≈ −8000 for star-forming galaxies; with E-MILES
(≥ 63 Myr) σ★ runs to its limit</td><td>starforming</td></tr>
<tr><td>two narrow-line systems</td><td>ΔBIC −74, −374, −97 (Seyfert 2); −8 composite; +10 LINER</td>
<td>seyfert2, composite</td></tr>
<tr><td>outflow wing incl. [S II]</td><td>ΔBIC −90 to −360 (Seyfert 2, composite), −400 to −1800
(star-forming); +16 LINER</td><td>all but liner, passive</td></tr>
<tr><td>Case B Balmer ratios + gas E(B-V)</td><td>physical E(B-V) for H II regions; for AGN it changed
[O III]/Hβ by up to 0.1 dex and [N II]/Hα by 0.15 dex (LINER, weak Hβ)</td><td>starforming</td></tr>
<tr><td>weak broad Hα in a LINER</td><td>not detected (S/N 3.5, width at the limit)</td><td>liner(broad=True)</td></tr>
</table>
<p>For AGN the line fluxes are free; the Balmer decrement then measures the reddening
(observed Hα/Hβ against 3.1). Passive galaxies get weak lines with limited widths, so that they cannot
absorb template mismatch. The sample was small (8 galaxies), so treat these as good defaults, not rules:
look at the residuals and compare alternatives with <code>fa.compare</code>.</p>
"""),
    ("multi", "Multi-epoch fitting", """
<p>Several spectra of one object (reverberation mapping, monitoring, changing-look AGN) are fitted
<b>jointly</b>: parameters matching the <i>shared</i> patterns (by default all narrow-line parameters,
<code>narrow.*</code>) are the same in every epoch, the rest (continuum, Fe II, broad lines) vary. With
<i>intercalibrate</i>, each epoch gets a flux scale that the shared narrow lines fix, which removes
differences in flux calibration, seeing and aperture losses.</p>
<pre>import glob
files = sorted(glob.glob("/data/sdss_rm/*.fits"))
spectra = [fa.read(f).deredden().to_rest_frame() for f in files]
me = fa.fit_epochs(spectra, fa.presets.hbeta(broad_components=2), n_jobs=-1)
me.light_curve()           # broad fluxes, widths, continuum per epoch
me.plot_light_curves(); me.plot_baldwin()</pre>
<p>First every epoch is fitted on its own (these fits run in parallel, <code>n_jobs</code> processes),
which gives the starting values; then all epochs are fitted together with a sparse Jacobian. The
reference epoch (flux scale 1) is the one with the highest S/N. For many unrelated spectra use
<code>fa.fit_many(spectra, model, n_jobs=-1)</code>, which returns one fit per spectrum.</p>
<p>In scripts on macOS and Windows, put the code under <code>if __name__ == "__main__":</code>, otherwise
the parallel part falls back to serial. In the app, the <b>Multi-epoch</b> view does all of this:
drop the files (or give a pattern), choose the model and press <i>Fit all epochs</i>.</p>
"""),
    ("params", "Parameters, limits and ties", """
<p>Every parameter has a value, limits and a status (free, fixed or tied). Names are
<code>component.parameter</code>, e.g. <code>narrow.fwhm</code> or
<code>narrow.[OIII]5007.flux</code>; <code>model.params</code> lists them.</p>
<pre>model.set("narrow.fwhm", value=400, min=100, max=900)   # start value and limits
model.fix("continuum.index", -1.7)                       # fixed
model.free("continuum.index")                            # free again
model.tie("broad2.shift", "broad.shift")                 # same velocity shift
model.tie("narrow.[NII]6548.flux", "narrow.[NII]6583.flux", factor=1/3.05)</pre>
<p>A tie sets <code>value = factor × value(target) + offset</code>. Fluxes can only be tied
to fluxes (then without offset), widths and shifts to widths and shifts. Patterns with
<code>*</code> change several parameters at once (<code>model.set("broad*.shift", min=-2000)</code>);
square brackets in line names are literal.</p>
<p>Fixed line ratios are part of a component: <code>ratios={"[OIII]4959": ("[OIII]5007", 1/2.98)}</code>;
upper limits on ratios: <code>max_ratios={"[NII]6583": ("Ha", 3)}</code>.</p>
<p>After a fit, <code>r.params</code> shows the fitted values with errors and their status;
parameters ending at a limit are listed in the summary, since their errors are then
not reliable.</p>
"""),
    ("fit", "Fitting and uncertainties", """
<pre>r = fa.fit(spec, model)                    # errors from the covariance matrix
r = fa.fit(spec, model, errors="mc")       # Monte Carlo: refit with added noise
r = fa.fit(spec, model, mcmc=True)         # Hamiltonian Monte Carlo posterior</pre>
<p>Amplitudes enter linearly and are solved exactly; widths, shifts and shapes are found
on a grid and refined with a bounded least-squares fit using analytic derivatives.
Components that overlap (outflow wings, extra broad components) are added one at a
time, so adding a component can only improve the fit.</p>
<p><b>Errors.</b> The covariance errors are scaled by √χ²<sub>ν</sub> when χ²<sub>ν</sub> &gt; 1.
Monte Carlo errors are more reliable for weak or blended lines. Errors of derived
quantities (FWHM of a multi-component line, black-hole mass) are propagated from all
parameters including their correlations.</p>
<p><b>Model choice</b> often matters more than noise: <code>fa.compare(spec, fa.presets.variants("hbeta"))</code>
fits several decompositions and reports a systematic error from the spread of the
acceptable models.</p>
"""),
    ("results", "Results", """
<pre>r.key_results()            # the main numbers
r.lines                    # every line: flux, FWHM, shift, EW with errors
r.measure("Hb")            # a line summed over its components
r.bh_mass("Hb")            # single-epoch black-hole mass
r.balmer_decrement()       # broad Hα/Hβ (typical ~3)
r.host_properties()        # σ★, v★, age, metallicity, host fraction
r.plot(); r.savefig("fit.pdf")
r.save("my_fit")           # tables as CSV</pre>
<p>The FWHM of a line with several broad components is measured on their sum, after
removing the instrumental broadening. The summary warns about parameters at their
limits and about an unusual Balmer decrement, which usually means that the continuum and
the broad-line wings are trading flux.</p>
"""),
]


# --------------------------------------------------------------- API reference
def _api_objects():
    import fantasy_agn as fa
    from fantasy_agn import presets
    from fantasy_agn.classify import Classification
    from fantasy_agn.result import FitResult
    from fantasy_agn.spectrum import Spectrum
    groups = [
        ("Reading data", [("fa.read", fa.read), ("Spectrum", Spectrum),
                          ("Spectrum.to_rest_frame", Spectrum.to_rest_frame),
                          ("Spectrum.deredden", Spectrum.deredden), ("Spectrum.crop", Spectrum.crop),
                          ("Spectrum.mask", Spectrum.mask), ("fa.desi.fetch", fa.desi.fetch),
                          ("fa.desi.find", fa.desi.find)]),
        ("Classification", [("fa.classify", fa.classify), ("Classification", Classification),
                            ("Classification.plot_steps", Classification.plot_steps)]),
        ("Models", [("fa.Model", fa.Model)]
         + [(f"fa.presets.{n}", getattr(presets, n)) for n in
            ("hbeta", "halpha", "optical", "mgii", "civ", "type2", "galaxy", "variants")]
         + [(f"fa.{n}", getattr(fa, n)) for n in
            ("Lines", "PowerLaw", "BrokenPowerLaw", "Polynomial", "FeII", "BalmerContinuum",
             "HighOrderBalmer", "StellarPopulation", "HostGalaxy", "Extinction")]),
        ("Fitting", [("fa.fit", fa.fit), ("fa.guess", fa.guess), ("fa.compare", fa.compare),
                     ("fa.fit_epochs", fa.fit_epochs)]),
        ("Results", [(f"FitResult.{n}", getattr(FitResult, n)) for n in
                     ("key_results", "lines", "measure", "bh_mass", "balmer_decrement",
                      "host_properties", "luminosity", "continuum", "plot", "savefig", "save",
                      "summary")]),
    ]
    return groups


def _doc_html(text):
    """Docstring (reST-like) -> simple HTML."""
    text = inspect.cleandoc(text or "")
    if not text:
        return "<p><i>no description</i></p>"
    out, para, in_code, params = [], [], False, []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if in_code:
            if ln.strip() and not ln.startswith(" "):
                in_code = False
                out.append("</pre>")
                continue
            out.append(html.escape(ln[4:] if ln.startswith("    ") else ln))
            i += 1
            continue
        m = re.match(r"\s*:(param|return|returns)\s*([^:]*):\s*(.*)", ln)
        if m:
            flush()
            body = [m.group(3)]
            while i + 1 < len(lines) and lines[i + 1].startswith(" ") and not re.match(r"\s*:", lines[i + 1]):
                i += 1
                body.append(lines[i].strip())
            label = m.group(2).strip() or "returns"
            params.append((label, " ".join(body)))
            i += 1
            continue
        if params and not ln.strip():
            i += 1
            continue
        if ln.rstrip().endswith("::"):
            para.append(ln.rstrip()[:-1])
            flush()
            out.append('<pre>')
            in_code = True
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue
        if re.match(r"\s*[-*] ", ln):
            flush()
            out.append("<li>" + _inline(ln.strip()[2:]) + "</li>")
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    if in_code:
        out.append("</pre>")
    flush()
    if params:
        out.append('<table class="doc params">' + "".join(
            f"<tr><td><code>{html.escape(a)}</code></td><td>{_inline(b)}</td></tr>" for a, b in params)
            + "</table>")
    body = "\n".join(out)
    return re.sub(r"(<li>.*?</li>\n?)+", lambda m_: "<ul>" + m_.group(0) + "</ul>", body, flags=re.S)


def _inline(text):
    t = html.escape(text)
    t = re.sub(r"``(.+?)``", r"<code>\1</code>", t)
    t = re.sub(r":(?:func|class|meth|mod):`~?([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"`([^`]+)`", r"<i>\1</i>", t)
    return t


def _signature(name, obj):
    target = obj.fget if isinstance(obj, property) else obj
    if isinstance(obj, property):
        return f"{name}"
    try:
        sig = str(inspect.signature(target))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("(self, ", "(").replace("(self)", "()")
    return f"{name}{sig}"


def api_pages():
    pages = []
    for group, objs in _api_objects():
        items = []
        for name, obj in objs:
            doc = (obj.fget.__doc__ if isinstance(obj, property) else obj.__doc__) or ""
            if inspect.isclass(obj) and obj.__init__.__doc__ and len(doc) < 40:
                doc += "\n\n" + obj.__init__.__doc__
            sig = _signature(name, obj.__init__ if inspect.isclass(obj) and name != "Classification" else obj)
            if inspect.isclass(obj):
                sig = name + sig[sig.index("("):] if "(" in sig else name
            anchor = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
            items.append(f'<div class="api" id="{anchor}"><h3><code>{html.escape(sig)}</code></h3>'
                         f"{_doc_html(doc)}</div>")
        pid = "api-" + re.sub(r"[^a-z]+", "-", group.lower()).strip("-")
        pages.append({"id": pid, "title": group, "section": "API reference", "html": "\n".join(items)})
    return pages


def line_list_page():
    from ..lines import LINES
    rows = "".join(f"<tr><td><code>{html.escape(n)}</code></td><td>{w:.2f}</td></tr>"
                   for n, w in sorted(LINES.items(), key=lambda kv: kv[1]))
    return {"id": "lines", "title": "Line list", "section": "Reference",
            "html": "<p>Names to use in <code>fa.Lines(name, [lines])</code>, with vacuum wavelengths "
                    "(Å). Lines above 2000 Å are converted from air.</p>"
                    f'<table class="doc"><tr><th>name</th><th>λ (vacuum, Å)</th></tr>{rows}</table>'}


def all_pages():
    pages = [{"id": i, "title": t, "section": "Guide", "html": h.strip()} for i, t, h in GUIDES]
    pages.append(line_list_page())
    pages += api_pages()
    return pages
