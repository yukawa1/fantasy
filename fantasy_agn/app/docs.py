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
type 2 AGN, LINERs, star-forming galaxies and nuclear transients. The app guides you
through five steps, shown at the top of the page:</p>
<ol>
<li><b>Read</b>: choose what you analyse (one spectrum, a sample, several epochs of one
object, or classification only) and where the spectra come from: files (drop them), a path or
pattern such as <code>~/data/AT2024zkr/*.txt</code>, DESI TARGETIDs or SDSS plate-mjd-fiber.
Epochs are ordered by the date in the file names (YYYYMMDD), the MJD in the headers or the
file order.</li>
<li><b>Prepare</b>: redshift (from the files or entered), Galactic extinction (looked up at the
position from the files or entered RA/Dec, or E(B&minus;V) by hand), the wavelength range and
masks: telluric bands, night-sky lines, narrow absorption lines and ranges dragged on the plot.
<i>Apply to</i> changes one epoch only. <i>Flux unit</i> shows the unit of the flux values as read from the file
(SDSS, DESI: 10<sup>&minus;17</sup> erg s<sup>&minus;1</sup> cm<sup>&minus;2</sup> &Aring;<sup>&minus;1</sup>; a text file: from its
header, e.g. &ldquo;flux in 1e-16 erg/s/cm2/A&rdquo;, or values already in erg s<sup>&minus;1</sup> cm<sup>&minus;2</sup>
&Aring;<sup>&minus;1</sup>) and lets you set it when the file does not say (<code>spec.with_flux_unit(1e-16)</code>).
Without it there are no luminosities or black-hole masses.</li>
<li><b>Model</b>: the card at the top classifies the spectrum (starlight, broad H&alpha;, BPT
diagrams) and recommends a model; <i>Use recommended model</i> applies it and <i>View classification</i>
shows the figure. Start from <i>a preset</i> (broad components, host and options) or from <i>your own
model</i> (a power-law continuum). <i>Add component</i> opens one dialog for everything: emission or
absorption lines chosen from the line list (broad, narrow or outflow; Gaussian, Lorentzian, Voigt,
Gauss&ndash;Hermite or stratified profile; width and shift limits; shift and width fitted, fitted for each
line, or the same as another component), continua (power law, broken power law, blackbody,
polynomial, Balmer continuum), templates (Fe II, high-order Balmer lines, host galaxy), absorption troughs
and dust extinction. Click a component to change its profile, tie its shift and width to another
component (e.g. Fe II to the narrow lines) or free them again, and edit the initial value, limits, fixed
status and tie of each parameter. The initial guess is drawn on the reference spectrum.</li>
<li><b>Fit</b>: what will be fitted (spectra, preparation, model, each with a link to change it); for several epochs
<i>together</i> (<code>fa.fit_epochs</code>: shared narrow lines and host, one flux scale per epoch, optionally the same
broad-line shape in all epochs) or <i>each on its own</i> (<code>fa.fit_many</code>); for a sample the model of each
spectrum (<code>fa.fit_sample</code>); the uncertainties (fast covariance, Monte Carlo, MCMC or none); and advanced
options (error scaling, model uncertainty, absorption troughs, host amplitude per epoch, shared parameters,
processes). When the fit is done, its quality and the parameters that ended at a limit are shown, with links to the
results and to the Model step.</li>
<li><b>Results</b>: the summary at the top gives the fit quality (reduced &chi;&sup2;) and the main numbers
with their errors (broad H&beta; FWHM and flux, black-hole mass, &sigma;&#9733;, ...); parameters that ended at a
limit are listed with a button that opens the component in the Model step. The views:
<i>Fit</i> (the fit with its components and residuals: drag to zoom, click the legend to hide a curve; the
lines grouped by component, with flags for undetected lines and tied widths; the continuum and host; all
parameters; for several epochs the table of all epochs; and <i>Does the model miss anything?</i>, the residual
analysis of <code>fa.analyze_residuals</code>, which marks missing lines, wings, absorption, continuum shape
and sky residuals on the plot and applies the suggested change; the continuum and host table also lists
the black-hole mass from every recipe that applies), <i>Publication figure</i> (wavelength and flux
range, title or none, width, font size, legend; what is drawn (data, model, each kind of component) and in
which colour; line labels for the strongest or all fitted lines, with their font, size, direction and colour,
joined to the lines by grey lines; redrawn as they change; PDF, PNG or EPS),
<i>Posterior</i> (with Monte Carlo or MCMC errors: a corner plot of the chosen parameters),
<i>Main sequence</i> (with Fe II and broad H&beta;: R<sub>Fe II</sub> against FWHM(H&beta;) for every epoch, with the
Population A/B boundary and spectral types),
<i>Classification</i> (when the spectrum was classified) and, for several epochs, <i>Light curves &amp; lags</i>
(light curves and reverberation mapping). <i>Download</i> gives the fitted lines (CSV or FITS), the model and
its components (CSV of the shown fit: data, model, every component, host, AGN, spectrum minus host and, with
extinction, the transmission and the intrinsic components), the fit figures (PDF), the Python script, the
notebook and the project.</li>
</ol>
<h3>The Python session</h3>
<p>Every step writes ordinary FANTASY code and runs it. The session panel shows that code
and its output; it is hidden by default: open it with the <b>python &middot; server</b> button
at the top (or <kbd>Ctrl</kbd>+<kbd>`</kbd>). Type code in the field at the bottom and run it
with <kbd>Shift</kbd>+<kbd>Enter</kbd>; click a cell to edit it and run it again. All code
runs in one namespace:</p>
<table class="doc"><tr><td><code>files</code>, <code>raw</code></td><td>the paths and the spectra as read</td></tr>
<tr><td><code>spectra</code>, <code>spec</code></td><td>the prepared spectra; the spectrum (one spectrum)</td></tr>
<tr><td><code>model</code></td><td>the model</td></tr>
<tr><td><code>r</code>, <code>fits</code>, <code>me</code></td><td>the fit of one spectrum; epochs fitted on their own; the joint fit</td></tr>
<tr><td><code>sample</code>, <code>classes</code></td><td>a fitted sample; classifications</td></tr></table>
<p>Changing these objects in code updates the steps, so code and buttons are equivalent.
<b>Copy</b> copies the session as a script, <b>.ipynb</b> downloads it as a notebook.
Reverberation mapping and the comparison of decompositions are in Python (<code>fa.rm</code>,
<code>fa.compare</code>); single lines are in Results, Line profiles.</p>
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
SPARCL downloads, OzDES and GAMA files, 1D FITS spectra and text files with columns wavelength, flux and
(optionally) error. It reads the redshift, the wavelength medium (air/vacuum), the
instrumental resolution and the flux unit when the file has them.</p>
<pre>spec = fa.read("spec-0615-52347-0108.fits").to_rest_frame()
spec = fa.read("my_spectrum.txt", z=0.123).to_rest_frame()
spec = fa.desi.fetch(39627963565609635).to_rest_frame()   # DESI DR1 via Data Lab</pre>
<h3>OzDES</h3>
<p>OzDES DR2 files (2dF/AAOmega) hold the stacked spectrum and every exposure with its variance
and bad-pixel mask; <code>fa.ozdes.download(name, folder)</code> gets them from VizieR.</p>
<pre>path = fa.ozdes.download("SVA1_COADD-2937976373", "ozdes/")[0]
spec = fa.read(path)                    # the stack
spec = fa.read(path, epoch=4)           # one exposure
nights = fa.ozdes.read_epochs(path, skip_qc=True, combine="night")   # for fit_epochs</pre>
<p>OzDES spectra are <b>not flux calibrated</b>: fluxes are in counts and differ between
nights, so FANTASY does not compute luminosities or black-hole masses for them. Widths,
shifts, EWs and line ratios are fine; for light curves use the multi-epoch fit with
intercalibration and also share what does not vary (e.g. <code>outflow.*</code>,
<code>feii.fwhm</code>, <code>feii.shift</code>). Single 40-min exposures are noisy:
<code>combine="night"</code> averages the exposures of each night. Spectra from the Data
Central SSA service have no errors; FANTASY estimates the noise from the spectrum.</p>
<h3>Galactic extinction</h3>
<p>Correct the Milky Way extinction before measuring fluxes and luminosities. Without it
fluxes and L5100 are too low by 5&ndash;20% for E(B-V) = 0.015&ndash;0.06 (typical at high
Galactic latitude) and the Balmer decrement is 2&ndash;6% too high; ratios of nearby lines
(BPT) change by less than 0.003 dex.</p>
<pre>spec = fa.read("spec-0615-52347-0108.fits").deredden().to_rest_frame()
spec.meta["ebv"], spec.meta["ebv_source"]      # 0.0122, IRSA dust service, SF11
spec = fa.read("my.txt").deredden(ebv=0.03).to_rest_frame(z=0.12)
# a file without coordinates (text, some FITS): give the position, then E(B-V) is looked up
spec = fa.read("my.txt", z=0.12, ra="12:34:56.7", dec="+12:34:56").deredden().to_rest_frame()
spec = fa.read("my.txt", z=0.12, ra=188.7362, dec=12.5822)          # degrees
spec = spec.with_coordinates("12:34:56.7 +12:34:56")                # or later</pre>
<p>In the app, the <b>RA</b> and <b>Dec</b> fields of the Read step set the position (degrees, or hours and
degrees in sexagesimal form) when the file has none or a wrong one.</p>
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
<h3>Stratified BLR profile for the broad lines</h3>
<p>Tick <b>stratified BLR profile</b> (Model step and Multi-epoch view) to describe each broad line by one
physical profile instead of a core plus a very broad Gaussian: a virialised broad-line region in which gas at
radius r has a Gaussian velocity distribution of FWHM ∝ r<sup>-1/2</sup>, summed from an inner to an outer
radius with emissivity per ln r ∝ r<sup>3-γ</sup>. Parameters: <code>fwhm</code> (width at the outer edge, the
core), <code>span</code> = log<sub>10</sub>(r<sub>out</sub>/r<sub>in</sub>) (the inner edge has FWHM
fwhm·10<sup>span/2</sup>) and <code>gamma</code>; span = 0 is a single Gaussian, γ ≈ 3 gives Lorentzian-like
wings. In code: <code>fa.presets.hbeta(broad_components=1, broad_profile="stratified")</code>.</p>
<p>The inner edge is often not measurable: the innermost gas is faint and its light is spread over very high
velocities, below the noise. <code>span</code> then reaches its upper limit (2.5) and is reported as a
<i>lower limit</i> of the radial extent ("inner edge not constrained"); the outer width and γ remain well
measured. A similar virialised, stratified description was used by Madau et al. (2026) for the broad wings of
JWST "little red dots".</p>

<p>A model is a sum of components. Start from a preset and change it:</p>
<pre>model = fa.presets.optical(physical=True, broad_components=2, outflow=True)
model.add(fa.Lines("broad_hei", ["HeI5876"], role="broad", fwhm=(4000, 1000, 15000)))
model.remove("feii")
model</pre>
<h3>Presets</h3>
<table class="doc">
<tr><td><code>hbeta</code></td><td>4150–5550 Å: power law, FeII (shift tied to broad Hβ), narrow and broad Hβ/Hγ, [O III]
with an outflow wing, HeII</td></tr>
<tr><td><code>halpha</code></td><td>6200–6900 Å: Hα, [N II], [S II], [O I], HeI; outflow can follow [O III]</td></tr>
<tr><td><code>optical</code></td><td>3700–6900 Å: everything above, Balmer continuum, high-order Balmer lines;
<code>physical=True</code>: one power law (α<sub>λ</sub> = −1.7), stellar host, broad FWHM ≤ 15000 km/s</td></tr>
<tr><td>options</td><td><code>coronal=True</code>: [Fe VII] 3759, 5159, 5721, 6087, [Fe X] 6375, [Fe XIV] 5303 (also
<code>hbeta</code>, <code>halpha</code>, <code>type2</code>); <code>oi6046=True</code>: broad O I 6046;
<code>red=True</code>: fit to 7500 Å with He I 7065, [Ar III] 7136, [O II] 7320, 7330. In the app these are the
<i>Also include</i> boxes of step 3. Look at the residuals: strong positive residuals at 3759, 5721, 6087 Å mean
coronal lines.</td></tr>
<tr><td><code>mgii</code>, <code>civ</code></td><td>UV lines: MgII with UV FeII (one broad component, no narrow MgII by
default); CIV with two broad components. Mask narrow absorption lines first (<code>spec.mask_absorption()</code>, or
<i>Mask narrow absorption</i> in step 1 of the app).</td></tr>
<tr><td><code>seyfert2</code>, <code>composite</code></td><td>stellar populations with dust, weak featureless
continuum (scattered light), narrow lines [O II] 3727&ndash;[O II] 7330 in two kinematic systems (high ionisation:
[Ne III], [O III], He II, [Ar III]; low: Balmer, [N II], [S II], [O I]&hellip;), outflow wing incl. [S II]
(<code>seyfert2</code>; off for <code>composite</code>)</td></tr>
<tr><td><code>liner</code></td><td>old stellar population with dust, one (possibly broad) narrow-line system;
<code>broad=True</code> for a LINER 1.9</td></tr>
<tr><td><code>starforming</code></td><td>BC03 young + old populations with dust, high-order Balmer series, He I,
auroral lines ([O III] 4363, [N II] 5755, [S III] 6312, [O II] 7320), Case B Balmer ratios + gas E(B-V);
<code>outflow=True</code> for a wing on the strong lines, <code>wolf_rayet=True</code> for WR galaxies</td></tr>
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
<tr><td><code>fa.FeII</code></td><td>Fe II template: <code>kind="optical"</code> (Kovačević et al. 2010, 3700–10 900 Å),
<code>"uv"</code> (Kovačević-Dojčinović &amp; Popović 2015, 2690–3005 Å) or <code>"uv_extended"</code> (2200–3500 Å, grouped by
lower term as the optical model, NIST atomic data)</td></tr>
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
<h3>Tidal disruption events and Bowen fluorescence flares</h3>
<p><code>fa.presets.tde()</code> (menu: <b>TDE (blackbody + Bowen)</b>): a hot blackbody continuum
(<code>fa.Blackbody</code>, T in K) and the host galaxy, broad Hδ/Hγ/Hβ and Hα (FWHM up to 20 000 km/s), broad
He II 4686 with the Bowen lines N III 4640 (and N III 4100, O III 3760, He II 5412 in range) sharing its
kinematics, broad He I 5876, 6678 and weak narrow host lines; options <code>broad_profile</code>
('lorentzian' or 'stratified' for extended wings), <code>feii</code>, <code>coronal</code>, <code>bowen</code>.
<code>fa.presets.bff()</code> (<b>Bowen flare / coronal</b>) adds Fe II and coronal lines with narrower broad lines
(from 1500 km/s), for Bowen fluorescence flares and extreme coronal line emitters (Baldini et al. 2025). The
optical spectrum constrains the blackbody temperature only loosely: fix it from UV data if available. Several
epochs of a transient are best fitted jointly (Multi-epoch), sharing the host galaxy and narrow-line fluxes,
which also calibrates spectra from different instruments.</p>

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
<h3>Spectra of high S/N: refinement</h3>
<p>On 160 SDSS narrow-line galaxies the presets give χ²/ν ≈ 1.0 below S/N 45 and 1.5 (median) above it:
there the data show what a simple model leaves out. For the narrow-line classes FANTASY therefore
masks the telluric bands and strong sky lines (<code>spec.mask_sky()</code>) and, if χ²/ν &gt; 1.2, tries
the physical extensions of the preset one at a time, keeping one only if the BIC improves by more than 10
and its new parameters do not end at a limit (then it absorbs template mismatch rather than describing
gas):</p>
<pre>r = fa.refine(spec.mask_sky(), "starforming")    # the app's Fit step does this
r.refinement                                     # what was tried, dBIC, kept or not</pre>
<table class="doc">
<tr><th>extension</th><th>what it describes</th><th>presets</th></tr>
<tr><td>flux-calibration polynomial (<code>calibration=10</code>)</td><td>smooth 1&ndash;3% errors of the
spectrophotometric calibration and the dust law (multiplicative Legendre polynomial, as pPXF
<code>mdegree</code>; without the linear term when a dust screen is fitted). On 390 galaxies it changed
σ★ by +1&ndash;7 km/s and the BPT ratios by &lt; 0.03 dex.</td><td>all</td></tr>
<tr><td>outflow wing</td><td>broad bases of the strong lines (winds, diffuse gas)</td><td>starforming, composite, liner</td></tr>
<tr><td>Gauss&ndash;Hermite profiles (<code>narrow_profile="gauss-hermite"</code>)</td><td>asymmetric and
peaked or flat-topped lines (rotation in the fibre)</td><td>starforming, seyfert2, composite, liner</td></tr>
<tr><td>broad Balmer lines</td><td>LINER 1.9</td><td>liner</td></tr>
</table>
<p>On the eight high-S/N examples χ²/ν went 3.4 → 2.0 and 2.2 → 1.8 (star-forming), 2.1 → 1.6 and 1.3 → 1.1
(Seyfert 2), 1.6 → 1.1 (LINER) and 2.1 → 1.3 (passive). H8 is no longer tied to Case B in
<code>starforming</code>: it is blended with He I 3889.</p>
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
me.plot_light_curves(); me.plot_baldwin()
me.plot(epoch=5)           # the fit of one epoch (me.epochs[5] is its full result)</pre>
<p>First every epoch is fitted on its own (these fits run in parallel, <code>n_jobs</code> processes),
which gives the starting values; then all epochs are fitted together with a sparse Jacobian. The
reference epoch (flux scale 1) is the one with the highest S/N. For many unrelated spectra use
<code>fa.fit_many(spectra, model, n_jobs=-1)</code>, which returns one fit per spectrum.</p>
<p>In scripts on macOS and Windows, put the code under <code>if __name__ == "__main__":</code>, otherwise
the parallel part falls back to serial. In the app, the <b>Multi-epoch</b> view does all of this:
drop the files (or give a pattern), choose the model and press <i>Fit all epochs</i>. A progress
bar shows the independent fits and the joint fit. Below the light curves, <i>Fit of each epoch</i>
shows the data, model, components and residuals of every epoch: step with the arrows (or the
← → keys), pick one from the list or click a row of the table. <i>All epoch fits (PDF)</i> gives
one page per epoch.</p>
"""),
    ("absorption", "Absorption troughs (CIV, MgII)", """
<p>Many CIV and MgII quasar spectra contain absorption: narrow intervening absorbers and, in BAL and
mini-BAL quasars, broad troughs from outflowing gas. They cut into the emission lines and dominate
&chi;&sup2;. Tick <b>Fit absorption troughs</b> in the Fit step: FANTASY fits the model, finds the troughs
in the residuals, multiplies the model by their transmission exp(&minus;&tau;) and refits while the BIC
improves. The emission components then describe the unabsorbed spectrum.</p>
<pre>r = fa.fit_absorbed(spec, fa.presets.civ())
r.absorbers                       # wavelength, FWHM and optical depth of every trough
r.no_absorbers                    # the fit without them, for comparison
r.plot_line("CIV1549")            # the line, unabsorbed and (dashed) as observed
model.add(fa.Absorption([(1510.5, 1200), (1547.0, 300)]))   # troughs by hand</pre>
<p>On the 150 CIV quasars of the calibration, the worst fits went from reduced &chi;&sup2; 8&ndash;24 to
1.0&ndash;1.9. Troughs are kept narrower than 6000 km/s (and than three times the width at which they were
found), so that a broad shallow &ldquo;absorber&rdquo; cannot trade flux with a too-broad emission line.
Widths and fluxes of the emission can change a lot when troughs are fitted: compare with
<code>r.no_absorbers</code> and look at the figure. For narrow absorbers only, masking is simpler:
<i>Mask narrow absorption</i> in step 1, or <code>spec.mask_absorption()</code>.</p>
"""),
    ("reverberation", "Reverberation mapping", """
<p>After a multi-epoch fit, <b>Lag, rms spectrum and mass</b> (Multi-epoch view, below the light curves)
measures the lag of the broad line behind the continuum, its width in the rms spectrum and the
reverberation black-hole mass.</p>
<pre>rm = me.reverberation("Hb")                                  # continuum: the fitted 5100 A light curve
rm = me.reverberation("Hb", continuum=(mjd, flux, err))       # better: photometry
rm = me.reverberation("Hb", mjd_range=(56600, 56850), detrend=True)   # one season, trend removed
rm; rm.table(); rm.plot()
me.rms_spectrum("Hb")                                        # mean and rms profiles and widths</pre>
<ul>
<li><b>Lag</b>: interpolated cross-correlation (ICCF), centroid of r &ge; 0.8 r<sub>max</sub>; errors from
flux randomisation and random subset selection (Peterson et al. 1998, 2004). The search window is &plusmn;5
times the lag expected from L5100 and the R&ndash;L relation (Bentz et al. 2013), at least 60 days.</li>
<li><b>rms spectrum</b>: the broad line alone in every epoch (calibrated data minus continuum, FeII, host
and narrow lines), noise-corrected rms (Park et al. 2012); widths with bootstrap errors.</li>
<li><b>Mass</b>: M = f c&tau; &sigma;<sub>line</sub>(rms)&sup2;/G with f = 4.47 (Woo et al. 2015); the
~0.3 dex scatter of f is not in the error.</li>
</ul>
<p><b>Wavelength-dependent intercalibration</b>: <code>fa.fit_epochs(..., calibration_degree=1)</code> (or 2)
gives every epoch a smooth calibration s<sub>k</sub>(λ) = scale<sub>k</sub> (1 + Σ c<sub>kj</sub> P<sub>j</sub>), anchored
by the shared narrow lines (and host) across the fitted range. PrepSpec (Shen et al. 2016) and the default
use one number per epoch, which leaves the colour-dependent part of the fibre and refraction losses in the
light curves. It needs narrow lines at both ends of the range ([O II] or Hβ/[O III] and Hα/[N II]/[S II]).
The coefficients are in <code>me.epochs[k].calibration</code>.</p>
<p><b>Intercalibration</b> on the narrow lines only helps when they are strong. With weak [O III] it adds
a few percent of noise per epoch, comparable to the variability; FANTASY warns when the median flux-scale
error exceeds 2%. For spectra with a good pipeline calibration (SDSS) compare with
<code>intercalibrate=False</code>: on SDSS-RM RMID 766 the uncalibrated light curve matched the published one
best.</p>
<p><b>Lag methods.</b> <code>me.reverberation</code> reports three lags (<code>rm.lag_table()</code>) and warns
when they disagree or when much of a lag distribution lies outside its primary peak (seasonal aliases,
Grier et al. 2017 weighting): the ICCF (adopted; <code>primary=</code> changes it), the von Neumann estimator
(Chelouche et al. 2017; no interpolation) and a damped-random-walk model (as JAVELIN) that also uses the
correlated errors of the line and continuum in each epoch's fit. On nine SDSS-RM quasars the DRW lag on
the published light curves agreed with PyROA in 5 of 8, with smaller errors, but it fell on a seasonal
alias in one; the ICCF stays the default.</p>
<p><b>Lags of the spectral components</b> (new): <code>fa.rm.component_reverberation(me, "Hb",
continuum=...)</code> measures a lag and a response for every component at once (the continuum, each broad
component, FeII), from their fluxes and full error covariance over all epochs (within an epoch the
components trade flux; between epochs shared parameters couple them). A component may not respond at all:
the data decide which reverberate. On SDSS-RM quasars the very broad component responded with a lag near
zero in six of nine objects (continuum leaking into the wings, or the innermost gas), and the main broad
component recovered the published Hβ lag in five of eight, e.g. RMID 772: 10.7 ± 1.3 d (PyROA 9.7 ± 1.4 d)
where the total Hβ light curve gave 4&ndash;6 d. The table also gives cleaned light curves
(decomposition noise removed).</p>
<p>Lags need well-sampled light curves: the continuum from the spectra is often too sparse and diluted
by host light, and a long baseline with seasonal gaps is dominated by slow trends. Check r<sub>max</sub>
(&gt; 0.6&ndash;0.7 for a reliable lag) and the CCF in the figure.</p>
"""),
    ("sample", "Fitting a sample", """
<p>The <b>Sample</b> view fits many spectra at once. Drop the files (or give a pattern such as
<code>/data/*.fits</code>) and press <i>Fit the sample</i>. Each spectrum is read, corrected for Galactic
extinction, moved to the rest frame, <b>classified</b> and fitted with the model its class needs (with the
host check for borderline starlight), in parallel processes. Choose a model in the menu instead of
<i>automatic</i> to fit all spectra with the same one.</p>
<p>The result is one <b>catalog</b>: class, model, reduced &chi;&sup2;, the main quantities with errors
(broad-line widths and black-hole masses, or the narrow-line diagnostics) per spectrum. Sort it by
clicking a column header (e.g. reduced &chi;&sup2;, to check the worst fits first); click a row to see
its fit and main line. Download the catalog as CSV; every fit is also saved as a <code>.fantasy</code> file.</p>
<pre>s = fa.fit_sample(glob.glob("data/*.fits"), out="fits/")   # in a script: under if __name__ == "__main__":
s.table                # the catalog
s.load(3)              # the full fit of row 3
s.gallery()            # HTML page with every fit</pre>
"""),
    ("diagnostics", "Narrow-line diagnostics", """
<p><code>r.diagnostics()</code> turns the narrow-line fluxes of a fit into physical quantities, with
errors propagated from the fit. The app shows them in the <b>Data</b> tab and the main ones in the
results.</p>
<table class="doc">
<tr><td>BPT ratios and class</td><td>[N II]/H&alpha;, [O III]/H&beta;, [S II]/H&alpha;, [O I]/H&alpha;;
Kauffmann et al. (2003), Kewley et al. (2001, 2006)</td></tr>
<tr><td>narrow H&alpha;/H&beta;, E(B-V) gas</td><td>against Case B 2.86 (H II regions) or 3.1 (AGN), Fitzpatrick
(1999); with the <code>starforming</code> preset E(B-V) is fitted and the fluxes are already dereddened</td></tr>
<tr><td>n<sub>e</sub> ([S II])</td><td>[S II] 6716/6731, Sanders et al. (2016), 10<sup>4</sup> K</td></tr>
<tr><td>T<sub>e</sub> ([O III]), T<sub>e</sub> ([N II])</td><td>auroral/nebular ratios (when [O III] 4363 or [N II] 5755 is
detected), dereddened, Osterbrock &amp; Ferland (2006)</td></tr>
<tr><td>EW(H&alpha;)</td><td>for the WHAN diagram (Cid Fernandes et al. 2011)</td></tr>
<tr><td>12+log(O/H)</td><td>star-forming galaxies only: O3N2 and N2 of Marino et al. (2013); their calibration
scatter (0.16&ndash;0.18 dex) is not in the error</td></tr>
</table>
<p>Fluxes are summed over the narrow components and the outflow wing (<code>outflow=False</code>
leaves the wing out).</p>
"""),
    ("uncertainty", "Model uncertainty", """
<p>Broad-line widths, fluxes and the continuum luminosity often depend more on how a spectrum is
decomposed (one, two or three broad components, an outflow wing, a host galaxy) than on the noise. Tick
<b>Model uncertainty</b> in the Fit step: FANTASY fits all these decompositions in parallel, keeps the
best one (lowest BIC) as the result, and adds a <b>model (systematic) error</b> to the results: the
spread over the acceptable decompositions (reduced &chi;&sup2; within 10% of the best).</p>
<pre>models = fa.presets.variants("hbeta", host=(False, True))
cmp = fa.compare(spec, models)        # fits in parallel
cmp                                    # table of the models and the summary
cmp.summary()                          # value, statistical, systematic and total error
r = cmp.result                         # the best fit</pre>
<p>Example (SDSS quasar, 12 decompositions): H&beta; FWHM 4522 &plusmn; 56 (stat) &plusmn; 52 (model)
km/s, but the line dispersion 3524 &plusmn; 143 &plusmn; 325 km/s and log L5100 44.854 &plusmn; 0.003
&plusmn; 0.039: for &sigma;<sub>line</sub> and L5100 the decomposition dominates the error.</p>
"""),
    ("save", "Saving and reopening work", """
<p>In the app, the <b>download</b> button above the code cells saves the whole project (all cells
with their outputs, the spectra, classifications, models and fits) as one <code>.fantasy</code> file;
the <b>upload</b> button next to it opens a saved project, which replaces the current session. The
<b>&lt;/&gt;</b> button downloads the session as a Python script instead.</p>
<p>In Python, any result can be saved and loaded:</p>
<pre>fa.save(r, "quasar.fantasy")         # FitResult, MultiEpochResult, Classification, Model, Spectrum
r = fa.load("quasar.fantasy")
r.plot_line("Hb"); r.measure("Hb")</pre>
<p>A <code>.fantasy</code> file is a compressed Python pickle: like any pickle it can run code when it is
opened, so open only files you or your collaborators made. Stellar templates are not stored (a fit with
a host is a few hundred kB); they are read from the template folder when the file is opened.</p>
"""),
    ("single-lines", "Looking at single lines", """
<p>When a line is fitted with several components, the figure shows each of them: the 1st broad
component dashed, the 2nd dotted, besides their sum (the interactive plot has them as
<i>broad 1</i>, <i>broad 2</i>; click a legend entry to hide one).</p>
<p>To look at one line alone, press <b>Single line</b> above the plot (after a fit). It shows the data
with everything else in the model subtracted (continuum, host, FeII, all other lines), each component
of the line with its FWHM, their sum, and the residuals, on a velocity axis. The list starts with the
lines that have several components and the main line is selected: broad H&beta;, H&alpha;, MgII or
CIV, or [O III] &lambda;5007 for narrow-line objects. Step through the lines with the arrows or the
&larr; &rarr; keys, or <b>Alt-click</b> (&#8997;-click) a line in the interactive plot. Each figure can
be downloaded as PDF or PNG.</p>
<pre>r.fitted_lines()                       # the lines and their components
r.plot_line()                          # the main line
r.plot_line("Ha", others=True)         # H-alpha; neighbouring [N II] dotted
r.plot_line("[OIII]5007", velocity=False, window=1000)
r.plot_lines(["Hb", "[OIII]5007", "Ha"])</pre>
"""),
    ("calibration", "How the presets were calibrated", """
<p>The defaults of the presets and of the classifier come from fits of <b>1500 SDSS DR17 spectra</b>, random within
nine groups: quasars at z &lt; 0.35 (300), with H&beta; at 0.35&ndash;0.75 (200), MgII (200) and CIV (150), and
MPA-JHU Seyfert 2s (150), LINERs (150), composites (100), star-forming (150) and passive galaxies (100). Each was
fitted with its preset and alternatives; start values and limits were taken from 70% of the sample and the
changes tested on the other 30%. The scripts and reports are in <code>calibration/</code> of the repository.</p>
<table class="doc">
<tr><th>change</th><th>why (held-out 30%)</th></tr>
<tr><td>[O III] outflow wing on in <code>optical</code> and <code>hbeta</code></td><td>BIC better for 82% / 66%;
median reduced &chi;&sup2; 1.35&rarr;1.25 and 1.02&rarr;0.96</td></tr>
<tr><td>stellar host for host-dominated quasars (<code>classify</code> &rarr; type1_host &rarr;
<code>optical(host=True)</code>)</td><td>with the classifier's choice: median &chi;&sup2;/&nu; 1.10, limits per fit
2.9&rarr;1.8, BIC better for 82%</td></tr>
<tr><td>FeII shift tied to broad H&beta; (<code>hbeta</code>)</td><td>BIC better for 72%; the free shift ended at its
limit in a quarter of the fits</td></tr>
<tr><td>no narrow MgII</td><td>BIC better for 77%; parameters at a limit 1.1&rarr;0.3 per fit</td></tr>
<tr><td>two broad CIV components, He II 1640 shift tied to CIV</td><td>BIC better for 98%; &chi;&sup2;/&nu;
1.76&rarr;1.52</td></tr>
<tr><td>no outflow wing for <code>composite</code> and <code>starforming</code></td><td>BIC better for 83% / 76%,
three times faster; the wing had absorbed template mismatch (width at the 2500 km/s limit)</td></tr>
<tr><td>broad-line test of <code>classify</code>: &Delta;BIC &gt; 25, S/N &gt; 5, EW &gt; 30 &Aring;, FWHM
1200&ndash;12000 km/s</td><td>type 1 found in 76% of the quasars (was 47%; the rest show no broad line),
0.9% of the narrow-line galaxies</td></tr>
</table>
<p>Unchanged after the tests: <code>seyfert2</code> (outflow and two narrow-line systems), <code>liner</code>,
<code>passive</code>, the <code>halpha</code> outflow (off: a free wing in the H&alpha; window is not constrained;
take it from an [O III] fit) and the He I and FeII kinematics of <code>optical</code> (tying them made the fits
worse). CIV spectra often contain narrow absorbers: <code>spec.mask_absorption()</code> removes them.</p>
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
r.bh_mass("Hb")            # single-epoch black-hole mass (Vestergaard & Peterson 2006)
r.bh_mass("Hb", recipe="db20")   # Dalla Bontà et al. 2020: FWHM and L(Hβ broad)
r.bh_mass("Ha")            # Greene & Ho 2005: FWHM and L(Hα broad)
r.bh_masses()              # every recipe that applies, in one table
r.feii()                   # Fe II 4434–4684 Å flux, EW and R_FeII
r.main_sequence()          # R_FeII, FWHM(Hβ), population and spectral type
r.balmer_decrement()       # broad Hα/Hβ (typical ~3)
r.host_properties()        # σ★, v★, age, metallicity, host fraction
r.plot(); r.savefig("fit.pdf")
r.plot(label_lines="all", colors={"broad": "navy"}, hide=("host",), title=None)
r.save("my_fit")           # tables as CSV</pre>
<p><b>Black-hole masses.</b> log M = a + c log(W/1000 km/s) + b log(L/10<sup>n</sup> erg/s), W the
FWHM (or σ<sub>line</sub>) of the summed broad line measured on the model:</p>
<table class="tbl"><tr><th>line</th><th>recipe</th><th>width, luminosity</th><th>reference</th></tr>
<tr><td>Hβ</td><td>vp06 (default)</td><td>FWHM, L5100</td><td>Vestergaard &amp; Peterson 2006</td></tr>
<tr><td>Hβ</td><td>db20, db20_sigma</td><td>FWHM or σ, L(Hβ broad)</td><td>Dalla Bontà et al. 2020, eqs. 40 and 38, log f = 0.683</td></tr>
<tr><td>Hβ</td><td>db20_l5100, db20_sigma_l5100</td><td>FWHM or σ, L5100</td><td>the same through their eq. 24</td></tr>
<tr><td>Hβ</td><td>shen24</td><td>FWHM, L5100</td><td>Shen et al. 2024 (SDSS-RM)</td></tr>
<tr><td>Hα</td><td>gh05 (default), rv15</td><td>FWHM, L(Hα broad)</td><td>Greene &amp; Ho 2005; Reines &amp; Volonteri 2015</td></tr>
<tr><td>Mg II</td><td>vo09 (default), shen11, shen24</td><td>FWHM, L3000</td><td>Vestergaard &amp; Osmer 2009; Shen et al. 2011, 2024</td></tr>
<tr><td>C IV</td><td>vp06 (default), shen24</td><td>FWHM, L1350</td><td>Vestergaard &amp; Peterson 2006; Shen et al. 2024</td></tr></table>
<p>The errors are statistical; the recipes scatter by 0.3–0.6 dex around reverberation masses.
Recipes with the broad-line luminosity need no host subtraction.</p>
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
                     ("key_results", "lines", "measure", "bh_mass", "bh_masses", "feii", "balmer_decrement",
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
    from ..lines import line_table
    t = line_table()
    e = html.escape
    rows = "".join(f"<tr><td><code>{e(n)}</code></td><td>{r.vacuum:.2f}</td><td>{r.wavelength:.2f} {e(r.medium)}</td>"
                   f"<td>{e(r.ion)}</td><td>{e(r.type)}</td><td>{e(r.series)}</td><td>{e(r.note)}</td></tr>"
                   for n, r in t.iterrows())
    return {"id": "lines", "title": "Line list", "section": "Reference",
            "html": f"<p>The {len(t)} lines of the library, with the names to use in "
                    "<code>fa.Lines(name, [lines])</code>. Wavelengths are used in vacuum; the tabulated value "
                    "and its medium (air or vacuum) are given next to it. In Python: <code>fa.line_table()</code>, "
                    "<code>fa.lines_in(wmin, wmax, type=..., ion=..., series=...)</code>, "
                    "<code>fa.line_info(name)</code>.</p>"
                    '<table class="doc"><tr><th>name</th><th>λ vacuum (Å)</th><th>tabulated</th><th>ion</th>'
                    f"<th>type</th><th>series</th><th>note</th></tr>{rows}</table>"}


def all_pages():
    pages = [{"id": i, "title": t, "section": "Guide", "html": h.strip()} for i, t, h in GUIDES]
    pages.append(line_list_page())
    pages += api_pages()
    return pages
