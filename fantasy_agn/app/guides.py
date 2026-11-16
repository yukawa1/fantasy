"""
The hand-written guides of the app's documentation: (id, section, title, HTML).

Conventions of the HTML (styled in static/wf.css, ``.docbody``):
``<div class="lead">`` the opening paragraph; ``<div class="note|tip|warn">`` call-outs; ``<div class="eq">``
a displayed equation, with ``<span class="frac"><span>numerator</span><span>denominator</span></span>``;
``<table class="doc">`` tables; ``<dl class="params">`` parameter lists; ``<ol class="steps">`` numbered steps.
"""

_PIPELINE = """
<svg class="diagram" viewBox="0 0 760 92" role="img" aria-label="The five steps: Read, Prepare, Model, Fit, Results">
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#898781"/></marker></defs>
  <g font-family="inherit" text-anchor="middle">
    <g><rect x="2" y="14" width="128" height="58" rx="9" class="dg-box"/><text x="66" y="40" class="dg-t">1 Read</text>
       <text x="66" y="58" class="dg-s">files, DESI, SDSS</text></g>
    <g><rect x="158" y="14" width="128" height="58" rx="9" class="dg-box"/><text x="222" y="40" class="dg-t">2 Prepare</text>
       <text x="222" y="58" class="dg-s">z, E(B&#8722;V), masks</text></g>
    <g><rect x="314" y="14" width="128" height="58" rx="9" class="dg-box dg-hi"/><text x="378" y="40" class="dg-t">3 Model</text>
       <text x="378" y="58" class="dg-s">classify, components</text></g>
    <g><rect x="470" y="14" width="128" height="58" rx="9" class="dg-box"/><text x="534" y="40" class="dg-t">4 Fit</text>
       <text x="534" y="58" class="dg-s">errors, epochs</text></g>
    <g><rect x="626" y="14" width="130" height="58" rx="9" class="dg-box"/><text x="691" y="40" class="dg-t">5 Results</text>
       <text x="691" y="58" class="dg-s">lines, masses, figures</text></g>
    <path d="M131,43 H156" class="dg-a" marker-end="url(#ar)"/><path d="M287,43 H312" class="dg-a" marker-end="url(#ar)"/>
    <path d="M443,43 H468" class="dg-a" marker-end="url(#ar)"/><path d="M599,43 H624" class="dg-a" marker-end="url(#ar)"/>
  </g>
</svg>"""

_FITFLOW = """
<svg class="diagram" viewBox="0 0 760 150" role="img" aria-label="How a fit proceeds">
  <defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#898781"/></marker></defs>
  <g text-anchor="middle">
    <rect x="2" y="10" width="170" height="70" rx="9" class="dg-box"/>
    <text x="87" y="36" class="dg-t">Initial guess</text><text x="87" y="54" class="dg-s">grid over widths, shifts</text>
    <text x="87" y="68" class="dg-s">amplitudes exact (NNLS)</text>
    <rect x="200" y="10" width="170" height="70" rx="9" class="dg-box dg-hi"/>
    <text x="285" y="36" class="dg-t">Variable projection</text><text x="285" y="54" class="dg-s">trust-region on nonlinear</text>
    <text x="285" y="68" class="dg-s">exact Jacobian, BVLS inside</text>
    <rect x="398" y="10" width="170" height="70" rx="9" class="dg-box"/>
    <text x="483" y="36" class="dg-t">Escape &amp; build-up</text><text x="483" y="54" class="dg-s">re-scan each component,</text>
    <text x="483" y="68" class="dg-s">add overlapping ones in turn</text>
    <rect x="596" y="10" width="160" height="70" rx="9" class="dg-box"/>
    <text x="676" y="36" class="dg-t">Uncertainties</text><text x="676" y="54" class="dg-s">covariance, Monte Carlo</text>
    <text x="676" y="68" class="dg-s">or HMC posterior</text>
    <path d="M173,45 H198" class="dg-a" marker-end="url(#ar2)"/><path d="M371,45 H396" class="dg-a" marker-end="url(#ar2)"/>
    <path d="M569,45 H594" class="dg-a" marker-end="url(#ar2)"/>
    <text x="380" y="118" class="dg-s">every derived number (FWHM, EW, L, M<tspan baseline-shift="sub" font-size="9">BH</tspan>) is propagated from the full parameter covariance or the samples</text>
  </g>
</svg>"""


GUIDES = [
# ======================================================================== Getting started
("start", "Getting started", "Overview", f"""
<div class="lead">FANTASY decomposes optical and ultraviolet spectra of active galaxies (quasars, Seyfert 1 and 2,
LINERs, star-forming and passive galaxies, nuclear transients) into physical components: continua, the host
galaxy, Fe&#8202;II, broad, narrow and outflowing emission lines, and absorption. It measures the lines,
derives black-hole masses and diagnostics, and propagates every uncertainty.</div>
{_PIPELINE}
<p>The app follows the five steps above. Each step writes ordinary FANTASY code into the Python session and runs
it, so whatever is done by clicking can be repeated, changed or scripted. The bar at the top shows which steps are done;
click a step to go back to it. <b>New analysis</b> clears the session; the <b>&#8943;</b> menu opens and saves projects and downloads
the session as a script or notebook.</p>
<p><b>The session is saved after every step</b> (to <code>~/.fantasy_agn/sessions</code>, or <code>$FANTASY_AUTOSAVE</code>; set it to
<code>0</code> to switch this off). When the app starts again, or after <b>New analysis</b>, the Read step offers <b>Restore last
session</b>, and <b>Open the one before</b> for the session before that: a server restart or a closed terminal no longer loses the
work. The saved file is a project like those of the &#8943; menu (a pickle: open only your own).</p>
<p><b>&#9632; Stop</b> appears in the top bar while a step runs: it ends the worker processes of a parallel run (fits of many epochs,
simulations) and interrupts the step at its next progress report or Python step (a single long call into compiled code finishes
first). The cell is marked <i>Stopped</i> and what it was computing is not kept; everything before it is.</p>
<h2>In five lines of Python</h2>
<pre>import fantasy_agn as fa
spec  = fa.read("spec-0615-52347-0108.fits").deredden().to_rest_frame()
c     = fa.classify(spec)                         # what kind of object, which model
r     = fa.fit(spec, fa.presets.optical(broad_components=2))
r.lines; r.measure("Hb"); r.bh_mass("Hb"); r.plot()</pre>
<h2>The session panel</h2>
<p>The <b>python &middot; server</b> button at the top (<kbd>Ctrl</kbd>+<kbd>`</kbd>) opens the session: every cell
the app ran, with its output. Type code at the bottom and run it with <kbd>Shift</kbd>+<kbd>Enter</kbd>; click a cell to
edit it and run it again. All cells share one namespace, and the steps read from it, so code and buttons are
interchangeable:</p>
<table class="doc"><tr><th>name</th><th>holds</th></tr>
<tr><td><code>files</code>, <code>raw</code></td><td>the paths and the spectra as read</td></tr>
<tr><td><code>spectra</code>, <code>spec</code></td><td>the prepared spectra; the spectrum when there is one</td></tr>
<tr><td><code>model</code></td><td>the model being built</td></tr>
<tr><td><code>c</code>, <code>classes</code></td><td>a classification; the classifications of several spectra</td></tr>
<tr><td><code>r</code>, <code>fits</code>, <code>me</code></td><td>the fit of one spectrum; epochs fitted one by one; the joint multi-epoch fit</td></tr>
<tr><td><code>sample</code></td><td>a fitted sample</td></tr></table>
<p><b>Copy</b> puts the session on the clipboard as a script; <b>.ipynb</b> downloads it as a notebook;
<b>&#8943; &rarr; Save project</b> keeps everything (cells, outputs, spectra, fits) in one <code>.fantasy</code> file.</p>
<h2>Keyboard</h2>
<table class="doc"><tr><td><kbd>Ctrl</kbd>+<kbd>`</kbd></td><td>show or hide the session</td></tr>
<tr><td><kbd>Shift</kbd>+<kbd>Enter</kbd></td><td>run the cell being typed</td></tr>
<tr><td><kbd>Esc</kbd></td><td>leave masking mode, close a dialog</td></tr>
<tr><td>drag on a plot</td><td>zoom (double-click: back); in Prepare, mark a range to mask</td></tr></table>
"""),

("install", "Getting started", "Installing and starting", r"""
<div class="lead">FANTASY is a Python package on PyPI; the app comes with it. Python 3.10 or newer;
the app is tested with Python 3.12.</div>
<h2>With pip</h2>
<pre>pip install fantasy_agn
fantasy test                  # check the installation
fantasy app                   # the app in the browser (or fantasy-app, or fa.app() in Python)
# Optional support for local SFD98 dust maps:
pip install "fantasy_agn[dust]"</pre>
<h2>With uv, in an environment of its own</h2>
<pre>uv tool install --python 3.12 fantasy_agn
fantasy app</pre>
<p>The first start after installing takes up to a minute while Python prepares the libraries; later
starts take a few seconds. New versions are announced in the app's header, with an Install button.</p>
<h2>From a source checkout</h2>
<p>In the directory containing <code>uv.lock</code> (the tested package versions):</p>
<pre>uv sync --locked --python 3.12
uv run fantasy test
uv run fantasy app</pre>
<div class="note"><b>Where things are kept.</b> Downloaded spectra and stellar templates, saved models
(<code>~/.fantasy_agn/models</code>) and uploads live under <code>~/.fantasy_agn</code>. Stellar template sets
(13&ndash;94 MB) are downloaded once, the first time a model uses them; <code>FANTASY_NO_DOWNLOAD=1</code>
switches downloads off.</div>
<div class="warn"><b>The app runs Python code.</b> It listens only on 127.0.0.1 and answers only requests that carry
the token of the page, so other web pages cannot use it. Do not expose its port.</div>
"""),

# ======================================================================== Workflow
("read", "The app, step by step", "1 &middot; Read", """
<div class="lead">Choose what you analyse and where the spectra come from. FANTASY recognises the format, the
redshift, the wavelength medium, the instrumental resolution and the flux unit when the file has them.</div>
<h2>What you analyse</h2>
<table class="doc"><tr><th>mode</th><th>what happens next</th><th>in Python</th></tr>
<tr><td>single spectrum</td><td>decompose one spectrum and measure its lines</td><td><code>fa.fit</code></td></tr>
<tr><td>sample</td><td>classify and fit many spectra in parallel; one catalogue</td><td><code>fa.fit_sample</code></td></tr>
<tr><td>multi-epoch</td><td>one object at several dates: joint fit, light curves, lags</td><td><code>fa.fit_epochs</code></td></tr>
<tr><td>classify only</td><td>type 1, host-dominated, Seyfert 2, LINER, star-forming or passive</td><td><code>fa.classify</code></td></tr></table>
<h2>Sources</h2>
<dl class="params">
<dt>Files</dt><dd>drop them on the page (any number). SDSS <code>spec-*.fits</code>, DESI coadd and SPARCL files, OzDES and
GAMA files, 1D FITS spectra, and text files with columns wavelength, flux and (optionally) error.</dd>
<dt>Path / glob</dt><dd>a path or pattern on this computer, e.g. <code>~/data/AT2024zkr/*.txt</code>.</dd>
<dt>DESI TARGETID</dt><dd>fetched from the DESI DR1 via Data Lab (<code>fa.desi.fetch</code>).</dd>
<dt>SDSS plate-mjd-fiber</dt><dd>fetched from the SDSS DR17 Science Archive Server (<code>fa.sdss.fetch</code>).</dd>
</dl>
<dl class="params">
<dt>File format</dt><dd><i>automatic</i> recognises the file; choose SDSS, DESI coadd, OzDES (all epochs or one public
spectrum), GAMA, 1D FITS or text when it guesses wrong.</dd>
<dt>Group epochs by</dt><dd>(multi-epoch) the date in the file name (YYYYMMDD), the MJD in the header, or the file order;
<i>automatic</i> tries them in that order.</dd>
<dt>Loaded</dt><dd>the table of what was read (date, name, redshift, position); tick rows and <b>Remove selected</b> to drop them.</dd>
</dl>
<p>DESI and SDSS spectra are downloaded with <b>Fetch</b> and kept under <code>~/.fantasy_agn</code>. OzDES files hold the stack and every exposure: <code>fa.read(path, epoch=4)</code> reads one exposure,
<code>fa.ozdes.read_epochs(path, combine="night")</code> averages each night.</p>
<div class="warn"><b>OzDES spectra are not flux calibrated</b> (counts, different per night). FANTASY then computes no
luminosities or masses; widths, shifts, EWs and ratios are fine, and light curves need the intercalibrated
multi-epoch fit.</div>
<pre>spec = fa.read("spec-0615-52347-0108.fits")
spec = fa.read("my_spectrum.txt", z=0.123, ra="12:34:56.7", dec="+12:34:56")
spec = fa.desi.fetch(39627963565609635)</pre>
"""),

("prepare", "The app, step by step", "2 &middot; Prepare", """
<div class="lead">Put every spectrum into the rest frame, correct the Milky Way extinction, set the flux unit and
remove what the model should not see.</div>
<h2>Redshift</h2>
<p><i>from file</i>, <i>manual</i>, or, with several spectra, <i>one for all (median)</i>: every spectrum is moved with the median of the
redshifts of the files (<code>z = float(np.median([s.z for s in raw if s.z]))</code>), the default for epochs of one object. Pipeline
redshifts differ from file to file (SDSS: about 20 km/s rms), and moving each epoch with its own shifts the same line between epochs;
the note under the choice says by how much the files differ. <b>Move to rest frame</b> applies it. Fits are done in the rest frame: <code>spec.to_rest_frame(z)</code> divides wavelengths by
1+<i>z</i> and multiplies flux densities by 1+<i>z</i>.</p>
<h2>Galactic extinction</h2>
<p><code>spec.deredden()</code> divides by the Fitzpatrick (1999) transmission with R<sub>V</sub> = 3.1:</p>
<div class="eq">F<sub>corr</sub>(&lambda;) = F(&lambda;) &middot; 10<sup>&nbsp;0.4&nbsp;E(B&minus;V)&nbsp;k(&lambda;)</sup></div>
<p>E(B&minus;V) comes from the file (DESI), the local SFD98 maps (<code>sfdmap2</code> and <code>SFD_DIR</code>) or the IRSA
dust service (this sends the coordinates to NASA/IPAC), recalibrated by 0.86 (Schlafly &amp; Finkbeiner 2011) unless
<code>sf11=False</code>. In the app: <i>From RA/Dec</i> (the position of the file, or typed RA and Dec in degrees or
sexagesimal), <i>Enter value</i>, or <i>None</i>. A spectrum is never corrected twice.</p>
<pre>spec = spec.with_coordinates("12:34:56.7 +12:34:56").deredden()</pre>
<div class="note">At high Galactic latitude E(B&minus;V) = 0.015&ndash;0.06: without the correction fluxes and L<sub>5100</sub> are
5&ndash;20% too low and the Balmer decrement 2&ndash;6% too high, while ratios of nearby lines (BPT) change by &lt; 0.003 dex.</div>
<h2>Flux unit</h2>
<p>Luminosities and black-hole masses need the flux unit. SDSS and DESI give
10<sup>&minus;17</sup> erg s<sup>&minus;1</sup> cm<sup>&minus;2</sup> &Aring;<sup>&minus;1</sup>; a text file may say so in its header
(&ldquo;flux in 1e-16 erg/s/cm2/A&rdquo;). Otherwise set it in <i>One flux unit of the file is</i> (10<sup>&minus;15</sup>&ndash;10<sup>&minus;19</sup>
or other) or with <code>spec.with_flux_unit(1e-16)</code>.</p>
<h2>Finding the redshift</h2>
<p><b>Find the redshift</b> (<code>fa.find_redshift(spec)</code>) is for a spectrum without a redshift, and a check of one that has
it. The spectrum as read is fitted with galaxy and quasar templates (Yip et al. 2004) and a smooth polynomial at every redshift;
the minimum of &chi;&sup2; is the redshift, as in the SDSS and DESI pipelines. A quasar redshift is then refined on the narrow
H&beta; and [O&#8202;III] lines when they are detected. It takes a few seconds.</p>
<p>The panel shows the best redshift with its error, <b>secure</b> or <b>not secure</b>, and the candidates: the best minima of
both template sets, each with how much worse it fits (&chi;&sup2; per pixel) and how many of the emission peaks of the spectrum sit on
a known line there. <b>Click a candidate</b> to use it. The figure shows &chi;&sup2; against redshift and the spectrum with the
template and the lines of the best redshift marked. A wrong redshift is nearly always one line taken for another, so look at the
candidates when the answer is not secure.</p>
<p>On 540 SDSS spectra of nine classes (z = 0.01&ndash;2.5, median S/N 14) the best redshift was right for 97%; of those called
secure (96%), 99.8% were right. Galaxies and AGN below z &asymp; 1.5 were right in 98&ndash;100% of the cases; quasars above
z &asymp; 1.7 in 83&ndash;90%, with the right value among the candidates for 95%. It cannot identify a single line in a narrow
range, nor featureless spectra; the galaxy templates reach z &asymp; 1 in an optical spectrum; stars are not included.</p>
<h2>Instrumental resolution</h2>
<p>The instrument broadens every line; FANTASY removes that width in quadrature, so fitted widths are intrinsic. SDSS and
DESI files carry the resolution along the spectrum. Text and plain FITS files do not: the widths then include the
instrument, and a narrow line can become narrower than a pixel. In <i>The lines are broadened by the instrument by</i> choose
<b>FWHM in km/s</b> (of an unresolved sky or arc line) or <b>resolving power R</b> (R = 2000 is 150 km/s); the note shows the
pixel size to compare with. In code: <code>spec.with_resolution(150)</code> or <code>spec.with_resolution(R=2000)</code>.</p>
<p>After a fit, <b>A line narrower than the data can sample</b> on the Results page lists widths below 1.2 pixels
(<code>r.undersampled()</code>). Line profiles are averaged over each pixel, as the spectrograph records them, so the flux of
such a line is kept wherever its peak falls; but a width below about a pixel barely changes the data, so it is poorly measured,
and the line can fit single noisy pixels. Set the resolution, or raise the lower limit of the width.</p>
<h2>Range and masks</h2>
<dl class="params">
<dt>Wavelength range</dt><dd>the fitted range (rest frame); presets set a sensible default.</dd>
<dt>Telluric bands, sky lines</dt><dd><code>spec.mask_sky()</code>: the telluric A, B and H<sub>2</sub>O absorption bands and the strongest night-sky emission lines (observed frame).</dd>
<dt>Narrow absorption</dt><dd><code>spec.mask_absorption()</code>: finds intervening absorbers (Mg&#8202;II, C&#8202;IV doublets, &hellip;)
and masks them; essential in the UV.</dd>
<dt>Ranges on the plot</dt><dd>drag on the spectrum to mask a range (<code>spec.mask((5570, 5585))</code>).</dd>
<dt>Apply to</dt><dd>with several epochs, change one epoch only.</dd>
</dl>
<p>The plot shows the <i>Raw</i> or the <i>Prepared</i> spectrum of the chosen <i>Epoch</i>, with the masked ranges hatched.</p>
<h2>In Python</h2>
<pre>spec.crop(4000, 7000)          # restrict the range
spec.subtract_host()           # remove the host with galaxy + quasar eigenspectra (Yip et al. 2004)
spec.to_air()                  # vacuum to air wavelengths</pre>
"""),

("model", "The app, step by step", "3 &middot; Model", """
<div class="lead">A model is a sum of components. Start from the recommended preset, a preset of your choice,
an empty model or a saved model, then add, remove and tune components.</div>
<h2>Classification card</h2>
<p><b>Classify spectrum</b> runs <code>fa.classify</code> (starlight, broad H&alpha;, BPT diagrams; see
<a data-doc="classify">Classification</a>) and recommends a model; <b>Use recommended model</b> applies it.</p>
<h2>Compare models on this spectrum</h2>
<p><b>Compare models</b> fits variants of the recommended preset in the background (about a minute): one broad component more
and fewer, and each on/off option of the preset switched on its own (host galaxy, outflow wing, coronal lines, O&#8202;I 6046,
Fe&#8202;II, Balmer continuum, ...), then the preset with every change that helped. Options that change the fitted range (the red
side) are left out, because the models are ranked by <b>BIC = &chi;&sup2; + k ln n</b>, which needs the same pixels: every free
parameter costs ln n, and <b>&Delta;BIC &lt; &minus;10</b> is strong evidence for a change (Kass &amp; Raftery 1995).</p>
<dl class="params">
<dt>recommended</dt><dd>the lowest BIC among the models that put no new parameter at a limit (a limit the preset already hits is
not held against them). A change must beat the preset by &Delta;BIC &lt; &minus;10; otherwise the preset stays recommended.</dd>
<dt>lowest BIC</dt><dd>shown when it is not the recommended one, with the number of its parameters at a limit: those values are set
by the limits, not by the data.</dd>
<dt>The model choice matters here</dt><dd>across the five best models, the main measurements (broad FWHM, M<sub>BH</sub>,
R<sub>Fe&#8202;II</sub>, Balmer decrement) move by more than their fit errors: quote the spread, or use <i>Model uncertainty</i> in the
Fit step.</dd>
</dl>
<p>Click a card or a row of <i>All candidates</i> to see its fit, its components, the comparison with the preset and its code.
<b>Use this model</b> runs that code as a cell of the session; <b>Customize&hellip;</b> opens it in the model builder.</p>
<h2>The model builder</h2>
<p><b>+ Your own model</b> (or <b>Open in the model builder</b> under the components) opens a full-screen workspace, starting from the
recommended model, the current model, any candidate or an empty one. The model is taken apart into its components, each with
every argument of its constructor:</p>
<ul>
<li>switch a component off (it keeps its settings), duplicate it (a third broad component is one click), remove or rename it, or
<b>+ Add component</b>: any component type, or a named line set as a new line component;</li>
<li>for a line component: its role, profile, lines (by name, ion or wavelength, or your own line as <code>name@5310</code>), fixed or
bounded flux ratios, FWHM and shift as start value and limits, and the same shift and width as another component;</li>
<li><b>Parameters</b>: the start value, limits, fixed flag and tie (with a factor) of every parameter, to any other parameter of
the same kind;</li>
<li>click the spectrum to see the catalogued lines there and add one to a component; the plot is the initial guess, split by
component, and follows every edit; undo and redo with &#8984;Z and &#8679;&#8984;Z; the fit range at the top.</li>
</ul>
<p>The code under <b>Code</b> is exactly the model: <b>Use this model</b> runs it as a session cell, and <b>Fit and rank it</b> fits it on
the spectrum in the background and adds it to the comparison as <i>your model</i>. <b>&larr; Back</b> (or Esc) keeps your edits:
<b>Continue your model</b> brings them back.</p>
<h2>Start from</h2>
<dl class="params">
<dt>A preset</dt><dd>broad components (1&ndash;3), host galaxy, outflow wing, stratified BLR, coronal lines, O&#8202;I 6046, the red side
to 7500 &Aring;. See <a data-doc="presets">Presets</a>.</dd>
<dt>Your own model</dt><dd>a power-law continuum, to which you add components.</dd>
<dt>A saved model</dt><dd>one of your saved models, or a model file from someone else. See <a data-doc="saved-models">Saved models</a>.</dd>
</dl>
<h2>Components</h2>
<p><b>+ Add component</b> opens one dialog for everything: emission or absorption lines from the line list (role broad,
narrow or outflow; profile Gaussian, Lorentzian, Voigt, Gauss&ndash;Hermite or stratified; width and shift limits;
kinematics shared, per line, or tied to another component), continua, Fe&#8202;II and high-order Balmer templates, the host
galaxy, absorption troughs and dust. Click a component to change its profile, tie its shift and width to another one, or
edit each parameter: initial value, limits, fixed, tied (<a data-doc="params">Parameters</a>).</p>
<p><b>Is a tie safe?</b> After a fit of one spectrum, the <i>Shift and width</i> choice of a component says for every possible tie
whether it would change the result, judged on that fit where both are free, without a new fit (<code>fa.model_confidence</code>): the
probability that the FWHM and flux of the main broad line stay within 5%. <i>safe</i> (above 95%), <i>changes it a little</i>,
<i>changes the result</i>, or <i>rejected by the data</i> (the two clearly differ: fit again with the tie and compare in Checks).
It is conditional on the free model being right (<a data-doc="compare">Model comparison</a>).</p>
<p>The plot shows the <b>initial guess</b> on the reference spectrum with the selected component highlighted: if the guess
is far off, the limits are probably wrong.</p>
<div class="tip"><b>Save model&hellip;</b> above the component list keeps the model (with every limit and tie) to use it
again; <b>Download</b> gives the JSON file to share.</div>
"""),

("fit-step", "The app, step by step", "4 &middot; Fit", """
<div class="lead">Check what will be fitted, choose how epochs are treated and how uncertainties are computed,
then run the fit.</div>
<h2>Several epochs</h2>
<dl class="params">
<dt>First, try the model on one epoch</dt><dd><b>Fit this epoch</b> fits one epoch alone (by default the one of highest S/N; any
epoch can be chosen) with the current model: <code>r = fa.fit(spectra[k], model)</code>. The Results step then shows that fit with all
its checks (residuals, robustness, compare, weak Fe&#8202;II, simulations), and a line that leads back: <b>Change the model</b>, or
<b>Fit all epochs</b>. Repeat until the model is right for the object; a look at the faintest epoch shows whether it is too rich for
the weak ones.</dd>
<dt>Start every epoch from the trial fit</dt><dd>shown once an epoch was tried: the epochs are fitted with
<code>model.starting_from(r)</code>, so each also starts from the values of the trial fit and keeps the better of that and its own
data-driven guess. The limits, fixed values and ties are those of the model. If the model was changed after the trial, the
parameters it still shares with the trial are used.</dd>
<dt>The same in all epochs, or one per epoch?</dt><dd>set in the <b>Model step</b> of a multi-epoch analysis, in the table under the
component: for every component, its <b>fluxes</b> and its <b>shape</b> (shift, width, profile parameters, slope) are <i>the same</i> in
all epochs or fitted <i>per epoch</i>. The shared fluxes are what the flux scale of each epoch is measured on, so they must not vary:
the narrow lines and a stellar host by default. A shared shape says the line looks the same in every epoch and only its flux changes:
the default for the broad lines and Fe&#8202;II, since free shapes per epoch trade flux with the continuum and can wash out the
reverberation signal. Everything else (the continuum, the outflow wing) is per epoch by default. These are assumptions about the
object; <i>changed</i> marks what differs from the default, <b>Defaults</b> restores it, and the Fit step repeats the choice in one
line. In code: <code>fa.fit_epochs(spectra, model, shared=("narrow.*", "broad.fwhm", ...))</code>. For a stellar host the flux choice is
<i>the same</i> or <i>amplitude per epoch</i>: one host spectrum with its own amplitude in every epoch, for spectra taken through
different apertures or in different seeing (<code>epoch_scaled=("host",)</code>).</dd>
<dt>Together (recommended)</dt><dd><code>fa.fit_epochs</code>: one joint fit, narrow lines (and a stellar host) shared, one flux scale per epoch
fixed by the shared lines (intercalibration). Best for light curves and reverberation mapping.</dd>
<dt>Each on its own</dt><dd><code>fa.fit_many</code>: independent fits in parallel, with the calibration of each spectrum as it is.</dd>
<dt>Wavelength scale of the epochs</dt><dd><b>fit a small offset per epoch</b> (recommended;
<code>fa.fit_epochs(..., wavelength_offsets=True)</code>): every epoch gets a velocity of up to &plusmn;150 km/s, added to all the shifts
of the model in that epoch, with the reference epoch at 0. The shared narrow lines measure it, as they measure the flux scale. The
epochs are never on exactly the same wavelength scale: the calibration differs by a few km/s (tens for slit spectra), and when the
files carry slightly different redshifts (SDSS: about 20 km/s rms) each spectrum is moved to the rest frame with its own; the note
under the choice says by how much they differ. In simulations, unmodelled offsets of 30 km/s raised the error of the flux scales
from 4% to 17%; with the offsets fitted it stayed at 3%, and they were recovered to 3 km/s. The table of all epochs lists the
offsets, and the Results tile <b>Redshift, narrow lines</b> gives the redshift of the object from the narrow lines of all epochs
together (<code>me.redshift()</code>), with the rms of the offsets. <b>as prepared</b> fits none.</dd>
<dt>Resolution of the epochs</dt><dd><b>fit an extra broadening per epoch</b> (<code>fa.fit_epochs(..., epoch_broadening=True)</code>):
every epoch gets an extra line width, a FWHM added in quadrature to every line width and to the stellar dispersion (up to 300 km/s; an
epoch may also be sharper than the reference, down to 80% of its own resolution), measured on the shared narrow lines. The resolution
differs between epochs (slit width, seeing, another spectrograph) and is often not known well; when it differs and is not modelled,
the shared narrow lines fit some epochs badly and pull their flux scales. In simulations with up to 250 km/s of extra broadening the
flux scales were off by up to 7%; with the broadening fitted, below 1%, and the broadening was recovered to about 10 km/s. Each
epoch's widths are then intrinsic (its spectrum carries the broadened resolution); the table of all epochs lists the broadening and
the Results tile gives its range. <b>as prepared</b> fits each epoch at the resolution of its spectrum.</dd>
<dt>Flux reference epoch</dt><dd>the epoch whose flux scale is 1; by default the one of highest S/N, since its uncertainty enters
every other epoch's calibration.</dd>
<dt>Calibration between epochs</dt><dd>one flux scale per epoch; scale and tilt; or scale, tilt and curvature
(<code>calibration_degree</code> 0, 1, 2): see <a data-doc="multi">Multi-epoch fitting</a>.</dd>
</dl>
<p>For a sample, <i>Model for each spectrum</i> is <i>automatic</i> (classify each and use the recommended model; a spectrum that ends
too blue for H&beta; gets Mg&#8202;II or C&#8202;IV) or one model for all.</p>
<h2>Uncertainties</h2>
<table class="doc"><tr><th>choice</th><th>method</th><th>cost</th></tr>
<tr><td>Fast</td><td>covariance from the Jacobian at the best fit</td><td>seconds</td></tr>
<tr><td>Monte Carlo</td><td>100 refits of the spectrum with noise added</td><td>&times;100</td></tr>
<tr><td>MCMC</td><td>Hamiltonian Monte Carlo posterior; corner plot</td><td>minutes</td></tr>
<tr><td>None</td><td>best fit only</td><td>fastest</td></tr></table>
<p>See <a data-doc="uncertainties">Uncertainties</a> for when each is appropriate.</p>
<h2>Advanced options</h2>
<dl class="params">
<dt>Scale errors</dt><dd>inflate the errors by &radic;&chi;<sup>2</sup><sub>&nu;</sub> when &chi;<sup>2</sup><sub>&nu;</sub> &gt; 1.</dd>
<dt>Model uncertainty</dt><dd>fit the alternative decompositions and add their spread as a systematic error (<a data-doc="compare">Model comparison</a>).</dd>
<dt>Absorption troughs</dt><dd>find troughs in the residuals and refit with them (<a data-doc="absorption">Absorption troughs</a>).</dd>
<dt>Rescale the errors of each epoch</dt><dd>by &radic;&chi;<sup>2</sup><sub>&nu;</sub> of its own fit, for epochs whose errors are under- or overestimated.</dd>
<dt>Shared parameters by name pattern</dt><dd>for experts: patterns such as <code>narrow.*, feii.fwhm</code> that replace the table of the
Model step (below) when the field is not empty.</dd>
<dt>Processes</dt><dd>parallel processes for independent fits and Monte Carlo.</dd>
</dl>
<p>The run button is in the bar at the bottom of the page, between <b>&larr; Model</b> and <b>See the results &rarr;</b>, so it is always
in view; it is the highlighted button until a fit of the current settings exists, then <b>See the results</b> is. When the fit is done,
its quality and the parameters that ended at a limit are listed, with a link to the component in the Model step.</p>
"""),

("results-step", "The app, step by step", "5 &middot; Results", """
<div class="lead">The summary at the top gives the fit quality and the main numbers with errors; parameters at a limit
are flagged with a button that opens their component. A joint fit adds <b>Flux scale between epochs</b>: the median error of the
flux scale of an epoch (<code>me.calibration_error</code>), marked when it is above 2%, since this noise enters every light curve.</div>
<h2>Views</h2>
<dl class="params">
<dt>Fit</dt><dd>data, model, components and residuals (drag to zoom, double-click to reset; click the legend to hide a curve).
Below, four tables: <i>Lines</i> (by component, with flags for undetected lines and tied widths; <i>Hide undetected</i>),
<i>Continuum and host</i> (continuum, luminosities, host properties and the black-hole mass of every recipe that applies),
<i>All parameters</i> (value, error, limits, status) and <i>All epochs</i>.</dd>
<dt>Degeneracies</dt><dd>which parameters trade against each other (<code>r.cov</code>, or <code>r.samples</code> after MCMC).
The free parameters sit around a wheel, grouped by component; a ribbon joins two whose correlation |&rho;| is at least the threshold,
red when they move together, blue when one goes up as the other goes down, wider for a stronger one. A dot's size is the relative error,
and a red ring marks a parameter at a limit. By default only pairs <i>between</i> components are shown (the ones that decide the
decomposition); <i>all pairs</i> adds those within a component (width against flux of one line). Hover a parameter for its partners,
a ribbon for the pair: the side panel gives the 68% and 95% regions of the two together and why they are linked. A thin diagonal
ellipse (|&rho;|&nbsp;&ge;&nbsp;0.9) means the data fix only a combination of the two: report their sum, tie one, or treat each error
as optimistic.</dd>
<dt>Atlas</dt><dd>(a sample, or several epochs) every fit as a small card: the data in grey, the model in the colour of its class
(for epochs, of its date, early to late), the broad lines and the host shaded, the residuals in &sigma; underneath. <i>Order by</i> puts
the cards in order of a number (for a sample &chi;&sup2;<sub>&nu;</sub>, residual structure, redshift, broad FWHM, R<sub>Fe&#8202;II</sub>,
M<sub>BH</sub>, Eddington ratio, luminosity, [O&#8202;III]/H&beta;; for epochs the date, continuum luminosity, broad-line flux, FWHM, fit
quality), and the strip above places every fit on that axis. <i>Worth a look</i> keeps the fits flagged for &chi;&sup2;<sub>&nu;</sub>&nbsp;&gt;&nbsp;2,
a run of 40 pixels more than 10&sigma; off the model (noise alone gives about 4&sigma;), a broad width at a limit, R<sub>Fe&#8202;II</sub>&nbsp;&gt;&nbsp;3,
a black-hole mass outside 10<sup>5.5</sup>&ndash;10<sup>10.5</sup>, or three or more parameters at limits; hover a flag for the reason.
<i>Common wavelength axis</i> lines the lines up down the grid, <i>same flux scale</i> (on for epochs) shows the variability. The class
chips filter a sample (alt-click: one class only). A card opens its fit in <i>Fit</i>. The Eddington ratio uses bolometric corrections
of 9.26 at 5100&nbsp;&Aring; and 5.15 at 3000&nbsp;&Aring;.</dd>
<dt>Anatomy</dt><dd>the fit taken apart (<code>r.model_flux(components=True)</code>): the components stacked from the continuum up
(host, Balmer continuum, Fe&#8202;II, broad, narrow, outflow), with the data over them. The <i>flux budget</i> above gives the share of
each in the flux of the range shown; hovering the spectrum gives the composition at that wavelength (a ring and a table, with data, model
and residual). <i>Peel layers</i> takes components off the top and shows the data minus them: what is left for the rest to explain
(peel the lines to see whether the continuum and Fe&#8202;II follow the data). Below, the residual (data&nbsp;&minus;&nbsp;model)/&sigma;
with the &plusmn;1&sigma; band, &plusmn;3&sigma; lines and a running mean over 15 pixels: noise averages out, a component the model
misses stays as a bump; the count of pixels beyond 3&sigma; is compared with what noise alone gives. Drag to zoom, double-click for the
full range, the windows (H&beta;, H&alpha;, ...) above; the strip at the bottom shows the whole spectrum. Click a legend chip to hide
a component, hover it (or click a budget segment) to isolate it.</dd>
<dt>Checks</dt><dd>five checks, numbered in the order to use them. Each is one line with its run button and a status
(<i>not run</i>, <i>nothing found</i>, <i>3 features</i>, <i>2 biased</i>, ...); click its title to open or close it. A check that needs
attention, or that is running, is open; the others are closed until you open them. The number on the <b>Checks</b> tab counts the
checks that need attention. 1. <b>Does the model miss anything?</b>: the residual
analysis (<code>fa.analyze_residuals</code>, threshold 3&ndash;6&sigma;) marks missing lines, wings, absorption, continuum shape and
sky residuals, suggests a change for each, and <b>Refit with the changes</b> applies the ones you keep. For one spectrum it
also shows <b>Lines to add or separate</b> at once, without running anything (<code>fa.suggest_lines(r)</code>, a few milliseconds): every library line the
model lacks is tested as a Gaussian with the fitted width and shift of a line component the model already has (forbidden and coronal lines
with the narrow and outflow components, permitted lines also with the broad ones), its best flux and the &chi;&sup2; it removes computed directly
on the residuals (no refit), scaled by the scatter of the residuals. A line is listed when it lowers the BIC by more than 10 (one parameter:
its flux) at S/N &ge; 3; lines are taken one at a time, each subtracted before the next is judged, so a blend is not counted twice. Lines on
a line the model already has, on night-sky lines or covered by the Fe II and high-order Balmer templates are not tested. <b>Add &amp; refit</b>
adds the line and fits again in one cell, with the options of the last fit, and you stay on Results: the code is one
<code>fa.Lines</code> per kinematic source, its shift and FWHM tied to that component, then <code>r = fa.fit(spec, model)</code>.
The same list tests the lines the model has: all lines of a component share one velocity, and a line (with its doublet) whose
velocity differs, typically [O III] blueshifted by a few tens of km/s against the other narrow lines, leaves an excess on one side
and a deficit on the other. The residuals projected on the velocity derivative of the line measure its offset; when its own
shift and width lower the BIC by more than 10 (two parameters), the row says by how much, and <b>Separate &amp; refit</b> moves
it to a component of its own (<code>model.split_lines</code>; fluxes, ratios, limits and ties go with it). The residual analysis
gives the same suggestion for a feature on such a line. <b>Apply all &amp; refit</b> does every change of the list at once; after
the trial fit of a multi-epoch analysis, only that epoch is fitted again. The list then comes from the new fit.
2. <b>How robust is this fit?</b> (<code>fa.robustness(r)</code>, one spectrum): <b>Check robustness</b> fixes each free width, shift and
ratio at values across its limits and refits everything else, from the best fit and from a fresh guess (a few hundred fits, 1&ndash;3 minutes).
It shows your results (broad-line FWHM, &sigma;, flux and shift, R<sub>FeII</sub>, total [O III]) over all acceptable fits (&Delta;&chi;&sup2; &le; 1 and &le; 4)
next to their error bars, and the parameters to look at: <i>weakly bounded</i>, <i>slides to its limit</i>, <i>second minimum</i>, with a small
&Delta;&chi;&sup2; scan for each. If the scans find a better fit, <b>Fit again from the better solution</b> starts the model there.
3. <b>Compare with another fit</b> (<code>fa.compare_fits(a, b)</code>): <b>Pin this fit</b> keeps it under a name; change the model (continuum,
line profile, ties), fit again and <b>Compare with current</b>. It shows whether the data can tell the two fits apart (RMS difference of the
total models in units of the error, &chi;&sup2; on the same pixels), the results of both with the change in errors, a figure with both fits,
where the flux went between the components and both broad profiles with their half-maximum crossings (PDF), and a warning when a broad
profile has a shoulder at half maximum, so that its FWHM is fragile and &sigma; is the safer width.
4. <b>Is weak Fe&#8202;II inflating the fit?</b> (<code>fa.prune_feii(r)</code>, one spectrum or epochs fitted one by one, with Fe&#8202;II and errors): the table lists every
Fe&#8202;II group with its strength, error and whether it is detected (2&sigma;). <b>Drop undetected groups</b> fixes the groups below the
threshold (2&sigma; or 3&sigma;) at zero and fits again; the new fit becomes the result, and a table shows what changed (Fe&#8202;II flux,
R<sub>Fe&#8202;II</sub>, the broad line, the continuum, BIC). <b>Prune again from the full fit</b> applies another threshold to the original fit;
<b>Back to the fit with every group</b> restores it. For epochs fitted one by one the card lists every epoch with its number of groups and how many are
below 2&sigma;; pruning runs on all epochs (<code>fits = fa.prune_feii(fits)</code>) and the light curves, lags and Main sequence tab then
use the pruned fits. A group dropped in one epoch and kept in the next adds scatter to the Fe&#8202;II light curve. A joint fit has no
pruning step. For a sample, tick <b>Drop undetected Fe&#8202;II groups and fit again</b> in the Fit step
(<code>fa.fit_sample(files, prune_feii=True)</code>): the list gets a <i>Fe II dropped</i> column, and <b>before pruning Fe&#8202;II</b> on
the main sequence diagram shows where the spectra were with every group. Why and when: <a data-doc="feii">Fe&#8202;II templates</a>, <i>Weak Fe&#8202;II</i>.
5. <b>Would this fit recover the truth?</b> (<code>fa.simulation_check(r)</code>, one spectrum): <b>Run simulations</b> takes the fitted
model as the truth, draws 30, 60 or 120 spectra from it with the noise of the spectrum and fits each again from scratch as the spectrum
was fitted (about a minute for 60). For every result the table gives the value in the fit, the mean of the simulations, the bias in units
of the scatter, the scatter (the real statistical error), the error bar the fits report and how often the truth lies within it (68% for
correct errors). <i>Biased</i>: the mean differs from the truth by more than 0.3 of the scatter and by more than chance allows. If the fit
was pruned of Fe&#8202;II groups (check 4), every simulation is pruned too, so running it before and after shows whether pruning helped
for this object. It tests the procedure for this model; what the model leaves out is for checks 1 and 3.
5 (joint fit). <b>Is the calibration between the epochs right?</b> <b>Check the calibration</b> (<code>fa.calibration_check(me)</code>)
fits every epoch again on its own, its spectrum divided by the flux scale of the joint fit, with everything free. If the scales are
right and the narrow lines really the same in every epoch, the fluxes of the shared lines agree within their errors: the table gives
per line the &chi;<sup>2</sup>/dof of the epochs around their mean, its p-value, the scatter against the error and the <i>excess</i>, the
scatter to add in quadrature for &chi;<sup>2</sup>/dof = 1: calibration noise that is not in the error bars. Epochs more than 3&sigma; off
are listed. With one strong narrow line the scales are set by it, so that line agrees almost by construction; the check has power
through the other lines and the widths (in simulations it found unmodelled resolution differences with &chi;<sup>2</sup>/dof = 26, and
passed once the broadening was fitted). <b>Run simulations</b> (<code>fa.simulation_check(me, n=20)</code>) draws sets of epochs from
the joint fit with their noise, fits each set jointly again exactly as the data were, and compares the flux scales, the broad-line
flux of every epoch (the light curve), the offsets and broadening, and the shared widths with the truth: the bias and the scatter of
(fit &minus; truth) in units of the error bar (0 and 1 for right error bars) and how often the truth is within one error bar. Each set is
a whole joint fit, so it takes minutes.</dd>
<dt>Correlations</dt><dd>several epochs (joint or one by one). Opening the tab measures every epoch once (<code>fa.epoch_table</code>: broad flux, FWHM, &sigma;,
shift and EW of every broad line, the continuum and log L, R<sub>Fe&#8202;II</sub>, [O&#8202;III]; all cores, about 20 s for 76 epochs). After that everything is immediate.
<b>The map</b> shows the Spearman &rho; of every pair of quantities, with the number where p &lt; 0.05 and &ldquo;=&rdquo; where a pair is the same number twice; with many pairs some reach p &lt; 0.05 by chance.
<b>Click a square</b> (or choose the two quantities at the top) to see that relation with its fit: slope and intrinsic scatter by maximum likelihood with errors in both quantities, the 68% range of the line
from a bootstrap of the epochs, fluxes, widths and EWs in log10 so that the slope is a power law. <b>Hover</b> over a point for its date and values, <b>click</b> it to leave that epoch out (Restore brings them back).
<b>Continuum earlier by</b> (observed-frame days, the rest-frame lag &times; (1 + z)) sets each line epoch against the continuum interpolated that much earlier between the epochs
(<code>fa.correlations.shift_continuum</code>), and the map and the fit follow as you drag; without it the quantities are taken at the same epoch, which biases the intrinsic Baldwin effect when the lag is comparable to the variability.
Look for the intrinsic Baldwin effect (EW against luminosity), the breathing of the BLR (width against luminosity) and a constant [O&#8202;III] (the flux calibration).
PDF, PNG and SVG of the relation or of the grid of all pairs, and their Python code, use the current lag and left-out epochs.</dd>
<dt>Publication figure</dt><dd>ranges, title, journal and width (MNRAS, A&amp;A, ApJ columns), zoom panels under the spectrum (quick H&beta;, H&alpha;,
Mg&#8202;II buttons; the whole fit, or with the continuum, Fe&#8202;II and host subtracted), legend position (above, inside where it is free, beside), font size, line labels; for every kind of component whether it is
drawn, its colour, fill, line width and style. PDF, PNG, EPS, SVG, and the Python code of the figure to copy. See <a data-doc="figures">Figures</a>.</dd>
<dt>Line profiles</dt><dd>tick lines (broad lines are listed first) and choose <b>Each in a panel</b> (each line alone with its components,
FWHM, &sigma; and flux, the half-maximum points and residuals) or <b>Overlaid</b> (the profiles on one velocity axis, scaled to their peaks,
to compare widths and shapes; with several epochs, <b>Compare: Every epoch</b> overlays the first chosen line of every epoch, in time order); a velocity range;
PDF, PNG, EPS, SVG and the Python code.</dd>
<dt>Posterior</dt><dd>with Monte Carlo or MCMC errors: the corner plot of chosen parameters.</dd>
<dt>Main sequence</dt><dd>with Fe&#8202;II and broad H&beta;: R<sub>Fe&#8202;II</sub> against FWHM(H&beta;) for every epoch; <i>Zoom to the
epochs</i> (<a data-doc="feii-ms">Fe&#8202;II and the main sequence</a>).</dd>
<dt>Classification</dt><dd>the <b>final class, from the values of the fit</b> (<code>fa.classify_fit(r)</code>, after every fit, for every epoch): the broad lines of the fit and its narrow-line ratios and EW(H&alpha;) on the BPT and WHAN diagrams, with the measurements. When the spectrum was classified in the Model step, that preliminary class is shown beside it (the table of epochs marks a disagreement with &ne;), with its three steps (starlight, broad H&alpha;, emission lines). Figures to download.</dd>
<dt>Light curves &amp; lags</dt><dd>(joint fit, or epochs fitted one by one through <code>fa.EpochSet(fits)</code>, which are not intercalibrated) for several epochs.
Choose a <b>driver</b> and a <b>responder</b> (a continuum window, the fitted AGN continuum, the Fe&#8202;II flux, or the flux, FWHM or EW of a broad line; the
chips give common pairs such as 5100&nbsp;&Aring; &rarr; H&beta;, or H&beta; flux &rarr; FWHM for breathing; &#8644; swaps them). Both are shown as
the % change from the mean of the epochs shown, with their fractional variability F<sub>var</sub>; the season buttons zoom to one
observing season, hovering gives an epoch and clicking opens it. The <b>&tau;</b> slider (or a drag along the curve below it, or
<b>sweep</b>) shifts the responder back by &tau; over the driver. The lag of the pair is measured with every method
(<code>fa.rm.ccf_lag</code>, <code>vn_lag</code>, <code>zdcf_lag</code>, <code>gp_lag</code>; a few seconds per pair), and <b>adopted lag</b>
picks the method whose own curve is drawn: the ICCF r(&tau;), live for the epochs shown; the ZDCF points (pyZDCF) with their
errors; the von Neumann statistic, whose minimum is the lag (its axis runs downwards); or the DRW posterior. The handle and
the readout follow that curve, the rows under it show each method's lag distribution, and clicking a method adopts it.
Continuum windows (<code>me.window_continuum()</code>) are measured on the data in line-free rest-frame windows (H&beta;:
4435&ndash;4700 and 5100&ndash;5535&nbsp;&Aring;; H&alpha;: 6000&ndash;6250 and 6800&ndash;7000&nbsp;&Aring;; Shen et al. 2011) inside the fitted
range: the mean of the intercalibrated flux minus the fitted lines and Fe&#8202;II (the blue H&beta; window lies on the Fe&#8202;II
4570 blend). They include the host light, which dilutes the variability but not the lag; the fitted continuum is
evaluated at one wavelength from the model. These lags use the two light curves alone. For the published lag, rms spectrum and mass, reverberation mapping: choose the
line (the responder's, when it is a broad-line flux), the adopted method (<code>primary</code>, from the switch above), the lag methods (ICCF, von Neumann, ZDCF, DRW), an MJD range and detrending, then <b>Measure lag</b>
(<a data-doc="reverberation">Reverberation mapping</a>). The table of lags gives the <b>significance</b> of each: the share of its
distribution (all Monte Carlo or posterior samples, aliases included) on its side of zero, as a one-sided Gaussian equivalent
(<code>fa.rm.lag_significance</code>). A lag of 6.0 (&minus;4.1, +4.9) days is about 1.3&sigma;: not a detection. The warnings say when the
adopted lag is below 2&sigma;, when the light curves are weakly correlated (r<sub>max</sub> &lt; 0.5), when methods disagree by more than
2&sigma; and when much of a distribution sits in seasonal aliases.</dd>
</dl>
<h2>Downloads</h2>
<p>The fitted lines (CSV or FITS), the model and its components (data, model, every component, host, AGN, spectrum minus host
and, with extinction, the transmission and intrinsic components), the fit figures (PDF), the script, the notebook and the project.</p>
<p><b>Report</b> is one HTML file of the current fit, to keep with the results or send to a colleague (it prints to PDF from the
browser): the figure, the fit statistics and warnings, the key results, the lines (with upper limits), every check that was run on
this fit with its outcome (residuals, robustness, comparison, Fe&#8202;II pruning, simulations; for a joint fit the calibration check,
its simulations and the lag), the settings of each step and the Python code that repeats the analysis.</p>
<p>In the lines table an undetected or weak line (below 2&sigma;) shows an <b>upper limit</b>: <code>r.line_limit(line)</code>, the fitted
flux (if positive) plus 3&sigma;<sub>F</sub>, where &sigma;<sub>F</sub> is the flux error of a line with the fitted width and shift from
the noise under it alone; with MCMC errors, the 95% limit of the posterior. Hover over it to see which.</p>
"""),

# ======================================================================== Modelling
("components", "Modelling", "Components", """
<div class="lead">Every component returns its flux on the wavelength grid together with exact derivatives with respect to its
parameters. Amplitudes (fluxes) enter linearly; widths, shifts and shapes are nonlinear.</div>
<h2>Emission lines: <code>fa.Lines</code></h2>
<p>A group of lines with shared kinematics (or one shift and width per line with <code>tie_kinematics=False</code>).
Parameters: the integrated flux of each line <code>&lt;name&gt;.&lt;line&gt;.flux</code>, the velocity shift
<code>&lt;name&gt;.shift</code> and FWHM <code>&lt;name&gt;.fwhm</code> (km/s). The line centre and Gaussian width are</p>
<div class="eq">&lambda;<sub>c</sub> = &lambda;<sub>0</sub>(1 + <span class="frac"><span>v</span><span>c</span></span>),
&nbsp;&nbsp; &sigma;<sub>&lambda;</sub> = &lambda;<sub>c</sub>&nbsp;<span class="frac"><span>&radic;(FWHM<sup>2</sup>/8 ln 2 + &sigma;<sub>inst</sub><sup>2</sup>)</span><span>c</span></span></div>
<p>so fitted widths are intrinsic: the instrumental resolution &sigma;<sub>inst</sub> of the spectrum is added in quadrature
(exactly for Gaussian, Lorentzian and Voigt profiles). Every profile integrates to one, so the flux parameter is the line flux.
Each pixel gets the mean of the profile over its width, not its value at the pixel centre (exactly for Gaussians, by
Simpson's rule for the other profiles): point values make lines a few pixels wide look wider, by 7% at FWHM 120 km/s and
3% at 200 km/s on SDSS (69 km/s pixels) with a 150 km/s resolution. <code>FANTASY_PIXEL=0</code> returns to point values.</p>
<table class="doc"><tr><th>profile</th><th>parameters</th><th>shape</th></tr>
<tr><td>gaussian</td><td>shift, fwhm</td><td>&prop; exp[&minus;(&lambda;&minus;&lambda;<sub>c</sub>)<sup>2</sup>/2&sigma;<sub>&lambda;</sub><sup>2</sup>]</td></tr>
<tr><td>lorentzian</td><td>shift, fwhm</td><td>&prop; [1 + (2(&lambda;&minus;&lambda;<sub>c</sub>)/&Gamma;)<sup>2</sup>]<sup>&minus;1</sup>, wings &prop; &Delta;&lambda;<sup>&minus;2</sup></td></tr>
<tr><td>voigt</td><td>shift, fwhm, fwhm_l</td><td>Gaussian &otimes; Lorentzian (Faddeeva function)</td></tr>
<tr><td>gauss-hermite</td><td>shift, fwhm, h3, h4</td><td>Gaussian &times; [1 + h<sub>3</sub>H<sub>3</sub>(y) + h<sub>4</sub>H<sub>4</sub>(y)] (van der Marel &amp; Franx 1993): skew and peakedness</td></tr>
<tr><td>stratified</td><td>shift, fwhm, span, gamma</td><td>a virialised BLR summed over radius (<a data-doc="stratified">Stratified BLR</a>)</td></tr></table>
<p>Options: fixed ratios <code>ratios={"[OIII]4959": ("[OIII]5007", 1/2.98)}</code>, upper limits
<code>max_ratios={"[NII]6583": ("Ha", 3)}</code>, <code>absorption=True</code> for absorption lines, custom lines as
<code>("name", vacuum_wavelength)</code>. The library of lines is in <i>Reference &rarr; Line list</i>; the FANTASY 1 lists are sets
(<code>fa.line_set("coronal")</code>, <code>fa.LINE_SETS</code>) and a CSV list of your own is read by <code>fa.read_line_list(path)</code>.</p>
<h2>Continua</h2>
<table class="doc"><tr><th>component</th><th>model</th><th>parameters</th></tr>
<tr><td><code>PowerLaw</code></td><td>f = A (&lambda;/&lambda;<sub>p</sub>)<sup>&alpha;</sup></td><td>amp, index</td></tr>
<tr><td><code>BrokenPowerLaw</code></td><td>&alpha;<sub>blue</sub> below the break, &alpha;<sub>red</sub> above; optionally fitted break</td><td>amp, index_blue, index_red</td></tr>
<tr><td><code>Polynomial</code></td><td>&Sigma; a<sub>k</sub> [(&lambda;&minus;&lambda;<sub>p</sub>)/s]<sup>k</sup></td><td>linear coefficients</td></tr>
<tr><td><code>Blackbody</code></td><td>f = A B<sub>&lambda;</sub>(&lambda;, T) / B<sub>&lambda;</sub>(&lambda;<sub>p</sub>, T)</td><td>amp, temperature</td></tr>
<tr><td><code>BalmerContinuum</code></td><td>see below</td><td>amp; T, &tau;, fwhm fixed by default</td></tr></table>
<p>The Balmer continuum (Grandi 1982; Dietrich et al. 2002) is an optically thick Planck function below the edge
&lambda;<sub>BE</sub> = 3646 &Aring;, smeared by the gas velocities (Gaussian of FWHM 5000 km/s, as QSFit):</p>
<div class="eq">F<sub>BaC</sub>(&lambda;) = A&nbsp;<span class="frac"><span>B<sub>&lambda;</sub>(&lambda;, T)</span><span>B<sub>&lambda;</sub>(&lambda;<sub>BE</sub>, T)</span></span>
<span class="frac"><span>1 &minus; e<sup>&minus;&tau;(&lambda;/&lambda;<sub>BE</sub>)<sup>3</sup></sup></span><span>1 &minus; e<sup>&minus;&tau;</sup></span></span>,
&nbsp; &lambda; &le; &lambda;<sub>BE</sub></div>
<h2>Templates</h2>
<dl class="params">
<dt><code>FeII</code></dt><dd>Fe&#8202;II lines in multiplet groups with fixed relative intensities, a free amplitude per group and one shift and FWHM
(<a data-doc="feii">Fe&#8202;II templates</a>).</dd>
<dt><code>HighOrderBalmer</code></dt><dd>H&epsilon; and above blending into the Balmer edge, fixed relative intensities, one amplitude.</dd>
<dt><code>StellarPopulation</code></dt><dd>the host as a non-negative sum of stellar population templates (E-MILES default; BC03, FSPS, XSL)
convolved to a stellar velocity v<sub>&#9733;</sub> and dispersion &sigma;<sub>&#9733;</sub>, optionally with a dust screen.</dd>
<dt><code>HostGalaxy</code></dt><dd>the host as a combination of the SDSS galaxy eigenspectra of Yip et al. (2004).</dd>
</dl>
<h2>Multiplicative components</h2>
<table class="doc"><tr><th>component</th><th>multiplies</th><th>by</th></tr>
<tr><td><code>Extinction</code></td><td>every component but the host</td><td>10<sup>&minus;0.4 E(B&minus;V) k(&lambda;)</sup>, F99 or Calzetti</td></tr>
<tr><td><code>Absorption</code></td><td>every component</td><td>exp[&minus;&Sigma;<sub>k</sub> &tau;<sub>k</sub>(&lambda;)], Gaussian troughs in velocity</td></tr>
<tr><td><code>Calibration</code></td><td>the whole model</td><td>1 + &Sigma;<sub>k</sub> c<sub>k</sub>P<sub>k</sub>(u), Legendre polynomials (as pPXF <code>mdegree</code>)</td></tr></table>
<p>With <code>Extinction</code>, fitted fluxes and luminosities are the intrinsic (dereddened) values, with the extinction
uncertainty in their errors; plots show the observed components.</p>
"""),

("feii", "Modelling", "Fe&#8202;II templates", """
<div class="lead">Fe&#8202;II emits thousands of blended lines. FANTASY models them as groups of lines that share a lower
energy level: within a group the intensities follow atomic physics, between groups the strengths are free.</div>
<p>Lines from the same lower term change together from object to object, because their strength depends on how many ions sit
in that (metastable) level. Within a group the relative intensities are those of a population at T = 10<sup>4</sup> K
(Kova&#269;evi&#263; et al. 2010):</p>
<div class="eq">I<sub>ik</sub> &prop; g<sub>i</sub> f<sub>ik</sub> &lambda;<sub>ik</sub><sup>&minus;3</sup> exp(&minus;<span class="frac"><span>E<sub>k</sub></span><span>kT</span></span>)</div>
<p>with g<sub>i</sub> the statistical weight of the lower level, f<sub>ik</sub> the oscillator strength and E<sub>k</sub> the
upper-level energy. All lines share one shift and one FWHM (by default tied as in the presets).</p>
<table class="doc"><tr><th><code>kind</code></th><th>range</th><th>groups</th><th>reference</th></tr>
<tr><td><code>optical</code></td><td>3700&ndash;10&#8202;900 &Aring;</td><td>by lower term (F, S, G, P), I&#8202;Zw&#8202;1 lines</td><td>Kova&#269;evi&#263; et al. 2010; Shapovalova et al. 2012</td></tr>
<tr><td><code>uv</code></td><td>2690&ndash;3005 &Aring;</td><td>multiplets UV 60&ndash;63, 78, I&#8202;Zw&#8202;1 lines</td><td>Kova&#269;evi&#263;-Doj&#269;inovi&#263; &amp; Popovi&#263; 2015</td></tr>
<tr><td><code>uv_extended</code></td><td>2200&ndash;3500 &Aring;</td><td>by lower term from NIST data, a<sup>4</sup>D split into UV 60&ndash;63, I&#8202;Zw&#8202;1 lines</td><td>this version</td></tr></table>
<pre>fa.FeII("feii_uv", kind="uv_extended", fwhm=(3000, 1000, 10000))
fa.presets.mgii(feii="uv_extended", wave_range=(2200, 3300))
r.feii()            # flux and EW in the standard band, R_FeII</pre>
<div class="note">Fit the UV model over a wide window (2200&ndash;3300 &Aring;) and mask intervening absorbers: in a narrow window
around Mg&#8202;II the pseudo-continuum is degenerate with the power law.</div>
<h2>Weak Fe&#8202;II</h2>
<p>The strength of every group is free and cannot be negative, so noise moves a weak or absent group only upwards: the summed
Fe&#8202;II comes out too high, the continuum too low and the neighbouring broad line too strong. In simulations of an SDSS
quasar at S/N 30 (<code>presets.hbeta()</code>) the Fe&#8202;II flux was 9% too high at R<sub>Fe&#8202;II</sub> = 0.25, 66% at 0.08, and
with no Fe&#8202;II at all the fit returned R<sub>Fe&#8202;II</sub> = 0.12 and a broad H&beta; flux 12% too high. With strong
Fe&#8202;II or high S/N the effect is below 2%.</p>
<pre>r = fa.fit(spec, fa.presets.hbeta())
r = fa.prune_feii(r)        # groups below 2 sigma fixed at 0, then fitted again
r.feii_pruning              # group, amplitude, error, kept or dropped</pre>
<p><code>fa.prune_feii</code> removes most of that bias where it is large (broad H&beta; with no Fe&#8202;II: +15% &rarr; +1%). Where
Fe&#8202;II is moderate it drops real weak groups too: at R<sub>Fe&#8202;II</sub> = 0.25 the Fe&#8202;II flux came out 13% too low
instead of 6% too high. It is a choice for weak Fe&#8202;II and low S/N, not a default; state it with the results.</p>
"""),

("presets", "Modelling", "Presets", """
<div class="lead">Ready-made models for the common spectral windows and object classes. Every preset is an ordinary
<code>fa.Model</code>: change it as you like.</div>
<table class="doc"><tr><th>preset</th><th>range</th><th>content</th></tr>
<tr><td><code>hbeta</code></td><td>4150&ndash;5550 &Aring;</td><td>power law; Fe&#8202;II (shift tied to broad H&beta;); narrow and broad H&beta;, H&gamma;; [O&#8202;III] with an outflow wing; He&#8202;II</td></tr>
<tr><td><code>halpha</code></td><td>6200&ndash;6900 &Aring;</td><td>H&alpha;, [N&#8202;II], [S&#8202;II], [O&#8202;I], He&#8202;I</td></tr>
<tr><td><code>optical</code></td><td>3700&ndash;6900 &Aring;</td><td>all of the above, Balmer continuum, high-order Balmer lines; <code>physical=True</code>: one power law, stellar host, broad FWHM &le; 15&#8202;000 km/s</td></tr>
<tr><td><code>mgii</code></td><td>2650&ndash;3050 &Aring;</td><td>Mg&#8202;II with UV Fe&#8202;II and the Balmer continuum</td></tr>
<tr><td><code>civ</code></td><td>1450&ndash;1720 &Aring;</td><td>C&#8202;IV with two broad components, He&#8202;II 1640 tied to C&#8202;IV</td></tr>
<tr><td><code>seyfert2</code>, <code>composite</code></td><td>optical</td><td>stellar populations with dust, weak scattered continuum, two narrow-line systems, outflow wing (seyfert2)</td></tr>
<tr><td><code>liner</code></td><td>optical</td><td>old population with dust, one narrow-line system; <code>broad=True</code> for a LINER 1.9</td></tr>
<tr><td><code>starforming</code></td><td>optical</td><td>young + old BC03 populations, Case&#8202;B Balmer ratios with gas E(B&minus;V), auroral lines</td></tr>
<tr><td><code>passive</code></td><td>optical</td><td>stellar populations with dust; weak lines with limited widths</td></tr>
<tr><td><code>tde</code>, <code>bff</code></td><td>optical</td><td>blackbody + host, broad Balmer and He&#8202;II with the Bowen lines; <code>bff</code> adds Fe&#8202;II and coronal lines</td></tr></table>
<h2>Options</h2>
<dl class="params">
<dt><code>broad_components</code></dt><dd>1&ndash;3 broad components per line (core, wings, very broad).</dd>
<dt><code>outflow</code></dt><dd>a blueshifted wing on the forbidden lines.</dd>
<dt><code>host</code></dt><dd>a stellar host (<code>True</code>, a template set such as <code>"bc03"</code>).</dd>
<dt><code>broad_profile</code></dt><dd><code>"lorentzian"</code> or <code>"stratified"</code> broad lines.</dd>
<dt><code>coronal</code>, <code>oi6046</code>, <code>red</code></dt><dd>[Fe&#8202;VII], [Fe&#8202;X], [Fe&#8202;XIV]; broad O&#8202;I 6046; the range to 7500 &Aring;.</dd>
<dt><code>feii</code>, <code>wave_range</code></dt><dd>(Mg&#8202;II) the Fe&#8202;II model and the window.</dd>
</dl>
<pre>model = fa.presets.optical(broad_components=2, outflow=True, host=True)
models = fa.presets.variants("hbeta", host=(False, True))   # alternatives for fa.compare
fa.presets.available(spec)                                  # the presets whose window the spectrum covers</pre>
"""),

("classes", "Modelling", "Models for each class", """
<div class="lead">The narrow-line presets were chosen by fitting SDSS examples of every class with models that differ in
one ingredient at a time, keeping what the data support (BIC) and what gives physical results.</div>
<table class="doc"><tr><th>ingredient</th><th>effect</th><th>used in</th></tr>
<tr><td>dust screen on the starlight</td><td>&chi;<sup>2</sup><sub>&nu;</sub> 4.6 &rarr; 1.5 (composite), 4.2 &rarr; 2.6 (Seyfert 2); &sigma;<sub>&#9733;</sub> close to SDSS</td><td>all</td></tr>
<tr><td>young populations (BC03 from 1 Myr)</td><td>&Delta;BIC &asymp; &minus;8000 for star-forming galaxies</td><td>starforming</td></tr>
<tr><td>two narrow-line systems</td><td>&Delta;BIC &minus;74 to &minus;374 (Seyfert 2); +10 LINER</td><td>seyfert2, composite</td></tr>
<tr><td>outflow wing incl. [S&#8202;II]</td><td>&Delta;BIC &minus;90 to &minus;360 (Seyfert 2, composite); +16 LINER</td><td>all but liner, passive</td></tr>
<tr><td>Case&#8202;B Balmer ratios + gas E(B&minus;V)</td><td>physical reddening for H&#8202;II regions</td><td>starforming</td></tr></table>
<h2>High S/N: refinement</h2>
<p>Above S/N &asymp; 45 the data show what a simple model leaves out. <code>fa.refine</code> (the app's Fit step does this for the
narrow-line classes) masks tellurics and sky lines, and if &chi;<sup>2</sup><sub>&nu;</sub> &gt; 1.2 tries the physical extensions one at a
time, keeping one only if &Delta;BIC &lt; &minus;10 and its new parameters do not end at a limit:</p>
<table class="doc"><tr><th>extension</th><th>describes</th></tr>
<tr><td>calibration polynomial</td><td>1&ndash;3% errors of the flux calibration and dust law</td></tr>
<tr><td>outflow wing</td><td>broad bases of the strong lines</td></tr>
<tr><td>Gauss&ndash;Hermite narrow lines</td><td>asymmetric, peaked or flat-topped lines</td></tr>
<tr><td>broad Balmer lines</td><td>LINER 1.9</td></tr></table>
<pre>r = fa.refine(spec.mask_sky(), "starforming")
r.refinement          # what was tried, &Delta;BIC, kept or not</pre>
"""),

("params", "Modelling", "Parameters, limits and ties", """
<div class="lead">Every parameter has a value, limits and a status: free, fixed or tied. Names are
<code>component.parameter</code>; patterns with <code>*</code> address several at once.</div>
<pre>model.set("narrow.fwhm", value=400, min=100, max=900)
model.fix("continuum.index", -1.7);  model.free("continuum.index")
model.tie("broad2.shift", "broad.shift")
model.tie("narrow.[NII]6548.flux", "narrow.[NII]6583.flux", factor=1/3.05)
model.set("broad*.shift", min=-2000)
model.untie("broad2.shift")
model.set_profile("broad", "lorentzian")      # change the line profile of a component
model.split_lines("narrow", ["[OIII]4959", "[OIII]5007"], "narrow_oiii")   # own shift and width for [O III]</pre>
<p>A tie is affine:</p>
<div class="eq">p = a &middot; p<sub>target</sub> + b</div>
<p>Amplitudes can be tied only to amplitudes (b = 0), widths and shifts only to widths and shifts; cycles are refused.
Limits are given as <code>(start, min, max)</code>, e.g. <code>fwhm=(3000, 1000, 9000)</code> km/s.</p>
<div class="warn"><b>Parameters at a limit.</b> When a parameter ends at its limit, the limit and not the data sets its value and
its error is not meaningful. The summary lists them; widen the limit, fix the parameter or change the model.</div>
<p>After a fit, <code>r.params</code> gives value, error, limits and status; <code>r.at_limit</code> the parameters at a limit;
<code>r.unconstrained</code> those without data (e.g. a line outside the fitted range);
<code>r.not_measured</code> the widths and shifts of components without flux (their values mean nothing, so <code>r.lines</code> shows NaN);
<code>r.unmeasured_fluxes</code> the line fluxes with an error at least as large as the flux.</p>
"""),

("saved-models", "Modelling", "Saved models", """
<div class="lead">A model you built can be kept and used again, or sent to a colleague. It is saved as a plain JSON recipe,
not as code.</div>
<pre>model.save("~/projects/agn/models/hbeta_two_broad.json")
model = fa.load_model("~/projects/agn/models/hbeta_two_broad.json")
d = model.to_dict()   # the same as data</pre>
<p>The file lists every component with the arguments it was made with (lines, profiles, templates, ranges) and every
parameter with its value, limits, fixed flag and tie. Loading rebuilds exactly the same model.</p>
<h2>In the app</h2>
<ol class="steps">
<li><b>Save model&hellip;</b> (Model step, above the components): a name and a folder, by default the library
<code>~/.fantasy_agn/models</code>; any folder works and is remembered.</li>
<li><b>Start from: a saved model</b> lists the library and the models saved elsewhere; <b>Use this model</b> loads one.</li>
<li><b>From a file&hellip;</b> loads a model someone sent you and adds it to your library; <b>Download</b> gives the current model as a file.</li>
</ol>
<div class="note"><b>Safe to share.</b> Loading only builds FANTASY's own component types; a file asking for anything else is refused.
<code>.fantasy</code> project files are different: they are Python pickles and can run code, so open only your own.</div>
"""),

("stratified", "Modelling", "Stratified BLR profile", """
<div class="lead">One physical profile for a broad line instead of a core plus a very broad Gaussian: a virialised broad-line
region in which the velocity width falls with radius.</div>
<p>Gas at radius r has a Gaussian velocity distribution with FWHM &prop; r<sup>&minus;1/2</sup>; the profile sums it from an inner to an
outer radius with an emissivity per ln&#8202;r &prop; r<sup>3&minus;&gamma;</sup>.</p>
<dl class="params">
<dt><code>fwhm</code></dt><dd>the width at the outer edge (the core).</dd>
<dt><code>span</code></dt><dd>log<sub>10</sub>(r<sub>out</sub>/r<sub>in</sub>); the inner edge has FWHM&middot;10<sup>span/2</sup>. span = 0 is a single Gaussian.</dd>
<dt><code>gamma</code></dt><dd>the emissivity slope; &gamma; &asymp; 3 gives Lorentzian-like wings.</dd>
</dl>
<pre>fa.presets.hbeta(broad_components=1, broad_profile="stratified")</pre>
<div class="note">The innermost gas is faint and spread over very high velocities, so the inner edge is often below the noise:
<code>span</code> then reaches its upper limit (2.5) and is a lower limit on the radial extent, while the outer width and
&gamma; remain well measured.</div>
"""),

("absorption", "Modelling", "Absorption troughs", """
<div class="lead">Intervening absorbers and the troughs of BAL and mini-BAL quasars cut into C&#8202;IV and Mg&#8202;II and dominate
&chi;<sup>2</sup>. FANTASY can find and fit them, so that the emission components describe the unabsorbed spectrum.</div>
<p>Every component is multiplied by the transmission</p>
<div class="eq">T(&lambda;) = exp[ &minus;&Sigma;<sub>k</sub> &tau;<sub>k</sub> exp(&minus;<span class="frac"><span>(v &minus; v<sub>k</sub>)<sup>2</sup></span><span>2&sigma;<sub>k</sub><sup>2</sup></span></span>) ]</div>
<p><b>Fit absorption troughs</b> (Fit step) fits the model, finds troughs in the residuals, adds them and refits while the BIC improves.
Troughs are kept narrower than 6000 km/s so that a shallow &ldquo;absorber&rdquo; cannot trade flux with a too-broad emission line.</p>
<pre>r = fa.fit_absorbed(spec, fa.presets.civ())
r.absorbers; r.no_absorbers
model.add(fa.Absorption([(1510.5, 1200), (1547.0, 300)]))   # by hand: (wavelength, FWHM)</pre>
<div class="tip">For narrow absorbers only, masking is simpler: <code>spec.mask_absorption()</code> (Prepare).</div>
"""),

# ======================================================================== Fitting
("algorithm", "Fitting", "How the fit works", f"""
<div class="lead">FANTASY minimises the weighted squared residuals with exact derivatives, solving the amplitudes exactly at every
step, after a data-driven search for the starting values.</div>
<div class="eq">&chi;<sup>2</sup>(&theta;) = &Sigma;<sub>i</sub> <span class="frac"><span>[F<sub>i</sub> &minus; M(&lambda;<sub>i</sub>; &theta;)]<sup>2</sup></span><span>&sigma;<sub>i</sub><sup>2</sup></span></span></div>
{_FITFLOW}
<ol class="steps">
<li><b>Initial guess.</b> The nonlinear parameters of each component (widths, shifts, shapes) are scanned on a grid of 9 values,
components carrying most flux first, in three passes. At every grid point the amplitudes are solved exactly by non-negative
least squares, so the scan costs only linear solves (<code>fa.guess</code> returns this start).</li>
<li><b>Variable projection.</b> The model is linear in the amplitudes <b>a</b> and nonlinear in the rest <b>&phi;</b>:
M = A(&phi;)&#8202;<b>a</b>. For given &phi;, <b>a</b> is found by bounded linear least squares (BVLS); a trust-region reflective
algorithm then optimises &phi; on the projected problem, with the exact Jacobian from the components.</li>
<li><b>Escape and build-up.</b> After convergence each component's nonlinear parameters are re-scanned; a better start restarts the fit.
Overlapping components (several broad components, outflow wings) are also fitted by adding them one at a time, and the better of
the two solutions is kept, so adding a component cannot make the fit worse.</li>
<li><b>Unconstrained parameters.</b> Amplitudes of lines without fitted pixels are fixed at zero and reported in <code>r.unconstrained</code>.</li>
</ol>
<h2>Machine-learned starting values</h2>
<p>Finding a good start takes most of the time of a fit. For a model with a trained initializer (a small neural network trained on
spectra simulated with the model itself), <code>guess="ml"</code> predicts the nonlinear parameters in about a millisecond; the
amplitudes, the fit and the uncertainties are unchanged, so a poor prediction costs time, never accuracy.</p>
<pre>r = fa.fit(spec, model, guess="ml")    # falls back to the data-driven guess without an initializer
fa.ml.available()                      # the trained initializers</pre>
<h2>In Python</h2>
<pre>r = fa.fit(spec, model, wave_range=(4150, 5550))
r.chi2, r.dof, r.rchi2, r.bic, r.n_free
fa.guess(spec, model)        # only the initial guess</pre>
"""),

("uncertainties", "Fitting", "Uncertainties", """
<div class="lead">Three ways to the errors of the parameters, and one rule for everything derived from them: propagate the full
covariance or the samples.</div>
<h2>Covariance (default)</h2>
<div class="eq">C = (J<sup>T</sup>WJ)<sup>&minus;1</sup> &middot; max(1, &chi;<sup>2</sup><sub>&nu;</sub>), &nbsp;&nbsp; W = diag(1/&sigma;<sub>i</sub><sup>2</sup>)</div>
<p>J is the exact Jacobian at the best fit. When &chi;<sup>2</sup><sub>&nu;</sub> &gt; 1 the errors are inflated by &radic;&chi;<sup>2</sup><sub>&nu;</sub>
(<code>scale_errors=False</code> switches this off): the error spectrum is then assumed underestimated.</p>
<h2>Monte Carlo</h2>
<p><code>errors="mc"</code>: the spectrum is perturbed by its errors <code>n_mc</code> times (100) and refitted in parallel; the spread of
the fits gives the errors. More reliable for weak or blended lines and parameters near limits.</p>
<h2>Posterior (HMC)</h2>
<p><code>errors="mcmc"</code>: Hamiltonian Monte Carlo with flat priors within the limits; 4 chains, 300 warm-up and 600 samples each.
<code>r.mcmc_info</code> gives the acceptance, split-R&#770; and effective sample size; the app shows the corner plot.
All free parameters are sampled, also those at a limit, and the chains run in parallel. Chains that have not converged are run once more with a 3&times; longer warm-up and twice the trajectory (<code>r.mcmc_info["attempts"]</code>). <code>r.upper_limit("broad.Hb.flux")</code> gives the 95% limit
on an undetected line; it assumes flat priors, so it depends on the parameter limits.</p>
<h2>Derived quantities</h2>
<p>The FWHM of a multi-component line, an EW, a luminosity or a black-hole mass is a function g(&theta;) of all parameters. Its error comes
from the samples when there are any, otherwise from linear propagation of the full covariance,</p>
<div class="eq">&sigma;<sub>g</sub><sup>2</sup> = &nabla;g<sup>T</sup> C &nabla;g,</div>
<p>so correlations between components (a broad wing against the continuum) are included: <code>r.propagate(func)</code>.</p>
<div class="warn">These are statistical errors. How a spectrum is decomposed often matters more: see <a data-doc="compare">Model comparison</a>.</div>
"""),

("compare", "Fitting", "Model comparison", """
<div class="lead">Broad-line widths, fluxes and luminosities often depend more on the decomposition than on the noise. FANTASY compares
decompositions with the Bayesian information criterion and turns their spread into a systematic error.</div>
<div class="eq">BIC = &chi;<sup>2</sup> + k&nbsp;ln&nbsp;N, &nbsp;&nbsp;&nbsp; AIC = &chi;<sup>2</sup> + 2k</div>
<p>k is the number of free parameters (tied and fixed ones excluded) and N the number of fitted pixels. A difference
&Delta;BIC &lt; &minus;6 strongly favours the model with the lower value; FANTASY uses &minus;10 to accept a refinement and &minus;25 to
detect a broad-line region.</p>
<pre>models = fa.presets.variants("hbeta", host=(False, True))
cmp = fa.compare(spec, models)       # fitted in parallel
cmp.summary()                        # value, statistical, systematic and total error
r = cmp.result                       # the best (lowest BIC) fit</pre>
<p>The systematic error is the spread over the acceptable decompositions (&chi;<sup>2</sup><sub>&nu;</sub> within 10% of the best).
Example (SDSS quasar, 12 decompositions): H&beta; FWHM 4522 &plusmn; 56 (stat) &plusmn; 52 (model) km/s, but the line dispersion
3524 &plusmn; 143 &plusmn; 325 km/s and log L<sub>5100</sub> 44.854 &plusmn; 0.003 &plusmn; 0.039.</p>
"""),

# ======================================================================== Measurements
("measurements", "Measurements", "Line measurements", """
<div class="lead">Line quantities are measured on the fitted model profile of the line, summed over its components (for example two
broad components), with errors propagated from the fit.</div>
<pre>r.key_results()              # the main numbers with errors (what the app's summary shows)
r.lines                      # every line of every component: flux, FWHM, shift, EW, errors
r.fitted_lines()             # the lines that can be shown alone, most interesting first
r.measure("Hb")              # summed over all components containing H&beta;
r.measure("Hb", ["broad", "broad2"])</pre>
<p>For a profile P(v) on a velocity grid around the rest wavelength &lambda;<sub>0</sub>:</p>
<dl class="params">
<dt>Flux</dt><dd>F = &int; P d&lambda;</dd>
<dt>Shift</dt><dd>the flux-weighted centroid v&#772; = &int; v P dv / &int; P dv</dd>
<dt>FWHM</dt><dd>between the outermost half-maximum crossings (interpolated), relative to the centroid</dd>
<dt>Line dispersion &sigma;</dt><dd>&sigma;<sup>2</sup> = &int; (v &minus; v&#772;)<sup>2</sup> P dv / &int; P dv</dd>
<dt>EW</dt><dd>F / F<sub>&lambda;,AGN</sub>(&lambda;<sub>c</sub>): against the AGN continuum at the line (host excluded)</dd>
</dl>
<p>Widths are intrinsic: the instrumental resolution was removed in the fit. Lines with several components get their FWHM from
the sum, not from the components.</p>
<h2>Continuum and luminosities</h2>
<pre>r.continuum(5100)            # AGN continuum flux density
r.luminosity(5100)           # &lambda;L&lambda; (erg/s), needs the flux unit and the redshift
r.host_properties()          # &sigma;&#9733;, v&#9733;, age, metallicity, host fraction at 5100 &Aring;
r.balmer_decrement()         # broad H&alpha;/H&beta;
r.model_flux(components=True)   # the model and each component on the data grid
r.to_dataframe()             # data, model, components, residuals in one table</pre>
"""),

("bh-mass", "Measurements", "Black-hole masses", """
<div class="lead">Single-epoch virial masses from the width of a broad line and a luminosity, with the main calibrations of the
literature.</div>
<div class="eq">log&nbsp;M<sub>BH</sub> = a + b&nbsp;log&nbsp;<span class="frac"><span>L</span><span>10<sup>n</sup> erg&nbsp;s<sup>&minus;1</sup></span></span>
+ c&nbsp;log&nbsp;<span class="frac"><span>W</span><span>1000 km&nbsp;s<sup>&minus;1</sup></span></span></div>
<p>W is the FWHM (or &sigma;<sub>line</sub>) of the summed broad line, L the continuum or broad-line luminosity.</p>
<table class="doc"><tr><th>line</th><th>recipe</th><th>width, luminosity</th><th>reference</th></tr>
<tr><td>H&beta;</td><td><code>vp06</code> (default)</td><td>FWHM, L<sub>5100</sub></td><td>Vestergaard &amp; Peterson 2006</td></tr>
<tr><td>H&beta;</td><td><code>db20</code>, <code>db20_sigma</code></td><td>FWHM or &sigma;, L(H&beta; broad)</td><td>Dalla Bont&agrave; et al. 2020</td></tr>
<tr><td>H&beta;</td><td><code>db20_l5100</code>, <code>db20_sigma_l5100</code></td><td>FWHM or &sigma;, L<sub>5100</sub></td><td>Dalla Bont&agrave; et al. 2020</td></tr>
<tr><td>H&beta;</td><td><code>shen24</code></td><td>FWHM, L<sub>5100</sub></td><td>Shen et al. 2024 (SDSS-RM)</td></tr>
<tr><td>H&alpha;</td><td><code>gh05</code> (default), <code>rv15</code></td><td>FWHM, L(H&alpha; broad)</td><td>Greene &amp; Ho 2005; Reines &amp; Volonteri 2015</td></tr>
<tr><td>Mg&#8202;II</td><td><code>vo09</code> (default), <code>shen11</code>, <code>shen24</code></td><td>FWHM, L<sub>3000</sub></td><td>Vestergaard &amp; Osmer 2009; Shen et al. 2011, 2024</td></tr>
<tr><td>C&#8202;IV</td><td><code>vp06</code> (default), <code>shen24</code></td><td>FWHM, L<sub>1350</sub></td><td>Vestergaard &amp; Peterson 2006; Shen et al. 2024</td></tr></table>
<pre>r.bh_mass("Hb"); r.bh_mass("Hb", recipe="db20"); r.bh_masses()</pre>
<div class="warn">The errors are statistical. The recipes scatter by 0.3&ndash;0.6 dex around reverberation masses, which is not in the
error. Recipes with the broad-line luminosity need no host subtraction.</div>
"""),

("feii-ms", "Measurements", "Fe&#8202;II and the main sequence", """
<div class="lead">The strength of Fe&#8202;II relative to broad H&beta; and the width of H&beta; place a quasar on the quasar main sequence
(Sulentic et al. 2000; Marziani et al. 2018).</div>
<div class="eq">R<sub>Fe&#8202;II</sub> = <span class="frac"><span>F(Fe&#8202;II, 4434&ndash;4684 &Aring;)</span><span>F(H&beta;<sub>broad</sub>)</span></span></div>
<p>The band is that of Boroson &amp; Green (1992); for the UV template the band is 2200&ndash;3090 &Aring; (Tsuzuki et al. 2006) and the
reference line broad Mg&#8202;II. Population A has FWHM(H&beta;) &lt; 4000 km/s; the spectral types A1&ndash;A4 are bins of 0.5 in
R<sub>Fe&#8202;II</sub>, and B1, B1+, B1++ bins of 4000 km/s in FWHM.</p>
<pre>r.feii()            # flux, EW and R_FeII with errors
r.main_sequence()   # R_FeII, FWHM(H&beta;), population, spectral type</pre>
"""),

("diagnostics", "Measurements", "Narrow-line diagnostics", """
<div class="lead"><code>r.diagnostics()</code> turns narrow-line fluxes into physical quantities, with errors from the fit.</div>
<table class="doc"><tr><th>quantity</th><th>from</th></tr>
<tr><td>BPT ratios and class</td><td>[N&#8202;II]/H&alpha;, [O&#8202;III]/H&beta;, [S&#8202;II]/H&alpha;, [O&#8202;I]/H&alpha;; Kauffmann et al. 2003; Kewley et al. 2001, 2006</td></tr>
<tr><td>Balmer decrement, gas E(B&minus;V)</td><td>narrow H&alpha;/H&beta; against Case&#8202;B 2.86 (H&#8202;II) or 3.1 (AGN), Fitzpatrick 1999</td></tr>
<tr><td>n<sub>e</sub></td><td>[S&#8202;II] 6716/6731, Sanders et al. 2016, at 10<sup>4</sup> K</td></tr>
<tr><td>T<sub>e</sub>([O&#8202;III]), T<sub>e</sub>([N&#8202;II])</td><td>auroral/nebular ratios when [O&#8202;III] 4363 or [N&#8202;II] 5755 is detected; Osterbrock &amp; Ferland 2006</td></tr>
<tr><td>EW(H&alpha;)</td><td>the WHAN diagram, Cid Fernandes et al. 2011</td></tr>
<tr><td>12 + log(O/H)</td><td>star-forming only: O3N2 and N2, Marino et al. 2013 (calibration scatter 0.16&ndash;0.18 dex not in the error)</td></tr></table>
<p>Fluxes are summed over the narrow components and the outflow wing (<code>outflow=False</code> leaves the wing out).</p>
"""),

("classify", "Measurements", "Classification", """
<div class="lead"><code>fa.classify(spec)</code> decides what kind of object a spectrum shows, before a model is chosen, with three
quick fits.</div>
<ol class="steps">
<li><b>Starlight.</b> E-MILES stellar populations with dust plus a weak AGN power law, emission lines masked: the host fraction at
5100 &Aring; and &sigma;<sub>&#9733;</sub>, and the stellar Balmer absorption under H&beta; removed before line ratios are measured.</li>
<li><b>Broad lines.</b> The starlight-subtracted spectrum with narrow lines only, and with broad H&beta;/H&alpha; added. A broad-line region
needs &Delta;BIC &gt; 25, S/N &gt; 5, EW &gt; 30 &Aring; and 1200 &lt; FWHM &lt; 12&#8202;000 km/s; weaker candidates are reported as type 1.8/1.9.</li>
<li><b>Narrow-line diagnostics.</b> BPT ([N&#8202;II], [S&#8202;II], [O&#8202;I]) and WHAN diagrams.</li>
</ol>
<table class="doc"><tr><th>class</th><th>meaning</th><th>recommended model</th></tr>
<tr><td>type1</td><td>broad lines, AGN-dominated</td><td>optical</td></tr>
<tr><td>type1_host</td><td>broad lines, host &ge; 50% at 5100 &Aring;</td><td>optical with host</td></tr>
<tr><td>seyfert2, liner, composite</td><td>narrow-line AGN</td><td>the preset of the same name</td></tr>
<tr><td>starforming</td><td>below the Kauffmann line</td><td>starforming</td></tr>
<tr><td>retired, passive</td><td>EW(H&alpha;) &lt; 3 &Aring; / &lt; 0.5 &Aring;</td><td>passive</td></tr>
<tr><td>uv_mgii, uv_civ</td><td>the spectrum ends below 5200 &Aring; in the rest frame, so the optical classes cannot be tested</td><td>mgii, or civ if Mg&#8202;II is not covered</td></tr></table>
<p>A UV label says which line can be fitted, not that the object has broad lines: nothing tests that. Look at the width of the
line and at the catalogs.</p>
<h2>Catalogs</h2>
<p><b>Look up</b> in step 1 (Read), below the list of spectra, or <code>fa.lookup_catalogs(files)</code>, shows a sky image and searches around the sky position in SIMBAD,
NED and Milliquas (name, type, redshift), AllWISE, FIRST and NVSS, and XMM, Chandra and 2RXS. Only the coordinates are sent, and only
when you press the button. It warns when a catalog lists a quasar classified here as passive or star-forming, a star, a galaxy fitted
with broad lines, a different redshift, or W1&minus;W2 &ge; 0.8 (Stern et al. 2012) for a class without an AGN. A catalog redshift
often comes from the same spectrum, so agreement is not always an independent test. A service that cannot be reached is named; its results are
missing, not negative.</p>
<p><code>c.uncertain</code> is set when a ratio is within its error of a boundary or a diagnostic is missing; <code>c.plot()</code> and
<code>c.plot_steps()</code> show the diagrams and the steps. The diagnostics need H&alpha;: up to z &asymp; 0.4 for SDSS and DESI.</p>
<h2>Preliminary and final class</h2>
<p><code>fa.classify</code> is <i>preliminary</i>: it chooses the model before the fit, from its own quick fits. The <i>final</i>
class comes from the fit, <code>fa.classify_fit(r)</code> (the Results page shows it after every fit), with the same rules on the
fitted values:</p>
<ul><li>a broad H&alpha; (else H&beta;, Mg&#8202;II, C&#8202;IV), all broad components summed (<code>r.measure</code>), with flux S/N &ge; 5
and FWHM &ge; 1200 km/s: <b>type1</b>, or <b>type1_host</b> when the fitted host is detected (&sigma;<sub>&#9733;</sub> S/N &ge; 5 or host
&ge; 50% at 5100 &Aring;). The position of the narrow lines is given as well (<code>c.measurements["narrow_class"]</code>);</li>
<li>otherwise the narrow (+ outflow) line ratios and EW(H&alpha;) of the fit (<code>r.diagnostics()</code>) on the BPT and WHAN diagrams,
a ratio counting when both lines have S/N &ge; 3. A broad component of the model that is not detected, or detected only weakly
(type 1.8/1.9?), is named in the reason.</li></ul>
<p>The final class can only see what the model contains: a model without broad lines cannot give a type 1, and a model without
H&alpha; gives no BPT class. With <code>preliminary=c</code> the reason says when the two disagree.</p>
"""),

# ======================================================================== Figures
("figures", "Figures", "Figures", """
<div class="lead">Every figure is drawn in one style: quiet grey axes with inward ticks on all sides, no grid, the model in ink over grey
data, Fe&#8202;II lightly filled, a &plusmn;1&sigma; band behind the residuals, and colours that stay distinct for colour-blind readers.</div>
<h2>The fit</h2>
<pre>r.plot()                                         # data, model, components, residuals
r.plot(xlim=(4600, 5100), label_lines="all", width="single")
r.plot(colors={"broad": "navy"}, hide=("host",), title=None)
r.plot(styles={"feii": {"fill": False, "ls": "--"},
               "broad": {"fill": True, "lw": 1.5},
               "model": {"lw": 0.6}})
r.plot(zoom=[(4650, 5150), (6350, 6800)])        # zoom panels of the H&beta; and H&alpha; regions under the spectrum
r.plot(zoom=[(4650, 5150)], zoom_subtract=("continuum", "feii"))   # ... with the continuum and Fe II subtracted
r.plot(width="mnras-double", legend="inside")    # a journal column; the legend where the plot is free
r.plot(style="sans-serif", height=0.8)           # sans-serif font; a compact (0.8) or tall (1.25) figure
r.savefig("fit.pdf")                             # PDF, EPS, SVG (editable text), PNG (300 dpi)</pre>
<dl class="params">
<dt><code>group</code></dt><dd><code>"auto"</code>: the sum of each kind and, when a line has several components of a kind, each of them (dashed, dotted); <code>True</code> only sums; <code>False</code> every component.</dd>
<dt><code>styles</code></dt><dd>per kind (<code>data</code>, <code>model</code>, <code>continuum</code>, <code>host</code>, <code>feii</code>, <code>narrow</code>, <code>broad</code>, <code>outflow</code>, &hellip;):
<code>fill</code>, <code>lw</code> (points), <code>ls</code> (<code>"-"</code>, <code>"--"</code>, <code>":"</code>, <code>"-."</code>).</dd>
<dt><code>width</code></dt><dd><code>"single"</code> (3.5 in, one journal column), <code>"double"</code> (7.1 in), a journal column
(<code>"mnras"</code> 84 mm, <code>"mnras-double"</code> 174 mm, <code>"aa"</code> 88 mm, <code>"aa-double"</code> 180 mm, <code>"apj"</code>, <code>"apj-double"</code>) or inches.</dd>
<dt><code>legend</code></dt><dd><code>"above"</code> (default), <code>"inside"</code> (where it covers the fewest data and line labels; above if nowhere is free), <code>"right"</code> or <code>False</code>.</dd>
<dt><code>zoom</code></dt><dd>wavelength ranges drawn again in a row of panels under the spectrum; the main panel marks them with a faint blue band and a letter.
<code>zoom_subtract</code>: kinds subtracted from the data and the model in these panels (<code>"continuum"</code>, <code>"feii"</code>, <code>"host"</code>), so the lines stand alone.</dd>
<dt><code>label_lines</code></dt><dd>name the strongest or all fitted lines; <code>label_options</code> for font, size, direction, colour, connectors.</dd>
</dl>
<h2>Single lines and grids</h2>
<pre>r.plot_line("Hb")                                # the line alone, its components and their sum
r.plot_line("Ha", others=True, velocity=False)
r.plot_lines(["Hg", "Hb", "Ha"], residuals=True) # a grid: 1, 2, 3 in a row, 4 as 2 &times; 2</pre>
<p>A single-line panel shows the data with everything else in the model subtracted, each component with its FWHM, their sum
(and the summed broad profile), the FWHM, line dispersion &sigma; and flux of the line rounded to their errors, a bar between the
half-maximum points and its residuals. A FWHM that a shoulder at half maximum makes fragile is marked; &sigma; is then the safer width.
Wide windows are labelled in 10<sup>3</sup> km/s.</p>
<pre>r.plot_profiles(["Hb", "Ha"])                    # broad profiles overlaid, each scaled to its peak
fa.plotting.plot_profiles([r1, r2], "Hb", labels=["power law", "broken"])   # one line in two fits</pre>
<p>The overlay compares the shapes of lines (is broad H&beta; wider than H&alpha;? more asymmetric?) or of one line in several fits or epochs;
the legend gives FWHM and &sigma;, the dots the half-maximum points. In the app: Results &rarr; <b>Line profiles</b>, tick the lines and choose
<b>Each in a panel</b> or <b>Overlaid</b>. Both figure tabs show the <b>Python code</b> of the figure, to copy into a script.</p>
<h2>Colours</h2>
<table class="doc"><tr><th>kind</th><th>colour</th></tr>
<tr><td>data, model</td><td><span class="sw" style="background:#9b9a95"></span> grey &nbsp; <span class="sw" style="background:#0b0b0b"></span> ink</td></tr>
<tr><td>broad lines (and their components)</td><td><span class="sw" style="background:#2a78d6"></span> blue, darker and lighter steps</td></tr>
<tr><td>narrow lines, outflow</td><td><span class="sw" style="background:#1baf7a"></span> teal &nbsp; <span class="sw" style="background:#eda100"></span> yellow</td></tr>
<tr><td>Fe&#8202;II, host</td><td><span class="sw" style="background:#e34948"></span> red &nbsp; <span class="sw" style="background:#008300"></span> green</td></tr>
<tr><td>Bowen + He&#8202;II, Balmer</td><td><span class="sw" style="background:#4a3aa7"></span> violet &nbsp; <span class="sw" style="background:#86b6ef"></span> light blue, dashed</td></tr></table>
"""),

# ======================================================================== Time domain and samples
("multi", "Time domain and samples", "Multi-epoch fitting", """
<div class="lead">Several spectra of one object are fitted jointly: what does not vary is shared, and the shared narrow lines calibrate the
epochs against each other.</div>
<p>Parameters matching the <i>shared</i> patterns (by default every narrow-line parameter and, with a stellar host, the host:
one set of template weights, &sigma;<sub>&#9733;</sub> and v<sub>&#9733;</sub>) are the same in every epoch; the continuum,
Fe&#8202;II and broad lines vary. With intercalibration each epoch k has a flux scale s<sub>k</sub> (the reference epoch, the one of highest
S/N, has s = 1):</p>
<div class="eq">M<sub>k</sub>(&lambda;) = s<sub>k</sub> [ N(&lambda;) + V<sub>k</sub>(&lambda;) ]</div>
<p>with N the shared and V<sub>k</sub> the varying part. <code>calibration_degree=1</code> or 2 makes s<sub>k</sub>(&lambda;) a smooth polynomial,
anchored by narrow lines at both ends of the range.</p>
<pre>me = fa.fit_epochs(spectra, fa.presets.hbeta(broad_components=2), n_jobs=-1)
me.light_curve(); me.plot_light_curves(); me.plot_baldwin()
me.plot(epoch=5)              # me.epochs[5] is the full result of epoch 5
me.scales                     # the flux scale of every epoch, with errors</pre>
<p>Every epoch is first fitted on its own (in parallel), which gives the starting values; then all epochs are fitted together with a
block-sparse Jacobian.</p>
<p>What else differs between epochs and can be fitted, measured like the flux scale on the shared narrow lines:</p>
<pre>me = fa.fit_epochs(spectra, model,
                   wavelength_offsets=True,     # a velocity per epoch: calibration, redshifts of the files
                   epoch_broadening=True,       # a FWHM per epoch, in quadrature: the resolution differs
                   epoch_scaled=("host",))      # one host spectrum, an amplitude per epoch: apertures, seeing
me.offsets; me.broadening; me.redshift()</pre>
<p>The broadening is exact for Gaussian lines: a line of FWHM w seen at resolution R has w<sup>2</sup> + R<sup>2</sup>, so an extra
b<sup>2</sup> on every width is a resolution &radic;(R<sup>2</sup> + b<sup>2</sup>). Each epoch's result keeps the intrinsic widths, and its
spectrum carries the broadened resolution.</p>
<p>Two checks of a joint fit:</p>
<pre>chk = fa.calibration_check(me)    # every epoch alone, with the joint flux scales: do the shared lines agree?
chk.table; chk.epochs; chk.excess  # excess: calibration noise not in the error bars
sim = fa.simulation_check(me, n=20)  # sets of epochs drawn from the fit and fitted jointly again</pre>
<div class="warn">Intercalibration on the narrow lines helps only when they are strong: with weak [O&#8202;III] it adds a few percent of noise per
epoch, and FANTASY warns when the median scale error exceeds 2%. For a good pipeline calibration (SDSS) compare with
<code>intercalibrate=False</code>.</div>
<div class="tip">In scripts on macOS and Windows put the code under <code>if __name__ == "__main__":</code>, otherwise the parallel part
runs serially.</div>
"""),

("reverberation", "Time domain and samples", "Reverberation mapping", """
<div class="lead">After a multi-epoch fit: the lag of a broad line behind the continuum, its width in the rms spectrum, and the
reverberation mass. In the app: Results &rarr; <b>Light curves &amp; lags</b> &rarr; <b>Measure lag</b>.</div>
<pre>rm = me.reverberation("Hb")                                # continuum from the fitted 5100 &Aring;
rm = me.reverberation("Hb", continuum=(mjd, flux, err))     # better: photometry
rm.table(); rm.lag_table(); rm.plot()
me.rms_spectrum("Hb")</pre>
<dl class="params">
<dt>Lag</dt><dd>interpolated cross-correlation (ICCF), centroid of r &ge; 0.8 r<sub>max</sub>; errors by flux randomisation and random subset
selection (Peterson et al. 1998, 2004). Also available: the von Neumann estimator (Chelouche et al. 2017), the z-transformed
discrete correlation function (ZDCF; Alexander 1997) and a damped-random-walk model that uses the correlated errors of each epoch's fit. The ICCF is adopted; FANTASY warns when the methods disagree or a lag distribution
has aliases.</dd>
<dt>Significance</dt><dd><code>rm.lag_table()</code> has <code>p_side</code>, the share of the lag distribution (all samples, aliases
included) on the side of zero where the lag is, and <code>significance</code>, its one-sided Gaussian equivalent
(<code>fa.rm.lag_significance(lag)</code>). Below 2&sigma; FANTASY warns that the lag is not significant, and below r<sub>max</sub> = 0.5 that
the light curves are weakly correlated.</dd>
<dt>rms spectrum</dt><dd>the broad line alone in every epoch, noise-corrected rms (Park et al. 2012), widths with bootstrap errors.</dd>
<dt>Mass</dt><dd>M = f c&tau; &sigma;<sub>line</sub>(rms)<sup>2</sup>/G with f = 4.47 (Woo et al. 2015); the 0.3 dex scatter of f is not in the error.</dd>
</dl>
<p><code>fa.rm.component_reverberation(me, "Hb", continuum=...)</code> measures a lag and response for every component (continuum, each
broad component, Fe&#8202;II) from their fluxes and full error covariance over all epochs.</p>
<div class="note">Check r<sub>max</sub> (&gt; 0.6&ndash;0.7 for a reliable lag) and the CCF: sparse continua diluted by host light, and long
baselines with seasonal gaps, give spurious lags.</div>
"""),

("sample", "Time domain and samples", "Fitting a sample", """
<div class="lead">Many spectra at once: each is read, corrected, classified and fitted with the model its class needs, in parallel.</div>
<pre>s = fa.fit_sample(glob.glob("data/*.fits"), out="fits/")
s.table              # the catalogue: class, model, &chi;&sup2;&nu;, main quantities with errors
s.load(3)            # the full fit of row 3
s.gallery()          # an HTML page with every fit</pre>
<p>In the app, sort the catalogue by a column (e.g. &chi;<sup>2</sup><sub>&nu;</sub>, to check the worst fits first) and click a row to see
its fit. Choose a model instead of <i>automatic</i> to fit all spectra with the same one. Every fit is also saved as a
<code>.fantasy</code> file.</p>
"""),

# ======================================================================== Reference
("save", "Reference", "Saving and reopening work", """
<div class="lead">Three kinds of files: projects (everything), results (one object), and models (a recipe).</div>
<table class="doc"><tr><th>file</th><th>holds</th><th>how</th></tr>
<tr><td><code>.fantasy</code> project</td><td>cells with outputs, spectra, models, fits</td><td>app: &#8943; &rarr; Save project / Open project</td></tr>
<tr><td><code>.fantasy</code> result</td><td>a FitResult, MultiEpochResult, Classification, Model or Spectrum</td><td><code>fa.save(r, "x.fantasy")</code>, <code>fa.load</code></td></tr>
<tr><td><code>.json</code> model</td><td>components, limits, fixed and tied parameters</td><td><code>model.save</code>, <code>fa.load_model</code></td></tr>
<tr><td>script, notebook</td><td>the session as code</td><td>Copy, .ipynb</td></tr></table>
<div class="warn"><code>.fantasy</code> files are compressed Python pickles: like any pickle they can run code when opened. Open only files you or your
collaborators made. Model files are plain data and safe to share.</div>
<p>Stellar templates are not stored in a result (a fit with a host is a few hundred kB); they are read from the template folder when it is opened.</p>
"""),

("calibration", "Reference", "How the presets were calibrated", """
<div class="lead">The defaults come from fits of 1500 SDSS DR17 spectra in nine groups; starting values and limits were set on 70% of each
group and the changes tested on the other 30%.</div>
<table class="doc"><tr><th>change</th><th>why (held-out 30%)</th></tr>
<tr><td>[O&#8202;III] outflow wing in <code>optical</code> and <code>hbeta</code></td><td>BIC better for 82% / 66%; &chi;<sup>2</sup><sub>&nu;</sub> 1.35 &rarr; 1.25 and 1.02 &rarr; 0.96</td></tr>
<tr><td>stellar host for host-dominated quasars</td><td>&chi;<sup>2</sup><sub>&nu;</sub> 1.10; limits per fit 2.9 &rarr; 1.8; BIC better for 82%</td></tr>
<tr><td>Fe&#8202;II shift tied to broad H&beta;</td><td>BIC better for 72%; the free shift ended at its limit in a quarter of the fits</td></tr>
<tr><td>no narrow Mg&#8202;II</td><td>BIC better for 77%; parameters at a limit 1.1 &rarr; 0.3 per fit</td></tr>
<tr><td>two broad C&#8202;IV components</td><td>BIC better for 98%; &chi;<sup>2</sup><sub>&nu;</sub> 1.76 &rarr; 1.52</td></tr>
<tr><td>broad-line test of <code>classify</code></td><td>type 1 found in 76% of the quasars, 0.9% of the narrow-line galaxies</td></tr></table>
<p>The scripts and reports are in <code>calibration/</code> of the repository.</p>
"""),
]
