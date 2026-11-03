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
<div class="lead">The launchers run the app with Python 3.12 and the tested package versions in an environment of
their own, made by <a href="https://docs.astral.sh/uv/">uv</a> on first start.</div>
<pre>launch/FANTASY.command        # macOS: double-click
launch/fantasy-app.sh         # Linux / macOS
launch\fantasy-app.bat        # Windows
fantasy-app                   # FANTASY installed with pip: the same app
python -c "import fantasy_agn as fa; fa.app()"</pre>
<p>As a package in your own environment (Python &ge; 3.10):</p>
<pre>pip install fantasy_agn             # then: fantasy test
pip install "fantasy_agn[dust]"     # + local SFD98 dust maps (sfdmap2)</pre>
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
<p><i>from file</i> or <i>manual</i>; <b>Move to rest frame</b> applies it. Fits are done in the rest frame: <code>spec.to_rest_frame(z)</code> divides wavelengths by
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
<dt>Together (recommended)</dt><dd><code>fa.fit_epochs</code>: one joint fit, narrow lines (and a stellar host) shared, one flux scale per epoch
fixed by the shared lines (intercalibration). Best for light curves and reverberation mapping.</dd>
<dt>Each on its own</dt><dd><code>fa.fit_many</code>: independent fits in parallel, with the calibration of each spectrum as it is.</dd>
<dt>Flux reference epoch</dt><dd>the epoch whose flux scale is 1; by default the one of highest S/N, since its uncertainty enters
every other epoch's calibration.</dd>
<dt>Calibration between epochs</dt><dd>one flux scale per epoch; scale and tilt; or scale, tilt and curvature
(<code>calibration_degree</code> 0, 1, 2): see <a data-doc="multi">Multi-epoch fitting</a>.</dd>
</dl>
<p>For a sample, <i>Model for each spectrum</i> is <i>automatic</i> (classify each and use the recommended model) or one model
for all.</p>
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
<dt>Host amplitude free in each epoch</dt><dd>for spectra taken with different apertures or seeing.</dd>
<dt>Rescale the errors of each epoch</dt><dd>by &radic;&chi;<sup>2</sup><sub>&nu;</sub> of its own fit, for epochs whose errors are under- or overestimated.</dd>
<dt>Parameters shared by the epochs</dt><dd>name patterns that replace the default (e.g. add <code>feii.fwhm</code>, <code>feii.shift</code>).</dd>
<dt>Processes</dt><dd>parallel processes for independent fits and Monte Carlo.</dd>
</dl>
<p>When the fit is done, its quality and the parameters that ended at a limit are listed, with links to the results and to
the component in the Model step.</p>
"""),

("results-step", "The app, step by step", "5 &middot; Results", """
<div class="lead">The summary at the top gives the fit quality and the main numbers with errors; parameters at a limit
are flagged with a button that opens their component.</div>
<h2>Views</h2>
<dl class="params">
<dt>Fit</dt><dd>data, model, components and residuals (drag to zoom, double-click to reset; click the legend to hide a curve).
Below, four tables: <i>Lines</i> (by component, with flags for undetected lines and tied widths; <i>Hide undetected</i>),
<i>Continuum and host</i> (continuum, luminosities, host properties and the black-hole mass of every recipe that applies),
<i>All parameters</i> (value, error, limits, status) and <i>All epochs</i>. Then <b>Does the model miss anything?</b>: the residual
analysis (<code>fa.analyze_residuals</code>, threshold 3&ndash;6&sigma;) marks missing lines, wings, absorption, continuum shape and
sky residuals, suggests a change for each, and <b>Refit with the changes</b> applies the ones you keep.</dd>
<dt>Publication figure</dt><dd>ranges, title, width, font size, legend, line labels; for every kind of component whether it is drawn, its colour,
fill, line width and style. PDF, PNG, EPS. See <a data-doc="figures">Figures</a>.</dd>
<dt>Line profiles</dt><dd>tick lines to draw them in a grid, each alone with its components, residuals, FWHM and flux.</dd>
<dt>Posterior</dt><dd>with Monte Carlo or MCMC errors: the corner plot of chosen parameters.</dd>
<dt>Main sequence</dt><dd>with Fe&#8202;II and broad H&beta;: R<sub>Fe&#8202;II</sub> against FWHM(H&beta;) for every epoch; <i>Zoom to the
epochs</i> (<a data-doc="feii-ms">Fe&#8202;II and the main sequence</a>).</dd>
<dt>Classification</dt><dd>the three steps (starlight, broad H&alpha;, emission lines) and the BPT and WHAN diagrams, as figures to download.</dd>
<dt>Light curves &amp; lags</dt><dd>for several epochs: light curves of fluxes and widths (CSV), and reverberation mapping: choose the
line, the lag methods (ICCF, von Neumann, ZDCF, DRW), an MJD range and detrending, then <b>Measure lag</b>
(<a data-doc="reverberation">Reverberation mapping</a>).</dd>
</dl>
<h2>Downloads</h2>
<p>The fitted lines (CSV or FITS), the model and its components (data, model, every component, host, AGN, spectrum minus host
and, with extinction, the transmission and intrinsic components), the fit figures (PDF), the script, the notebook and the project.</p>
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
(exactly for Gaussian, Lorentzian and Voigt profiles). Every profile integrates to one, so the flux parameter is the line flux.</p>
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
model.set_profile("broad", "lorentzian")      # change the line profile of a component</pre>
<p>A tie is affine:</p>
<div class="eq">p = a &middot; p<sub>target</sub> + b</div>
<p>Amplitudes can be tied only to amplitudes (b = 0), widths and shifts only to widths and shifts; cycles are refused.
Limits are given as <code>(start, min, max)</code>, e.g. <code>fwhm=(3000, 1000, 9000)</code> km/s.</p>
<div class="warn"><b>Parameters at a limit.</b> When a parameter ends at its limit, the limit and not the data sets its value and
its error is not meaningful. The summary lists them; widen the limit, fix the parameter or change the model.</div>
<p>After a fit, <code>r.params</code> gives value, error, limits and status; <code>r.at_limit</code> the parameters at a limit;
<code>r.unconstrained</code> those without data (e.g. a line outside the fitted range).</p>
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
<code>r.mcmc_info</code> gives the acceptance, split-R&#770; and effective sample size; the app shows the corner plot.</p>
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
<tr><td>retired, passive</td><td>EW(H&alpha;) &lt; 3 &Aring; / &lt; 0.5 &Aring;</td><td>passive</td></tr></table>
<p><code>c.uncertain</code> is set when a ratio is within its error of a boundary or a diagnostic is missing; <code>c.plot()</code> and
<code>c.plot_steps()</code> show the diagrams and the steps. The diagnostics need H&alpha;: up to z &asymp; 0.4 for SDSS and DESI.</p>
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
r.savefig("fit.pdf")                             # PDF, EPS, SVG, PNG (300 dpi)</pre>
<dl class="params">
<dt><code>group</code></dt><dd><code>"auto"</code>: the sum of each kind and, when a line has several components of a kind, each of them (dashed, dotted); <code>True</code> only sums; <code>False</code> every component.</dd>
<dt><code>styles</code></dt><dd>per kind (<code>data</code>, <code>model</code>, <code>continuum</code>, <code>host</code>, <code>feii</code>, <code>narrow</code>, <code>broad</code>, <code>outflow</code>, &hellip;):
<code>fill</code>, <code>lw</code> (points), <code>ls</code> (<code>"-"</code>, <code>"--"</code>, <code>":"</code>, <code>"-."</code>).</dd>
<dt><code>width</code></dt><dd><code>"single"</code> (3.5 in, one journal column), <code>"double"</code> (7.1 in) or inches.</dd>
<dt><code>label_lines</code></dt><dd>name the strongest or all fitted lines; <code>label_options</code> for font, size, direction, colour, connectors.</dd>
</dl>
<h2>Single lines and grids</h2>
<pre>r.plot_line("Hb")                                # the line alone, its components and their sum
r.plot_line("Ha", others=True, velocity=False)
r.plot_lines(["Hg", "Hb", "Ha"], residuals=True) # a grid: 1, 2, 3 in a row, 4 as 2 &times; 2</pre>
<p>A single-line panel shows the data with everything else in the model subtracted, each component with its FWHM, their sum,
the FWHM and flux of the line and its residuals. In the app: Results &rarr; <b>Line profiles</b>, tick the lines.</p>
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
