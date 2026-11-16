"""Generate the static website documentation from this checkout, including local edits.

From the project root: .venv/bin/python tools/build_website_docs.py
"""
import hashlib
import html
import inspect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fantasy_agn as fa
from fantasy_agn.app.docs import all_pages

OUT = ROOT / "website" / "docs"
FIRST_FIT = '''import fantasy_agn as fa

spectrum = fa.read(fa.examples.path("type1")).to_rest_frame()
model = fa.presets.hbeta(broad_components=2, feii=True)
result = fa.fit(spectrum, model)
hb = result.measure("Hb", ["broad", "broad2"])
print(hb["fwhm"], hb["fwhm_err"])  # km/s
print(hb["flux"], hb["flux_err"])  # in the spectrum's integrated flux units
result.plot()'''

def fingerprint():
    digest = hashlib.sha256()
    for p in sorted((ROOT / "fantasy_agn").rglob("*.py")):
        digest.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return digest.hexdigest()[:12]

def text(value):
    return html.escape(html.unescape(value))

def home():
    return f'''<p class="lead">Fit your first Hβ spectrum, then use the guides to prepare your own data, choose components and check your measurements.</p>
<div class="links"><a href="install.html">Install and launch →</a><a href="start.html">Use the browser app →</a><a href="api-fitting.html">Python API →</a></div>
<section id="app-walkthrough"><h2>Watch a first fit in the app</h2>
<p>Follow a bundled SDSS quasar from the file chooser to its broad Hβ width and flux. To use the same spectrum, run <code>fantasy examples fantasy-examples</code>, then <code>fantasy app</code>.</p>
<video id="first-fit-video" controls playsinline preload="none" width="1440" height="960" style="width:100%;height:auto;aspect-ratio:3/2" poster="../assets/first-fit-poster.jpg" aria-label="Your first Hβ fit in the FANTASY app">
<source src="../assets/first-fit.mp4" type="video/mp4"><track kind="captions" srclang="en" label="English" src="../assets/first-fit.en.vtt">
<a href="../assets/first-fit.mp4">Download the walkthrough</a></video>
<p>1 min 41 s · English narration and captions · <a href="../assets/first-fit.mp4" download>Download MP4</a></p>
<details><summary>Chapters and written walkthrough</summary><ol><li><a href="../assets/first-fit.mp4#t=0.000" data-seek="0.000">0:00 · Read</a> — Choose Single spectrum. Under Files, load spec-1059-52618-0453.fits and confirm the redshift in the loaded table.</li>
<li><a href="../assets/first-fit.mp4#t=14.250" data-seek="14.250">0:14 · Prepare</a> — Keep the redshift, flux unit and resolution from the file. Move to rest frame. This offline example uses no Galactic extinction correction.</li>
<li><a href="../assets/first-fit.mp4#t=31.000" data-seek="31.000">0:31 · Model</a> — Choose hbeta with two broad components. Inspect the initial guess and the continuum, Fe II, narrow-line and broad-line components.</li>
<li><a href="../assets/first-fit.mp4#t=43.708" data-seek="43.708">0:43 · Fit</a> — Open Set up fit. Select Fast for covariance errors, run the fit and open the results.</li>
<li><a href="../assets/first-fit.mp4#t=53.000" data-seek="53.000">0:53 · Results</a> — Read the combined broad Hβ profile. This run gives FWHM 4,552 ± 69 km/s and flux (3.94 ± 0.13) × 10⁻¹⁴ erg s⁻¹ cm⁻².</li>
<li><a href="../assets/first-fit.mp4#t=70.208" data-seek="70.208">1:10 · Checks</a> — Inspect the parameter-at-limit warning. In Checks, expand “Does the model miss anything?” and run Analyse residuals. This fit flags one feature for inspection.</li>
<li><a href="../assets/first-fit.mp4#t=86.333" data-seek="86.333">1:26 · Export</a> — Use Download → Fitted lines for the CSV table. The top-right menu offers Download script (.py) to reproduce the session.</li></ol></details>
<script defer src="../assets/walkthrough.js"></script></section>
<h2 id="first-fit">Measure broad Hβ in Python</h2>
<p>After <a href="install.html">installing this source version</a>, run this example with the bundled SDSS quasar. The FITS file supplies the redshift. This offline example uses the recorded flux. For extinction-corrected fluxes, apply <code>Spectrum.deredden()</code> in the observed frame before <code>to_rest_frame()</code>. The Hβ preset fits the local continuum, Fe II and the line components together.</p>
<pre><code>{html.escape(FIRST_FIT)}</code></pre>
<p>Passing both broad component names excludes narrow Hβ from the flux and width. Check <code>hb["fwhm_fragile"]</code> for a shoulder near half maximum. For a galaxy with visible stellar absorption, enable the host in the model, for example <code>fa.presets.hbeta(host="pca")</code>. The <a href="measurements.html">measurement guide</a> explains line width, dispersion and equivalent width.</p>
<h2>Work with the current fitting tools</h2><dl>
<dt><a href="api-reading-data.html#fa-find-redshift">Find an unknown redshift</a></dt><dd>Scan the observed spectrum before moving to rest frame. Inspect the alternative redshifts and the lines expected at each one.</dd>
<dt><a href="api-fitting.html#fa-prune-feii">Test weak Fe II multiplets</a></dt><dd>Compare the original fit with the refit after pruning. Weak groups can bias the blend in either direction depending on which groups the data constrain.</dd>
<dt><a href="api-fit-checks.html">Check your fit</a></dt><dd>Inspect residuals, simulate spectra to test coverage, and compare broad-line measurements across alternative fits.</dd>
<dt><a href="api-epochs-and-reverberation.html">Work with repeated observations</a></dt><dd>Inspect the relative flux scales and wavelength offsets from a joint fit, build light curves and estimate reverberation lags.</dd></dl>
<h2>About this edition</h2><p>These pages use the app guides, line list, function signatures and docstrings from the FANTASY {fa.__version__} working tree. To use this edition, install from the matching checkout or tester bundle. The public repository can be at a different revision.</p>'''

def route(page):
    return page['id'] + '.html'

def build():
    assert Path(inspect.getfile(fa)).resolve().parent == ROOT / "fantasy_agn"
    OUT.mkdir(exist_ok=True)
    pages = [dict(id='index', title='First fit', section='Getting started', html=home()), *all_pages()]
    groups = defaultdict(list)
    entries = []
    anchors = {}
    for page in pages:
        groups[page['section']].append(page)
        url = route(page)
        entries.append([html.unescape(page['title']), page['section'], url])
        for anchor, signature in re.findall(r'<div class="api" id="([^"]+)"><h3><code>(.*?)</code>', page['html']):
            name = html.unescape(signature.split('(')[0])
            entries.append([name, page['title'], f'{url}#{anchor}'])
        anchors[page['id']] = url

    snapshot = fingerprint()
    for page in pages:
        nav = []
        for section, group in groups.items():
            links = []
            for item in group:
                active = ' aria-current="page"' if item is page else ''
                links.append(f'<a href="{route(item)}"{active}>{text(item["title"])}</a>')
            nav.append(f'<section><h3>{text(section)}</h3>{"".join(links)}</section>')
        body = re.sub(r'data-doc="([^"]+)"', lambda m: f'href="{anchors[m[1]]}"', page['html'])
        body = re.sub(r'<table\b.*?</table>', lambda m: '<div class="table-scroll">'+m[0]+'</div>', body, flags=re.S)
        title = text(page['title'])
        content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · FANTASY documentation</title><link rel="icon" href="../assets/mark.png">
<link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="docs.css">
<script defer src="search-index.js"></script><script defer src="docs.js"></script></head>
<body><a class="skip" href="#content">Skip to article</a>
<header class="mast"><nav class="wrap" aria-label="Main navigation"><a class="brand" href="../index.html"><img src="../assets/mark.png" alt="">FANTASY</a>
<div class="doc-mast-links"><a href="../index.html#explore">Spectrum explorer</a><a href="index.html">Documentation</a></div></nav></header>
<div class="doc-layout"><aside><details id="contents" open><summary>Documentation menu</summary><div class="menu">
<label for="search">Find a guide or function</label><input type="search" id="search" autocomplete="off" placeholder="e.g. prune_feii"><nav id="matches" aria-label="Search results" hidden></nav>
<nav id="topics" aria-label="Documentation topics">{"".join(nav)}</nav></div></details></aside>
<main id="content" class="article"><p class="kicker"><a href="index.html">FANTASY {fa.__version__}</a> / {text(page["section"])}</p>
<h1>{title}</h1>{body}
<footer class="doc-foot"><a href="index.html">Documentation home</a><span>Source snapshot {snapshot}</span></footer></main></div></body></html>
'''
        (OUT / route(page)).write_text(content, encoding='utf-8')
    (OUT / 'search-index.js').write_text('window.docEntries = ' + json.dumps(entries, ensure_ascii=False) + ';\n', encoding='utf-8')
    print(f'{len(pages)} pages, {len(entries)-len(pages)} API entries, source {snapshot}')

if __name__ == "__main__":
    build()
