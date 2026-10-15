FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis
===============================================================

Installing
----------
macOS     double-click  "Install FANTASY.command"
          (if macOS refuses to open it: right-click > Open, or run  sh install.sh  in Terminal)
Linux     sh install.sh
Windows   double-click  install.bat

The installer
  1. uses uv (https://docs.astral.sh/uv/) - and offers to install it if it is missing;
  2. installs FANTASY with Python 3.12 and the tested package versions (constraints.txt)
     in an environment of its own: no other Python installation is used or changed;
  3. adds the commands  fantasy  and  fantasy-app  (uv puts its tool folder on your PATH);
  4. runs the self-test  fantasy test --full  (downloads the stellar templates, 13 MB, once)
     and offers to start the app.

Using it
--------
  fantasy app              the FANTASY app in your browser (read -> classify -> fit -> results)
  fantasy test             check the installation again (offline)
  fantasy info             versions, processor cores, templates
  fantasy examples DIR     example spectra and a first Python script
  fantasy python           Python with FANTASY (also: fantasy python first_fit.py)

In Python:
  import fantasy_agn as fa
  spec = fa.read(fa.examples.path("type1")).deredden().to_rest_frame()
  r = fa.fit(spec, fa.presets.hbeta()); r.plot()

Problems
--------
If a check fails, run  fantasy test --full --verbose  and send the output to the authors.
Uninstall:  uv tool uninstall fantasy_agn   (data and templates: ~/.fantasy_agn)
