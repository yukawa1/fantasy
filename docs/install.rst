************
Installation
************

::

    pip install fantasy_agn

FANTASY needs only numpy, scipy, astropy, pandas and matplotlib, and works on Linux,
macOS and Windows with Python 3.10 or newer.

Optional extras::

    pip install "fantasy_agn[dust]"   # E(B-V) lookup in the SFD98 dust maps (sfdmap2)

For the dust-map lookup, also download the SFD map files and point the environment
variable ``SFD_DIR`` to them. Alternatively, pass ``ebv=...`` to
``Spectrum.deredden``.
