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

Version 1
=========

FANTASY 1.x (based on Sherpa) stays available for reproducing earlier work and gets only
critical fixes::

    pip install "fantasy_agn<2"    # Python 3.10 to 3.12

Its code is on the `1.x branch <https://github.com/yukawa1/fantasy/tree/1.x>`_ and its
tutorials in the `1.x documentation <https://fantasy-agn.readthedocs.io/en/1.x/>`_. How to
move a 1.x script to version 2 is in the
`CHANGELOG <https://github.com/yukawa1/fantasy/blob/main/CHANGELOG.md#moving-a-1x-script>`_.
