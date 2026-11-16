# Releasing FANTASY

FANTASY is published on PyPI as [`fantasy-agn`](https://pypi.org/project/fantasy-agn/), so users install
and update it with

```bash
pip install -U fantasy_agn
```

and the app (`fantasy app`) comes with it. Installed copies look for a newer release once a day: the app
shows **New: FANTASY x.y.z** in the header, with an Install button, and `fantasy update` does the same in a
terminal (`fantasy update --check` only checks; `FANTASY_NO_UPDATE_CHECK=1` turns the check off).

## Every release

1. Set the version in **one** place: `__version__` in `fantasy_agn/__init__.py` (`pyproject.toml` reads it).
   Use `MAJOR.MINOR.PATCH`: patch for fixes, minor for new features, major for breaking changes.
2. Give the release a section in `CHANGELOG.md` (the app's update notice links to it).
3. Commit, push, and tag:

   ```bash
   git tag v2.0.1
   git push origin main v2.0.1
   ```

   The tag starts `.github/workflows/release.yml`: tests on Linux and macOS, a check that the tag matches
   `__version__`, the build (wheel and source archive), the upload to PyPI and a GitHub release with
   the files. Nothing is uploaded if a test fails.
4. Check https://pypi.org/project/fantasy-agn/ and try it in a fresh environment:
   `pip install -U fantasy_agn && fantasy test`.

A version number can be uploaded to PyPI only once. If a release is broken, publish the next patch
version; a broken release can be *yanked* on PyPI (it stays installable only when asked for exactly).

## Once: let GitHub publish to PyPI (Trusted Publishing)

The workflow uploads without a password or token: PyPI is told to trust this one workflow.

1. On https://pypi.org, log in with the account that owns `fantasy-agn`, open **Your projects ›
   fantasy-agn › Manage › Publishing**, and add a GitHub publisher:
   owner `yukawa1`, repository `fantasy`, workflow `release.yml`, environment `pypi`.
2. On GitHub, **Settings › Environments › New environment** `pypi`. Optionally add yourself as a
   required reviewer: every upload then waits for your approval.

## Without GitHub Actions (by hand)

```bash
uv build                         # dist/fantasy_agn-X.whl and .tar.gz
uvx twine check dist/*
uvx twine upload dist/*          # asks for a PyPI API token (pypi.org › Account settings › API tokens)
```

To rehearse first, upload to TestPyPI (`uvx twine upload --repository testpypi dist/*`, with a
test.pypi.org token) and install from there:
`pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fantasy_agn`.
