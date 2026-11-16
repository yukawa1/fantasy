"""
Cross-check objects against public astronomical catalogs.

:func:`lookup_catalogs` searches around the sky position of each object in
SIMBAD, NED, Milliquas, AllWISE, FIRST, NVSS, 4XMM/5XMM, the Chandra Source
Catalog and 2RXS, and flags disagreements with the fit (a catalog quasar
classified as passive, a different redshift). Only coordinates are sent. The
services are free and need no key; a service that cannot be reached is named in
the ``unreachable`` column and the others are still used.
"""
import http.client
import io
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import _net
from . import progress

__all__ = ("lookup_catalogs", "with_fit_classes", "sky_image")

# TAP endpoints; the mirrors are tried in turn, starting from the last one that worked
_SERVICES = {
    "simbad": ("https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
               "https://simbad.harvard.edu/simbad/sim-tap/sync"),
    "ned": ("https://ned.ipac.caltech.edu/tap/sync",),
    "heasarc": ("https://heasarc.gsfc.nasa.gov/xamin/vo/tap/sync",),
}
_WISE_URL = "https://irsa.ipac.caltech.edu/SCS"
_NED_API = "https://ned.ipac.caltech.edu/NED::API/"
_USER_AGENT = "fantasy-agn catalog lookup"
_LAST_GOOD = {}
_LOCK = threading.Lock()

CHUNK = 40          # positions per query
WORKERS = 6         # simultaneous requests: these are shared public services
MIRROR_WAIT = 12    # seconds before moving on to the next mirror

# key: service, table, columns, extra condition, search radius (arcsec; None: the
# radius of the lookup), column for choosing among several matches (None: nearest)
_SOURCES = {
    "simbad": ("simbad", "basic", "main_id,otype,rvz_redshift,nbref", "", None, "nbref"),
    "ned": ("ned", "NEDTAP.objdir", "prefname,prefphytype,z,zflag,n_crosref", "prefphytype<>'AbLS'", None, "n_crosref"),
    "milliquas": ("heasarc", "milliquas", "name,broad_type,redshift", "", None, None),
    "first": ("heasarc", "first", "name,int_flux_20_cm", "", 3.0, None),
    "nvss": ("heasarc", "nvss", "name,flux_20_cm", "", 15.0, None),
    "xmm": ("heasarc", "xmmssc", "name,ep_flux", "", 5.0, None),
    "chandra": ("heasarc", "csc", "name", "", 3.0, None),
    "rosat": ("heasarc", "rass2rxs", "name,plaw_flux", "", 30.0, None),
}

_NOT_A_CLASS = {"", "X", "Rad", "IR", "UV", "mm", "cm", "Opt", "?", "gam", "ev", "err", "Unknown", "UvS", "IrS",
                "RadioS", "XrayS", "VisS", "smm", "FIR", "NIR", "MIR", "blu", "HI"}
_SIMBAD_BROAD = {"QSO", "Q?", "Sy1", "Bla", "Bla?", "BLL", "BLL?", "Bz?", "QSO_Candidate"}
_SIMBAD_NARROW = {"Sy2", "Sy?", "SyG", "AGN", "AG?", "LIN", "LI?"}
_SIMBAD_TYPE2 = {"Sy2", "LIN", "LI?"}

FAMILY_LABEL = {"broad": "broad-line AGN", "narrow": "AGN (no broad lines listed)", "galaxy": "galaxy", "star": "star"}
_OURS_NO_AGN = {"passive", "retired", "starforming"}
_OURS_NARROW = {"seyfert2", "liner", "composite", "unknown"}
_OURS_BROAD = {"type1", "type1_host", "uv_mgii", "uv_civ"}
_Z_TOLERANCE = 0.01             # |dz| / (1 + z)
_STERN_W1W2, _STERN_W2_MAX = 0.8, 15.05       # Stern et al. (2012)


def _fetch(service, params, timeout):
    """POST ``params`` to a service, trying its mirrors; returns the reply as bytes."""
    urls = _SERVICES[service]
    data = urllib.parse.urlencode(params).encode()
    with _LOCK:
        first = _LAST_GOOD.get(service, 0)
    error = None
    for i in [first] + [j for j in range(len(urls)) if j != first]:
        req = urllib.request.Request(urls[i], data=data, headers={"User-Agent": _USER_AGENT})
        wait = min(timeout, MIRROR_WAIT) if len(urls) > 1 and i == first else timeout    # a mirror is left to try
        try:
            with _net.urlopen(req, timeout=wait) as reply:
                body = reply.read()
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            error = exc
            continue
        with _LOCK:
            _LAST_GOOD[service] = i
        return body
    raise OSError(f"{service}: {error}")


def _tap(service, query, timeout):
    """Rows of an ADQL query, as a list of dicts."""
    from astropy.io.votable import parse
    body = _fetch(service, {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query}, timeout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        votable = parse(io.BytesIO(body), verify="ignore")
        try:
            table = votable.get_first_table().to_table(use_names_over_ids=True)
        except IndexError:          # an error reply holds no table
            raise OSError(f"{service}: the query was refused") from None
    return [{c: _plain(table[c][i]) for c in table.colnames} for i in range(len(table))]


def _plain(value):
    """A number, a stripped string or None (masked and empty values)."""
    if value is None or value is np.ma.masked:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value.strip() or None
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sep(ra, dec, ra0, dec0):
    """Separation in arcsec (small angles)."""
    dra = (np.asarray(ra, float) - ra0 + 180.0) % 360.0 - 180.0
    return 3600.0 * np.hypot(dra * np.cos(np.radians(dec0)), np.asarray(dec, float) - dec0)


def _query_source(key, chunk, radius, timeout):
    """One catalog, one chunk of objects: {object index: best match}."""
    service, table, columns, where, source_radius, rank = _SOURCES[key]
    r = source_radius if source_radius is not None else radius
    circles = " OR ".join(f"CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{o['ra']:.6f},{o['dec']:.6f},{r / 3600:.6f}))=1"
                          for _, o in chunk)
    rows = _tap(service, f"SELECT {columns},ra,dec FROM {table} WHERE ({circles})" + (f" AND {where}" if where else ""),
                timeout)
    found = {}
    for k, o in chunk:
        near = []
        for row in rows:
            if row.get("ra") is None or row.get("dec") is None:
                continue
            sep = float(_sep(row["ra"], row["dec"], o["ra"], o["dec"]))
            if sep <= r:
                near.append({**row, "sep": sep})
        if near:
            found[k] = (max(near, key=lambda m: (m.get(rank) or 0, -m["sep"])) if rank
                        else min(near, key=lambda m: m["sep"]))
    return found


def _query_ned_activity(k, name, timeout):
    """NED's activity type of a named object ('Sy 1.5', 'LINER', ...): {object index: text} (empty if it has none)."""
    from astropy.io.votable import parse
    url = _NED_API + "OverviewOfObject?" + urllib.parse.urlencode({"TARGET": name})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with _net.urlopen(req, timeout=timeout) as reply:
        body = reply.read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            table = parse(io.BytesIO(body), verify="ignore").get_first_table().to_table(use_names_over_ids=True)
        except IndexError:          # NED holds no overview under this name
            return {}
    text = _plain(table["Activity Type"][0]) if len(table) and "Activity Type" in table.colnames else None
    return {k: text} if text else {}


def _query_wise(k, o, radius, timeout):
    """The nearest AllWISE source: {object index: match} (the cone search handles one position)."""
    url = _WISE_URL + "?" + urllib.parse.urlencode({"table": "allwise_p3as_psd", "RA": f"{o['ra']:.6f}",
                                                    "DEC": f"{o['dec']:.6f}", "SR": f"{radius / 3600:.6f}",
                                                    "FORMAT": "csv"})
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(2):         # the service sometimes answers with an error page once
        try:
            with _net.urlopen(req, timeout=timeout) as reply:
                body = reply.read()
            if body[:1] != b"<":
                break
            error = OSError("wise: the query was refused")
        except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            error = exc
    else:
        raise error
    if not body.strip():
        return {}
    t = pd.read_csv(io.BytesIO(body))
    if t.empty:
        return {}
    seps = _sep(t.ra.values, t.dec.values, o["ra"], o["dec"])
    i = int(np.argmin(seps))
    return {k: {**{c: _plain(t[c].iloc[i]) for c in t.columns}, "sep": float(seps[i])}}


# ---------------------------------------------------------------- the objects
def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _from_table(t):
    """Objects of a table with columns ra, dec and optionally name, z, cls."""
    if "ra" not in t and "path" in t:          # a sample fitted before ra and dec were stored
        from .spectrum import Spectrum
        t = t.copy()
        pos = []
        for path in t["path"]:
            try:
                s = Spectrum.read(path)
                pos.append((s.ra, s.dec))
            except Exception:
                pos.append((None, None))
        t["ra"], t["dec"] = [p[0] for p in pos], [p[1] for p in pos]
    if "ra" not in t or "dec" not in t:
        raise ValueError("the table needs columns ra and dec (degrees)")
    get = lambda row, c: row[c] if c in t and not (_is_number(row[c]) and not np.isfinite(row[c])) else None
    return [{"name": get(row, "name"), "ra": get(row, "ra"), "dec": get(row, "dec"), "z": get(row, "z"),
             "cls": get(row, "cls")} for _, row in t.iterrows()]


def _objects(objects, cls):
    """A list of dicts (name, ra, dec, z, cls) from whatever was given."""
    if isinstance(objects, pd.DataFrame):
        return _from_table(objects)
    if hasattr(objects, "table") and isinstance(objects.table, pd.DataFrame) and hasattr(objects, "failed"):
        return _from_table(objects.table)                       # a SampleResult
    if isinstance(objects, (tuple, list)) and len(objects) == 2 and all(_is_number(v) for v in objects):
        return [{"name": None, "ra": float(objects[0]), "dec": float(objects[1]), "z": None, "cls": cls}]
    if isinstance(objects, (str, os.PathLike)):
        from .spectrum import Spectrum
        try:
            return _objects(Spectrum.read(os.fspath(objects)), cls)
        except Exception:
            name = os.path.splitext(os.path.basename(os.fspath(objects)))[0]
            return [{"name": name, "ra": None, "dec": None, "z": None, "cls": cls}]
    if isinstance(objects, (tuple, list)):
        many = [o for item in objects for o in _objects(item, cls)]
        if not many:
            raise ValueError("nothing to look up")
        return many
    label = getattr(objects, "label", None) if hasattr(objects, "RECOMMENDED") else None
    spec = getattr(objects, "spectrum", objects)                # a fit or a classification
    if not hasattr(spec, "ra"):
        raise TypeError("give a Spectrum, a fit, a classification, a sample, a table with ra and dec, or (ra, dec)")
    return [{"name": spec.name, "ra": spec.ra, "dec": spec.dec, "z": spec.z, "cls": cls or label}]


# ------------------------------------------------------------ one result row
def _family_of(source, otype):
    """'broad', 'narrow', 'galaxy', 'star' or None, from a catalog's own type code."""
    if not otype:
        return None
    t = str(otype).strip()
    if source == "milliquas":
        if set(t) & set("QAB"):
            return "broad"
        if set(t) & set("KN"):
            return "narrow"
        return "star" if "S" in t else None
    if t in _NOT_A_CLASS:
        return None
    if source == "ned":
        return "broad" if t == "QSO" else "star" if "*" in t else "galaxy" if t.startswith(("G", "PofG")) else None
    if t in _SIMBAD_BROAD:
        return "broad"
    if t in _SIMBAD_NARROW:
        return "narrow"
    return "star" if "*" in t else "galaxy"


def _activity_family(text):
    """'broad' or 'narrow' from NED's activity type; None if it does not say."""
    t = str(text or "").strip()
    if t.startswith(("Sy 2", "LINER")) or t in ("Sy", "AGN"):
        return "narrow"
    low = t.lower()
    if t.startswith("Sy 1") or any(word in low for word in ("qso", "quasar", "bl lac", "blazar")):
        return "broad"
    return None


def _ned_view(row):
    """What NED says: (family, in its words). The activity type is more specific than the physical type."""
    family = _activity_family(row.get("ned_activity"))
    if family:
        return family, row["ned_activity"]
    return _family_of("ned", row.get("ned_type")), row.get("ned_type")


def _combine_family(families):
    if "broad" in families:
        return "broad"
    if "narrow" in families:
        return "narrow"
    if "galaxy" in families:
        return "galaxy"
    return "star" if "star" in families else None


def _catalog_redshifts(row):
    """(source, z) of the spectroscopic catalog redshifts; NED's photometric ones are left out."""
    out = []
    if row.get("ned_z") is not None and not str(row.get("ned_z_flag") or "").upper().startswith("PHOT"):
        out.append(("NED", float(row["ned_z"])))
    for source, label in (("simbad_z", "SIMBAD"), ("milliquas_z", "Milliquas")):
        if row.get(source) is not None:
            out.append((label, float(row[source])))
    return [(s, z) for s, z in out if z > 0]


def _listed_as(row, family):
    """Which catalogs say ``family``, in words: 'NED: QSO, SIMBAD: BLL'."""
    ned_family, ned_text = _ned_view(row)
    parts = [f"NED: {ned_text}"] if ned_family == family else []
    return ", ".join(parts + [f"{label}: {row[f'{key}_type']}" for key, label in
                              (("simbad", "SIMBAD"), ("milliquas", "Milliquas"))
                              if _family_of(key, row.get(f"{key}_type")) == family])


def _flags(row, z, cls):
    """Disagreements between the catalogs and the fit, in words."""
    flags = []
    zs = _catalog_redshifts(row)
    if z is not None and zs and z > 0 and row.get("catalog_family") != "star":
        src, cat = min(zs, key=lambda it: abs(it[1] - z))
        if abs(cat - z) / (1 + z) > _Z_TOLERANCE:
            flags.append(f"Redshift differs: {z:.4f} used here, {cat:.4f} in {src}.")
    fam = row.get("catalog_family")
    if cls in _OURS_NO_AGN | _OURS_NARROW and fam == "broad":
        flags.append(f"Listed as a broad-line AGN ({_listed_as(row, 'broad')}), classified here as {cls}.")
    elif cls in _OURS_NO_AGN and fam == "narrow":
        flags.append(f"Listed as an AGN ({_listed_as(row, 'narrow')}), classified here as {cls}.")
    elif cls in _OURS_BROAD and fam == "narrow":
        flags.append(f"Listed without broad lines ({_listed_as(row, 'narrow')}), fitted here with broad lines. "
                     "It may be a wrong entry or a real change of state.")
    elif cls in _OURS_BROAD and fam == "galaxy":
        flags.append(f"Listed as a galaxy ({_listed_as(row, 'galaxy')}), fitted here with broad lines.")
    if fam == "star":
        flags.append(f"Listed as a star ({_listed_as(row, 'star')}).")
    w12, w2 = row.get("wise_w1_w2"), row.get("wise_w2")
    if cls in _OURS_NO_AGN and w12 is not None and w12 >= _STERN_W1W2 and (w2 is None or w2 < _STERN_W2_MAX):
        flags.append(f"WISE colour W1−W2 = {w12:.2f} looks like an AGN (0.8 or more), classified here as {cls}.")
    return flags


def _row(obj, m, radius, unreachable):
    """The flat result row of one object from its matches ``m`` {source: match}."""
    row = {"name": obj["name"], "ra": obj["ra"], "dec": obj["dec"], "z": obj["z"], "cls": obj["cls"]}
    if "simbad" in m:
        s = m["simbad"]
        row.update(simbad_name=s.get("main_id"), simbad_type=s.get("otype"), simbad_z=s.get("rvz_redshift"),
                   simbad_refs=s.get("nbref"), simbad_sep=s["sep"])
    if "ned" in m:
        s = m["ned"]
        row.update(ned_name=s.get("prefname"), ned_type=s.get("prefphytype"), ned_z=s.get("z"),
                   ned_z_flag=s.get("zflag"), ned_refs=s.get("n_crosref"), ned_sep=s["sep"],
                   ned_activity=s.get("activity"))
    if "milliquas" in m:
        s = m["milliquas"]
        row.update(milliquas_name=s.get("name"), milliquas_type=s.get("broad_type"), milliquas_z=s.get("redshift"),
                   milliquas_sep=s["sep"])
    if "wise" in m:
        s = m["wise"]
        row.update(wise_name=s.get("designation"), wise_sep=s["sep"])
        quality = str(s.get("ph_qual") or "UUUU")
        for i, band in enumerate(("w1", "w2", "w3", "w4")):
            usable = quality[i:i + 1] in ("A", "B", "C")
            row[f"wise_{band}"] = s.get(f"{band}mpro") if usable else None
        if row["wise_w1"] is not None and row["wise_w2"] is not None:
            row["wise_w1_w2"] = float(row["wise_w1"]) - float(row["wise_w2"])
        if row["wise_w2"] is not None and row["wise_w3"] is not None:
            row["wise_w2_w3"] = float(row["wise_w2"]) - float(row["wise_w3"])
    if "first" in m:
        s = m["first"]
        row.update(first_name=s.get("name"), first_mjy=s.get("int_flux_20_cm"), first_sep=s["sep"])
    if "nvss" in m:
        s = m["nvss"]
        row.update(nvss_name=s.get("name"), nvss_mjy=s.get("flux_20_cm"), nvss_sep=s["sep"])
    if "xmm" in m:
        s = m["xmm"]
        row.update(xmm_name=s.get("name"), xmm_flux=s.get("ep_flux"), xmm_sep=s["sep"])
    if "chandra" in m:
        row.update(chandra_name=m["chandra"].get("name"), chandra_sep=m["chandra"]["sep"])
    if "rosat" in m:
        s = m["rosat"]
        row.update(rosat_name=s.get("name"), rosat_flux=s.get("plaw_flux"), rosat_sep=s["sep"])
    families = [_ned_view(row)[0]] + [_family_of(k, row.get(f"{k}_type")) for k in ("simbad", "milliquas")]
    type2 = row.get("simbad_type") in _SIMBAD_TYPE2 or str(row.get("ned_activity") or "").startswith(("Sy 2", "LINER"))
    row["catalog_family"] = "narrow" if type2 else _combine_family([f for f in families if f])    # the most specific
    row["catalog_class"] = FAMILY_LABEL.get(row["catalog_family"])
    row["catalog_name"] = row.get("simbad_name") or row.get("ned_name") or row.get("milliquas_name")
    zs = _catalog_redshifts(row)
    if zs:
        row["catalog_z_source"], row["catalog_z"] = zs[0]
    row["radio_mjy"] = row.get("first_mjy") if row.get("first_mjy") is not None else row.get("nvss_mjy")
    xray = [row.get(k) for k in ("xmm_name", "chandra_name", "rosat_name") if row.get(k)]
    row["xray"] = ", ".join(xray) if xray else None
    flags = _flags(row, obj["z"], obj["cls"])
    row["disagreements"] = " ".join(flags) if flags else None
    row["n_disagreements"] = len(flags)
    row["unreachable"] = ", ".join(sorted(unreachable)) if unreachable else None
    return row


def lookup_catalogs(objects, cls=None, radius=3.0, timeout=30, verbose=False):
    """
    What public catalogs know about one or many objects, and where they
    disagree with the fit.

    Searches ``radius`` arcsec around each position in SIMBAD, NED, Milliquas
    and AllWISE; radio and X-ray catalogs use radii that suit their positional
    accuracy (FIRST 3", NVSS 15", XMM 5", Chandra 3", 2RXS 30"). Only the
    coordinates leave the computer. A service that cannot be reached is named
    in the ``unreachable`` column; the rest of the row still holds.

    The redshift check is independent only if the catalog redshift does not come
    from the same spectrum, which is often the case for SDSS objects.

    :param objects: a Spectrum, a file path, a fit, a Classification, a sample
        (``fa.fit_sample``), a table with columns ra, dec (degrees; optionally
        name, z, cls), an (ra, dec) pair, or a list of those.
    :param cls: the class given to a single object (the label of ``fa.classify``),
        used for the disagreement flags.
    :param radius: search radius in arcsec for the optical and infrared catalogs.
    :param timeout: seconds to wait for one reply.
    :return: pandas DataFrame, one row per object in the input order, with
        ``catalog_name``, ``catalog_class``, ``catalog_z``, the matches of each
        catalog (``simbad_*``, ``ned_*``, ``milliquas_*``, ``wise_*``,
        ``first_*``, ``nvss_*``, ``xmm_*``, ``chandra_*``, ``rosat_*``),
        ``disagreements`` (in words), ``n_disagreements`` and ``unreachable``.

    Example::

        fa.lookup_catalogs((187.27792, 2.05239))      # by position
        table = fa.lookup_catalogs(sample)            # a whole fitted sample
        table[table.n_disagreements > 0][["name", "disagreements"]]
    """
    objs = _objects(objects, cls)
    good = [(k, o) for k, o in enumerate(objs) if o["ra"] is not None and o["dec"] is not None]
    for _, o in good:
        o["ra"], o["dec"] = float(o["ra"]), float(o["dec"])
    chunks = [good[i:i + CHUNK] for i in range(0, len(good), CHUNK)]
    tasks = [(key, [k for k, _ in chunk], _query_source, (key, chunk, radius, timeout))
             for chunk in chunks for key in _SOURCES]
    tasks += [("wise", [k], _query_wise, (k, o, radius, timeout)) for k, o in good]
    matches, unreachable = {k: {} for k, _ in good}, {k: set() for k, _ in good}
    task = "Looking up catalogs"
    progress.report(task, 0, len(tasks))
    finished = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(fn, *args): (key, ks) for key, ks, fn, args in tasks}
        for future in as_completed(futures):
            finished += 1
            key, ks = futures[future]
            try:
                for k, match in future.result().items():
                    matches[k][key] = match
            except Exception as exc:
                for k in ks:
                    unreachable[k].add(key)
                if verbose:
                    print(f"  {key}: {exc}")
            progress.report(task, finished, len(tasks), key)
        # NED files most Seyferts under 'G'; only those need the slower per-object call
        named = [(k, m["ned"]["prefname"]) for k, m in matches.items()
                 if m.get("ned", {}).get("prefname") and str(m["ned"].get("prefphytype") or "").startswith("G")]
        futures = {pool.submit(_query_ned_activity, k, name, timeout): k for k, name in named}
        for future in as_completed(futures):
            finished += 1
            k = futures[future]
            try:
                for i, text in future.result().items():
                    matches[i]["ned"]["activity"] = text
            except Exception as exc:
                unreachable[k].add("ned activity")
                if verbose:
                    print(f"  ned activity: {exc}")
            progress.report(task, finished, len(tasks) + len(named), "ned activity")
    rows = []
    for k, o in enumerate(objs):
        if o["ra"] is None or o["dec"] is None:
            rows.append({"name": o["name"], "ra": None, "dec": None, "z": o["z"], "cls": o["cls"], "n_disagreements": 0,
                         "unreachable": None, "note": "no coordinates"})
        else:
            rows.append(_row(o, matches[k], radius, unreachable[k]))
    table = pd.DataFrame(rows)
    if verbose:
        found = int(table.get("catalog_name", pd.Series(dtype=object)).notna().sum())
        print(f"{found} of {len(table)} objects found in the catalogs; {int(table.n_disagreements.sum())} disagreements")
    return table


def _missing(value):
    return value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))


def with_fit_classes(table, z=None, cls=None):
    """
    A copy of a :func:`lookup_catalogs` table with the disagreements worked out
    again for new redshifts and classes (no query is made), e.g. a lookup made
    before the fit and the classes that the fit gave. ``z`` and ``cls`` are
    sequences of the length of the table; a missing value keeps the one in the table.
    """
    out = table.copy()
    if out.empty:
        return out
    z = [None] * len(out) if z is None else list(z)
    cls = [None] * len(out) if cls is None else list(cls)
    texts, counts, zs, classes = [], [], [], []
    for (_, row), zi, ci in zip(table.iterrows(), z, cls):
        r = {k: v for k, v in row.items() if not _missing(v)}
        zi = r.get("z") if _missing(zi) else zi
        ci = r.get("cls") if _missing(ci) else ci
        flags = _flags(r, zi, ci) if "ra" in r else []
        texts.append(" ".join(flags) if flags else None)
        counts.append(len(flags))
        zs.append(zi)
        classes.append(ci)
    out["disagreements"], out["n_disagreements"], out["z"], out["cls"] = texts, counts, zs, classes
    return out


# ------------------------------------------------------------- image of the sky
def _image_ok(body):
    """True for a decodable image with something on it (outside a survey the image is one flat colour)."""
    from PIL import Image
    try:
        return float(np.std(np.asarray(Image.open(io.BytesIO(body)).convert("L"), float))) > 2.0
    except Exception:
        return False


def _get(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with _net.urlopen(req, timeout=timeout) as reply:
        return reply.read()


def _sdss_image(ra, dec, fov, size, timeout):
    return _get("https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?" + urllib.parse.urlencode(
        {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "scale": f"{fov / size:.4f}", "width": size, "height": size}), timeout)


def _panstarrs_image(ra, dec, fov, size, timeout):
    base = "https://ps1images.stsci.edu/cgi-bin/"
    listing = _get(f"{base}ps1filenames.py?" + urllib.parse.urlencode({"ra": f"{ra:.6f}", "dec": f"{dec:.6f}",
                                                                       "filters": "gri"}), timeout)
    t = pd.read_csv(io.BytesIO(listing), sep=r"\s+")
    files = dict(zip(t["filter"], t["filename"]))
    return _get(f"{base}fitscut.cgi?" + urllib.parse.urlencode(
        {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "size": int(round(fov / 0.25)), "output_size": size, "format": "jpg",
         "red": files["i"], "green": files["r"], "blue": files["g"]}), timeout)


def _legacy_image(ra, dec, fov, size, timeout):
    return _get("https://www.legacysurvey.org/viewer/jpeg-cutout?" + urllib.parse.urlencode(
        {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "layer": "ls-dr10", "pixscale": f"{fov / size:.4f}", "size": size}), timeout)


def _dss_image(ra, dec, fov, size, timeout):
    return _get("https://alasky.cds.unistra.fr/hips-image-services/hips2fits?" + urllib.parse.urlencode(
        {"hips": "CDS/P/DSS2/color", "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "width": size, "height": size,
         "fov": f"{fov / 3600:.6f}", "projection": "TAN", "format": "jpg"}), timeout)


_IMAGE_SURVEYS = (("SDSS", _sdss_image), ("Pan-STARRS", _panstarrs_image), ("Legacy Surveys", _legacy_image),
                  ("DSS", _dss_image))


def sky_image(ra, dec, fov=60.0, size=300, timeout=15):
    """
    A colour image of the sky around a position, north up and east left, from
    the first of SDSS, Pan-STARRS, the Legacy Surveys and DSS that covers it.

    :param ra, dec: position in degrees.
    :param fov: side of the image in arcsec.
    :param size: side of the image in pixels.
    :param timeout: seconds to wait for all the surveys together.
    :return: (JPEG bytes, name of the survey), or (None, None) if none has it.
    """
    deadline, bodies = time.monotonic() + timeout, {}

    def fetch(survey, get):
        try:
            bodies[survey] = get(float(ra), float(dec), float(fov), int(size), timeout)
        except Exception:
            bodies[survey] = None

    # all at once, in threads that cannot hold up the caller or the end of the program
    threads = [(survey, threading.Thread(target=fetch, args=(survey, get), daemon=True))
               for survey, get in _IMAGE_SURVEYS]
    for _, thread in threads:
        thread.start()
    for survey, thread in threads:          # in order of preference
        thread.join(max(0.0, deadline - time.monotonic()))
        if bodies.get(survey) and _image_ok(bodies[survey]):
            return bodies[survey], survey
    return None, None
