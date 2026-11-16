"""HTTPS for the online services, with a fallback for missing root certificates.

Python verifies servers against OpenSSL's certificate file. On macOS that file
(``/etc/ssl/cert.pem``) lags behind the system keychain, and python.org's Python
ships with none until "Install Certificates" is run, so servers signed by newer
roots (IRSA, for example, uses SSL.com's 2022 root) are rejected although the
browser trusts them. When that happens the request is tried once more with the
roots the operating system trusts, and with certifi's when it is installed. The
check stays strict: only the set of trusted roots grows.
"""

import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

_CONTEXT = None

MACOS_KEYCHAINS = ("/System/Library/Keychains/SystemRootCertificates.keychain",
                   "/Library/Keychains/System.keychain")


def _system_roots():
    """PEM text of the roots the operating system trusts that OpenSSL may not see."""
    pems = []
    if sys.platform == "darwin":
        for keychain in MACOS_KEYCHAINS:
            if not os.path.exists(keychain):
                continue
            try:
                out = subprocess.run(["/usr/bin/security", "find-certificate", "-a", "-p", keychain],
                                     capture_output=True, text=True, timeout=20)
                pems.append(out.stdout)
            except (OSError, subprocess.SubprocessError):
                pass
    try:
        import certifi
        with open(certifi.where()) as f:
            pems.append(f.read())
    except (ImportError, OSError):
        pass
    return "\n".join(pems)


def _load(ctx, pem):
    """Add PEM certificates to an SSL context, skipping any it cannot read."""
    try:
        ctx.load_verify_locations(cadata=pem)
        return
    except (ssl.SSLError, ValueError):
        pass
    end = "-----END CERTIFICATE-----"
    for block in pem.split(end)[:-1]:
        start = block.find("-----BEGIN CERTIFICATE-----")
        if start >= 0:
            try:
                ctx.load_verify_locations(cadata=block[start:] + end + "\n")
            except (ssl.SSLError, ValueError):
                pass


def fallback_context():
    """A verifying SSL context with Python's default roots plus the system's (built once)."""
    global _CONTEXT
    if _CONTEXT is None:
        ctx = ssl.create_default_context()
        pem = _system_roots()
        if pem.strip():
            _load(ctx, pem)
        _CONTEXT = ctx
    return _CONTEXT


def _certificate_error(e):
    if isinstance(e, ssl.SSLCertVerificationError):
        return True
    reason = getattr(e, "reason", None)
    return isinstance(e, urllib.error.URLError) and isinstance(reason, ssl.SSLCertVerificationError)


def urlopen(url, timeout=None):
    """``urllib.request.urlopen``, tried again with the system's roots if the certificate is refused."""
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except (urllib.error.URLError, ssl.SSLError) as e:
        if not _certificate_error(e):
            raise
    return urllib.request.urlopen(url, timeout=timeout, context=fallback_context())


def download(url, path, timeout=120):
    """Save ``url`` to ``path`` (``urllib.request.urlretrieve``, with the same fallback)."""
    try:
        urllib.request.urlretrieve(url, path)
        return
    except (urllib.error.URLError, ssl.SSLError) as e:
        if not _certificate_error(e):
            raise
    with urllib.request.urlopen(url, timeout=timeout, context=fallback_context()) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
