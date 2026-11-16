"""
A helper process for one fit: it runs the staged search (``fitting._staged_fit``) while the
fitting process runs the direct one, so a fit takes the time of the longer search instead of
both. Started by ``fitting._Helper`` as ``python -m fantasy_agn._worker`` (not by
multiprocessing, which would import the user's script again); it reads length-prefixed
pickled requests from stdin and writes the results to stdout, one at a time, and ends when
stdin closes. Only the functions in ``ALLOWED`` are run.
"""
import os
import pickle
import struct
import sys

ALLOWED = {"_staged_fit"}


def _read(stream, n):
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def main():
    inp = sys.stdin.buffer
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                         # anything printed goes to stderr, not into the replies
    import warnings
    warnings.simplefilter("ignore")       # the fitting process reports its own warnings
    from . import fitting
    while True:
        head = _read(inp, 8)
        if head is None:
            return
        body = _read(inp, struct.unpack("<Q", head)[0])
        if body is None:
            return
        try:
            name, args = pickle.loads(body)
            if name not in ALLOWED:
                raise ValueError(f"not allowed: {name}")
            reply = (True, getattr(fitting, name)(*args))
        except Exception as exc:          # the caller falls back to running it itself
            reply = (False, f"{type(exc).__name__}: {exc}")
        data = pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL)
        out.write(struct.pack("<Q", len(data)) + data)
        out.flush()


if __name__ == "__main__":
    main()
