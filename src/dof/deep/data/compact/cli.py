"""Command line of the compact store.

  python -m dof.deep.data.compact build  --src SRC --dst DST --pattern GLOB --context NAME [NAME ...]
                                         [--rate-mb-s 10] [--pause-s 0.05] [--free-path PATH] [--min-free-gib 0]
  python -m dof.deep.data.compact verify --dst DST [--src SRC] [--rate-mb-s 10] [--pause-s 0.05]
  python -m dof.deep.data.compact show   --dst DST

build   converts every file of SRC that matches GLOB, then seals the store. Run again after a stop, it continues:
        converted files are skipped. On a sealed store it only verifies; it never converts again.
verify  reads a sealed store against its manifest, without loading it into memory.
show    prints the totals of a sealed store.

--rate-mb-s is in units of 1,000,000 bytes per second; 0 means no rate limit. --min-free-gib is in units of
2**30 bytes; without --free-path no floor is checked.

Every line of build and verify starts with an ISO time. The last line is the verdict: ``BUILD_DONE ...``,
``VERIFY_OK ...`` or ``<CMD>_FAILED: <message>``, and the exit status is 1 after a failure. A log of a run that
was not killed therefore always ends with a verdict.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime
import importlib
import traceback
from pathlib import Path

from dof.deep.data.compact.errors import CompactError
from dof.deep.data.compact.layout import manifest_path

# Through the module, never through names imported from it: the tests replace functions of the module. (The
# package's own name ``convert`` is the function, so the module is taken from the import system.)
convert_mod = importlib.import_module("dof.deep.data.compact.convert")

# glibc's malloc: after the first large block is freed, glibc raises its mmap threshold and serves the next large
# blocks from the heap, and the heap does not shrink again. A fixed threshold turns that adjustment off
# (mallopt(3)): every block of BIG_ALLOC bytes or more gets its own mapping and is returned when freed.
M_MMAP_THRESHOLD = -3           # <malloc.h>
BIG_ALLOC = 128 * 1024          # glibc's own starting value


def log(msg: str) -> None:
    print(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def plain(msg: str) -> None:
    print(msg, flush=True)


def _verify_ok(s: dict) -> str:
    return (f"VERIFY_OK files={s['files']} rows={s['rows']} compressed_bytes={s['compressed_bytes']} "
            f"manifest_sha256={s['manifest_sha256']}")


def _build_done(s: dict) -> str:
    return (f"BUILD_DONE files={s['files']} rows={s['rows']} source_bytes={s['source_bytes']} "
            f"compressed_bytes={s['compressed_bytes']} bytes_per_row={s['bytes_per_row']:.3f} "
            f"verified_rows={s['verified_rows']} manifest_sha256={s['manifest_sha256']}")


def cmd_build(args) -> None:
    src, dst, rate = Path(args.src), Path(args.dst), args.rate_mb_s * 1_000_000
    if manifest_path(dst).exists():
        # Sealed: verify, say so, then the same verdict line, so that the last line of a log means the same after a
        # second run of the same command.
        summary = convert_mod.verify(dst, src_root=src, rate=rate, pause=args.pause_s, log=log)
        log(_verify_ok(summary))
    else:
        convert_mod.convert(src, dst, args.pattern, args.context, rate=rate, pause=args.pause_s,
                            free_path=None if args.free_path is None else Path(args.free_path),
                            min_free_bytes=int(args.min_free_gib * 2**30), log=log)
        summary = convert_mod.finalize(src, dst, args.pattern, args.context, rate=rate, pause=args.pause_s, log=log)
    log(_build_done(summary))


def cmd_verify(args) -> None:
    summary = convert_mod.verify(Path(args.dst), src_root=None if args.src is None else Path(args.src),
                                 rate=args.rate_mb_s * 1_000_000, pause=args.pause_s, log=log)
    log(_verify_ok(summary))


def cmd_show(args) -> None:
    s = convert_mod.show(Path(args.dst))
    for name in ("root", "manifest_sha256", "files", "rows", "source_bytes", "compressed_bytes"):
        plain(f"{name} {s[name]}")
    plain(f"bytes_per_row {s['bytes_per_row']:.3f}")
    plain(f"verified_rows {s['verified_rows']}")
    for name, digest in s["context"].items():
        plain(f"context {name} {digest}")


def _fix_mmap_threshold() -> int | None:
    """mallopt's return value (1 on success); None where the C library has no mallopt."""
    name = ctypes.util.find_library("c")
    if name is None:
        return None
    mallopt = getattr(ctypes.CDLL(name), "mallopt", None)
    return None if mallopt is None else mallopt(M_MMAP_THRESHOLD, BIG_ALLOC)


def main(argv: list[str] | None = None) -> int:
    fixed = _fix_mmap_threshold()
    ap = argparse.ArgumentParser(prog="python -m dof.deep.data.compact", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--context", nargs="+", required=True)
    p.add_argument("--rate-mb-s", type=float, default=10.0)
    p.add_argument("--pause-s", type=float, default=0.05)
    p.add_argument("--free-path")
    p.add_argument("--min-free-gib", type=float, default=0.0)
    p = sub.add_parser("verify")
    p.add_argument("--dst", required=True)
    p.add_argument("--src")
    p.add_argument("--rate-mb-s", type=float, default=10.0)
    p.add_argument("--pause-s", type=float, default=0.05)
    p = sub.add_parser("show")
    p.add_argument("--dst", required=True)
    args = ap.parse_args(argv)
    say = plain if args.cmd == "show" else log
    if fixed is not None and fixed != 1:
        log(f"mallopt(M_MMAP_THRESHOLD, {BIG_ALLOC}) returned {fixed}, not 1: glibc may keep freed memory; not fatal")
    try:
        {"build": cmd_build, "verify": cmd_verify, "show": cmd_show}[args.cmd](args)
    except (CompactError, ValueError) as e:
        say(f"{args.cmd.upper()}_FAILED: {e}")
        return 1
    except Exception as e:      # the log must end with a verdict, whatever went wrong
        traceback.print_exc()
        say(f"{args.cmd.upper()}_FAILED: unexpected {type(e).__name__}: {e}")
        return 1
    return 0
