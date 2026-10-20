"""Command line: ``python -m dof.lob build-day ...``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from dof.lob.tables.build import build_day
from dof.lob.tables.manifest import load_symbols_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dof.lob")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-day", help="build per-stock events/states Parquet for one ITCH file")
    b.add_argument("--itch", required=True, type=Path)
    b.add_argument("--symbols-file", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    b.add_argument("--limit-bytes", type=int, default=None)
    b.add_argument("--overwrite", action="store_true")
    manifest = b.add_mutually_exclusive_group(required=True)
    manifest.add_argument("--manifest", type=Path, default=None,
                          help="MANIFEST.csv of the downloads: the input must be listed there as complete")
    manifest.add_argument("--no-manifest-check", action="store_true",
                          help="skip the MANIFEST.csv completeness check (smoke tests only)")
    b.add_argument("--flush-rows", type=int, default=32768)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    summary: dict[str, object] = {"status": "failed", "day_dir": None, "symbols_valid": None,
                                  "symbols_invalid": None, "symbols_missing": None, "total_s": None}
    try:
        res = build_day(args.itch, load_symbols_file(args.symbols_file), args.out,
                        manifest_csv=None if args.no_manifest_check else args.manifest,
                        limit_bytes=args.limit_bytes, overwrite=args.overwrite, flush_rows=args.flush_rows)
    except Exception as exc:  # noqa: BLE001 — reported on stdout and via the exit code
        summary.update({"error": repr(exc), "total_s": round(time.perf_counter() - t0, 3)})
        print(json.dumps(summary))
        return 1
    summary.update({
        "status": res.status,
        "day_dir": str(res.day_dir),
        "symbols_valid": len(res.files) - len(res.invalid_symbols),
        "symbols_invalid": len(res.invalid_symbols),
        "symbols_missing": len(res.missing_symbols),
        "total_s": round(time.perf_counter() - t0, 3),
    })
    print(json.dumps(summary))
    return 0 if res.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
