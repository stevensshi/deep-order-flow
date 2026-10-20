"""Exceptions of the experiment framework. Library code raises these and never ``SystemExit``; the command line maps
them to exit code 1."""

from __future__ import annotations


class SpecError(ValueError):
    """A spec is invalid, or asks for a step this package does not run."""


class ExportError(RuntimeError):
    """A record contradicts its spec (protocol hash, duplicate id, held-out order), or an output folder exists."""


class AnalysisError(RuntimeError):
    """A table cannot be built from the result tables (a missing price ratio, an ambiguous record)."""


class ReportError(RuntimeError):
    """A headline names a table, row or column that the report does not build, or a cell without a value."""
