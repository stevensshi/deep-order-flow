"""Experiments as specs: one spec, then gate, fit, score once, collect and report.

- ``spec``: ``ExperimentSpec``, the fits, seeds, rounds, parts, record layouts, protocol hashes and report sections of
  an experiment;
- ``store``, ``cells``, ``export``: stored records into the result tables (score cells, stock-days, records, fits,
  checks), refusing a record that names another protocol hash or was scored too early;
- ``analyses``: pure functions of the result tables; ``report`` and ``render``: the report built from them;
- ``gate``, ``ledger``, ``runner``: ``run(spec)`` for the specs whose steps the training package provides;
- ``specs``: the four stored experiments (main run, seeds, adjacent days, decay).

Importing this package does not import torch: the names below are loaded on first use.
"""

from __future__ import annotations

import importlib

from dof.experiments.errors import AnalysisError, ExportError, ReportError, SpecError
from dof.experiments.spec import ExperimentSpec

# name -> (module, attribute or None for the module itself), imported on first access (PEP 562)
_LAZY = {
    "SPECS": ("dof.experiments.specs", "SPECS"),
    "export": ("dof.experiments.export", "export"),
    "read_tables": ("dof.experiments.cells", "read_tables"),
    "ResultTables": ("dof.experiments.cells", "ResultTables"),
    "render": ("dof.experiments.render", "render"),
    "analyses": ("dof.experiments.analyses", None),
    "run": ("dof.experiments.runner", "run"),
}

__all__ = ["AnalysisError", "ExperimentSpec", "ExportError", "ReportError", "SpecError", *_LAZY]


def __getattr__(name: str):
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'dof.experiments' has no attribute {name!r}") from None
    mod = importlib.import_module(module)
    return mod if attr is None else getattr(mod, attr)
