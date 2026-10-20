"""The analyses: pure functions of the result tables (score cells, stock-days, fits and checks).

No file access, no clock, no randomness. Every table function takes ``(rt, spec, *, ..., id, title)`` and returns one
``Table``. The primitives repeat the reference implementation's statement order, numpy calls and iteration order, so
every float equals the stored report's bit for bit.
"""

from __future__ import annotations

from dof.experiments.analyses.adjacent import pair_table, part_table, reference_table
from dof.experiments.analyses.decay import decay_days, decay_outside, decay_table
from dof.experiments.analyses.fits import checks_table, fits_table, linear_table
from dof.experiments.analyses.rescore import rescore_table
from dof.experiments.analyses.scores import left_out_table, no_r2_table, older_rows, records_table, score_rows
from dof.experiments.analyses.seeds import finite_table, seed_older, seed_rule, seed_scores

ANALYSES = {f.__name__: f for f in (
    score_rows, older_rows, no_r2_table, left_out_table, records_table,
    fits_table, checks_table, linear_table,
    seed_rule, seed_scores, seed_older, finite_table,
    part_table, pair_table, reference_table,
    decay_table, decay_days, decay_outside,
    rescore_table,
)}
