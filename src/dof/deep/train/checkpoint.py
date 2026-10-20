"""The state of a fit after its last check: what a stopped fit needs to continue with the batches an uninterrupted
fit would have used, in the same order.

The state holds the weights, the optimizer state, the best weights so far, the per-check record, the counters of
the stopping rule, and the batch stream's place: the generator state the present pass was drawn from and the
number of batches of that pass already used. ``ident`` names the fit, its identity and its training set; a
checkpoint of another one is refused, never used or overwritten.

Format 2 also holds the global random-number state of the CPU (and of every CUDA device) after the check, so a
model with dropout draws the same masks after a restart as without it. A checkpoint without it is refused.
"""

from __future__ import annotations

from pathlib import Path

from dof.deep.errors import CheckpointConflict
from dof.deep.runtime.io import torch_save_atomic

FORMAT = 2
KEYS = ("format", "ident", "model", "optimizer", "keras_t", "scaler", "best_state", "best", "wait", "stop", "checks",
        "steps", "pass_start_steps", "gen_state_pass_start", "sec_fit", "started", "resumes", "saved", "rng_cpu",
        "rng_cuda")


def checkpoint_state(**fields) -> dict:
    """The checkpoint dict, with exactly the keys of ``KEYS``."""
    if set(fields) != set(KEYS):
        raise TypeError(f"checkpoint fields {sorted(set(fields) ^ set(KEYS))} are missing or unknown")
    return {k: fields[k] for k in KEYS}


def save_checkpoint(path: Path, state: dict) -> None:
    """Written under another name and renamed: a checkpoint that exists is whole."""
    torch_save_atomic(path, state)


def load_checkpoint(path: Path, ident: dict) -> dict | None:
    """The state saved after the last check of an interrupted fit, or None."""
    if not path.exists():
        return None
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != FORMAT:
        raise CheckpointConflict(f"{path}: the checkpoint has no random-number state (format 1); move it away")
    if ck["ident"] != ident:
        raise CheckpointConflict(f"{path} was saved for another fit, identity or training set; move it away before running")
    return ck
