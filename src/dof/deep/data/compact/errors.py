"""Refusals of the compact store."""

from __future__ import annotations


class CompactError(RuntimeError):
    """Every refusal of this package that is not a bad argument."""


class StoreMissing(CompactError):
    """A manifest, sidecar, data file or key is not there."""


class StoreMismatch(CompactError):
    """A hash, size, context, dtype, row count or block table differs."""


class NotEnoughMemory(CompactError):
    """The machine, or this process's cgroup, would be left with too little memory after the load."""


class VerificationError(CompactError):
    """``CompactStore.windows`` did not give back the source bytes."""
