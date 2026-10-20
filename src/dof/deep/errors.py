"""Exceptions of the training package. Library code raises these and never ``SystemExit``; the command line maps
them to exit code 1."""

from __future__ import annotations


class ConfigError(ValueError):
    """A configuration or local machine file is invalid or incomplete."""


class CacheError(RuntimeError):
    """The feature cache is missing, incomplete, or was made under another rule."""


class TestDayRefused(RuntimeError):
    """A test day was asked for without the conditions of the test gate."""

    __test__ = False        # not a test class, whatever pytest's name rule says


class Diverged(RuntimeError):
    """The training loss or a validation loss is not finite: the fit is recorded and never scored on the test
    days."""


class RecordConflict(RuntimeError):
    """A record exists under another identity, or a test file was scored with another corporate-action file."""


class CheckpointConflict(RuntimeError):
    """A checkpoint belongs to another fit, identity or training set, or has an old format."""


class DeviceError(RuntimeError):
    """The configured device cannot be used."""


class DiskFloor(RuntimeError):
    """The disk guard path has less free space than its floor."""


class RestoreMismatch(RuntimeError):
    """The saved best weights do not reproduce the best validation loss."""
