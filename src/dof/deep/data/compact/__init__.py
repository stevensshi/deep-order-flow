"""Lossless block-compressed store of fixed-size records, held in memory: the optional compact backend of the
training stream. Linux only (it reads /proc/meminfo and the cgroup files, and marks its memory with madvise); with
``machine.compact_store = false``, the default, the record files are read and this package is not imported.

A one-dimensional ``.npy`` file of any fixed-size dtype is cut into blocks of ``BLOCK_ROWS`` records. Every block
is one zstd frame, and the frames lie back to back in one ``.zst`` file. A sidecar (``.zst.json``) holds the block
sizes and the hashes of one file; one ``MANIFEST.json`` seals a store. ``CompactStore.load`` puts the frames of a
set of files into anonymous memory, and ``CompactStore.windows`` gives back the bytes of any run of rows.

Why it exists: a training set of tens of GB of records does not fit in memory, so every training step waits on
the disk. Block-compressed, the records take about a tenth of that, and a step reads no disk.

The package imports numpy, zstandard and the standard library only. It draws no sample, orders no batch and
scales nothing. No function raises ``SystemExit``; the command line decides exit codes.
"""

from dof.deep.data.compact.convert import Pacer, convert, convert_file, finalize, show, verify, verify_windows
from dof.deep.data.compact.errors import CompactError, NotEnoughMemory, StoreMismatch, StoreMissing, VerificationError
from dof.deep.data.compact.layout import BLOCK_ROWS, FORMAT, data_path, key_of, manifest_path, read_npy_header, sidecar_path
from dof.deep.data.compact.store import CompactStore, encode_blocks

__all__ = ["BLOCK_ROWS", "CompactError", "CompactStore", "FORMAT", "NotEnoughMemory", "Pacer",
           "StoreMismatch", "StoreMissing", "VerificationError", "convert", "convert_file", "data_path",
           "encode_blocks", "finalize", "key_of", "manifest_path", "read_npy_header", "show", "sidecar_path", "verify",
           "verify_windows"]
