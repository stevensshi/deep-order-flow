"""ITCH 5.0 message types, lengths and body layouts (spec revision of April 28, 2023)."""

from __future__ import annotations

import struct

SPEC_URL = "https://nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf"
MSG_LEN: dict[int, int] = {  # payload length (without the 2-byte prefix), spec §1.x
    ord("S"): 12, ord("R"): 39, ord("H"): 25, ord("Y"): 20, ord("L"): 26, ord("V"): 35,
    ord("W"): 12, ord("K"): 28, ord("J"): 35, ord("h"): 21, ord("A"): 36, ord("F"): 40,
    ord("E"): 31, ord("C"): 36, ord("X"): 23, ord("D"): 19, ord("U"): 35, ord("P"): 44,
    ord("Q"): 40, ord("B"): 19, ord("I"): 50, ord("N"): 20, ord("O"): 48,
}
BOOK_TYPES = frozenset("AFECXDU")
EVENT_TYPES = frozenset("AFECXDUPQBH")   # types written to the event table
DEFAULT_CHUNK = 64 * 1024 * 1024

# Flat lookup tables for the hot loop: indexing a list by the type byte is
# cheaper than a dict lookup, and 0 marks "type not in the spec".
_LEN_BY_BYTE = [0] * 256
for _t, _n in MSG_LEN.items():
    _LEN_BY_BYTE[_t] = _n
_CHR = [chr(_i) for _i in range(256)]

# Body layouts, unpacked from payload offset 11 (after type, locate, tracking, ts).
_TS = struct.Struct(">HI")                       # u48 timestamp @5 as (high 16, low 32)
_R = struct.Struct(">8sccIcc2scccccIc")          # §1.2.1 Stock Directory
_H = struct.Struct(">8scc4s")                    # §1.2.2 Stock Trading Action
_ADD = struct.Struct(">QcI8sI")                  # §1.3.1/1.3.2 A and F (F's MPID @36 unused)
_E = struct.Struct(">QIQ")                       # §1.4.1 Order Executed
_C = struct.Struct(">QIQcI")                     # §1.4.2 Order Executed With Price
_X = struct.Struct(">QI")                        # §1.4.3 Order Cancel
_REF = struct.Struct(">Q")                       # §1.4.4 D ref; §1.5.3 B match number
_U = struct.Struct(">QQII")                      # §1.4.5 Order Replace
_P = struct.Struct(">QcI8sIQ")                   # §1.5.1 Trade (non-cross)
_Q = struct.Struct(">Q8sIQc")                    # §1.5.2 Cross Trade (shares are u64)

_T_S, _T_R, _T_H = ord("S"), ord("R"), ord("H")
_T_A, _T_F, _T_E, _T_C = ord("A"), ord("F"), ord("E"), ord("C")
_T_X, _T_D, _T_U = ord("X"), ord("D"), ord("U")
_T_P, _T_Q, _T_B = ord("P"), ord("Q"), ord("B")
