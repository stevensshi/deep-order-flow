"""The five stored experiments as specs: the main run on all held days, its seeds, the adjacent-days experiment, the
decay measurement, and the earlier stage's pooled models on 2019 days. Each spec pins the sha256 of its protocol
documents and of the corporate-action file (the earlier stage had none); the export refuses a stored record that
names another hash."""

from __future__ import annotations

PROTOCOL_MAIN = "babfaf66c72cd5627f6b28cf271c12c8837078722216f9c91570bafd157607e2"
PROTOCOL_SEEDS = "33f9cef8db47c6ebda964c38489f503c446f3bbe94368e9bc81e76a81e0e5e39"
PROTOCOL_ADJACENT = "1f1417e75640153875e0a9eebada72be1243a496932f590520092d34a5ca20a3"
PROTOCOL_DECAY = "ad9015c9c6fca945ad22fe8fd2e276d9f7d9bf05000e3a2d14b783267152a97b"
CORPORATE_ACTIONS = "df7ef9adb2c52d4af1d112e5ae8a326d2408956ebe43e1d953e1b034f9f6ea4a"
MAIN_DOCUMENT = "protocol/preregistration.md"

FITS = ("LSTM-MLP-OF-recent", "LSTM-MLP-OF-alldays", "CNN-LSTM-OF-alldays")
LINEAR = ("Ridge-OF-alldays", "Ridge-OF-recent")
VALID_DAYS = ("2025-12-11", "2025-12-12")
TEST_DAYS = ("2026-05-15", "2026-05-18", "2026-06-12")
TEST_DAYS_LABEL = ",".join(TEST_DAYS)                      # the row key of a table row over the three test days
TEST_DAYS_WORDS = "2026-05-15, 2026-05-18 and 2026-06-12"

from dof.experiments.specs import adjacent, alldays, decay, earlier, seeds  # noqa: E402  (they read the constants above)

SPECS = {"alldays": alldays.SPEC, "seeds": seeds.SPEC, "adjacent": adjacent.SPEC, "decay": decay.SPEC,
         "earlier": earlier.SPEC}
