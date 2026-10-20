"""A synthetic training run for the hermetic tests: three stocks on four seeded random days, built into order-book
tables and a feature cache, with a configuration of two tiny networks and two linear references.

Days: two training days (the second one also "recent"), one validation day, one test day. The corporate-action file
leaves CCC out on the test day. No torch import here.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from dof.deep.cache import build
from dof.deep.config import CachePaths, MachineConfig, RunConfig, load
from dof.lob.tables import build_day
from dof.lob.testing.synth import write_random_day

SYMBOLS = ("AAA", "BBB", "CCC")
DAYS = {"2019-01-30": 1, "2019-01-31": 2, "2019-02-01": 3, "2019-02-04": 4}   # ISO date -> random_day seed
CA_HEADER = "ticker,event,ratio,effective_date,status,source_url,quote\n"
CA_ROW = "CCC,forward_split,2:1,2019-02-01,confirmed,,synthetic\n"

TEMPLATE = """\
[split]
name = "synth"
train_days = ["2019-01-30", "2019-01-31"]
recent_days = ["2019-01-31"]
valid_days = ["2019-02-01"]
test_days = ["2019-02-04"]

[data]
universe = "universe.csv"

[protocol]
corporate_actions = "ca.csv"
corporate_actions_sha256 = "{ca_sha}"
test_days_need_document = false

[features]
convention = "cks"
dedupe = true
history = 100
latency_ns = 10000000
horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gap_days = 300
ratio_range = [0.6, 1.6]

[train]
batch_size = 256
check_steps = 4
max_checks = 2
patience = 5
adam_betas = [0.9, 0.999]
eval_batch = 512
restore_best = true
fit_order = ["TinyMLP-all", "TinyLSTM-recent"]
eval_row_cap = 512

[ridge]
lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
chunk_rows = 16384

[models.TinyMLP]
family = "mlp"
input = "of"
input_dim = 20
hidden = [8]
init = "keras"
precision = "float32"

[models.TinyLSTM]
family = "lstm_mlp"
input = "of"
input_dim = 20
units = 8
hidden = 8
init = "keras"
precision = "float32"

[fits.TinyMLP-all]
arch = "TinyMLP"
optimizer = "adam"
lr = 1e-3
eps = 1e-7
days = "all"

[fits.TinyLSTM-recent]
arch = "TinyLSTM"
optimizer = "keras_adam"
lr = 1e-3
eps = 1e-7
days = "recent"

[linear.Ridge-all]
days = "all"

[linear.Ridge-recent]
days = "recent"
"""


@dataclass(frozen=True)
class SynthRun:
    root: Path
    cfg: RunConfig
    cache: CachePaths


def build_synth_run(root: Path) -> SynthRun:
    root = Path(root)
    (root / "itch").mkdir(parents=True, exist_ok=True)
    for date, seed in DAYS.items():
        path = write_random_day(root / "itch", date, seed)
        res = build_day(path, list(SYMBOLS), root / "book", manifest_csv=None, date=datetime.date.fromisoformat(date),
                        overwrite=True)
        assert res.status == "complete", (date, res.status)
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "universe.csv").write_text("ticker\n" + "".join(s + "\n" for s in SYMBOLS), encoding="utf-8")
    ca = configs / "ca.csv"
    ca.write_text(CA_HEADER + CA_ROW, encoding="utf-8")
    cfg_path = configs / "synth.toml"
    cfg_path.write_text(TEMPLATE.format(ca_sha=hashlib.sha256(ca.read_bytes()).hexdigest()), encoding="utf-8")
    cache = CachePaths(root / "cache" / "train", root / "cache" / "eval")
    machine = MachineConfig(book_root=root / "book", device="cpu", torch_threads=1, loader_workers=0,
                            cache={"synth": cache})
    cfg = replace(load(cfg_path), machine=machine)
    build(cfg, cfg.split, cache, workers=0, log=lambda msg: None)
    return SynthRun(root, cfg, cache)
