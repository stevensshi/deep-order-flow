"""The test-day gate: a test day is read only for a scored model whose fit has ended with its weights saved.

``TestGate.token(record)`` issues a token only when the model is one the configuration scores, its record says
the fit has finished, the record was made under the present protocol document, the saved weights are the ones the
record names (sha256), and the corporate-action file is there. A token is bound to (split, part, model, seed,
weights sha256) and to the gate instance that issued it; there is no module-level token set. A spawned worker
builds its own gate and asks its own tokens.

``check(tokens, stage, part, models)`` refuses a test part unless every model being scored holds a token of this
gate for this split and that model. The caller that loads the weights reads their bytes once, compares them with
the token's hash, and loads from those bytes, so a file changed after the token was issued is caught.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Iterable, Mapping

from dof.deep.config import RunConfig, Split, protocol_sha256
from dof.deep.errors import TestDayRefused
from dof.deep.train.record import FitRecord
from dof.lob.provenance import sha256_file


@dataclass(frozen=True)
class TestToken:
    __test__ = False        # not a test class, whatever pytest's name rule says
    stage: str
    part: str
    model: str
    seed: int
    weights_sha256: str
    nonce: str


class TestGate:
    __test__ = False

    def __init__(self, cfg: RunConfig, split: Split) -> None:
        self.cfg, self.split = cfg, split
        self._issued: set[TestToken] = set()

    def token(self, record: FitRecord | None) -> TestToken:
        if record is None:
            raise TestDayRefused("no fit record: the fit has not ended")
        model = record.model
        if model not in self.cfg.scored_models:
            raise TestDayRefused(f"{model} is not a model the configuration scores: the test days are not read for it")
        if record.status != "finished":
            raise TestDayRefused(f"{model}: the fit record has status {record.status!r}, not 'finished'")
        protocol = protocol_sha256(self.cfg)
        if self.cfg.protocol.test_days_need_document and protocol is None:
            raise TestDayRefused("the configuration names no protocol document, and the test days need one")
        if record.doc.get("protocol_sha256") != protocol:
            raise TestDayRefused(f"{model}: the fit was made under another protocol document")
        w = record.weights_path
        if not w.exists() or sha256_file(w) != record.weights_sha256:
            raise TestDayRefused(f"{model}: the saved weights are missing or are not the ones of the fit record")
        if not self.cfg.protocol.corporate_actions.exists():
            raise TestDayRefused(f"{self.cfg.protocol.corporate_actions.name} is missing: the test days are not scored "
                                 "without the corporate-action file")
        tok = TestToken(self.split.name, "test", model, record.seed, record.weights_sha256, secrets.token_hex(8))
        self._issued.add(tok)
        return tok

    def check(self, tokens: Mapping[str, TestToken] | None, stage: str, part: str, models: Iterable[str]) -> None:
        if part != "test":
            return
        models = tuple(models)
        if not models:
            raise TestDayRefused("refused to load a test day: no model is being scored")
        for name in models:
            tok = (tokens or {}).get(name)
            if tok is None or tok not in self._issued or (tok.stage, tok.part, tok.model) != (stage, part, name):
                raise TestDayRefused(f"refused to load a test day for {name}: no token of this gate for split {stage} "
                                     "and this model")
