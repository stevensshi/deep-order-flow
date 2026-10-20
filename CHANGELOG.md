# Changelog

## 0.4.0.dev0 — descriptive names, first public version

- Names no longer abbreviate the authors of a paper; the papers are cited in full in the README and in the
  docstrings. The order-flow series branch is `dof.lob.features.flow_series` (`FlowConfig`, `FlowFrame`,
  `flow_pipeline`, `FLOW_COLUMNS`), its cleaning is `flow_clean` and its labels are `horizon_labels`. The one-minute
  branch is `dof.lob.features.minute_ofi` (`MinuteConfig`, `ChangeEvents`, `change_events`, `minute_buckets`,
  `window_ofi`). The sign convention of the two papers is `current_size`; `cks` is unchanged. The universe file is
  `configs/universe_115.csv`. The entries below use the new names.
- The golden recipes use the new names, so the golden keys and `recipe_sha256` changed. Every golden value is the
  same, bit for bit.
- A fifth experiment spec, `earlier`: the earlier stage's pooled models on 115 tickers (LSTM-OF, LSTM-MLP-OF,
  CNN-LSTM-OF-lr3e-5-eps1e-7 and the pooled Ridge-OF), trained on three days of 2019, scored on three 2019 test days;
  no protocol document. Its rows are added to the result tables (the rows of the other four specs are unchanged, and
  so are their reports), with a report and a report golden. A record source may now split one stored file by part,
  name the model of a pattern without `{model}`, and keep only rows with given values; sums leave out missing sse and
  sst. `export --root NAME=DIR` reads one spec from another folder. A table of one day no longer repeats that day.
- MIT licence. `protocols/` holds edited copies of the four pre-registration documents, with the sha256 of each
  original. The README reports the results, and a test checks each of its numbers against `results/`.

## 0.3.0.dev0 — the experiment framework

- `dof.experiments`: an experiment is a spec (`ExperimentSpec`): its fits, seeds and rounds, its parts (held-out ones
  declared by day and time), where its records are, the hashes of its protocol documents, and the tables of its
  report. `export` turns stored records into the tables of `results/` (score cells, stock-days, records, fits, checks)
  and refuses a record that names another protocol hash or was scored before every fit of its round had ended.
  The analyses are pure functions of those tables; the renderer only formats their output. `run` fits, gates and
  scores once, then reports, for specs whose steps the training package provides. The four stored experiments (main
  run, seeds, adjacent days, decay) are specs. New report goldens; existing goldens unchanged.

## 0.2.0.dev4 — a Transformer family

- New model family `transformer` (`dof.deep.models.transformer`) with the catalogue entry `[models.Transformer-OF]`:
  each row of the window projected to `d_model`, a learned position embedding, pre-norm encoder layers with
  dropout, the last row's state to the horizons; no attention mask, since the window ends at row t. The family is
  one model file and one catalogue entry; a fit stopped at its second check resumes to the weights of an
  uninterrupted fit with dropout on. No fit of `smoke.toml` or `alldays.toml` uses it. Golden changes: none.

## 0.2.0.dev3 — the restore check is enforced

- A fit whose saved weights do not reproduce the best validation loss (relative tolerance 1e-6, `train.restore_rtol`)
  stops before its record is written. Golden changes: none.

## 0.2.0.dev2 — strict evaluation list

- The evaluation days are checked against the cache summary; a missing or extra stock-day stops the run instead of
  being skipped. Golden changes: none.

## 0.2.0.dev1 — random-number state in checkpoints

- Checkpoints hold the CPU (and CUDA) random-number state and are resumed with it; the loader draws its worker seed
  from its own generator. A model with dropout now resumes to the same weights as an uninterrupted fit. Checkpoints
  of 0.2.0.dev0 are refused. Golden changes: none.

## 0.2.0.dev0 — the training package `dof.deep`

- `dof.deep`: run configurations from TOML with machine settings in a separate local file (`config`), the feature
  cache of a split (`cache`), the training stream and the evaluation days (`data`), the model registry (`models`),
  Adam in the Keras form (`optim`), the fit loop (`train`), score cells and the test-day gate (`evaluate`), the
  pooled ridge references (`linear`), the optional compact backend (`data.compact`), and `python -m dof.deep`.
  Caches, batches, initial weights, fits, restarts, ridge weights and score cells are identical to the reference
  implementation (golden tests).
- Record format, against the reference implementation's records:
  - `prereg_sha256` is `protocol_sha256`, the sha256 of the protocol document the configuration names (None when it
    names none); a cache summary no longer holds it;
  - new keys `identity` and `identity_sha256` (what defines the one result: no machine setting, no path, no other
    fit), `seeds` (`init` and `data`) and `device_settings` (read from torch); `config` keeps the reference's
    configuration dict for compatibility;
  - `progress.json` is written in the fit's own folder;
  - a record is reused, and a checkpoint continued, only under an equal identity; the checkpoint's `ident` holds the
    identity hash;
  - test-day tokens belong to the gate instance that issued them and are bound to the model, the seed and the
    weights hash;
  - the default device is the CPU;
  - the compact backend is off by default (`compact_store = false`); its memory rules default to 4 GiB when it is on.

## 0.1.0.dev3 — no NaN in the sufficient statistics

- `suff_stats_update` raises `ValueError("non-finite value in a sufficient-statistics chunk")` when X or Y holds
  a NaN or an infinity, before anything is added. Valid chains hold no such value, so no result changes.

## 0.1.0.dev2 — partial builds

- A build is `complete` only when no byte limit stopped it and the end-of-messages event (S "C") was seen;
  otherwise the build log and `BuildResult.status` say `partial`. The command line exits 0 only for `complete`.
- `load_states` raises `InvalidStockDay` with "build status=<s>" unless the day's build is complete, and with
  "files missing" when a stock's table file is absent.
- The build-log layout and `SCHEMA_VERSION` are unchanged.

## 0.1.0.dev1 — trading sessions and half days

- New `dof.lob.features.Session` (regular trading `open_ns`..`close_ns`, the trimmed session `start_ns`..`end_ns`
  ten minutes inside) with `Session.full_day()`, `Session.half_day("13:00")` and `session_for(date, half_days)`.
- `flow_pipeline(frame, session, config)`: the session is an argument; `FlowConfig` loses `session_start_ns` and
  `session_end_ns` and gains `dedupe`; `FlowFrame` carries `session` instead of `dedupe`. Labels end before
  `session.end_ns`, so on a half day closing at 13:00 they end before 12:50.
- `count_mid_changes(frame, session)` counts over regular trading (`open_ns`..`close_ns`).
- `estimate_delta_t(frames, dates, sessions)`: Δt is the total regular trading time over the total count of
  mid-price changes, so a half day enters with its 3.5 hours. For full days the result is unchanged to the last
  bit.
- `minute_buckets(frame, *, config, session=None)`: (close - open) / 1 minute buckets; 390 on a full day.
- No number changes on full days; the half-day results equal the reference implementation's half-day rule.

## 0.1.0.dev0

- The order-book library as `dof.lob`: ITCH 5.0 parsing (`dof.lob.itch`), the 10-level book (`dof.lob.book`),
  per-stock Parquet tables and build logs (`dof.lob.tables`), order-flow features, labels and scalers
  (`dof.lob.features`), linear statistics (`dof.lob.stats`) and the LOBSTER reconciliation
  (`dof.lob.validation`). Outputs are identical to the reference implementation (golden tests).
- `build_day` takes `manifest_csv` as a required keyword; the command line takes `--manifest PATH` or
  `--no-manifest-check`, one of the two.
