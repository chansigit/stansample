# stansample — Design Spec

**Date:** 2026-06-21

## Goal

A standalone, pip-installable Python package **and** CLI that, given an AnnData
(or just its `.obs` + obs_names), **ranks** which `.obs` column — or composite
key of columns, or a grouping derived from cell barcodes — identifies the
**sample** each cell came from. It returns ranked candidates with scores and
human-readable reasons. It ranks; it does **not** pick a single winner for the
caller.

Primary path: a **single structured LLM call** (`claude-opus-4-8` via the
`anthropic` SDK) over a compact, deterministic digest of `.obs`. When no API key
is present, the network is unavailable, or the LLM call fails, it **falls back**
to a deterministic heuristic ranker over the same digest, so it always returns
an answer offline (e.g. on a Sherlock compute node with no egress).

## Non-goals

- **Not** a Claude Code skill. No `SKILL.md`, no `.claude-plugin/`, no
  `/stansample` trigger. It is a plain library + CLI.
- Does **not** modify or write the AnnData, and writes no output files. Pure
  read.
- Does **not** authoritatively choose one column — returns a ranked list (top 5
  by default). The caller decides.
- The heuristic path requires **no network and no API key**.

## Definition of "sample"

The **natural grouping unit**: "which sample each cell came from" — the unit
used for per-sample QC, batch grouping, or pseudobulk. The tool does not try to
distinguish biological (donor/patient) from technical (library/10x channel); it
surfaces whatever column best partitions cells into samples and lets the caller
judge.

## Package location

`/home/users/chensj16/s/projects/stansample` — a standalone repo, sibling to
`stancounts` and `stangene`.

## Architecture / data flow

```
adata / obs ──▶ profile.py ──▶ ObsDigest  (compact, deterministic, JSON-able;
               (per-column features +              NEVER contains the matrix)
                composite-pair candidates +
                barcode-pattern analysis)
                     │
                     ▼
                rank.py ──┬─ key present & use_llm ─▶ llm.py: messages.parse(opus-4-8, schema)
                          │                              │  (raises LLMUnavailable on key/net/API error)
                          └─ else / on LLMUnavailable ─▶ heuristic.py: weighted score
                                                          │
                                                          ▼
                                  RankResult.candidates : [Candidate(column, kind, score, reason, source)]
                                                          (sorted desc by score, truncated to top_k)
```

The LLM only ever sees the **digest** (column names, dtypes, cardinalities, a
few example values, group-balance stats, barcode-pattern summary, composite
candidates). It never sees the expression matrix.

## Modules (`src/stansample/`)

### `schema.py`
Types shared across modules.

- `Candidate` (dataclass): `column: str` (for composite, a `"a + b"` join;
  for barcode, a synthetic label like `"<barcode:prefix>"`), `kind:
  Literal["single", "composite", "barcode"]`, `score: float` (0–1),
  `reason: str`, `source: Literal["llm", "heuristic"]`.
- `RankResult` (dataclass): `candidates: list[Candidate]` (sorted desc by
  score), `method: str` (e.g. `"llm"` or `"heuristic (llm unavailable: no API
  key)"`), `digest: ObsDigest`. Method `top() -> Candidate | None` returns the
  highest-scored candidate or `None` if empty.
- `RankedCandidates` / `RankedCandidate` (Pydantic models): the structured
  output schema the LLM is constrained to. `RankedCandidate` =
  `{column: str, kind: str, score: float, reason: str}`; `RankedCandidates =
  {candidates: list[RankedCandidate]}`.
- `ObsDigest` (dataclass): `n_obs: int`, `columns: list[ColumnProfile]`,
  `composite_candidates: list[CompositeProfile]`,
  `barcode: BarcodeProfile | None`. Has `to_prompt_dict()` returning a plain
  JSON-serializable dict for the prompt.
- `ColumnProfile`, `CompositeProfile`, `BarcodeProfile` (dataclasses) — see
  `profile.py`.
- Exceptions: `LLMUnavailable(Exception)`.

### `profile.py`
Deterministic feature extraction. **No LLM, no network, no mutation.**

- `profile_obs(obs, obs_names=None, *, max_example_values=8, max_composite_pairs=8) -> ObsDigest`
  - `obs`: a pandas DataFrame. `obs_names`: optional sequence of cell names
    (index); if omitted, taken from `obs.index`.
- Per column → `ColumnProfile`:
  - `name: str`
  - `dtype: str` — one of `"categorical"`, `"string"`, `"integer"`,
    `"float"`, `"bool"`
  - `n_unique: int`
  - `n_missing: int`
  - `example_values: list[str]` — up to `max_example_values`, sorted, stringified
  - `cells_per_group: dict` — `{"min": int, "max": int, "median": float}` over
    value counts
  - `balance: float` — `min_group / max_group` (1.0 = perfectly balanced; ~0 =
    one giant group + slivers)
  - `unique_per_cell: bool` — `n_unique == n_obs` (looks like a barcode/index,
    not a sample label)
  - `single_value: bool` — `n_unique <= 1`
  - `looks_like_barcode: bool` — fraction of values matching a barcode-ish regex
    (e.g. `^[ACGT]{8,}(-\d+)?$`) exceeds 0.5
- Composite candidates → `CompositeProfile` (`columns: list[str]`,
  `n_unique: int`, `cells_per_group`, `balance`): generated from pairs of
  "moderate-cardinality, non-unique-per-cell" categorical/string/int columns
  whose **product cardinality** stays plausible (`2 <= combined_unique <
  n_obs`); capped at `max_composite_pairs`, ranked by a cheap balance heuristic
  so only the most promising pairs are emitted.
- Barcode analysis → `BarcodeProfile | None`: split each `obs_name` on the last
  `_` and the trailing `-<int>`; if a delimiter yields `2 <= distinct_groups <
  n_obs` with reasonable balance, emit `{delimiter: str, position:
  "prefix"|"suffix", n_groups: int, cells_per_group, balance,
  example_groups: list[str]}`. `None` if no delimiter produces a usable
  grouping.

### `prompts.py`
- `SYSTEM_PROMPT: str` — explains the task: "you are given a digest of an
  AnnData `.obs`; rank which entries identify the sample each cell came from
  (the natural grouping unit for per-sample QC / batch / pseudobulk). Prefer
  columns whose name and values look like sample/donor/library/GEO identifiers
  with moderate cardinality and reasonable group balance; penalize per-cell-unique
  columns (barcodes/indices), single-value columns, and continuous measurements.
  Consider the provided composite-key and barcode-derived candidates too. Return
  every plausible candidate with a 0–1 score and a one-sentence reason." Includes
  the alias hint list (below). Instructs JSON-only via the schema.
- `build_user_prompt(digest: ObsDigest) -> str` — embeds
  `json.dumps(digest.to_prompt_dict(), sort_keys=True)`.
- `ALIAS_HINTS: list[str]` — `["sample", "sample_id", "sampleid", "donor",
  "donor_id", "patient", "patient_id", "subject", "individual", "specimen",
  "orig.ident", "library", "library_id", "gsm", "geo_accession", "srr", "batch",
  "channel", "well", "lane", "replicate"]` (case-insensitive, also matched on
  normalized name with `_`/`.`/spaces stripped).

### `llm.py`
The single structured call.

- `rank_with_llm(digest, *, model="claude-opus-4-8", client=None, max_tokens=2048) -> list[Candidate]`
  - Lazily `import anthropic` inside the function; if the import fails, raise
    `LLMUnavailable("anthropic not installed")`.
  - `client = client or anthropic.Anthropic()` — if construction fails (no key),
    raise `LLMUnavailable`.
  - Calls `client.messages.parse(model=model, max_tokens=max_tokens,
    system=SYSTEM_PROMPT, messages=[{"role": "user", "content":
    build_user_prompt(digest)}], output_format=RankedCandidates)`.
  - On any `anthropic.APIError` / connection error / parse failure (incl.
    `parsed_output is None`), raise `LLMUnavailable(<reason>)`.
  - Maps the validated `RankedCandidates` → `list[Candidate]` with
    `source="llm"`, clamping scores to [0, 1] and dropping candidates whose
    `column`/`kind` don't correspond to a real digest entry (guard against
    hallucinated columns).

### `heuristic.py`
Deterministic fallback ranker. **No network.**

- `rank_heuristic(digest) -> list[Candidate]` — weighted score per candidate
  (single columns, composites, barcode), each in [0, 1], with a generated
  reason string and `source="heuristic"`:
  - **+ name signal**: column name (normalized) matches `ALIAS_HINTS` (strong)
  - **+ cardinality plausibility**: `2 <= n_unique <= max(50, n_obs * 0.5)` and
    not `unique_per_cell` (Gaussian-ish bump peaking at small-but-not-tiny group
    counts)
  - **+ balance**: higher `balance` scores better
  - **− penalties**: `unique_per_cell`, `single_value`, `dtype == "float"`,
    high `n_missing`
  - composites get a modest discount vs. an equally-good single column; barcode
    candidate scored from its grouping balance and used mainly as a fallback
    when no obs column scores well.
  - Returns sorted desc by score.

### `rank.py`
Public orchestrator.

- `rank_sample_columns(data, *, use_llm=True, model="claude-opus-4-8",
  client=None, top_k=5) -> RankResult`
  - `data`: an `anndata.AnnData` **or** a pandas `DataFrame` (`.obs`). If
    AnnData, uses `data.obs` and `data.obs_names`; never mutates it.
  - Builds the digest via `profile_obs`.
  - If `use_llm`: try `rank_with_llm(...)`; on `LLMUnavailable(e)`, fall back to
    `rank_heuristic` and set `method = f"heuristic (llm unavailable: {e})"`.
    Otherwise `method = "llm"`.
  - If not `use_llm`: `rank_heuristic`, `method = "heuristic"`.
  - Sorts candidates desc by score; truncates to `top_k` (`top_k=0` or `None`
    → return all).
  - Returns `RankResult`.
  - Empty obs / no columns → empty candidate list, no exception.

### `__init__.py`
Exports: `rank_sample_columns`, `profile_obs`, `Candidate`, `RankResult`,
`ObsDigest`, `LLMUnavailable`, `__version__ = "0.1.0"`.

### `__main__.py`
CLI entry. `main(argv=None) -> int`.

- Usage: `stansample PATH.h5ad [--no-llm] [--top K] [--json] [--model NAME]`
  - `PATH.h5ad`: read with `anndata.read_h5ad(path, backed="r")` (memory-light;
    only `.obs`/`.obs_names` are touched).
  - `--no-llm`: force the heuristic path (`use_llm=False`).
  - `--top K`: default **5**; `--top 0` → all candidates.
  - `--json`: emit machine-readable JSON (`{"method": ..., "candidates":
    [{column, kind, score, reason, source}, ...]}`); otherwise a human table.
  - `--model NAME`: override `claude-opus-4-8`.
- **Exit codes**: `0` if at least one candidate is returned; `2` if none
  (lets scripts branch); `1` on usage/IO error (bad path, unreadable file).
- Console-script entry point: `stansample = "stansample.__main__:main"`, so both
  `stansample x.h5ad` and `python -m stansample x.h5ad` work.

## Error handling summary

| Situation | Behavior |
|---|---|
| No API key / `anthropic` not installed / network down / API error / parse fail | `LLMUnavailable` → automatic heuristic fallback; reason recorded in `method` |
| Hallucinated column from LLM | dropped (not in digest → filtered out) |
| Empty `.obs` / no columns | empty `candidates`, no crash, CLI exit `2` |
| Bad path / unreadable h5ad (CLI) | message to stderr, exit `1` |
| Input AnnData | never mutated; no files written |

## Dependencies (`pyproject.toml`)

- Build backend: `hatchling` (mirror stancounts).
- Core: `pandas>=1.5`, `numpy>=1.22`, `pydantic>=2`.
- Optional extras:
  - `anndata = ["anndata>=0.8"]` (needed for the CLI's h5ad reading and for
    passing AnnData objects; the library accepts a bare DataFrame without it)
  - `llm = ["anthropic>=0.40"]` (the LLM path; lazily imported)
  - `test = ["pytest>=7.0", "anndata>=0.8", "anthropic>=0.40"]`
- `requires-python = ">=3.9"`.
- `[project.scripts] stansample = "stansample.__main__:main"`.

## Testing (`tests/`)

- `test_profile.py`: synthetic `obs` with known columns (a `sample_id`, a
  per-cell barcode index, a `tissue`, a continuous `pct_mito`, a
  `donor`×`timepoint` pair) → assert each `ColumnProfile` field, that the
  composite candidate `donor + timepoint` is generated, and barcode parsing
  from synthetic obs_names.
- `test_heuristic.py`: same synthetic obs → assert ranking puts `sample_id`
  first, `donor`/composite above `tissue`, per-cell-unique and continuous
  columns near the bottom.
- `test_llm.py`: inject a **mock** `client` whose `messages.parse(...)` returns
  a stub `RankedCandidates`; assert (a) the user prompt contains the digest
  JSON, (b) the result maps to `Candidate(source="llm")`, (c) a hallucinated
  column in the stub is filtered out, (d) a mock that raises
  `anthropic.APIError` surfaces as `LLMUnavailable`. **No real API calls.**
- `test_rank.py`: `use_llm=False` → heuristic method; `use_llm=True` with a
  mock client → llm method; `use_llm=True` with a client that raises →
  `method` starts with `"heuristic (llm unavailable"`; `top_k` truncation;
  empty obs → empty result.
- `test_cli.py`: build a tiny h5ad in a tmp dir, run `main([...])` with
  `--no-llm --json`; assert exit code 0 and JSON shape; assert exit code 2 on
  an obs with no usable columns.

## Open decisions deferred to implementation

- Exact heuristic weights (tuned against the synthetic fixtures in
  `test_heuristic.py`).
- Barcode regex and delimiter set (start with `_` and trailing `-<int>`; widen
  only if a fixture needs it).
