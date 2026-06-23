# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `organ` and `tissue` metadata roles (10 roles total). Each carries its own
  specific role type (`organ` / `tissue`) — no generic `categorical` bucket —
  routed to a shared vocabulary-based heuristic scorer with near-disjoint value
  vocabularies. Two-stage LLM path gains organ-vs-tissue discrimination guidance.

## [0.2.0] - 2026-06-22

Renamed from **stansample** and generalized from ranking the single `sample`
column to identifying **eight metadata-column roles** across three role types.

### Added
- **Eight roles in three types:** `sample` (grouping); `pct_mt`, `pct_hb`,
  `doublet_score`, `n_counts`, `n_genes` (numeric per-cell — `pct_*` and
  `doublet_score` are fractions in `[0, 1]`); `cell_type_coarse`,
  `cell_type_fine` (cell-type label columns). Any role may be absent.
- **Role registry** (`roles.py`): per-role name aliases, token rules, numeric
  value-shape checks, and a cell-type value vocabulary.
- **Two-stage LLM path:** one holistic ranking call over all roles, then a
  focused numeric **adjudication** call (margin Δ = 0.15) that disambiguates
  look-alike numeric columns (e.g. `total_counts` vs `total_counts_mt`).
- **`--hint`:** optional free-text user guidance threaded into both LLM stages
  to steer column identification (LLM-only; ignored on `--no-llm`; default empty
  ⇒ unchanged behavior).
- **Pluggable LLM provider:** `--provider anthropic` (native `messages.parse`)
  or `openai` (any OpenAI-compatible `/chat/completions`, incl. Volcengine ARK),
  with `--base-url` and `--api-key-env`.
- **`--roles`** to restrict the run to a subset of roles.
- **Numeric value statistics** in the digest (`frac_unit`, `is_integer_valued`,
  `v_min`/`v_max`/`v_median`/`v_mean`, `frac_nonneg`).
- `LICENSE` (MIT) and this changelog.

### Changed
- Package and CLI renamed `stansample` → `stanmetacols`.
- CLI is **JSON-only on stdout**, keyed by role:
  `{ "method": ..., "roles": { "<role>": [ {role, column, kind, score, reason, source}, ... ] } }`.
- Public API: `rank_sample_columns(...)` → `rank_meta_columns(...)`, returning
  `MetaColsResult` with `.top(role)`.
- Heuristic scoring is per role type: grouping (name + cardinality + balance −
  penalties), numeric (`0.6·name + 0.4·value`, gated on a name hit), and
  cell-type (`0.4·name + 0.4·vocabulary + 0.2·cardinality-fit`).

### Fixed
- `RankedCandidate.kind` is optional: a model reply that omits it no longer
  collapses the entire LLM ranking to the heuristic fallback.

## [0.1.0] - stansample

Initial release as **stansample**: ranked which `.obs` column (single column,
composite key, or barcode-derived grouping) identifies the sample each cell came
from, via a single structured LLM call with a deterministic heuristic fallback.

[0.2.0]: https://github.com/chansigit/stanmetacols/releases/tag/v0.2.0
