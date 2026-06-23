# stanmetacols

[![Version](https://img.shields.io/badge/version-0.2.0-orange)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Identify which AnnData `.obs` columns fill standard **metadata roles** in a
single-cell dataset — sample grouping, per-cell QC fractions, count/gene
statistics, and cell-type labels. It **ranks** candidates for each role; it does
not decide.

See [CHANGELOG.md](CHANGELOG.md) for release notes.

**10 roles, 4 types:** `sample` (grouping); `pct_mt`, `pct_hb`,
`doublet_score`, `n_counts`, `n_genes` (numeric per-cell); `cell_type_coarse`,
`cell_type_fine` (cell-type labels); `organ`, `tissue` (categorical anatomical
metadata). Any role may be absent from the result.

**Two paths, same digest:** a deterministic heuristic (name aliases + per-type
value-shape checks) and a structured LLM pass (two-stage: holistic ranking then
numeric adjudication). The heuristic is the fallback when the LLM is
unavailable.

The CLI speaks **JSON only** — stdout is always a single JSON object, ready to
pipe into `jq` or load in another program.

---

## Install

```bash
pip install -e .                # core + heuristic only
pip install -e ".[llm]"         # add the Anthropic backend (Claude, default)
pip install -e ".[openai]"      # add the OpenAI-compatible backend (OpenAI, ARK, …)
pip install -e ".[anndata]"     # add .h5ad reading / AnnData inputs
```

The Anthropic backend reads `ANTHROPIC_API_KEY`; the OpenAI-compatible backend
reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` (see [Providers](#providers)).

---

## Roles

| Role | Type | Description |
|---|---|---|
| `sample` | grouping | the sample each cell came from (grouping unit for per-sample QC / pseudobulk) |
| `pct_mt` | numeric | per-cell mitochondrial-gene fraction, a float in [0, 1] |
| `pct_hb` | numeric | per-cell hemoglobin-gene fraction, a float in [0, 1] |
| `doublet_score` | numeric | per-cell doublet detection score, a float in [0, 1] |
| `n_counts` | numeric | total counts / UMIs per cell (non-negative integer, large) |
| `n_genes` | numeric | number of genes detected per cell (non-negative integer) |
| `cell_type_coarse` | celltype | coarse / broad cell-type or lineage label (fewer categories) |
| `cell_type_fine` | celltype | fine-grained cell-type / subtype label (more categories) |
| `organ` | organ | the solid anatomical organ a cell came from (heart, liver, kidney, lung, brain) |
| `tissue` | tissue | the sampled biological material / anatomical site (blood, PBMC, bone marrow, tumor, biopsy, CSF) |

`pct_mt`, `pct_hb`, and `doublet_score` are fractions in `[0, 1]`, **not**
percentages in `[0, 100]`. Any role may be absent (its list is empty).

---

## CLI

```bash
stanmetacols sample.h5ad                  # LLM if key present, else heuristic
stanmetacols sample.h5ad --no-llm         # force offline heuristic
stanmetacols sample.h5ad --top 0          # all candidates (default: top 5)
stanmetacols sample.h5ad --roles pct_mt,n_counts   # only these roles
python -m stanmetacols sample.h5ad        # equivalent module form
```

| Flag | Default | Meaning |
|---|---|---|
| `--roles ROLES` | all 10 | comma-separated subset of the 10 role keys |
| `--no-llm` | off | force the offline heuristic ranker (no API call) |
| `--top K` | `5` | keep the K highest-scored candidates per role; `0` = all |
| `--provider P` | `anthropic` | LLM backend: `anthropic` or `openai` (see [Providers](#providers)) |
| `--model ID` | `claude-opus-4-8` | LLM model ID; set this when `--provider openai` |
| `--base-url URL` | `$OPENAI_BASE_URL` | OpenAI-compatible endpoint base URL |
| `--api-key-env VAR` | SDK default | name of the env var holding the API key |
| `--hint TEXT` | `""` | free-text guidance for the LLM to locate columns (LLM path only; ignored with `--no-llm`) |

---

## Providers

The primary ranker runs through one of two backends, chosen by `--provider`.
The heuristic fallback is provider-independent.

**`anthropic`** (default) — native `messages.parse` with a strict structured
output schema. Reads `ANTHROPIC_API_KEY`. Best structured-output guarantee; this
is the path for Claude.

```bash
stanmetacols sample.h5ad                  # claude-opus-4-8
```

**`openai`** — any OpenAI-compatible `/chat/completions` endpoint (OpenAI,
Volcengine ARK, DeepSeek, vLLM, Ollama, …). The reply is parsed as JSON and
validated against the same schema, with the same hallucination guard. Reads
`OPENAI_API_KEY` and `OPENAI_BASE_URL` by default; `--base-url` and
`--api-key-env` override them.

```bash
# OpenAI
export OPENAI_API_KEY=sk-…
stanmetacols sample.h5ad --provider openai --model gpt-4o-mini

# Volcengine ARK (Doubao) — endpoint id as the model
export OPENAI_API_KEY=$ARK_API_KEY
stanmetacols sample.h5ad --provider openai \
    --base-url https://ark.cn-beijing.volces.com/api/v3 \
    --model ep-xxxxxxxxxxxx

# …or keep ARK's own key var and let stanmetacols read it:
stanmetacols sample.h5ad --provider openai \
    --base-url https://ark.cn-beijing.volces.com/api/v3 \
    --api-key-env ARK_API_KEY --model ep-xxxxxxxxxxxx
```

---

## Output

The CLI emits **one JSON object on stdout** — always, including when nothing is
found. Diagnostics for unreadable files go to **stderr** only, so a consumer can
parse stdout whenever the exit code is `0` or `2`.

```jsonc
{
  "method": "llm (anthropic)",           // see method values table below
  "roles": {
    "sample": [
      {
        "role": "sample",
        "column": "donor_id",            // .obs column; composite is "a + b"; barcode is "<barcode:prefix:_>"
        "kind": "single",                // "single" | "composite" | "barcode"
        "score": 0.95,                   // 0..1, full precision
        "reason": "exact alias, n_unique=8, balance=0.87",
        "source": "llm"                  // "llm" | "heuristic"
      }
    ],
    "pct_mt": [
      {
        "role": "pct_mt",
        "column": "pct_counts_mt",
        "kind": "single",
        "score": 0.97,
        "reason": "canonical mitochondrial fraction column, frac_unit=1.0",
        "source": "llm"
      }
    ],
    "pct_hb": [],
    "doublet_score": [],
    "n_counts": [],
    "n_genes": [],
    "cell_type_coarse": [],
    "cell_type_fine": [],
    "organ": [],
    "tissue": []
  }
}
```

### Output field reference

| Field | Type | Notes |
|---|---|---|
| `method` | string | Which path ran end-to-end (see table below). |
| `roles` | object | One key per requested role; value is a list of candidates sorted by `score` desc, truncated to `--top`. An absent or unidentified role has an empty list. |
| `roles.<role>[].role` | string | Role key this candidate was scored for. |
| `roles.<role>[].column` | string | Column name, `"a + b"` composite, or `"<barcode:POSITION:DELIM>"` barcode grouping. |
| `roles.<role>[].kind` | enum | `single`, `composite`, or `barcode`. |
| `roles.<role>[].score` | float | Confidence in `0..1` (full precision; not rounded). |
| `roles.<role>[].reason` | string | Human-readable one-liner explaining the score. |
| `roles.<role>[].source` | enum | `llm` or `heuristic` — the ranker that emitted this row. |

### Method values

| `method` string | Meaning |
|---|---|
| `"llm (anthropic)"` | Stage-1 holistic LLM ran, Anthropic backend. |
| `"llm (anthropic) + adjudication"` | Stage-1 + stage-2 numeric adjudication both ran, Anthropic backend. |
| `"llm (openai)"` | Stage-1 holistic LLM ran, OpenAI-compatible backend. |
| `"llm (openai) + adjudication"` | Stage-1 + stage-2 numeric adjudication both ran, OpenAI-compatible backend. |
| `"heuristic"` | `--no-llm` was passed; deterministic heuristic was used. |
| `"heuristic (llm unavailable: …)"` | LLM path failed (no key / no network / API error); fell back to heuristic. The `…` names the reason. |

**Exit codes:** `0` at least one candidate found · `2` no candidates at all (still valid JSON on stdout) · `1` IO error or bad `--roles` argument (message on stderr).

```bash
# Take the best pct_mt column name, or empty if none:
stanmetacols sample.h5ad --no-llm | jq -r '.roles.pct_mt[0].column // empty'
```

---

## Library

```python
from stanmetacols import rank_meta_columns

result = rank_meta_columns(adata)           # or pass a pandas .obs DataFrame

# Top candidate for each role (or None if absent):
best_sample  = result.top("sample")
best_pct_mt  = result.top("pct_mt")

print(result.method)                        # "llm (anthropic)" or "heuristic (...)"
if best_pct_mt:
    print(best_pct_mt.column, best_pct_mt.score, best_pct_mt.reason)

# Iterate all candidates for a role:
for c in result.roles.get("n_counts", []):
    print(c.score, c.column, c.reason)
```

**Full signature** (copied verbatim from `rank.py`):

```python
def rank_meta_columns(data, *, roles=None, use_llm: bool = True,
                      adjudicate: bool = True, hint: str = "",
                      provider: str = "anthropic",
                      model: str = "claude-opus-4-8", client=None,
                      base_url: str | None = None, api_key: str | None = None,
                      top_k: int | None = 5) -> MetaColsResult:
```

| Parameter | Default | Meaning |
|---|---|---|
| `data` | — | `AnnData` or a pandas `DataFrame` (`.obs`) |
| `roles` | `None` (all 10) | list of role keys to identify; `None` = all 10 |
| `use_llm` | `True` | set `False` to force the offline heuristic |
| `adjudicate` | `True` | run stage-2 numeric adjudication when stage-1 is ambiguous (Δ ≤ 0.15) |
| `hint` | `""` | free-text guidance threaded into both LLM stages (ignored when `use_llm=False`) |
| `provider` | `"anthropic"` | `"anthropic"` or `"openai"` |
| `model` | `"claude-opus-4-8"` | LLM model ID |
| `client` | `None` | pre-built SDK client (skips client construction) |
| `base_url` | `None` | OpenAI-compatible base URL |
| `api_key` | `None` | API key (alternative to env var) |
| `top_k` | `5` | candidates per role; `None` or `0` = all |

Returns a `MetaColsResult` with `.roles` (`dict[str, list[Candidate]]`),
`.method` (string), and `.digest`. Never mutates the input; writes no files.

**`--hint` / provider example:**

```python
# Pass user guidance to both LLM stages and use the OpenAI backend:
result = rank_meta_columns(
    adata,
    hint="This dataset uses 'Donor' for sample and 'leiden_coarse' for broad cell types.",
    provider="openai",
    model="gpt-4o-mini",
)
```

---

## How it works

For the precise math — the digest, the role registry, all scoring formulas, and
the two-stage LLM pipeline — see
[`docs/formulation.md`](docs/formulation.md). This section is the prose tour.

The pipeline is **profile → rank → fall back → report**. A single deterministic
*digest* of `.obs` is built once, then handed to one of two interchangeable
rankers. The LLM never sees raw cell data — only the compact digest — so the
call is cheap, private, and reproducible, and the offline heuristic scores the
*exact same* digest when the LLM is unavailable.

### Stage 1 — Digest (`profile.py`, deterministic)

`profile_obs(obs, obs_names)` reduces `.obs` to an `ObsDigest`. For grouping
candidates it computes per-column cardinality/balance, composite keys, and
barcode groupings. For every column it also records **numeric value stats**
(`v_min`, `v_max`, `v_median`, `frac_nonneg`, `frac_unit`,
`is_integer_valued`) needed by the numeric-role scorers.

### Stage 2 — Rank (two paths, same digest)

**LLM path** — a two-stage structured call:

- **Stage 1 (holistic ranking):** one LLM call over all requested roles at
  once. For cell-type roles the model uses value examples and cardinality to
  resolve coarse vs. fine; for numeric roles it uses the value stats to
  distinguish look-alikes (e.g. `total_counts` vs. `total_counts_mt`).
- **Stage 2 (numeric adjudication):** if two candidates for the same numeric
  role are within Δ = 0.15 of each other in stage-1 score, a second focused
  call re-evaluates only those tied columns with their value statistics. Stage 2
  is non-fatal — if it fails, the stage-1 ranking is kept unchanged.

Any `--hint` text is injected as an authoritative block at the top of the user
prompt in **both** stages.

**Heuristic path** — deterministic, offline. Scoring is role-type-specific (see
[`docs/formulation.md`](docs/formulation.md) for exact formulas):

- **Grouping** (`sample`): `clip(0.5·name + 0.25·card + 0.25·balance − penalties)`.
- **Numeric** (`pct_mt`, etc.): `clip(0.6·name + 0.4·value)` — requires a name
  hit (`name > 0`); skips columns with no recognizable name.
- **Cell-type** (`cell_type_coarse`, `cell_type_fine`):
  `clip(0.4·name + 0.4·vocab + 0.2·card_fit)` — uses both name signals and a
  vocabulary scan of the actual cell values.
- **Categorical label** (`organ`, `tissue`): `clip(0.4·name + 0.4·vocab + 0.2·card_fit)` — name aliases plus a per-role value vocabulary (organ names vs sampled-material names); near-disjoint vocabularies keep the two roles from bleeding into each other.

### Stage 3 — Report

Per-role candidate lists are sorted by score (desc) and truncated to `top_k`.
The result carries `method` so you always know which path produced it.
The CLI serializes to a single JSON object on stdout (see [Output](#output));
the library returns a `MetaColsResult`, whose `.top(role)` gives the single
best candidate for any role.
