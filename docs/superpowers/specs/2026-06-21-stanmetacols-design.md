# stanmetacols — design spec

**Date:** 2026-06-21
**Supersedes:** `stansample` (single-target sample-column ranker) — renamed and
generalized to multi-role metadata-column identification.

## 1. Goal

Given an AnnData `.obs` table (or a bare pandas DataFrame), identify, for each of
a fixed set of **metadata roles**, which `.obs` column best fills that role. The
tool **ranks** candidates per role; it does not decide. Same dual-path design as
`stansample`: a structured LLM pass over a deterministic digest, with a
deterministic offline heuristic fallback over the same digest.

The LLM path is **two-stage** (new in this version): a single holistic call ranks
every role, then — when a *numeric* role still has two or more close, high-scoring
candidates — one focused **adjudication** call decides which column is the
canonical one for that role. This catches the common case where
`sc.pp.calculate_qc_metrics` emits look-alike columns (`total_counts`,
`total_counts_mt`, `total_counts_hb`, …) that a single pass might not fully
disambiguate.

This is a rename (`stansample` → `stanmetacols`) plus a generalization from one
target (`sample`) to six roles.

## 2. Roles

Six roles in two **types**. Type drives the detection logic.

| Role key | Type | What it is | Canonical value shape |
|---|---|---|---|
| `sample` | grouping | the sample each cell came from | categorical, moderate cardinality, balanced groups |
| `pct_mt` | numeric | mitochondrial-gene fraction per cell | float in **[0, 1]** |
| `pct_hb` | numeric | hemoglobin-gene fraction per cell | float in **[0, 1]** |
| `doublet_score` | numeric | doublet detection score per cell | float in **[0, 1]** |
| `n_counts` | numeric | total counts / UMIs per cell | non-negative **integer-valued**, large magnitude |
| `n_genes` | numeric | genes detected per cell | non-negative **integer-valued**, smaller magnitude |

**Hard constraint (user):** every `pct_*` value is a fraction in `[0, 1]`, NOT a
percent in `[0, 100]`. The `pct_mt` / `pct_hb` / `doublet_score` value checks all
expect `[0, 1]`; these three are disambiguated **by name**.

**Out of scope (this spec):** `pct_ribo`, `pct_hb`-beyond, or any other QC role.
Adding one later is a single entry in the role registry — that extensibility is a
design goal, not a feature to build now. No backward-compatible `stansample`
import alias (clean rename).

## 3. Architecture

```
src/stanmetacols/
  __init__.py      exports rank_meta_columns, profile_obs, Candidate,
                   MetaColsResult, ROLES, LLMUnavailable, __version__
  roles.py    NEW  Role registry: alias/token rules + numeric value checks
  schema.py        dataclasses + Pydantic schemas (Candidate gains `role`;
                   RankResult → MetaColsResult; RankedCandidates + Adjudications)
  profile.py       digest builder; ColumnProfile gains numeric value stats
  prompts.py       multi-role system prompt + per-role hints + user prompt
  heuristic.py     per-role deterministic scorer (grouping vs numeric)
  llm.py           stage 1: holistic ranking call (anthropic | openai);
                   stage 2: focused numeric adjudication call
  rank.py          orchestrator: rank_meta_columns(...)
  __main__.py      CLI: JSON-only, --roles filter
```

Pipeline: **profile → rank → adjudicate (numeric, LLM only) → group by role →
report**. One digest is built once and scored for every requested role. The
adjudication stage runs only on the LLM path and only for numeric roles that
remain ambiguous after stage 1 (Section 7.2).

## 4. Role registry (`roles.py`)

```python
@dataclass(frozen=True)
class Role:
    key: str
    type: str                  # "grouping" | "numeric"
    aliases: tuple             # normalized-exact names -> name score 1.0
    include_tokens: tuple      # any present (substring of normalized name) -> token hit
    exclude_tokens: tuple      # any present -> token rule fails (disambiguation)
    measure_tokens: tuple      # for pct roles: a measure word must co-occur
    value_check: Callable      # (ColumnProfile) -> float in [0,1], numeric roles only
```

Name normalization `ν(s)`: lowercase, delete `_`, `.`, spaces (same as today).

### 4.1 Name signal per role

```
name(col, role) =
  1.0  if ν(col) in role.aliases                       # exact
  0.8  if role token rule matches ν(col)               # see below
  0.6  if some alias is a substring of ν(col), or vice versa
  0.0  otherwise
```

**Token rule** (matches iff): at least one `include_tokens` is a substring of
`ν(col)`, AND no `exclude_tokens` is a substring, AND (for pct roles) at least
one `measure_tokens` is also a substring.

### 4.2 Role definitions (concrete)

- **`sample`** (grouping) — aliases: `sample, sample_id, donor, donor_id,
  patient, patient_id, subject, individual, specimen, orig.ident, library,
  library_id, gsm, geo_accession, srr, batch, channel, well, lane, replicate`.
  No token/value rule (grouping scorer handles it).
- **`pct_mt`** (numeric) — measure_tokens: `pct, percent, frac, fraction,
  proportion`; include_tokens: `mt, mito, mitochond`; aliases: `pct_counts_mt,
  pct_mt, percent.mt, percent_mt, percent_mito, pct_mito, mito_frac, mt_frac`.
- **`pct_hb`** (numeric) — measure_tokens: same as `pct_mt`; include_tokens:
  `hb, hemo, haemo, hemoglobin`; aliases: `pct_counts_hb, pct_hb, percent.hb,
  percent_hb, hb_frac, hemo_frac`.
- **`doublet_score`** (numeric) — include_tokens: `doublet, scrublet`; aliases:
  `doublet_score, doublet_scores, scrublet_score, scrublet, df_score,
  doubletfinder_score, doublet_probability, predicted_doublet`. No measure rule.
- **`n_counts`** (numeric) — include_tokens: `count, counts, umi, libsize,
  librarysize`; **exclude_tokens: `gene, genes, feature, features`** (so
  `n_genes_by_counts` does NOT match here); aliases: `n_counts, total_counts,
  ncount_rna, numi, n_umi, umi_count, library_size`.
- **`n_genes`** (numeric) — include_tokens: `gene, genes, feature, features`;
  aliases: `n_genes, n_genes_by_counts, nfeature_rna, n_features, num_genes,
  genes_detected, detected_genes`. No measure rule.

### 4.3 Numeric value checks → `[0, 1]`

Computed from `ColumnProfile` numeric stats (Section 5). Each returns a
plausibility in `[0, 1]`; non-numeric columns score `0`.

- **pct roles + doublet_score** (`[0,1]` floats):
  `1.0` if numeric, `frac_nonneg ≥ 0.99` and `frac_unit ≥ 0.99` (≥99% of values
  in `[0,1]`) and not integer-valued-only; `0.3` if numeric and `frac_unit`
  between 0.5 and 0.99 (e.g. a percent-scale column, which the user says won't
  occur but we degrade gracefully); else `0.0`.
- **`n_counts`**: `1.0` if numeric, `is_integer_valued`, `frac_nonneg ≥ 0.99`,
  and `v_median ≥ 100`; `0.5` if integer non-negative but small median; else
  `0.0`.
- **`n_genes`**: `1.0` if numeric, `is_integer_valued`, `frac_nonneg ≥ 0.99`,
  and `2 ≤ v_median ≤ 20000`; `0.5` if integer non-negative outside that band;
  else `0.0`.

The value check is dual-purpose: positive evidence **and** a guard against a
name false-positive (e.g. a `[0,1]` `pct_counts` column scores low for
`n_counts` because its values aren't large integers).

## 5. Digest (`profile.py`)

`ColumnProfile` keeps all current fields and gains numeric value stats, computed
only for numeric columns (defaults otherwise):

```python
is_numeric: bool            # dtype in {integer, float}
v_min: float                # over non-missing; 0.0 if non-numeric
v_max: float
v_median: float
v_mean: float
frac_nonneg: float          # fraction of non-missing values >= 0, in [0,1]
frac_unit: float            # fraction of non-missing values in [0,1]
is_integer_valued: bool     # all non-missing values are whole numbers
```

`profile_obs(...)` is otherwise unchanged. Composite-key and barcode candidates
remain (they feed only the `sample` role). `to_prompt_dict()` includes the new
numeric fields so the LLM sees value shape, not just names.

## 6. Scoring (`heuristic.py`)

`rank_heuristic(digest, roles) -> dict[str, list[Candidate]]`. For each requested
role, score every column; drop `score <= 0`; sort descending. Each emitted
`Candidate` carries its `role`.

**Grouping role (`sample`)** — unchanged from `stansample`: single columns
(skipping single-value / unique-per-cell), composite keys, and the barcode
grouping, with the existing formula
`0.5·name + 0.25·card + 0.25·balance − penalties`.

**Numeric roles** — over every column (numeric per-cell metrics are often
near-unique, so `unique_per_cell` is NOT a disqualifier here):

```
score = clip( 0.6·name(col, role) + 0.4·value_check(col, role) )   # only if name > 0
```

**Gate:** the heuristic requires `name(col, role) > 0` for numeric roles — drop a
column with no name hit. Value shape alone cannot assign a role here: a bare
`[0,1]` float is equally consistent with `pct_mt`, `pct_hb`, and `doublet_score`,
so without a name signal the offline path would pollute all three. (The LLM path
has no such gate — it reasons over names *and* values together.) After the gate,
a perfect name with a wrong value shape still scores `0.6·1.0 + 0.4·0 = 0.6` and
surfaces, but ranks below a column matching both; value evidence breaks ties and
guards against name false-positives.

## 7. LLM path (`llm.py`)

Two stages. Stage 1 always runs on the LLM path; stage 2 runs only when stage 1
leaves a numeric role ambiguous. Provider abstraction is preserved verbatim for
both calls: `provider="anthropic"` (native `messages.parse`) or
`provider="openai"` (any OpenAI-compatible `/chat/completions`, JSON parsed
against the schema). Same lazy imports, same `base_url`/`api_key` handling, same
tolerant JSON parsing (`_extract_json`/`_parse_ranked`).

### 7.1 Stage 1 — holistic ranking

One structured call over the digest returns candidates for **all** requested
roles (one call, not one per role).

```python
class RankedCandidate(BaseModel):
    role: str        # must be one of the requested role keys
    column: str
    score: float
    reason: str

class RankedCandidates(BaseModel):
    candidates: List[RankedCandidate]
```

Post-processing guard (both backends): drop any candidate whose `role` is not a
requested role, or whose `column` is not a valid label in the digest
(hallucination guard); clip score to `[0,1]`; `kind` is taken from the digest
(`single`/`composite`/`barcode`; numeric columns are `single`); group by role;
sort each role descending. Any stage-1 failure raises `LLMUnavailable` →
heuristic fallback (same as today), and stage 2 does not run.

### 7.2 Stage 2 — numeric adjudication

After stage 1, a numeric role is **ambiguous** when it has ≥2 candidates whose
top-two score gap is `≤ Δ` (`Δ = 0.15`). Its **contention set** is every
candidate within `Δ` of that role's top score. If no numeric role is ambiguous,
stage 2 is skipped (total LLM calls = 1).

Otherwise **one** focused call adjudicates all ambiguous numeric roles at once.
The prompt gives, per ambiguous role: the role's intent and expected value shape,
and each contention-set column with its value stats from the digest
(`v_min/v_max/v_median`, `frac_unit`, `is_integer_valued`, …). The model picks
the single canonical column per role.

```python
class Adjudication(BaseModel):
    role: str        # one of the ambiguous roles
    column: str      # must be in that role's contention set
    reason: str

class Adjudications(BaseModel):
    verdicts: List[Adjudication]
```

Apply: for each verdict whose `column` is in the offered contention set, move that
column to rank 1 of its role and replace its `reason` with the adjudication
reason; leave the rest of the order intact. A verdict naming a column outside the
contention set is ignored. **Stage 2 is non-fatal:** if the adjudication call
fails or returns nothing usable, keep the stage-1 ranking unchanged (do not fall
back to the heuristic — stage 1 already succeeded).

Adjudication is **numeric-only** (the user's concern; the `sample` grouping role
keeps its stage-1 ranking) and runs only on the LLM path — `--no-llm` and the
heuristic fallback never adjudicate.

## 8. Output schema

`schema.py`:

```python
@dataclass
class Candidate:
    role: str
    column: str
    kind: str        # "single" | "composite" | "barcode"
    score: float
    reason: str
    source: str      # "llm" | "heuristic"

@dataclass
class MetaColsResult:
    roles: dict      # role_key -> list[Candidate] (sorted desc, truncated to top_k)
    method: str      # "llm (anthropic)" | "llm (openai)" | "heuristic" |
                     # "heuristic (llm unavailable: …)"; when stage-2 ran, the llm
                     # form gains " + adjudication", e.g. "llm (openai) + adjudication"
    digest: ObsDigest
    def top(self, role: str) -> Candidate | None: ...
```

CLI JSON on stdout (always valid JSON, even when every role is empty):

```json
{
  "method": "llm (anthropic)",
  "roles": {
    "sample":        [{"role":"sample","column":"sample","kind":"single","score":0.9,"reason":"…","source":"llm"}],
    "pct_mt":        [{"role":"pct_mt","column":"pct_counts_mt","kind":"single","score":0.97,"reason":"…","source":"llm"}],
    "pct_hb":        [],
    "doublet_score": [{"…"}],
    "n_counts":      [{"…"}],
    "n_genes":       [{"…"}]
  }
}
```

Every requested role appears as a key (empty array if no candidate). Per-role
list sorted by score desc, truncated to `--top`.

## 9. CLI (`__main__.py`)

```bash
stanmetacols x.h5ad                          # all 6 roles, JSON on stdout
stanmetacols x.h5ad --roles sample,pct_mt    # subset (comma-separated)
stanmetacols x.h5ad --no-llm                 # offline heuristic
stanmetacols x.h5ad --top 3
stanmetacols x.h5ad --provider openai --base-url … --model … --api-key-env ARK_API_KEY
```

Flags: existing `--no-llm`, `--top`, `--provider`, `--model`, `--base-url`,
`--api-key-env`, plus new `--roles` (default: all six; unknown role name → exit 1
with a stderr message). Exit codes: `0` at least one candidate across all roles,
`2` none found anywhere (still valid JSON), `1` IO error / bad `--roles`.

## 10. Public API

```python
rank_meta_columns(data, *, roles=None, use_llm=True, adjudicate=True,
                  provider="anthropic", model="claude-opus-4-8", client=None,
                  base_url=None, api_key=None, top_k=5) -> MetaColsResult
```

`roles=None` ⇒ all six. `adjudicate=True` enables stage 2 (no effect when
`use_llm=False` or stage 1 fails); exposed mainly so tests can isolate stage 1.
Never mutates input; writes no files.

## 11. Rename mechanics

- `gh repo rename stanmetacols` (from the repo) — `chansigit/stansample` →
  `chansigit/stanmetacols`; GitHub redirects the old URL, history/issues/stars
  preserved. Update the local `origin` URL.
- `git mv src/stansample src/stanmetacols`; rename the local project directory.
- `pyproject.toml`: `name = "stanmetacols"`, `version = "0.2.0"`,
  `[project.scripts] stanmetacols = "stanmetacols.__main__:main"`, update
  `[tool.hatch.build.targets.wheel] packages`, description, URLs.
- Update all imports (`stansample` → `stanmetacols`), README title/body, and
  `docs/formulation.md`.

## 12. Testing

Mirror the current per-module test layout; reuse the dependency-injection
pattern (`client=` stubs for both providers).

- `test_roles.py` — name signal (exact / token rule / substring / miss);
  `n_genes_by_counts` resolves to `n_genes` not `n_counts`; value checks return
  expected `[0,1]` for representative profiles.
- `test_profile.py` — numeric stats (`frac_unit`, `is_integer_valued`,
  `v_median`) on float-in-[0,1], integer-count, and categorical columns;
  existing grouping/composite/barcode tests unchanged.
- `test_heuristic.py` — per role on a synthetic `.obs` containing `sample`,
  `pct_counts_mt`, `pct_counts_hb`, `total_counts`, `n_genes_by_counts`,
  `doublet_score`: each role's top candidate is the right column; a `[0,1]`
  `pct_counts` does not win `n_counts`.
- `test_llm.py` — stage-1 multi-role parse + role/column hallucination filtering,
  for both anthropic and openai stubs; stage-2 adjudication: a verdict reorders
  the role's candidates, an out-of-contention verdict is ignored, an adjudication
  failure leaves the stage-1 ranking intact.
- `test_rank.py` — `rank_meta_columns` grouping by role, `--roles` subset,
  LLM-failure fallback, top_k truncation, input-not-mutated; ambiguity trigger
  (a numeric role with two close candidates fires stage 2; a clear winner does
  not — `adjudicate=False` isolates stage 1); `method` gains `+ adjudication`
  when stage 2 runs.
- `test_cli.py` — JSON shape `{method, roles:{…}}`, `--roles`, exit codes.

## 13. Success criteria

- On a synthetic `.obs` with all six columns present, both `--no-llm` and the LLM
  path put the correct column first for every role.
- With look-alike numeric columns present (`total_counts` + `total_counts_mt`),
  the LLM path fires stage-2 adjudication and ranks the canonical column first.
- Roles with no matching column return `[]`, not a wrong guess.
- `pct_*` detection respects the `[0,1]` convention; a mis-scaled `[0,100]`
  column degrades gracefully (name can still surface it, lower value score).
- The rename leaves no `stansample` references in code, packaging, or docs.
- Full test suite green.
