# stanmetacols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `stansample` → `stanmetacols` and generalize it from ranking the single `sample` column to identifying six metadata-column roles (`sample`, `pct_mt`, `pct_hb`, `doublet_score`, `n_counts`, `n_genes`) via a two-stage LLM path (holistic ranking + numeric adjudication) with a deterministic heuristic fallback.

**Architecture:** A deterministic digest of `.obs` (now carrying numeric value stats) is scored per role by either the LLM path or the offline heuristic. A role registry declares each role's aliases, token rules, and value checks. The LLM path runs a holistic ranking call, then a focused adjudication call for any numeric role left with ≥2 close candidates. Output is keyed by role.

**Tech Stack:** Python ≥3.9, pandas, numpy, pydantic v2, hatchling; optional `anthropic` / `openai` / `anndata`; pytest.

## Global Constraints

- Package & repo name: `stanmetacols` (rename existing `chansigit/stansample`). Package version `0.2.0`. CLI entry point `stanmetacols`.
- Every `pct_*` value and `doublet_score` is a fraction in `[0, 1]`, NOT a percent in `[0, 100]`.
- Six roles, two types: `sample` (grouping) + `pct_mt`/`pct_hb`/`doublet_score`/`n_counts`/`n_genes` (numeric).
- Numeric-role heuristic requires a name hit (`name_signal > 0`); value alone never assigns a numeric role offline.
- LLM provider abstraction preserved verbatim: `provider="anthropic"` (native `messages.parse`) or `provider="openai"` (OpenAI-compatible `/chat/completions`, JSON parsed). Same lazy imports, `base_url`/`api_key` handling, tolerant JSON parsing.
- Stage-2 adjudication: numeric-only, batched into one call, triggered when a numeric role's top-2 score gap ≤ `Δ = 0.15`, non-fatal (keep stage-1 ranking on failure), LLM-path-only.
- CLI is JSON-only on stdout; `{method, roles:{...}}`. Exit `0` ≥1 candidate anywhere, `2` none, `1` IO error / bad `--roles`. IO/arg errors print to stderr.
- `from __future__ import annotations` at the top of every module that uses `X | None` (Python 3.9 floor).
- Never mutate input; write no files (except the CLI's stdout).
- The Python venv for all test commands: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python`. Run tests with `<venv>/bin/python -m pytest`.
- Reference spec: `docs/superpowers/specs/2026-06-21-stanmetacols-design.md`.

---

## File Structure

```
src/stanmetacols/
  __init__.py      exports + __version__ = "0.2.0"
  roles.py    NEW  Role dataclass, ROLES registry, name_signal, value_check
  schema.py        + Candidate.role; MetaColsResult; RankedCandidate.role; Adjudications
  profile.py       + numeric value stats on ColumnProfile
  prompts.py       multi-role ranking prompt + adjudication prompt
  heuristic.py     rank_heuristic(digest, roles) -> dict[role -> list[Candidate]]
  llm.py           rank_with_llm(stage 1) + adjudicate_numeric(stage 2)
  rank.py          rank_meta_columns(...)
  __main__.py      CLI: --roles, JSON {method, roles}
tests/             one test module per source module (mirrors existing layout)
```

---

## Task 1: Rename stansample → stanmetacols

**Files:**
- Rename: `src/stansample/` → `src/stanmetacols/` (the whole package dir)
- Modify: `pyproject.toml`
- Modify: every `tests/*.py` import (`stansample` → `stanmetacols`)
- Modify: `src/stanmetacols/*.py` docstrings/headers mentioning `stansample`
- Modify: `README.md`, `docs/formulation.md` (string replace `stansample` → `stanmetacols`)
- Repo: GitHub rename + local project dir rename

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `stanmetacols` with the *current* (sample-only) API intact: `rank_sample_columns`, `profile_obs`, `Candidate`, `RankResult`, `ObsDigest`, `LLMUnavailable`, `__version__ == "0.2.0"`.

This task is a pure rename: behavior and the public API are unchanged, only the package name and version. The multi-role API arrives in later tasks.

- [ ] **Step 1: Move the package directory (preserves git history)**

```bash
cd /scratch/users/chensj16/projects/stansample
git mv src/stansample src/stanmetacols
```

- [ ] **Step 2: Update imports in source and tests**

Replace every occurrence of the import path `stansample` with `stanmetacols` across the package and tests (module-relative imports like `from .schema import …` are unaffected; only absolute `stansample…` references change — these live in `tests/*.py` and in `src/stanmetacols/__init__.py`/`__main__.py` docstrings).

```bash
grep -rl 'stansample' src/stanmetacols tests | xargs sed -i 's/stansample/stanmetacols/g'
```

- [ ] **Step 3: Update pyproject.toml**

In `pyproject.toml` set the name, version, script, and wheel package:

```toml
[project]
name = "stanmetacols"
version = "0.2.0"
description = "Identify which AnnData .obs columns fill standard metadata roles (sample, pct_mt, pct_hb, doublet_score, n_counts, n_genes)"
# ... keep readme, requires-python, license, authors ...
keywords = ["single-cell", "scRNA-seq", "anndata", "metadata", "sample", "qc"]

[project.scripts]
stanmetacols = "stanmetacols.__main__:main"

[project.urls]
Homepage = "https://github.com/chansigit/stanmetacols"
Repository = "https://github.com/chansigit/stanmetacols"

[tool.hatch.build.targets.wheel]
packages = ["src/stanmetacols"]
```

- [ ] **Step 4: Reinstall editable so the new package + entry point resolve**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pip install -e ".[test]"`
Expected: installs `stanmetacols 0.2.0`, no errors.

- [ ] **Step 5: Run the full suite to verify the rename is behavior-preserving**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest -q`
Expected: PASS — same count as before the rename (41 passed).

- [ ] **Step 6: Sweep docs for the old name**

Replace `stansample` → `stanmetacols` in `README.md` and `docs/formulation.md` (these get a full rewrite in Task 7; this is only the name sweep so nothing references a dead package).

```bash
sed -i 's/stansample/stanmetacols/g' README.md docs/formulation.md
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor!: rename package stansample -> stanmetacols (v0.2.0)"
```

- [ ] **Step 8: Rename the GitHub repo and local project dir, update origin**

```bash
gh repo rename stanmetacols -R chansigit/stansample --yes
git remote set-url origin https://github.com/chansigit/stanmetacols.git
git push origin master
cd /scratch/users/chensj16/projects
mv stansample stanmetacols
cd stanmetacols
```

Expected: `git remote -v` shows the `stanmetacols` URL; `git push` succeeds. (The local project dir is now `/scratch/users/chensj16/projects/stanmetacols`; all later tasks run there.)

---

## Task 2: Numeric value stats in the digest

**Files:**
- Modify: `src/stanmetacols/schema.py` (add fields to `ColumnProfile`)
- Modify: `src/stanmetacols/profile.py` (compute them in `_profile_column`)
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: existing `ColumnProfile`, `_profile_column(name, s, n_obs, max_example_values)`.
- Produces: `ColumnProfile` with new fields `is_numeric: bool`, `v_min/v_max/v_median/v_mean: float`, `frac_nonneg: float`, `frac_unit: float`, `is_integer_valued: bool`. `to_prompt_dict()` carries them automatically (it serializes `vars(c)`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_profile.py`:

```python
import numpy as np
import pandas as pd
from stanmetacols.profile import profile_obs


def test_numeric_stats_unit_float():
    obs = pd.DataFrame({"pct": [0.0, 0.1, 0.2, 0.9, 1.0]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is True
    assert col.is_integer_valued is False
    assert col.frac_unit == 1.0
    assert col.frac_nonneg == 1.0
    assert col.v_min == 0.0 and col.v_max == 1.0


def test_numeric_stats_integer_counts():
    obs = pd.DataFrame({"total_counts": [1000, 2000, 3000, 50000]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is True
    assert col.is_integer_valued is True
    assert col.frac_unit == 0.0           # none in [0,1]
    assert col.v_median == 2500.0


def test_numeric_stats_absent_for_categorical():
    obs = pd.DataFrame({"sample": ["A", "B", "A", "B"]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is False
    assert col.frac_unit == 0.0 and col.is_integer_valued is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_profile.py -q`
Expected: FAIL with `AttributeError: 'ColumnProfile' object has no attribute 'is_numeric'`.

- [ ] **Step 3: Add the fields to ColumnProfile**

In `src/stanmetacols/schema.py`, add to the `ColumnProfile` dataclass (after `looks_like_barcode`):

```python
    is_numeric: bool = False
    v_min: float = 0.0
    v_max: float = 0.0
    v_median: float = 0.0
    v_mean: float = 0.0
    frac_nonneg: float = 0.0
    frac_unit: float = 0.0          # fraction of non-missing values in [0,1]
    is_integer_valued: bool = False
```

(Defaults keep every existing `ColumnProfile(...)` construction valid.)

- [ ] **Step 4: Compute them in `_profile_column`**

In `src/stanmetacols/profile.py`, inside `_profile_column`, before the `return ColumnProfile(...)`, add:

```python
    dtype = _classify_dtype(s)
    is_numeric = dtype in ("integer", "float", "bool")
    v_min = v_max = v_median = v_mean = frac_nonneg = frac_unit = 0.0
    is_integer_valued = False
    if is_numeric:
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
        vals = vals[~np.isnan(vals)]
        if vals.size:
            v_min = float(vals.min()); v_max = float(vals.max())
            v_median = float(np.median(vals)); v_mean = float(vals.mean())
            frac_nonneg = float((vals >= 0).mean())
            frac_unit = float(((vals >= 0.0) & (vals <= 1.0)).mean())
            is_integer_valued = bool(np.all(vals == np.round(vals)))
```

Then pass `dtype=dtype` (reuse the variable) and the new fields into the `ColumnProfile(...)` constructor:

```python
    return ColumnProfile(
        name=name,
        dtype=dtype,
        n_unique=n_unique,
        n_missing=n_missing,
        example_values=example_values,
        cells_per_group=cells_per_group,
        balance=balance,
        unique_per_cell=(n_obs > 0 and n_unique == n_obs),
        single_value=(n_unique <= 1),
        looks_like_barcode=(frac_bc > 0.5),
        is_numeric=is_numeric,
        v_min=v_min, v_max=v_max, v_median=v_median, v_mean=v_mean,
        frac_nonneg=frac_nonneg, frac_unit=frac_unit,
        is_integer_valued=is_integer_valued,
    )
```

(The existing line `return ColumnProfile(... looks_like_barcode=(frac_bc > 0.5))` is replaced by the block above; `_classify_dtype(s)` was previously called inline in the constructor — it is now stored in `dtype` and reused.)

- [ ] **Step 5: Run to verify it passes**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_profile.py -q`
Expected: PASS (all profile tests, new and existing).

- [ ] **Step 6: Commit**

```bash
git add src/stanmetacols/schema.py src/stanmetacols/profile.py tests/test_profile.py
git commit -m "feat: numeric value stats on ColumnProfile (frac_unit, is_integer_valued, v_*)"
```

---

## Task 3: Role registry (`roles.py`)

**Files:**
- Create: `src/stanmetacols/roles.py`
- Test: `tests/test_roles.py` (new)

**Interfaces:**
- Consumes: `ColumnProfile` from `schema.py` (no cycle: `schema.py` does not import `roles.py`).
- Produces:
  - `ROLE_KEYS: tuple` = `("sample","pct_mt","pct_hb","doublet_score","n_counts","n_genes")`
  - `NUMERIC_ROLE_KEYS: tuple` (the five numeric)
  - `ROLES: dict[str, Role]`
  - `Role` (frozen dataclass: `key, type, aliases, include_tokens, exclude_tokens, measure_tokens`)
  - `normalize(name: str) -> str`
  - `name_signal(col: str, role: Role) -> float` (1.0 / 0.8 / 0.6 / 0.0)
  - `value_check(profile: ColumnProfile, role: Role) -> float` (`[0,1]`, numeric only)

- [ ] **Step 1: Write the failing test**

Create `tests/test_roles.py`:

```python
import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.roles import ROLES, ROLE_KEYS, name_signal, value_check


def _profile(values):
    return profile_obs(pd.DataFrame({"x": values})).columns[0]


def test_role_keys():
    assert ROLE_KEYS == ("sample", "pct_mt", "pct_hb", "doublet_score",
                         "n_counts", "n_genes")


def test_name_exact_and_token_and_substring():
    assert name_signal("pct_counts_mt", ROLES["pct_mt"]) == 1.0      # exact alias
    assert name_signal("MyMitoPercent", ROLES["pct_mt"]) == 0.8      # token rule
    assert name_signal("nonsense", ROLES["pct_mt"]) == 0.0


def test_n_genes_by_counts_resolves_to_genes_not_counts():
    # contains "counts" but also "genes" -> excluded from n_counts, matches n_genes
    assert name_signal("n_genes_by_counts", ROLES["n_counts"]) == 0.0
    assert name_signal("n_genes_by_counts", ROLES["n_genes"]) >= 0.8


def test_value_check_unit_for_pct():
    prof = _profile([0.0, 0.05, 0.1, 0.3])
    assert value_check(prof, ROLES["pct_mt"]) == 1.0
    assert value_check(prof, ROLES["n_counts"]) == 0.0     # not integer/large


def test_value_check_integer_counts():
    prof = _profile([1000, 2000, 8000, 50000])
    assert value_check(prof, ROLES["n_counts"]) == 1.0
    assert value_check(prof, ROLES["pct_mt"]) == 0.0       # not in [0,1]


def test_value_check_genes_band():
    prof = _profile([200, 1500, 4000, 9000])
    assert value_check(prof, ROLES["n_genes"]) == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_roles.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stanmetacols.roles'`.

- [ ] **Step 3: Create `roles.py`**

```python
"""Role registry: each metadata role's name aliases/token rules and (for numeric
roles) a value-shape check. Pure functions, no LLM, no network."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import ColumnProfile


@dataclass(frozen=True)
class Role:
    key: str
    type: str                       # "grouping" | "numeric"
    aliases: tuple = ()             # raw names; matched after normalization
    include_tokens: tuple = ()      # any present (substring of norm name) -> token hit
    exclude_tokens: tuple = ()      # any present -> token rule fails
    measure_tokens: tuple = ()      # for pct roles: a measure word must co-occur


def normalize(name: str) -> str:
    return name.lower().replace("_", "").replace(".", "").replace(" ", "")


_PCT_MEASURE = ("pct", "percent", "frac", "fraction", "proportion")

ROLES: dict = {
    "sample": Role(
        key="sample", type="grouping",
        aliases=("sample", "sample_id", "donor", "donor_id", "patient",
                 "patient_id", "subject", "individual", "specimen", "orig.ident",
                 "library", "library_id", "gsm", "geo_accession", "srr", "batch",
                 "channel", "well", "lane", "replicate")),
    "pct_mt": Role(
        key="pct_mt", type="numeric",
        aliases=("pct_counts_mt", "pct_mt", "percent.mt", "percent_mt",
                 "percent_mito", "pct_mito", "mito_frac", "mt_frac"),
        include_tokens=("mt", "mito", "mitochond"),
        measure_tokens=_PCT_MEASURE),
    "pct_hb": Role(
        key="pct_hb", type="numeric",
        aliases=("pct_counts_hb", "pct_hb", "percent.hb", "percent_hb",
                 "hb_frac", "hemo_frac"),
        include_tokens=("hb", "hemo", "haemo", "hemoglobin"),
        measure_tokens=_PCT_MEASURE),
    "doublet_score": Role(
        key="doublet_score", type="numeric",
        aliases=("doublet_score", "doublet_scores", "scrublet_score", "scrublet",
                 "df_score", "doubletfinder_score", "doublet_probability",
                 "predicted_doublet"),
        include_tokens=("doublet", "scrublet")),
    "n_counts": Role(
        key="n_counts", type="numeric",
        aliases=("n_counts", "total_counts", "ncount_rna", "numi", "n_umi",
                 "umi_count", "library_size"),
        include_tokens=("count", "counts", "umi", "libsize", "librarysize"),
        exclude_tokens=("gene", "genes", "feature", "features")),
    "n_genes": Role(
        key="n_genes", type="numeric",
        aliases=("n_genes", "n_genes_by_counts", "nfeature_rna", "n_features",
                 "num_genes", "genes_detected", "detected_genes"),
        include_tokens=("gene", "genes", "feature", "features")),
}

ROLE_KEYS = ("sample", "pct_mt", "pct_hb", "doublet_score", "n_counts", "n_genes")
NUMERIC_ROLE_KEYS = ("pct_mt", "pct_hb", "doublet_score", "n_counts", "n_genes")


def _token_rule(n: str, role: Role) -> bool:
    if not role.include_tokens:
        return False
    if not any(t in n for t in role.include_tokens):
        return False
    if any(t in n for t in role.exclude_tokens):
        return False
    if role.measure_tokens and not any(t in n for t in role.measure_tokens):
        return False
    return True


def name_signal(col: str, role: Role) -> float:
    n = normalize(col)
    aliases = {normalize(a) for a in role.aliases}
    if n in aliases:
        return 1.0
    if _token_rule(n, role):
        return 0.8
    if any(a in n or n in a for a in aliases):
        return 0.6
    return 0.0


def _unit_value_check(p: ColumnProfile) -> float:
    if p.frac_nonneg >= 0.99 and p.frac_unit >= 0.99 and not p.is_integer_valued:
        return 1.0
    if 0.5 <= p.frac_unit < 0.99:        # e.g. a percent-scale column (degrade)
        return 0.3
    return 0.0


def _count_value_check(p: ColumnProfile) -> float:
    if p.is_integer_valued and p.frac_nonneg >= 0.99 and p.v_median >= 100:
        return 1.0
    if p.is_integer_valued and p.frac_nonneg >= 0.99:
        return 0.5
    return 0.0


def _genes_value_check(p: ColumnProfile) -> float:
    if p.is_integer_valued and p.frac_nonneg >= 0.99 and 2 <= p.v_median <= 20000:
        return 1.0
    if p.is_integer_valued and p.frac_nonneg >= 0.99:
        return 0.5
    return 0.0


def value_check(profile: ColumnProfile, role: Role) -> float:
    if role.type != "numeric" or not profile.is_numeric:
        return 0.0
    if role.key in ("pct_mt", "pct_hb", "doublet_score"):
        return _unit_value_check(profile)
    if role.key == "n_counts":
        return _count_value_check(profile)
    if role.key == "n_genes":
        return _genes_value_check(profile)
    return 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_roles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stanmetacols/roles.py tests/test_roles.py
git commit -m "feat: role registry (aliases/token rules + [0,1] value checks)"
```

---

## Task 4: Multi-role types in `schema.py`

**Files:**
- Modify: `src/stanmetacols/schema.py`
- Modify: `src/stanmetacols/heuristic.py` (tag `role="sample"` on its Candidates)
- Modify: `src/stanmetacols/llm.py` (RankedCandidate gains `role`; tag Candidates)
- Modify: `tests/test_schema.py`, `tests/test_llm.py`, `tests/test_rank.py` (constructors gain `role`)

**Interfaces:**
- Consumes: existing `Candidate`, `RankResult`, `RankedCandidate`, `RankedCandidates`.
- Produces:
  - `Candidate` gains `role: str` (keyword field; see Step 3).
  - `MetaColsResult(roles: dict, method: str, digest: ObsDigest)` with `.top(role) -> Candidate | None`.
  - `RankedCandidate` gains `role: str`.
  - `Adjudication(role, column, reason)` and `Adjudications(verdicts: List[Adjudication])` Pydantic models.

This task introduces the type surface while behavior stays sample-only and the suite stays green. `RankResult` and `rank_sample_columns` remain until Task 5 replaces them.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schema.py`:

```python
from stanmetacols.schema import Candidate, MetaColsResult, Adjudications


def test_candidate_has_role():
    c = Candidate(role="sample", column="s", kind="single", score=0.9,
                  reason="r", source="heuristic")
    assert c.role == "sample"


def test_metacolsresult_top(digest_fixture):
    c = Candidate(role="pct_mt", column="pct_counts_mt", kind="single",
                  score=0.9, reason="r", source="llm")
    res = MetaColsResult(roles={"pct_mt": [c], "sample": []},
                         method="heuristic", digest=digest_fixture)
    assert res.top("pct_mt") is c
    assert res.top("sample") is None
    assert res.top("n_genes") is None        # role absent -> None


def test_adjudications_schema():
    a = Adjudications(verdicts=[{"role": "n_counts", "column": "total_counts",
                                 "reason": "canonical total"}])
    assert a.verdicts[0].column == "total_counts"
```

If `tests/test_schema.py` has no `digest_fixture`, add this fixture to that file:

```python
import pytest
import pandas as pd
from stanmetacols.profile import profile_obs

@pytest.fixture
def digest_fixture():
    return profile_obs(pd.DataFrame({"sample": ["A", "B"]}))
```

- [ ] **Step 2: Run to verify it fails**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_schema.py -q`
Expected: FAIL — `TypeError` on `role=` / `ImportError` for `MetaColsResult`/`Adjudications`.

- [ ] **Step 3: Edit `schema.py`**

Add `role` as the FIRST field of `Candidate`:

```python
@dataclass
class Candidate:
    role: str
    column: str
    kind: str              # "single" | "composite" | "barcode"
    score: float
    reason: str
    source: str            # "llm" | "heuristic"
```

Add `MetaColsResult` (keep the existing `RankResult` for now):

```python
@dataclass
class MetaColsResult:
    roles: dict            # role_key -> list[Candidate], sorted desc, truncated
    method: str
    digest: ObsDigest

    def top(self, role: str):
        cands = self.roles.get(role) or []
        return cands[0] if cands else None
```

Extend the Pydantic schemas:

```python
class RankedCandidate(BaseModel):
    role: str
    column: str
    kind: str
    score: float
    reason: str


class RankedCandidates(BaseModel):
    candidates: List[RankedCandidate]


class Adjudication(BaseModel):
    role: str
    column: str
    reason: str


class Adjudications(BaseModel):
    verdicts: List[Adjudication]
```

(Note `RankedCandidate` already had `kind`; if it did not, this adds it — match the existing fields plus `role`.)

- [ ] **Step 4: Tag `role="sample"` at the existing Candidate construction sites**

In `src/stanmetacols/heuristic.py`, every `Candidate(...)` call gains `role="sample"` as the first argument (there are three: single, composite, barcode). Example:

```python
        out.append(Candidate(
            role="sample", column=c.name, kind="single", score=score,
            source="heuristic",
            reason=(f"name match={name_sig:.1f}, n_unique={c.n_unique}, "
                    f"balance={c.balance:.2f}")))
```

In `src/stanmetacols/llm.py`, the loop building Candidates from `parsed.candidates` gains `role`. Because `RankedCandidate` now requires `role`, set it from the parsed value and tag the Candidate:

```python
    for rc in parsed.candidates:
        kind = labels.get(rc.column)
        if kind is None:
            continue
        score = max(0.0, min(1.0, float(rc.score)))
        out.append(Candidate(role=rc.role, column=rc.column, kind=kind,
                             score=score, reason=rc.reason, source="llm"))
```

- [ ] **Step 5: Update existing test constructors to pass `role`**

In `tests/test_llm.py` and `tests/test_rank.py`, every `RankedCandidates(candidates=[{...}])` dict and every `Candidate(...)` gains `"role": "sample"` / `role="sample"`. Example in `tests/test_llm.py`:

```python
    parsed = RankedCandidates(candidates=[
        {"role": "sample", "column": "sample_id", "kind": "single",
         "score": 0.9, "reason": "looks like a sample id"},
        {"role": "sample", "column": "made_up_column", "kind": "single",
         "score": 0.8, "reason": "hallucinated"},
    ])
```

In `tests/test_schema.py`, the existing `RankResult(candidates=[c], ...)` test constructs `Candidate` — add `role="sample"`.

- [ ] **Step 6: Run the full suite**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest -q`
Expected: PASS (behavior unchanged; everything tagged `role="sample"`).

- [ ] **Step 7: Commit**

```bash
git add src/stanmetacols/schema.py src/stanmetacols/heuristic.py src/stanmetacols/llm.py tests/
git commit -m "feat: multi-role type surface (Candidate.role, MetaColsResult, Adjudications)"
```

---

## Task 5: Multi-role pipeline (heuristic + LLM stage 1 + orchestrator + CLI)

**Files:**
- Modify: `src/stanmetacols/prompts.py`
- Modify: `src/stanmetacols/heuristic.py`
- Modify: `src/stanmetacols/llm.py`
- Modify: `src/stanmetacols/rank.py`
- Modify: `src/stanmetacols/__main__.py`
- Modify: `src/stanmetacols/__init__.py`
- Test: `tests/test_heuristic.py`, `tests/test_llm.py`, `tests/test_rank.py`, `tests/test_cli.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `ROLES, ROLE_KEYS, NUMERIC_ROLE_KEYS, name_signal, value_check` (roles.py); `profile_obs` (profile.py); `Candidate, MetaColsResult, RankedCandidates` (schema.py).
- Produces:
  - `rank_heuristic(digest, roles) -> dict[str, list[Candidate]]`
  - `rank_with_llm(digest, roles, *, provider="anthropic", model="claude-opus-4-8", client=None, base_url=None, api_key=None, max_tokens=2048) -> dict[str, list[Candidate]]`
  - `rank_meta_columns(data, *, roles=None, use_llm=True, adjudicate=True, provider="anthropic", model="claude-opus-4-8", client=None, base_url=None, api_key=None, top_k=5) -> MetaColsResult` (adjudicate is accepted but is a no-op until Task 6)
  - CLI emitting `{method, roles:{...}}`
- This task removes `rank_sample_columns`, `RankResult`, and the single-role `rank_with_llm`/`rank_heuristic` signatures.

This is the coupled core: the per-role data contract (`dict[role] -> list[Candidate]`) flows through heuristic, llm, rank, and the CLI together, so they change as one task.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_heuristic.py` with multi-role tests:

```python
import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.heuristic import rank_heuristic
from stanmetacols.roles import ROLE_KEYS


def _digest():
    n = 60
    return profile_obs(pd.DataFrame({
        "sample": ["S1"] * 30 + ["S2"] * 30,
        "pct_counts_mt": [i / 100 for i in range(n)],          # [0,1) floats
        "pct_counts_hb": [i / 1000 for i in range(n)],
        "total_counts": [1000 + 10 * i for i in range(n)],     # large ints
        "n_genes_by_counts": [200 + i for i in range(n)],      # mid ints
        "doublet_score": [i / 100 for i in range(n)],
    }, index=[f"c{i}" for i in range(n)]))


def test_each_role_top_is_correct_column():
    out = rank_heuristic(_digest(), list(ROLE_KEYS))
    assert out["sample"][0].column == "sample"
    assert out["pct_mt"][0].column == "pct_counts_mt"
    assert out["pct_hb"][0].column == "pct_counts_hb"
    assert out["n_counts"][0].column == "total_counts"
    assert out["n_genes"][0].column == "n_genes_by_counts"
    assert out["doublet_score"][0].column == "doublet_score"


def test_numeric_role_requires_name_hit():
    # a bare [0,1] float column with no pct/doublet name must NOT appear for pct_mt
    d = profile_obs(pd.DataFrame({"score_x": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    assert rank_heuristic(d, ["pct_mt"])["pct_mt"] == []


def test_value_guard_rejects_unit_column_for_counts():
    # a [0,1] column literally named like counts should not win n_counts on value
    d = profile_obs(pd.DataFrame({"pct_counts": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    cands = rank_heuristic(d, ["n_counts"])["n_counts"]
    assert all(c.score < 0.7 for c in cands)    # name may hit, value does not
```

Replace `tests/test_rank.py` with:

```python
import pandas as pd
from stanmetacols.rank import rank_meta_columns
from stanmetacols.schema import RankedCandidates


def _obs():
    n = 40
    return pd.DataFrame({
        "sample": ["S1"] * 20 + ["S2"] * 20,
        "pct_counts_mt": [i / 100 for i in range(n)],
        "total_counts": [1000 + 5 * i for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])


class _StubClient:
    def __init__(self, parsed):
        class _M:
            def parse(_s, **kw):
                class _R: parsed_output = parsed
                return _R()
        self.messages = _M()


class _Boom:
    class messages:
        @staticmethod
        def parse(**kw):
            raise RuntimeError("no network")


def test_no_llm_heuristic_groups_by_role():
    res = rank_meta_columns(_obs(), use_llm=False)
    assert res.method == "heuristic"
    assert res.top("sample").column == "sample"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("n_counts").column == "total_counts"


def test_roles_subset():
    res = rank_meta_columns(_obs(), use_llm=False, roles=["pct_mt"])
    assert set(res.roles) == {"pct_mt"}


def test_llm_path_with_mock_client():
    parsed = RankedCandidates(candidates=[
        {"role": "pct_mt", "column": "pct_counts_mt", "kind": "single",
         "score": 0.95, "reason": "ok"}])
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False,
                            client=_StubClient(parsed))
    assert res.method == "llm (anthropic)"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("pct_mt").source == "llm"


def test_llm_failure_falls_back():
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False, client=_Boom())
    assert res.method.startswith("heuristic (llm unavailable")
    assert res.top("sample").column == "sample"


def test_top_k_truncation_per_role():
    res = rank_meta_columns(_obs(), use_llm=False, top_k=1)
    assert all(len(v) <= 1 for v in res.roles.values())


def test_input_not_mutated():
    obs = _obs(); before = obs.copy()
    rank_meta_columns(obs, use_llm=False)
    pd.testing.assert_frame_equal(obs, before)
```

Replace `tests/test_cli.py` with:

```python
import json
import anndata
import numpy as np
import pandas as pd
from stanmetacols.__main__ import main


def _write(path, obs, names):
    a = anndata.AnnData(X=np.zeros((len(obs), 2), dtype="float32"),
                        obs=obs.set_index(pd.Index(names)))
    a.write_h5ad(path)


def test_cli_emits_roles_json(tmp_path, capsys):
    p = tmp_path / "x.h5ad"
    n = 20
    obs = pd.DataFrame({"sample": ["S1"] * 10 + ["S2"] * 10,
                        "pct_counts_mt": [i / 100 for i in range(n)]})
    _write(p, obs, [f"c{i}" for i in range(n)])
    code = main([str(p), "--no-llm"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "heuristic"
    assert out["roles"]["pct_mt"][0]["column"] == "pct_counts_mt"


def test_cli_roles_subset(tmp_path, capsys):
    p = tmp_path / "y.h5ad"
    obs = pd.DataFrame({"sample": ["S1"] * 5 + ["S2"] * 5})
    _write(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm", "--roles", "sample"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["roles"]) == {"sample"}


def test_cli_exit_2_when_nothing(tmp_path, capsys):
    p = tmp_path / "z.h5ad"
    obs = pd.DataFrame({"tissue": ["lung"] * 5})       # nothing matches any role
    _write(p, obs, ["aa", "bb", "cc", "dd", "ee"])
    code = main([str(p), "--no-llm"])
    assert code == 2
    out = json.loads(capsys.readouterr().out)
    assert all(v == [] for v in out["roles"].values())


def test_cli_bad_role_exit_1(tmp_path):
    p = tmp_path / "w.h5ad"
    _write(p, pd.DataFrame({"sample": ["A", "B"]}), ["a", "b"])
    assert main([str(p), "--no-llm", "--roles", "bogus"]) == 1


def test_cli_bad_path_exit_1():
    assert main(["/no/such/file.h5ad", "--no-llm"]) == 1
```

In `tests/test_prompts.py`, update the assertion to the multi-role prompt: replace any `assert "sample" in SYSTEM_PROMPT.lower()` style check with:

```python
from stanmetacols.prompts import SYSTEM_PROMPT, ADJUDICATION_SYSTEM_PROMPT

def test_prompts_mention_roles_and_json():
    assert "json" in SYSTEM_PROMPT.lower()
    for token in ("sample", "pct_mt", "n_counts", "n_genes", "doublet"):
        assert token in SYSTEM_PROMPT
    assert "canonical" in ADJUDICATION_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run to verify they fail**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_heuristic.py tests/test_rank.py tests/test_cli.py tests/test_prompts.py -q`
Expected: FAIL — wrong signatures / `rank_meta_columns` and `ADJUDICATION_SYSTEM_PROMPT` not defined.

- [ ] **Step 3: Rewrite `prompts.py`**

```python
"""Prompts for the holistic ranking call and the numeric adjudication call."""

from __future__ import annotations

import json

from .schema import ObsDigest
from .roles import ROLES

_ROLE_DESCRIPTIONS = {
    "sample": "the sample each cell came from (grouping unit for per-sample QC / pseudobulk)",
    "pct_mt": "per-cell mitochondrial-gene fraction (a float in [0,1])",
    "pct_hb": "per-cell hemoglobin-gene fraction (a float in [0,1])",
    "doublet_score": "per-cell doublet detection score (a float in [0,1])",
    "n_counts": "total counts / UMIs per cell (a non-negative integer, large)",
    "n_genes": "number of genes detected per cell (a non-negative integer)",
}


def _roles_block(roles) -> str:
    lines = []
    for k in roles:
        aliases = ", ".join(ROLES[k].aliases[:6])
        lines.append(f"- {k}: {_ROLE_DESCRIPTIONS[k]}. Common names: {aliases}.")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are given a digest of an AnnData .obs table from a single-cell dataset. "
    "For EACH requested role, rank which .obs column best fills it. The roles:\n"
    + _roles_block(ROLES.keys()) + "\n\n"
    "All pct_* values and doublet_score are fractions in [0,1], NOT percents in "
    "[0,100]. Use the provided value stats (v_min/v_max/v_median, frac_unit, "
    "is_integer_valued) as evidence, not just the column name: counts/genes are "
    "non-negative integers (counts >> genes), the fractions live in [0,1]. "
    "Distinguish look-alikes: total_counts is the per-cell total, while "
    "total_counts_mt / total_counts_hb are subset counts and are NOT n_counts; "
    "n_genes_by_counts is n_genes, not n_counts.\n\n"
    "Return JSON only, matching the schema: a list of candidates, each with "
    "`role` (one of the requested roles), `column` (the .obs column name; for the "
    "sample role a composite may use its exact 'a + b' label or the barcode "
    "label), `kind` ('single'|'composite'|'barcode'), `score` in 0..1, and a "
    "one-sentence `reason`. Only include plausible candidates; omit a role "
    "entirely if no column fits."
)

ADJUDICATION_SYSTEM_PROMPT = (
    "You are disambiguating look-alike numeric columns in a single-cell .obs "
    "table. For each role below you are given several candidate columns with "
    "their value statistics. Pick the SINGLE canonical column for each role. "
    "Remember: pct_* and doublet_score are fractions in [0,1]; n_counts is the "
    "per-cell TOTAL counts/UMIs (largest), not a per-subset count like "
    "total_counts_mt; n_genes is genes detected per cell. Return JSON only: a "
    "list of verdicts, each with `role`, `column` (must be one of that role's "
    "given candidates), and a one-sentence `reason`."
)


def build_user_prompt(digest: ObsDigest, roles) -> str:
    return (
        "Requested roles: " + ", ".join(roles) + "\n\n"
        "Here is the .obs digest (JSON):\n\n"
        + json.dumps(digest.to_prompt_dict(), sort_keys=True, indent=2)
        + "\n\nRank the columns that fill each requested role."
    )


def build_adjudication_prompt(digest: ObsDigest, contention) -> str:
    # contention: dict[role_key -> list[Candidate]]
    by_col = {c.name: c for c in digest.columns}
    blocks = []
    for role, cands in contention.items():
        lines = [f"Role {role} — candidates:"]
        for cand in cands:
            p = by_col.get(cand.column)
            stats = ("" if p is None else
                     f" [v_min={p.v_min:.3g}, v_max={p.v_max:.3g}, "
                     f"v_median={p.v_median:.3g}, frac_unit={p.frac_unit:.2f}, "
                     f"is_integer_valued={p.is_integer_valued}]")
            lines.append(f"  - {cand.column}{stats}")
        blocks.append("\n".join(lines))
    return ("Pick the canonical column for each role.\n\n" + "\n\n".join(blocks)
            + "\n\nReturn one verdict per role.")
```

- [ ] **Step 4: Rewrite `heuristic.py`**

```python
"""Deterministic per-role scorer over an ObsDigest. No network, no API key."""

from __future__ import annotations

from .schema import ObsDigest, Candidate
from .roles import ROLES, name_signal, value_check
from .prompts import ROLES as _UNUSED  # noqa: F401  (keeps import graph explicit)

# --- sample (grouping) scorer: the original stansample logic ---

from .roles import normalize as _normalize

_SAMPLE_ALIASES = {_normalize(a) for a in ROLES["sample"].aliases}


def _sample_name_signal(name: str) -> float:
    n = _normalize(name)
    if n in _SAMPLE_ALIASES:
        return 1.0
    if any(a in n for a in _SAMPLE_ALIASES):
        return 0.6
    return 0.0


def _cardinality_signal(n_unique: int, n_obs: int) -> float:
    if n_obs == 0 or n_unique < 2:
        return 0.0
    upper = max(50, int(n_obs * 0.2))
    return 1.0 if n_unique <= upper else 0.3


def _sample_score(c, n_obs):
    name_sig = _sample_name_signal(c.name)
    card = _cardinality_signal(c.n_unique, n_obs)
    penalty = 0.0
    if c.dtype == "float":
        penalty += 0.5
    if c.looks_like_barcode:
        penalty += 0.5
    if n_obs and (c.n_missing / n_obs) > 0.5:
        penalty += 0.3
    raw = 0.5 * name_sig + 0.25 * card + 0.25 * c.balance - penalty
    return max(0.0, min(1.0, raw)), name_sig


def _rank_sample(digest: ObsDigest) -> list:
    out = []
    n_obs = digest.n_obs
    for c in digest.columns:
        if c.single_value or c.unique_per_cell:
            continue
        score, name_sig = _sample_score(c, n_obs)
        if score <= 0:
            continue
        out.append(Candidate(
            role="sample", column=c.name, kind="single", score=score,
            source="heuristic",
            reason=(f"name match={name_sig:.1f}, n_unique={c.n_unique}, "
                    f"balance={c.balance:.2f}")))
    for comp in digest.composite_candidates:
        name_sig = sum(_sample_name_signal(col) for col in comp.columns) / len(comp.columns)
        card = _cardinality_signal(comp.n_unique, n_obs)
        score = max(0.0, min(1.0, 0.85 * (0.5 * name_sig + 0.25 * card + 0.25 * comp.balance)))
        if score <= 0:
            continue
        out.append(Candidate(
            role="sample", column=comp.label, kind="composite", score=score,
            source="heuristic",
            reason=(f"composite of {comp.columns}, n_unique={comp.n_unique}, "
                    f"balance={comp.balance:.2f}")))
    if digest.barcode is not None:
        bc = digest.barcode
        score = max(0.0, min(1.0, 0.45 * bc.balance + 0.1))
        if score > 0:
            out.append(Candidate(
                role="sample", column=bc.label, kind="barcode", score=score,
                source="heuristic",
                reason=(f"barcode {bc.position} on '{bc.delimiter}', "
                        f"{bc.n_groups} groups, balance={bc.balance:.2f}")))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _rank_numeric(digest: ObsDigest, role) -> list:
    out = []
    for c in digest.columns:
        ns = name_signal(c.name, role)
        if ns <= 0:                       # numeric roles require a name hit
            continue
        vc = value_check(c, role)
        score = max(0.0, min(1.0, 0.6 * ns + 0.4 * vc))
        if score <= 0:
            continue
        out.append(Candidate(
            role=role.key, column=c.name, kind="single", score=score,
            source="heuristic",
            reason=(f"name match={ns:.1f}, value_fit={vc:.1f}, "
                    f"median={c.v_median:.3g}, frac_unit={c.frac_unit:.2f}")))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def rank_heuristic(digest: ObsDigest, roles) -> dict:
    out = {}
    for key in roles:
        role = ROLES[key]
        out[key] = _rank_sample(digest) if role.type == "grouping" else _rank_numeric(digest, role)
    return out
```

(Remove the now-unused `from .prompts import ROLES as _UNUSED` line if your linter objects — it is only there to be explicit and may be deleted.)

- [ ] **Step 5: Rewrite `llm.py` stage-1 to multi-role**

Change `_valid_labels` (unchanged), and replace `rank_with_llm` so it takes `roles`, builds the multi-role prompt, and groups results by role. The provider plumbing (`_call_anthropic`, `_call_openai`, `_extract_json`, `_parse_ranked`) is unchanged. The post-processing now also drops candidates whose `role` is not requested:

```python
def rank_with_llm(digest: ObsDigest, roles, *, provider: str = "anthropic",
                  model: str = "claude-opus-4-8", client=None,
                  base_url: str | None = None, api_key: str | None = None,
                  max_tokens: int = 2048) -> dict:
    if provider == "anthropic":
        parsed = _call_anthropic(digest, roles, model, client, max_tokens)
    elif provider == "openai":
        parsed = _call_openai(digest, roles, model, client, base_url, api_key, max_tokens)
    else:
        raise LLMUnavailable(f"unknown provider: {provider!r}")

    labels = _valid_labels(digest)
    requested = set(roles)
    out = {k: [] for k in roles}
    for rc in parsed.candidates:
        if rc.role not in requested:
            continue
        kind = labels.get(rc.column)
        if kind is None:                  # hallucinated column -> drop
            continue
        score = max(0.0, min(1.0, float(rc.score)))
        out[rc.role].append(Candidate(role=rc.role, column=rc.column, kind=kind,
                                      score=score, reason=rc.reason, source="llm"))
    for k in out:
        out[k].sort(key=lambda c: c.score, reverse=True)
    return out
```

Update `_call_anthropic` / `_call_openai` to take `roles` and pass it to `build_user_prompt(digest, roles)` (the only change to those two functions: the user-content line becomes `build_user_prompt(digest, roles)`).

- [ ] **Step 6: Rewrite `rank.py`**

```python
"""Public orchestrator: build digest, rank per role (LLM stage 1 + heuristic
fallback). Numeric adjudication (stage 2) is wired in Task 6."""

from __future__ import annotations

from .schema import MetaColsResult, LLMUnavailable
from .profile import profile_obs
from .roles import ROLE_KEYS
from .llm import rank_with_llm
from .heuristic import rank_heuristic


def _extract(data):
    if hasattr(data, "obs_names") and hasattr(data, "obs"):
        return data.obs, list(data.obs_names)
    return data, list(data.index)


def rank_meta_columns(data, *, roles=None, use_llm: bool = True,
                      adjudicate: bool = True, provider: str = "anthropic",
                      model: str = "claude-opus-4-8", client=None,
                      base_url: str | None = None, api_key: str | None = None,
                      top_k: int | None = 5) -> MetaColsResult:
    role_keys = list(roles) if roles else list(ROLE_KEYS)
    obs, obs_names = _extract(data)
    digest = profile_obs(obs, obs_names)

    if use_llm:
        try:
            ranked = rank_with_llm(digest, role_keys, provider=provider,
                                   model=model, client=client,
                                   base_url=base_url, api_key=api_key)
            method = f"llm ({provider})"
        except LLMUnavailable as exc:
            ranked = rank_heuristic(digest, role_keys)
            method = f"heuristic (llm unavailable: {exc})"
    else:
        ranked = rank_heuristic(digest, role_keys)
        method = "heuristic"

    for k in ranked:
        ranked[k] = sorted(ranked[k], key=lambda c: c.score, reverse=True)
        if top_k and top_k > 0:
            ranked[k] = ranked[k][:top_k]
    return MetaColsResult(roles=ranked, method=method, digest=digest)
```

- [ ] **Step 7: Rewrite `__main__.py`**

```python
"""CLI: identify metadata-role columns of an .h5ad file. JSON on stdout."""

import argparse
import json
import os
import sys
from dataclasses import asdict

from .rank import rank_meta_columns
from .roles import ROLE_KEYS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stanmetacols",
        description="Identify which .obs columns fill standard metadata roles. "
                    "Emits a JSON object on stdout.")
    parser.add_argument("path", help="path to an .h5ad file")
    parser.add_argument("--roles", default=None,
                        help="comma-separated subset of: " + ",".join(ROLE_KEYS))
    parser.add_argument("--no-llm", action="store_true",
                        help="force the offline heuristic ranker (no API call)")
    parser.add_argument("--top", type=int, default=5,
                        help="keep top K candidates per role (default 5; 0 = all)")
    parser.add_argument("--provider", choices=["anthropic", "openai"],
                        default="anthropic")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    args = parser.parse_args(argv)

    roles = None
    if args.roles:
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        bad = [r for r in roles if r not in ROLE_KEYS]
        if bad:
            print(f"error: unknown role(s): {', '.join(bad)}; "
                  f"valid: {', '.join(ROLE_KEYS)}", file=sys.stderr)
            return 1

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None

    try:
        import anndata
        adata = anndata.read_h5ad(args.path, backed="r")
    except Exception as exc:
        print(f"error: cannot read {args.path!r}: {exc}", file=sys.stderr)
        return 1

    result = rank_meta_columns(
        adata, roles=roles, use_llm=not args.no_llm, provider=args.provider,
        model=args.model, base_url=args.base_url, api_key=api_key, top_k=args.top)

    print(json.dumps(
        {"method": result.method,
         "roles": {k: [asdict(c) for c in v] for k, v in result.roles.items()}},
        indent=2))

    any_found = any(v for v in result.roles.values())
    return 0 if any_found else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Update `__init__.py`**

```python
"""stanmetacols — identify which .obs columns fill standard metadata roles."""

from .schema import Candidate, MetaColsResult, ObsDigest, LLMUnavailable
from .profile import profile_obs
from .roles import ROLES, ROLE_KEYS
from .rank import rank_meta_columns

__version__ = "0.2.0"

__all__ = [
    "rank_meta_columns", "profile_obs", "Candidate", "MetaColsResult",
    "ObsDigest", "ROLES", "ROLE_KEYS", "LLMUnavailable", "__version__",
]
```

- [ ] **Step 9: Delete the dead single-role API**

In `src/stanmetacols/schema.py`, remove the `RankResult` dataclass (now unused). Confirm nothing imports it:

```bash
grep -rn 'RankResult\|rank_sample_columns' src tests
```
Expected: no matches (fix any that remain).

- [ ] **Step 10: Run the full suite**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest -q`
Expected: PASS — all multi-role tests plus the unchanged profile/roles/schema tests.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: multi-role pipeline (heuristic + LLM stage-1 + rank_meta_columns + CLI)"
```

---

## Task 6: Stage-2 numeric adjudication

**Files:**
- Modify: `src/stanmetacols/llm.py` (add `adjudicate_numeric`)
- Modify: `src/stanmetacols/rank.py` (wire stage 2)
- Test: `tests/test_llm.py`, `tests/test_rank.py`

**Interfaces:**
- Consumes: `Adjudications` (schema.py), `ADJUDICATION_SYSTEM_PROMPT`, `build_adjudication_prompt` (prompts.py), `NUMERIC_ROLE_KEYS` (roles.py), the provider helpers in llm.py.
- Produces:
  - `adjudicate_numeric(digest, contention, *, provider="anthropic", model="claude-opus-4-8", client=None, base_url=None, api_key=None, max_tokens=1024) -> dict[str, tuple]` returning `role -> (column, reason)`.
  - `rank_meta_columns` now runs stage 2 when `adjudicate=True`, stage 1 succeeded, and a numeric role is ambiguous; `method` gains `" + adjudication"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_rank.py`:

```python
from stanmetacols.schema import Adjudications


class _TwoStageClient:
    """messages.parse returns the stage-1 ranking first, the adjudication second."""
    def __init__(self, stage1, stage2):
        self._responses = [stage1, stage2]
        class _M:
            def parse(_s, **kw):
                payload = self._responses.pop(0)
                class _R: parsed_output = payload
                return _R()
        self.messages = _M()


def _ambiguous_obs():
    n = 40
    return pd.DataFrame({
        "total_counts": [1000 + 5 * i for i in range(n)],
        "total_counts_mt": [10 + i for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])


def test_adjudication_reorders_numeric_role():
    from stanmetacols.schema import RankedCandidates
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.80, "reason": "looks like counts"},
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.78, "reason": "also counts-like"}])
    stage2 = Adjudications(verdicts=[
        {"role": "n_counts", "column": "total_counts",
         "reason": "per-cell total, not the mt subset"}])
    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_TwoStageClient(stage1, stage2))
    assert res.method == "llm (anthropic) + adjudication"
    assert res.top("n_counts").column == "total_counts"
    assert "subset" in res.top("n_counts").reason


def test_no_adjudication_when_clear_winner():
    from stanmetacols.schema import RankedCandidates
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.95, "reason": "clear"},
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.40, "reason": "weak"}])
    # second response would raise if called
    class _OneCall:
        def __init__(self, payload):
            self._p = [payload]
            class _M:
                def parse(_s, **kw):
                    class _R: parsed_output = self._p.pop(0)
                    return _R()
            self.messages = _M()
    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_OneCall(stage1))
    assert res.method == "llm (anthropic)"     # no " + adjudication"
    assert res.top("n_counts").column == "total_counts"


def test_adjudication_failure_keeps_stage1():
    from stanmetacols.schema import RankedCandidates
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.80, "reason": "a"},
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.78, "reason": "b"}])

    class _Stage2Boom:
        def __init__(self, s1):
            self._first = [s1]
            outer = self
            class _M:
                def parse(_s, **kw):
                    if outer._first:
                        payload = outer._first.pop(0)
                        class _R: parsed_output = payload
                        return _R()
                    raise RuntimeError("adjudication network error")
            self.messages = _M()

    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_Stage2Boom(stage1))
    assert res.method == "llm (anthropic)"     # stage-1 kept, non-fatal
    assert res.top("n_counts").column == "total_counts"   # unchanged top
```

- [ ] **Step 2: Run to verify they fail**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_rank.py -q -k adjudic`
Expected: FAIL — adjudication not wired (`method` lacks `+ adjudication`).

- [ ] **Step 3: Add `adjudicate_numeric` to `llm.py`**

```python
from .schema import Adjudications
from .prompts import ADJUDICATION_SYSTEM_PROMPT, build_adjudication_prompt


def _call_anthropic_adjudication(prompt, model, client, max_tokens) -> Adjudications:
    if client is None:
        try:
            import anthropic
        except Exception as exc:
            raise LLMUnavailable(f"anthropic not installed: {exc}") from exc
        try:
            client = anthropic.Anthropic()
        except Exception as exc:
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc
    try:
        resp = client.messages.parse(
            model=model, max_tokens=max_tokens,
            system=ADJUDICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=Adjudications)
    except Exception as exc:
        raise LLMUnavailable(str(exc)) from exc
    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("adjudication returned no parseable output")
    return parsed


def _call_openai_adjudication(prompt, model, client, base_url, api_key, max_tokens) -> Adjudications:
    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMUnavailable(f"openai not installed: {exc}") from exc
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        try:
            client = OpenAI(**kwargs)
        except Exception as exc:
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc
    try:
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}])
    except Exception as exc:
        raise LLMUnavailable(str(exc)) from exc
    try:
        text = resp.choices[0].message.content
    except Exception as exc:
        raise LLMUnavailable(f"unexpected response shape: {exc}") from exc
    blob = _extract_json(text)
    try:
        import json as _json
        data = _json.loads(blob)
    except Exception as exc:
        raise LLMUnavailable(f"adjudication is not valid JSON: {exc}") from exc
    if isinstance(data, list):
        data = {"verdicts": data}
    try:
        return Adjudications.model_validate(data)
    except Exception as exc:
        raise LLMUnavailable(f"adjudication does not match schema: {exc}") from exc


def adjudicate_numeric(digest, contention, *, provider: str = "anthropic",
                       model: str = "claude-opus-4-8", client=None,
                       base_url: str | None = None, api_key: str | None = None,
                       max_tokens: int = 1024) -> dict:
    prompt = build_adjudication_prompt(digest, contention)
    if provider == "anthropic":
        parsed = _call_anthropic_adjudication(prompt, model, client, max_tokens)
    elif provider == "openai":
        parsed = _call_openai_adjudication(prompt, model, client, base_url, api_key, max_tokens)
    else:
        raise LLMUnavailable(f"unknown provider: {provider!r}")
    offered = {role: {c.column for c in cands} for role, cands in contention.items()}
    verdicts = {}
    for v in parsed.verdicts:
        if v.role in offered and v.column in offered[v.role]:
            verdicts[v.role] = (v.column, v.reason)
    return verdicts
```

- [ ] **Step 4: Wire stage 2 into `rank.py`**

Add the imports and helpers, and extend the `use_llm` success branch:

```python
from .roles import ROLE_KEYS, NUMERIC_ROLE_KEYS
from .llm import rank_with_llm, adjudicate_numeric

_ADJ_MARGIN = 0.15


def _ambiguous_numeric(ranked, margin: float) -> dict:
    amb = {}
    for key in NUMERIC_ROLE_KEYS:
        cands = ranked.get(key) or []
        if len(cands) >= 2 and (cands[0].score - cands[1].score) <= margin:
            top = cands[0].score
            amb[key] = [c for c in cands if c.score >= top - margin]
    return amb


def _apply_verdicts(ranked, verdicts) -> None:
    for role, (column, reason) in verdicts.items():
        cands = ranked.get(role) or []
        chosen = next((c for c in cands if c.column == column), None)
        if chosen is None:
            continue
        chosen.score = max(c.score for c in cands)   # force rank-1 under stable sort
        chosen.reason = reason
        cands.remove(chosen)
        cands.insert(0, chosen)
```

In the `use_llm` success branch (right after `method = f"llm ({provider})"`), add:

```python
            if adjudicate:
                amb = _ambiguous_numeric(ranked, _ADJ_MARGIN)
                if amb:
                    try:
                        verdicts = adjudicate_numeric(
                            digest, amb, provider=provider, model=model,
                            client=client, base_url=base_url, api_key=api_key)
                        if verdicts:
                            _apply_verdicts(ranked, verdicts)
                            method += " + adjudication"
                    except LLMUnavailable:
                        pass        # stage-2 is non-fatal; keep stage-1 ranking
```

(Because `_apply_verdicts` sets the chosen score to the role's max and moves it first, the final per-role `sorted(..., reverse=True)` in the orchestrator is stable and keeps it at rank 1.)

- [ ] **Step 5: Run the adjudication tests, then the full suite**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_rank.py -q -k adjudic`
Expected: PASS.
Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest -q`
Expected: PASS (whole suite).

- [ ] **Step 6: Commit**

```bash
git add src/stanmetacols/llm.py src/stanmetacols/rank.py tests/test_rank.py tests/test_llm.py
git commit -m "feat: stage-2 numeric adjudication (focused LLM tie-break, non-fatal)"
```

---

## Task 7: User guidance hint (`--hint`)

**Files:**
- Modify: `src/stanmetacols/prompts.py`
- Modify: `src/stanmetacols/llm.py`
- Modify: `src/stanmetacols/rank.py`
- Modify: `src/stanmetacols/__main__.py`
- Test: `tests/test_prompts.py`, `tests/test_llm.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `build_user_prompt(digest, roles)`, `build_adjudication_prompt(digest, contention)`, `rank_with_llm(digest, roles, ...)`, `adjudicate_numeric(digest, contention, ...)`, `rank_meta_columns(...)`.
- Produces: an optional free-text `hint: str = ""` threaded through `rank_meta_columns` → `rank_with_llm` AND `adjudicate_numeric` → the LLM prompts. `build_user_prompt(digest, roles, hint="")` and `build_adjudication_prompt(digest, contention, hint="")` gain a trailing `hint` parameter. CLI gains `--hint` (default `""`). The hint is LLM-only: NO effect on `--no-llm` / the heuristic fallback (silently ignored). Default `""` ⇒ byte-identical prompts to before.

Rationale: when both the heuristic and the LLM miss a column, the user steers the LLM at runtime (e.g. "the mito fraction column is named `mt.frac`"). The hint is injected as an authoritative block at the TOP of the LLM user prompt so the model weights it heavily.

- [ ] **Step 1: Write the failing prompt tests**

Add to `tests/test_prompts.py`:

```python
import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.prompts import build_user_prompt, build_adjudication_prompt


def _d():
    return profile_obs(pd.DataFrame({"sample": ["A", "B"]}))


def test_user_prompt_includes_hint_block():
    p = build_user_prompt(_d(), ["sample"], hint="mito col is mt.frac")
    assert "User guidance" in p
    assert "mito col is mt.frac" in p


def test_user_prompt_omits_block_when_hint_empty():
    p = build_user_prompt(_d(), ["sample"])
    assert "User guidance" not in p


def test_adjudication_prompt_includes_hint():
    p = build_adjudication_prompt(_d(), {}, hint="counts are in total_umis")
    assert "User guidance" in p and "total_umis" in p
```

- [ ] **Step 2: Run to verify they fail**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_prompts.py -q`
Expected: FAIL — `build_user_prompt()`/`build_adjudication_prompt()` got an unexpected keyword argument `hint`.

- [ ] **Step 3: Add the hint block to `prompts.py`**

```python
def _hint_block(hint: str) -> str:
    hint = (hint or "").strip()
    if not hint:
        return ""
    return ("User guidance (authoritative — follow this to locate the columns):\n"
            + hint + "\n\n")


def build_user_prompt(digest: ObsDigest, roles, hint: str = "") -> str:
    return (
        _hint_block(hint)
        + "Requested roles: " + ", ".join(roles) + "\n\n"
        "Here is the .obs digest (JSON):\n\n"
        + json.dumps(digest.to_prompt_dict(), sort_keys=True, indent=2)
        + "\n\nRank the columns that fill each requested role."
    )
```

And in `build_adjudication_prompt`, add the `hint` param and prepend the block to its returned string:

```python
def build_adjudication_prompt(digest: ObsDigest, contention, hint: str = "") -> str:
    # contention: dict[role_key -> list[Candidate]]
    by_col = {c.name: c for c in digest.columns}
    blocks = []
    for role, cands in contention.items():
        lines = [f"Role {role} — candidates:"]
        for cand in cands:
            p = by_col.get(cand.column)
            stats = ("" if p is None else
                     f" [v_min={p.v_min:.3g}, v_max={p.v_max:.3g}, "
                     f"v_median={p.v_median:.3g}, frac_unit={p.frac_unit:.2f}, "
                     f"is_integer_valued={p.is_integer_valued}]")
            lines.append(f"  - {cand.column}{stats}")
        blocks.append("\n".join(lines))
    return (_hint_block(hint) + "Pick the canonical column for each role.\n\n"
            + "\n\n".join(blocks) + "\n\nReturn one verdict per role.")
```

- [ ] **Step 4: Run prompt tests to verify pass**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_prompts.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing llm-threading test**

Add to `tests/test_llm.py`:

```python
def test_hint_reaches_user_prompt():
    parsed = RankedCandidates(candidates=[])
    client = _StubClient(parsed)
    rank_with_llm(_digest(), ["sample"], hint="HINTTOKEN", client=client)
    content = client.messages.kwargs["messages"][0]["content"]
    assert "HINTTOKEN" in content
```

(`_StubClient` / `_digest` already exist in `tests/test_llm.py`. `rank_with_llm`'s first positional after `digest` is `roles`.)

- [ ] **Step 6: Run to verify it fails**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_llm.py::test_hint_reaches_user_prompt -q`
Expected: FAIL — `rank_with_llm()` got an unexpected keyword argument `hint`.

- [ ] **Step 7: Thread `hint` through `llm.py`**

`rank_with_llm` gains `hint`, passes it to the backend callers, which pass it to `build_user_prompt`:

```python
def rank_with_llm(digest: ObsDigest, roles, *, hint: str = "",
                  provider: str = "anthropic", model: str = "claude-opus-4-8",
                  client=None, base_url: str | None = None,
                  api_key: str | None = None, max_tokens: int = 2048) -> dict:
    if provider == "anthropic":
        parsed = _call_anthropic(digest, roles, hint, model, client, max_tokens)
    elif provider == "openai":
        parsed = _call_openai(digest, roles, hint, model, client, base_url, api_key, max_tokens)
    else:
        raise LLMUnavailable(f"unknown provider: {provider!r}")
    # ... (post-processing unchanged) ...
```

`_call_anthropic` / `_call_openai` gain a `hint` parameter (right after `roles`) and call `build_user_prompt(digest, roles, hint)` instead of `build_user_prompt(digest, roles)`.

`adjudicate_numeric` gains `hint` and passes it into the prompt builder:

```python
def adjudicate_numeric(digest, contention, *, hint: str = "",
                       provider: str = "anthropic", model: str = "claude-opus-4-8",
                       client=None, base_url: str | None = None,
                       api_key: str | None = None, max_tokens: int = 1024) -> dict:
    prompt = build_adjudication_prompt(digest, contention, hint)
    # ... (rest unchanged) ...
```

- [ ] **Step 8: Run to verify the llm test passes**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest tests/test_llm.py -q`
Expected: PASS.

- [ ] **Step 9: Thread `hint` through `rank_meta_columns` (`rank.py`)**

Add `hint: str = ""` to the signature (after `adjudicate`), and pass it to both LLM calls:

```python
def rank_meta_columns(data, *, roles=None, use_llm: bool = True,
                      adjudicate: bool = True, hint: str = "",
                      provider: str = "anthropic", model: str = "claude-opus-4-8",
                      client=None, base_url: str | None = None,
                      api_key: str | None = None, top_k: int | None = 5) -> MetaColsResult:
    ...
    ranked = rank_with_llm(digest, role_keys, hint=hint, provider=provider,
                           model=model, client=client, base_url=base_url, api_key=api_key)
    ...
    verdicts = adjudicate_numeric(digest, amb, hint=hint, provider=provider,
                                  model=model, client=client, base_url=base_url, api_key=api_key)
```

(The heuristic branch ignores `hint` — that is the spec'd "LLM-only" behavior.)

- [ ] **Step 10: Add `--hint` to the CLI and write the CLI test**

In `src/stanmetacols/__main__.py`, add the argument and pass it through:

```python
    parser.add_argument("--hint", default="",
                        help="optional free-text guidance for the LLM to locate "
                             "columns (LLM path only; ignored with --no-llm)")
```

and in the `rank_meta_columns(...)` call add `hint=args.hint`.

Add to `tests/test_cli.py`:

```python
def test_cli_hint_accepted_offline(tmp_path, capsys):
    p = tmp_path / "h.h5ad"
    obs = pd.DataFrame({"sample": ["S1"] * 5 + ["S2"] * 5})
    _write(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm", "--hint", "ignore me offline"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["roles"]["sample"][0]["column"] == "sample"
```

(`_write` is the helper already defined in `tests/test_cli.py`.)

- [ ] **Step 11: Run the full suite**

Run: `/scratch/users/chensj16/venvs/dl2025/.venv/bin/python -m pytest -q`
Expected: PASS (55 + 5 new = 60).

- [ ] **Step 12: Commit**

```bash
git add src/stanmetacols/prompts.py src/stanmetacols/llm.py src/stanmetacols/rank.py src/stanmetacols/__main__.py tests/
git commit -m "feat: --hint user guidance threaded into both LLM calls (LLM-only, default empty)"
```

---

## Task 8: Documentation rewrite (README + formulation)

**Files:**
- Modify: `README.md`
- Modify: `docs/formulation.md`

**Interfaces:**
- Consumes: the final public API and CLI from Tasks 5–6.
- Produces: docs that describe the six roles, the two-stage LLM path, the role registry, the output schema, and the CLI — no `stansample`/`rank_sample_columns`/single-role language remains.

- [ ] **Step 1: Rewrite README.md**

Replace the body so it documents: the six roles (table from the spec §2), Install (the `[llm]`/`[openai]`/`[anndata]` extras), CLI (`--roles`, `--no-llm`, `--top`, `--provider`/`--base-url`/`--api-key-env`), Providers (unchanged from before), Output (the `{method, roles:{...}}` shape with a field table and the `llm (<provider>) + adjudication` method values), Library (`rank_meta_columns(...)` signature and a `.top("pct_mt")` example), and a "How it works" section pointing at `docs/formulation.md`. Keep a `jq` example, e.g.:

```bash
stanmetacols sample.h5ad --no-llm | jq -r '.roles.pct_mt[0].column // empty'
```

- [ ] **Step 2: Rewrite docs/formulation.md**

Generalize the formulation to multi-role: the digest now carries numeric value stats; per-role scoring splits into the grouping scorer (sample, unchanged formula) and the numeric scorer `clip(0.6·name + 0.4·value)` gated on `name>0`; the role registry's name signal (exact/token/substring) and value checks; the two-stage LLM path (holistic ranking + numeric adjudication with the `Δ=0.15` ambiguity rule); orchestration and output keyed by role.

- [ ] **Step 3: Verify no stale single-role references remain**

```bash
grep -rn 'rank_sample_columns\|RankResult\|single .obs column' README.md docs/formulation.md
```
Expected: no matches.

- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/formulation.md
git commit -m "docs: rewrite README + formulation for multi-role stanmetacols"
git push origin master
```

---

## Self-Review (completed by plan author)

**Spec coverage:** §2 roles → Task 3 (registry) + Task 5 (scoring) + Task 4 (types). §3 architecture → file structure + Tasks 1–7. §4 role registry → Task 3. §5 digest numeric stats → Task 2. §6 scoring (grouping + numeric, name gate) → Task 5. §7.1 holistic LLM → Task 5. §7.2 adjudication → Task 6. §8 output → Task 4 (MetaColsResult) + Task 5 (CLI JSON). §9 CLI → Task 5. §10 API → Task 5. §11 rename → Task 1. §12 testing → tests in every task. §13 success criteria → covered by Task 5/6 tests + Task 7 doc sweep. No gaps.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows code; every test step shows assertions. (The `from .prompts import ROLES as _UNUSED` line in Task 4 is flagged as deletable, not a placeholder.)

**Type consistency:** `Candidate(role, column, kind, score, reason, source)` used identically in heuristic.py, llm.py, and tests. `rank_heuristic(digest, roles)->dict` and `rank_with_llm(digest, roles, ...)->dict` consumed consistently by `rank_meta_columns`. `MetaColsResult.roles` is `dict[str,list[Candidate]]` everywhere. `adjudicate_numeric(...)->dict[str,tuple]` consumed by `_apply_verdicts`. `ROLE_KEYS`/`NUMERIC_ROLE_KEYS` tuples match between roles.py and consumers.
