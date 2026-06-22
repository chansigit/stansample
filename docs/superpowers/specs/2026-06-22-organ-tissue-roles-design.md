# stanmetacols — organ & tissue roles design spec

**Date:** 2026-06-22
**Extends:** the 8-role stanmetacols (`2026-06-21-stanmetacols-design.md`).
Adds two new metadata roles — `organ` and `tissue` — taking the role count from
**8 → 10** and the role-type count from **3 → 4**.

## 1. Goal

Identify, from an AnnData `.obs` table, which column records the **organ** a cell
came from (a solid anatomical organ) and which records the **tissue** (the
sampled biological material / anatomical site). As with every other role, the
tool **ranks** candidate columns per role; it does not decide. Any role may be
absent (empty list).

The two roles are deliberately **split** (not one "organ/tissue" role) with
**near-disjoint vocabularies**, because the two concepts answer different
questions and datasets often carry both:

- **`organ`** — the solid anatomical organ of origin.
  Values: `heart`, `liver`, `kidney`, `lung`, `brain`, `spleen`, `pancreas`,
  `colon`, `intestine`, `stomach`, `skin`, `thyroid`, `breast`, `thymus`,
  `placenta`, …
- **`tissue`** — the sampled biological material / anatomical site.
  Values: `blood`, `PBMC`, `bone marrow`, `lymph node`, `tumor`, `biopsy`,
  `CSF`, `BAL`, `ascites`, `plasma`, `serum`, `cord blood`, `buffy coat`, …

## 2. Role typing — no generic "categorical" bucket

**Design rule (user, durable):** role `type` must stay specific and semantic.
We do **not** introduce a generic `categorical` type — a catch-all that
collapses distinct families and is hard to extend. Each new family gets its own
`type`, exactly as `celltype` already does.

- `organ` → `type = "organ"`
- `tissue` → `type = "tissue"`

`Role.type` is **internal only** — it routes scoring; it is **not** emitted in
the output JSON (each candidate keeps `role/column/kind/score/reason/source`).
Surfacing `type` in the output is explicitly out of scope (YAGNI).

Implementation is shared even though the types are distinct: both roles route to
one **vocabulary-based scorer** (`_rank_vocab`). "Specific type label" and
"shared implementation" are orthogonal — we keep both.

## 3. Components & changes

| File | Change |
|---|---|
| `roles.py` | add `vocab` field to `Role`; register `organ` + `tissue`; add `vocab_value_frac` / `vocab_name_base` helpers; extend `ROLE_KEYS`; add `VOCAB_ROLE_KEYS`. |
| `heuristic.py` | add `_rank_vocab(digest, role)`; route `type in ("organ","tissue")` to it in `rank_heuristic`. |
| `prompts.py` | add `organ` + `tissue` to `_ROLE_DESCRIPTIONS`; add an organ-vs-tissue discrimination paragraph to `SYSTEM_PROMPT`. |
| `rank.py`, `llm.py`, `__main__.py` | **no logic change** — already role-agnostic (verified); adjudication stays numeric-only. CLI `--roles` help auto-includes the new keys via `ROLE_KEYS`. |
| docs | README, `docs/formulation.md`, CHANGELOG → "10 roles, 4 types" + the vocab scorer formula + two role-table rows. |
| tests | new/extended `test_roles.py`, `test_heuristic.py`, `test_rank.py`, `test_public_api.py`, `test_prompts.py`. |

### 3.1 `roles.py`

- Extend the `Role` dataclass with `vocab: tuple = ()` (the role's value
  vocabulary; normalized-substring matched against example values).
- Register:

  ```python
  "organ": Role(
      key="organ", type="organ",
      aliases=("organ", "organ_type", "source_organ", "organ_of_origin",
               "organ_name"),
      include_tokens=("organ",),
      exclude_tokens=("organism",),        # guard: "organism" contains "organ"
      vocab=("heart", "liver", "kidney", "lung", "brain", "spleen", "pancreas",
             "stomach", "intestine", "colon", "ileum", "jejunum", "duodenum",
             "esophagus", "skin", "muscle", "bladder", "prostate", "ovary",
             "uterus", "testis", "thyroid", "adrenal", "trachea", "tongue",
             "gallbladder", "breast", "placenta", "thymus")),
  "tissue": Role(
      key="tissue", type="tissue",
      aliases=("tissue", "tissue_type", "source_tissue", "anatomical_site",
               "body_site", "sample_site", "biomaterial", "biosample",
               "biopsy_site", "sampling_site"),
      include_tokens=("tissue", "anatomicalsite", "bodysite", "biomaterial",
                      "biosample"),
      vocab=("blood", "pbmc", "peripheralblood", "wholeblood", "bonemarrow",
             "marrow", "lymphnode", "tumor", "tumour", "biopsy", "csf",
             "cerebrospinal", "bal", "bronchoalveolar", "ascites", "pleural",
             "synovial", "plasma", "serum", "cordblood", "buffycoat",
             "adipose")),
  ```

  Vocab terms are stored **normalized** (no spaces/underscores, lowercased) so
  they substring-match against `normalize(value)` — exactly how `CELLTYPE_VOCAB`
  works today. The two vocabularies are disjoint by construction; the
  cross-talk test (§5) locks this in.

- `ROLE_KEYS` appends `"organ", "tissue"` (output order: after
  `cell_type_fine`). Add `VOCAB_ROLE_KEYS = ("organ", "tissue")`.
- Helpers (parameterized generalizations of the existing celltype helpers):
  - `vocab_value_frac(profile, role) -> float` — fraction of `example_values`
    whose normalized form contains any term in `role.vocab`.
  - `vocab_name_base(col, role) -> float` — `1.0` if the normalized column name
    contains one of `role.include_tokens` **and** none of `role.exclude_tokens`
    (so an `organism` column does not register as `organ`), else `0.0`.

### 3.2 `heuristic.py` — `_rank_vocab`

Same shape as the proven celltype scorer, parameterized by the role's own
vocab/aliases:

```
name_score = max(name_signal(col, role), 0.6 * vocab_name_base(col, role))
vocab      = vocab_value_frac(profile, role)
card       = card_fit(n_unique)                # 1.0 if 2 <= n_unique <= 50 else 0.3
score      = clip(0.4*name_score + 0.4*vocab + 0.2*card, 0, 1)
```

Gating (identical philosophy to celltype):
- skip `single_value` / `unique_per_cell` columns;
- only `dtype in ("categorical", "string")`;
- drop the column if `name_score <= 0 and vocab < 0.5` (must *look* like the
  role by name or by values).

`card_fit` band `[2, 50]` covers focused studies (a few categories) through
multi-organ atlases (tens). Outside the band → `0.3` (degrade, not exclude).

Wire-in:

```python
elif role.type in ("organ", "tissue"):
    out[key] = _rank_vocab(digest, role)
```

(Keep `celltype` on its existing `_rank_celltype`; do not refactor it — out of
scope.)

### 3.3 `prompts.py`

- `_ROLE_DESCRIPTIONS`:
  - `organ`: "the solid anatomical organ a cell came from (e.g. Heart, Liver,
    Kidney, Lung, Brain)".
  - `tissue`: "the sampled biological material / anatomical site (e.g. Blood,
    PBMC, Bone marrow, Lymph node, Tumor, Biopsy, CSF)".
- Add to `SYSTEM_PROMPT`: organ = solid anatomical organ; tissue = the sampled
  material / site; the two are distinct and a dataset may have organ, tissue,
  both, or neither. Judge by the column's values (and name), mirroring the
  coarse/fine guidance already present.

## 4. Data flow (unchanged)

`profile_obs` already records `dtype`, `n_unique`, `example_values`,
`single_value`, `unique_per_cell` — everything `_rank_vocab` needs. No change to
`profile.py` or the digest schema. The orchestrator and both LLM backends iterate
roles generically, so the new roles flow through stage-1 ranking and the
heuristic fallback with zero changes there.

## 5. Testing (TDD)

- **`test_roles.py`** — `organ`/`tissue` registered with `type` `"organ"`/
  `"tissue"` (assert **not** `"categorical"`); `name_signal` hits their aliases;
  `vocab_value_frac` ≈ 1.0 on in-vocab values, ≈ 0.0 cross-vocab (organ vocab vs
  tissue values and vice versa).
- **`test_heuristic.py`** — fixture `.obs` with an `organ` column
  (`heart/liver/lung`) and a `tissue` column (`blood/PBMC/bone marrow`):
  - `organ` role's top candidate is the organ column;
  - `tissue` role's top candidate is the tissue column;
  - **no cross-talk** — the organ column does not top the tissue role and vice
    versa (key assertion validating the disjoint vocabularies).
- **`test_rank.py` / `test_public_api.py`** — `len(ROLE_KEYS) == 10`; default
  run includes `organ` and `tissue` keys in `result.roles`.
- **`test_prompts.py`** — both new role descriptions appear in the built prompt.

## 6. Out of scope (YAGNI)

- No `categorical` umbrella type, ever.
- No coarse/fine split within organ or tissue.
- No `type` field in the output JSON.
- No refactor of the existing `celltype` scorer.
- No new `profile.py` stats.
