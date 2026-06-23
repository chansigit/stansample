# stanmetacols — algorithm formulation

A precise statement of what the code computes. Every symbol below maps to a
named quantity in `profile.py`, `heuristic.py`, `llm.py`, `roles.py`, or
`rank.py`.

GitHub renders the math inline. If you read this in a plain editor, the `$…$`
spans are LaTeX.

---

## 1. Problem

The input is an `.obs` table: $n$ cells (rows), a set of columns $\mathcal{C}$,
and a list of cell names (`obs_names`) $b_1,\dots,b_n$. We want to **identify
and rank candidate columns** for each of 10 standard metadata roles.

A role is one of four types:

- a **grouping** role (`sample`) — assigns each cell to the sample it came from;
- a **numeric** role (`pct_mt`, `pct_hb`, `doublet_score`, `n_counts`,
  `n_genes`) — carries a per-cell numeric QC measurement;
- a **cell-type** role (`cell_type_coarse`, `cell_type_fine`) — carries a
  per-cell label string;
- an **organ** role (`organ`) or a **tissue** role (`tissue`) — carries a
  per-cell categorical anatomical label. Each has its own specific type
  (`organ` / `tissue`); there is no generic `categorical` type bucket.

For the grouping role, a candidate induces a partition

$$P=\{G_1,\dots,G_k\},\qquad \textstyle\bigsqcup_i G_i \subseteq \{1,\dots,n\},\quad G_i\neq\varnothing,$$

with group sizes $g_i=|G_i|$. Numeric and cell-type roles only admit single
columns (kind `"single"`). The tool **scores and ranks** candidates; it never
commits to one.

---

## 2. Partition statistics

For any partition $P$ with group sizes $g_1,\dots,g_k$ ($k=|P|$ nonempty
groups), define the **cardinality** $k$ and the **balance**

$$\beta(P)=\frac{\min_i g_i}{\max_i g_i}\in(0,1],\qquad \beta:=0 \text{ if } \max_i g_i=0.$$

$\beta=1$ is a perfectly even split; $\beta\to 0$ is one dominant group plus
slivers.

---

## 3. Column profile

Fix a column $c$ with non-missing value-counts $\mathrm{vc}(c)$ (a map *distinct
value* $\mapsto$ *count*). Write:

- $u_c=\lvert\mathrm{vc}(c)\rvert$ — number of distinct non-missing values;
- $m_c$ — number of missing entries;
- $\beta_c=\beta$ of the partition induced by $c$'s distinct values;
- $\tau_c\in\{\text{categorical},\text{string},\text{integer},\text{float},\text{bool}\}$ — dtype.

Three boolean flags drive eligibility downstream:

$$\mathrm{unique}_c=\mathbf{1}\!\left[n>0 \wedge u_c=n\right]
\quad(\text{a per-cell ID, never a sample}),$$

$$\mathrm{single}_c=\mathbf{1}\!\left[u_c\le 1\right]
\quad(\text{one value, nothing to split}),$$

$$\mathrm{barcode}_c=\mathbf{1}\!\left[\phi_c>0.5\right],\qquad
\phi_c=\frac{\bigl|\{v\in S_c:\, R\text{ matches }v\}\bigr|}{|S_c|},$$

where $S_c$ is the first $\le 1000$ distinct values of $c$ and $R$ is the barcode
pattern

$$R=\texttt{\textasciicircum[ACGTN]\{8,\}(-\textbackslash d+)?\$}\quad(\text{case-insensitive}).$$

**Numeric value stats.** For numeric columns (`is_numeric=True`), `profile.py`
additionally computes:

| Stat | Meaning |
|---|---|
| `v_min`, `v_max`, `v_median`, `v_mean` | value range and central tendency |
| `frac_nonneg` | fraction of non-missing values that are $\ge 0$ |
| `frac_unit` | fraction of non-missing values in $[0,1]$ |
| `is_integer_valued` | `True` if all non-missing values are integers (int dtype or float with no fractional part) |

These stats are carried in the digest and used by both the numeric heuristic
scorer and the LLM prompts.

---

## 4. Candidate families (grouping role only)

### 4.1 Single columns

Every column is profiled. A column is a scorable single candidate iff
$\neg\,\mathrm{single}_c \wedge \neg\,\mathrm{unique}_c$ (Section 5 then scores it
and may still drop it). Numeric and cell-type roles only use single columns.

### 4.2 Composite keys

Eligible columns:

$$\mathcal{E}=\Bigl\{\,c\in\mathcal{C}\;:\;
\neg\,\mathrm{unique}_c\,\wedge\,\neg\,\mathrm{single}_c\,\wedge\,\neg\,\mathrm{barcode}_c
\,\wedge\,\tau_c\in\{\text{cat},\text{str},\text{int},\text{bool}\}
\,\wedge\, 2\le u_c\le 0.5\,n \,\Bigr\}.$$

Sort $\mathcal{E}$ by $\beta_c$ descending and keep the top $12$ (this bounds the
$O(k^2)$ pair enumeration). For each unordered pair $\{a,b\}$ form the joint
partition (`groupby([a,b])`, nonempty cells only); let $u_{ab}$ be its number of
groups and $\beta_{ab}$ its balance. **Keep** the pair iff

$$2\le u_{ab}<n.$$

Rank kept pairs by $\beta_{ab}$ descending; keep the top $8$. The label of a kept
pair is the string `"a + b"`.

### 4.3 Barcode grouping

From the names $b_1,\dots,b_n$ (as strings), consider two rules; let
$\mathrm{frac}(\cdot)$ denote the fraction of names satisfying a predicate.

$$\textbf{prefix on }\verb|_| :\quad
\text{applicable if } \mathrm{frac}(\verb|_|\in b_i)>0.9,\quad
\text{key}(b_i)=b_i\ \text{before its last }\verb|_|.$$

$$\textbf{suffix on }\verb|-| :\quad
t_i=b_i\ \text{after its last }\verb|-|,\quad
\text{applicable if } \mathrm{frac}(t_i\in\verb|\d+|)>0.9,\quad
\text{key}(b_i)=t_i.$$

For each applicable rule build the partition with $k$ groups and **keep** it iff
$2\le k<n$. Among the kept rules choose the one of **maximum balance** $\beta$.
Its label is `"<barcode:POSITION:DELIM>"`, e.g. `<barcode:prefix:_>`.

### 4.4 The digest

The deterministic output of `profile_obs` is

$$\mathcal{D}=\bigl(n,\ \{\text{column profiles (incl. numeric stats)}\},\ \{\text{composite profiles}\},\ \text{barcode profile or }\varnothing\bigr).$$

$\mathcal{D}$ is computed without any network or LLM and without mutating the
input. Both rankers consume the *same* $\mathcal{D}$.

---

## 5. Role registry (name signal and value checks)

All name matching operates on a **normalized** string: lowercase, strip `_`, `.`,
and spaces.

### 5.1 Name signal

`name_signal(col, role)` returns a score in $\{0.0, 0.6, 0.8, 1.0\}$:

$$\mathrm{name}(c, r)=
\begin{cases}
1.0 & \nu(c)\in\nu(A_r)\quad(\text{exact alias})\\
0.8 & \text{any include-token of }r\text{ is a substring of }\nu(c),\\
    & \quad\wedge\;\text{no exclude-token present,}\\
    & \quad\wedge\;\text{for pct roles: a measure token is also present}\\
0.6 & \exists\,a\in\nu(A_r):\ a\subseteq\nu(c) \vee \nu(c)\subseteq a\quad(\text{substring})\\
0.0 & \text{otherwise.}
\end{cases}$$

Three signal tiers per role type:

| Tier | Value | Condition |
|---|---|---|
| Exact alias | 1.0 | normalized name is in the role's alias table |
| Token rule | 0.8 | include-token substring match (with exclude-token guard; pct roles require a measure word: `pct`, `percent`, `frac`, `fraction`, `proportion`) |
| Substring | 0.6 | normalized name contains or is contained by any normalized alias |

### 5.2 Numeric value checks

`value_check(profile, role)` returns a score in $[0,1]$ (or 0 if the column is
not numeric):

| Role | Check | Score |
|---|---|---|
| `pct_mt`, `pct_hb`, `doublet_score` | `frac_nonneg ≥ 0.99` **and** `frac_unit ≥ 0.99` **and** not integer-valued | 1.0 |
| `pct_mt`, `pct_hb`, `doublet_score` | `0.5 ≤ frac_unit < 0.99` (percent-scale, degraded) | 0.3 |
| `n_counts` | integer-valued, `frac_nonneg ≥ 0.99`, `v_median ≥ 100` | 1.0 |
| `n_counts` | integer-valued, `frac_nonneg ≥ 0.99`, `v_median < 100` | 0.5 |
| `n_genes` | integer-valued, `frac_nonneg ≥ 0.99`, `2 ≤ v_median ≤ 20000` | 1.0 |
| `n_genes` | integer-valued, `frac_nonneg ≥ 0.99`, outside band | 0.5 |
| any | column is not numeric | 0.0 |

### 5.3 Cell-type vocabulary check

`celltype_value_frac(profile)` scans `profile.example_values` against a
vocabulary of ~40 cell-type terms (e.g. `epithelial`, `macrophage`, `tcell`,
`cyte`, `blast`, …) and returns the fraction of example values that contain any
term. `celltype_name_base(col)` returns `1.0` if the normalized column name
contains a generic token such as `celltype`, `annotation`, `celllabel`, etc.

---

## 6. Heuristic scorer

Let $\operatorname{clip}(x)=\max\!\bigl(0,\min(1,x)\bigr)$.

### 6.1 Grouping role (`sample`)

**Name signal** — uses the sample-specific alias table directly (same formula as
§5.1 exact/substring only; no token tier for this role):

$$\mathrm{name}_{\mathrm{grp}}(c)=
\begin{cases}
1.0 & \nu(c)\in\nu(A_{\mathrm{sample}})\\
0.6 & \exists\,a\in\nu(A_{\mathrm{sample}}):\ a\subseteq\nu(c)\\
0.0 & \text{otherwise.}
\end{cases}$$

**Cardinality signal.** With soft ceiling $\theta(n)=\max(50,\,0.2\,n)$:

$$\mathrm{card}(u)=
\begin{cases}
0.0 & u<2\\
1.0 & 2\le u\le \theta(n)\\
0.3 & u>\theta(n).
\end{cases}$$

**Penalty** (single columns only):

$$\rho_c = 0.5\cdot\mathbf{1}[\tau_c=\text{float}]
\;+\;0.5\cdot\mathbf{1}[\mathrm{barcode}_c]
\;+\;0.3\cdot\mathbf{1}\!\left[\tfrac{m_c}{n}>0.5\right].$$

**Scores.** For a single column ($\neg\,\mathrm{single}_c\wedge\neg\,\mathrm{unique}_c$):

$$\boxed{\,s_c=\operatorname{clip}\bigl(0.5\,\mathrm{name}_{\mathrm{grp}}(c)+0.25\,\mathrm{card}(u_c)+0.25\,\beta_c-\rho_c\bigr)\,}$$

For a composite key $\{a,b\}$, with member-averaged name signal
$\overline{\mathrm{name}}=\tfrac12\bigl(\mathrm{name}_{\mathrm{grp}}(a)+\mathrm{name}_{\mathrm{grp}}(b)\bigr)$
and a $0.85$ discount, **no** penalty term:

$$\boxed{\,s_{ab}=\operatorname{clip}\Bigl(0.85\cdot\bigl(0.5\,\overline{\mathrm{name}}+0.25\,\mathrm{card}(u_{ab})+0.25\,\beta_{ab}\bigr)\Bigr)\,}$$

For the barcode grouping with balance $\beta_{\mathrm{bc}}$:

$$\boxed{\,s_{\mathrm{bc}}=\operatorname{clip}\bigl(0.45\,\beta_{\mathrm{bc}}+0.1\bigr)\,}$$

Any candidate with score $\le 0$ is dropped.

### 6.2 Numeric roles (`pct_mt`, `pct_hb`, `doublet_score`, `n_counts`, `n_genes`)

A column is considered only if it has a name signal hit ($\mathrm{name}(c,r)>0$).
The score is:

$$\boxed{\,s_c=\operatorname{clip}\bigl(0.6\,\mathrm{name}(c,r)+0.4\,\mathrm{value}(c,r)\bigr)\,}$$

where $\mathrm{value}(c,r)$ is the value check from §5.2. Columns with
$\mathrm{name}=0$ are **skipped entirely** (the name acts as a gate: a column
that does not look like the role at all is never a numeric candidate, regardless
of its value shape).

### 6.3 Cell-type roles (`cell_type_coarse`, `cell_type_fine`)

Only categorical or string columns that are neither single-valued nor
unique-per-cell are considered.

**Name score** combines the role-specific alias signal and the generic cell-type
name signal:

$$\mathrm{name\_score}(c,r)=\max\!\bigl(\mathrm{name}(c,r),\ 0.6\cdot\mathrm{celltype\_name\_base}(c)\bigr).$$

A column is **admitted** if $\mathrm{name\_score}>0 \vee \mathrm{vocab}(c)\ge 0.5$
(the value vocabulary scan can carry a column whose name is uninformative).

**Cardinality fit.** Expected cardinality bands:

| Role | Band | Score |
|---|---|---|
| `cell_type_coarse` | $2\le u_c\le 25$ | 1.0 |
| `cell_type_coarse` | outside band | 0.3 |
| `cell_type_fine` | $5\le u_c\le 200$ | 1.0 |
| `cell_type_fine` | outside band | 0.3 |

**Score:**

$$\boxed{\,s_c=\operatorname{clip}\bigl(0.4\,\mathrm{name\_score}(c,r)+0.4\,\mathrm{vocab}(c)+0.2\,\mathrm{card\_fit}(u_c,r)\bigr)\,}$$

Candidates with score $\le 0$ are dropped.

### 6.4 Organ role (`organ`) and tissue role (`tissue`)

These two roles each carry their **own specific type** (`organ` and `tissue`
respectively) — there is no generic `categorical` type bucket in the registry.

Both are scored by a shared vocabulary-based heuristic `_rank_vocab`, but with
per-role alias tables and per-role value vocabularies.

**Aliases.** `vocab_name_base(col, role)` returns `1.0` when the normalized
column name matches a role-specific alias (e.g. `organ`, `tissue`,
`anatomicallocation`, `samplesite` for `tissue`; `bodypart`, `sourceorgan` for
`organ`).

**Value vocabulary.** `vocab_value_frac(profile, role)` scans
`profile.example_values` against a normalized per-role vocabulary and returns
the fraction of example values that contain any term.

- `organ` vocabulary: heart, liver, kidney, lung, brain, pancreas, spleen,
  colon, intestine, stomach, thymus, ovary, testis, prostate, uterus, bladder,
  adrenal, thyroid, tonsil, appendix.
- `tissue` vocabulary: blood, pbmc, bonemarrow, lymphnode, tumor, biopsy, csf,
  adipose, skin, muscle, nerve, cornea, retina, placenta, cord, ascites,
  pleural, synovial, saliva, nasal.

The two vocabularies are **near-disjoint**, which prevents `organ` candidates
from scoring well on `tissue` columns and vice versa.

**Cardinality fit.** A shared band applies to both roles:

$$\mathrm{card\_fit}(u_c)=
\begin{cases}
1.0 & 2\le u_c\le 50\\
0.3 & \text{otherwise.}
\end{cases}$$

**Name score:**

$$\mathrm{name\_score}(c,r)=\max\!\bigl(\mathrm{name}(c,r),\ 0.6\cdot\mathrm{vocab\_name\_base}(c,r)\bigr).$$

A column is **admitted** if $\mathrm{name\_score}>0 \vee \mathrm{vocab}(c,r)\ge 0.5$.

**Score:**

$$\boxed{\,s_c=\operatorname{clip}\bigl(0.4\,\mathrm{name\_score}(c,r)+0.4\,\mathrm{vocab}(c,r)+0.2\,\mathrm{card\_fit}(u_c)\bigr)\,}$$

**Rejection gate.** A column is **dropped** (score forced to 0) if
$\mathrm{name\_score}(c,r)\le 0$ and $\mathrm{vocab}(c,r)<0.5$ — neither the
name nor the value content is informative enough.

**Exclude-token guard (organ only).** If the normalized column name contains the
token `organism` (e.g. `organism_id`, `source_organism`), the column is
**excluded** from the `organ` role regardless of other signals, to avoid
confusing taxonomy labels with anatomical organ labels.

---

## 7. Two-stage LLM scorer

### 7.1 Stage 1 — holistic ranking

One structured LLM call over all requested roles simultaneously:

$$\mathcal{R}=\mathrm{LLM}\bigl(\text{system prompt},\ \text{[hint block] + roles + JSON}(\mathcal{D})\bigr).$$

Two backends, selected by `provider`:

- **`anthropic`** (default) — native `messages.parse` constrains the reply to the
  Pydantic schema `RankedCandidates` directly.
- **`openai`** — any OpenAI-compatible `/chat/completions` endpoint. The reply
  text is parsed: strip Markdown fences, slice the outermost JSON object/array,
  then validate against the same Pydantic schema. A bare array is wrapped as
  `{"candidates": […]}`.

The schema for each candidate is $(\texttt{role}, \texttt{column},
\texttt{kind}, \texttt{score}, \texttt{reason})$. Let $L$ be the set of **valid
labels** in $\mathcal{D}$. Post-processing (identical for both backends):

$$\widehat{\mathcal{R}}=\Bigl\{\bigl(r,\ \ell,\ \mathrm{kind}(\ell),\ \operatorname{clip}(s),\ t\bigr)\ :\ (r,\ell,\cdot,s,t)\in\mathcal{R},\ r\in\text{requested},\ \ell\in L\Bigr\}.$$

That is: a returned role not in the requested set is dropped; a returned label
not in $L$ is **dropped** (hallucination guard); the score is clipped to
$[0,1]$; and `kind` is taken from $\mathcal{D}$, not from the model. Results
are tagged $\mathrm{source}=\texttt{llm}$ and sorted by score descending per
role. Any failure — missing SDK, no API key, network/API error, or unparseable
output — raises `LLMUnavailable`.

**Cell-type coarse vs. fine resolution.** The system prompt instructs the model
to distinguish the two cell-type roles by cardinality (fewer broad categories =
coarse; more fine-grained subtypes = fine), using the column's value examples
and `n_unique`.

**Numeric disambiguation.** The system prompt supplies value stats
(`v_min`, `v_max`, `v_median`, `frac_unit`, `is_integer_valued`) and explicit
guidance on look-alikes: `total_counts` is `n_counts`; `total_counts_mt` and
`total_counts_hb` are subset counts, not `n_counts`; `n_genes_by_counts` is
`n_genes`; `pct_*` and `doublet_score` are fractions in $[0,1]$, not
percentages.

**Organ vs. tissue discrimination.** The system prompt instructs the model to
distinguish the two anatomical roles: `organ` is a solid anatomical structure
(heart, liver, kidney, lung, brain) while `tissue` is the sampled biological
material or anatomical site (blood, PBMC, bone marrow, lymph node, tumor,
biopsy, CSF). A column that names a solid organ should be routed to `organ`;
a column describing the sampled compartment or preparation method should be
routed to `tissue`.

### 7.2 Stage 2 — numeric adjudication (Δ = 0.15)

After stage 1, for each numeric role the top-2 candidates are compared. If
their scores are within $\Delta=0.15$ of each other, that role is **contended**
and enters stage 2. Stage 2 issues a second, focused LLM call (system prompt:
`ADJUDICATION_SYSTEM_PROMPT`) with only the contended roles and their candidate
columns plus full value stats.

The model returns one `verdict` per contended role: `(role, column, reason)`.
The verdict's chosen column is moved to rank 1 for that role (its score is set
to the maximum of the contended candidates to preserve sort stability).

Stage 2 is **non-fatal**: `LLMUnavailable` from the adjudication call is caught
and the stage-1 ranking is used unchanged.

When stage 2 actually runs and produces verdicts, the method string gains the
suffix `" + adjudication"` (e.g. `"llm (anthropic) + adjudication"`).

### 7.3 `--hint` injection

When a non-empty `hint` string is supplied, it is prepended to the **user
prompt** in both stage-1 and stage-2 calls as an authoritative block:

```
User guidance (authoritative — follow this to locate the columns):
<hint text>

<rest of prompt>
```

`hint` is ignored (has no effect) when `use_llm=False` / `--no-llm`.

---

## 8. Orchestration and output

`rank_meta_columns` selects a ranker, then sorts and truncates per role:

$$\text{ranked}=
\begin{cases}
\mathrm{LLM}_1(\mathcal{D}) \xrightarrow{+\,\mathrm{LLM}_2\text{ if ambiguous}} \text{adjusted} & \texttt{use\_llm} \wedge \text{LLM succeeds},\\
\mathrm{heuristic}(\mathcal{D}) & \neg\,\texttt{use\_llm}\ \vee\ \texttt{LLMUnavailable},
\end{cases}$$

$$\text{result}[r]=\operatorname{sort}_{\downarrow\,\mathrm{score}}\bigl(\text{ranked}[r]\bigr)\big[:\!k\big]\quad\forall r\in\text{roles}\quad(\text{truncate only if } k>0).$$

The returned `MetaColsResult` carries:

- `.roles` — `dict[str, list[Candidate]]`, one key per requested role, each list
  sorted descending by score, truncated to `top_k`.
- `.method` — one of `"llm (<provider>)"`, `"llm (<provider>) + adjudication"`,
  `"heuristic"`, or `"heuristic (llm unavailable: …)"`.
- `.digest` — the `ObsDigest` $\mathcal{D}$.
- `.top(role)` — returns `roles[role][0]` or `None` if the list is empty.

The CLI serializes this to one JSON object on stdout (see the README's *Output*
section); exit code is $0$ if any candidate survived across all roles, $2$ if
none, $1$ on an IO error or bad `--roles` argument.
