# stansample — algorithm formulation

A precise statement of what the code computes. Every symbol below maps to a
named quantity in `profile.py`, `heuristic.py`, `llm.py`, or `rank.py`.

GitHub renders the math inline. If you read this in a plain editor, the `$…$`
spans are LaTeX.

---

## 1. Problem

The input is an `.obs` table: $n$ cells (rows), a set of columns $\mathcal{C}$,
and a list of cell names (`obs_names`) $b_1,\dots,b_n$. We want to **rank
candidate partitions** of the $n$ cells, where each candidate is a plausible way
to assign every cell to the *sample* it came from — the grouping unit used for
per-sample QC, batch grouping, or pseudobulk.

A candidate induces a partition

$$P=\{G_1,\dots,G_k\},\qquad \textstyle\bigsqcup_i G_i \subseteq \{1,\dots,n\},\quad G_i\neq\varnothing,$$

with group sizes $g_i=|G_i|$. The tool **scores and ranks** candidates; it never
commits to one.

There are three candidate families:

| Family | Partition rule |
|---|---|
| **single** column $c$ | cells grouped by the distinct values of $c$ |
| **composite** key $\{a,b\}$ | cells grouped by distinct value-*pairs* of $(a,b)$ |
| **barcode** | groups extracted from `obs_names` by a delimiter + position rule |

---

## 2. Partition statistics

For any partition $P$ with group sizes $g_1,\dots,g_k$ ($k=|P|$ nonempty
groups), define the **cardinality** $k$ and the **balance**

$$\beta(P)=\frac{\min_i g_i}{\max_i g_i}\in(0,1],\qquad \beta:=0 \text{ if } \max_i g_i=0 .$$

$\beta=1$ is a perfectly even split; $\beta\to 0$ is one dominant group plus
slivers. (`_group_stats` also returns $\min$, $\max$, and the median group size,
carried in the digest for the prompt but not used by the scorer.)

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

---

## 4. Candidate families

### 4.1 Single columns

Every column is profiled. A column is a scorable single candidate iff
$\neg\,\mathrm{single}_c \wedge \neg\,\mathrm{unique}_c$ (Section 5 then scores it
and may still drop it).

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

$$2\le u_{ab}<n .$$

Rank kept pairs by $\beta_{ab}$ descending; keep the top $8$. The label of a kept
pair is the string `"a + b"`.

### 4.3 Barcode grouping

From the names $b_1,\dots,b_n$ (as strings), consider two rules; let
$\mathrm{frac}(\cdot)$ denote the fraction of names satisfying a predicate.

$$\textbf{prefix on }\verb|_| :\quad
\text{applicable if } \mathrm{frac}(\verb|_|\in b_i)>0.9,\quad
\text{key}(b_i)=b_i\ \text{before its last }\verb|_| .$$

$$\textbf{suffix on }\verb|-| :\quad
t_i=b_i\ \text{after its last }\verb|-|,\quad
\text{applicable if } \mathrm{frac}(t_i\in\verb|\d+|)>0.9,\quad
\text{key}(b_i)=t_i .$$

For each applicable rule build the partition with $k$ groups and **keep** it iff
$2\le k<n$. Among the kept rules choose the one of **maximum balance** $\beta$.
Its label is `"<barcode:POSITION:DELIM>"`, e.g. `<barcode:prefix:_>`.

### 4.4 The digest

The deterministic output of `profile_obs` is

$$\mathcal{D}=\bigl(n,\ \{\text{column profiles}\},\ \{\text{composite profiles}\},\ \text{barcode profile or }\varnothing\bigr).$$

$\mathcal{D}$ is computed without any network or LLM and without mutating the
input. Both rankers below consume the *same* $\mathcal{D}$.

---

## 5. Heuristic scorer

Let $\operatorname{clip}(x)=\max\!\bigl(0,\min(1,x)\bigr)$.

**Name signal.** Let $A$ be the alias list — `sample, sample_id, donor,
donor_id, patient, subject, individual, specimen, orig.ident, library,
library_id, gsm, geo_accession, srr, batch, channel, well, lane, replicate, …` —
and let $\nu(s)$ lowercase $s$ and delete `_`, `.`, and spaces. Then

$$\mathrm{name}(c)=
\begin{cases}
1.0 & \nu(c)\in\nu(A)\quad(\text{exact alias})\\[2pt]
0.6 & \exists\,a\in\nu(A):\ a\subseteq\nu(c)\quad(\text{alias substring})\\[2pt]
0.0 & \text{otherwise.}
\end{cases}$$

**Cardinality signal.** With the soft ceiling $\theta(n)=\max(50,\,0.2\,n)$,

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

$$\boxed{\,s_c=\operatorname{clip}\bigl(0.5\,\mathrm{name}(c)+0.25\,\mathrm{card}(u_c)+0.25\,\beta_c-\rho_c\bigr)\,}$$

For a composite key $\{a,b\}$, with member-averaged name signal
$\overline{\mathrm{name}}=\tfrac12\bigl(\mathrm{name}(a)+\mathrm{name}(b)\bigr)$ and a
$0.85$ discount, **no** penalty term:

$$\boxed{\,s_{ab}=\operatorname{clip}\Bigl(0.85\cdot\bigl(0.5\,\overline{\mathrm{name}}+0.25\,\mathrm{card}(u_{ab})+0.25\,\beta_{ab}\bigr)\Bigr)\,}$$

For the barcode grouping with balance $\beta_{\mathrm{bc}}$:

$$\boxed{\,s_{\mathrm{bc}}=\operatorname{clip}\bigl(0.45\,\beta_{\mathrm{bc}}+0.1\bigr)\,}$$

Any candidate with score $\le 0$ is dropped. The kept candidates, tagged
$\mathrm{source}=\texttt{heuristic}$, are returned sorted by score descending.

> **Reading the weights.** A single column's score is a convex blend
> $0.5\,\text{name}+0.25\,\text{card}+0.25\,\text{balance}$ minus disqualifiers:
> the *name* carries half the weight (a column literally called `sample` wins
> even when imbalanced), while *cardinality* and *balance* together encode "many
> groups, far fewer than one per cell, reasonably even." Composites inherit the
> same blend at a $15\%$ discount; the barcode path is balance-driven with a
> small floor so a usable grouping never scores exactly zero.

---

## 6. LLM scorer

The primary path issues **one** structured call

$$\mathcal{R}=\mathrm{LLM}\bigl(\text{system prompt},\ \mathrm{JSON}(\mathcal{D})\bigr),$$

where $\mathcal{R}$ is forced to the schema *list of* $(\texttt{column},
\texttt{kind}, \texttt{score}, \texttt{reason})$. Let $L$ be the set of **valid
labels** in $\mathcal{D}$ — every column name, every composite label `"a + b"`,
and the barcode label — together with the map $\ell\mapsto\mathrm{kind}(\ell)$.
Post-processing is a guard:

$$\widehat{\mathcal{R}}=\Bigl\{\bigl(\ell,\ \mathrm{kind}(\ell),\ \operatorname{clip}(s),\ r\bigr)\ :\ (\ell,\cdot,s,r)\in\mathcal{R},\ \ell\in L\Bigr\}.$$

That is: a returned label not in $L$ is **dropped** (hallucination guard); the
score is clipped to $[0,1]$; and `kind` is taken from $\mathcal{D}$, not from the
model. Results are tagged $\mathrm{source}=\texttt{llm}$ and sorted by score
descending. Any failure — `anthropic` missing, no API key, network/API error, or
unparseable output — raises `LLMUnavailable`.

---

## 7. Orchestration and output

`rank_sample_columns` selects a ranker, then sorts and truncates:

$$\text{candidates}=
\begin{cases}
\mathrm{LLM}(\mathcal{D}) & \texttt{use\_llm} \ \wedge\ \text{LLM succeeds},\\
\mathrm{heuristic}(\mathcal{D}) & \neg\,\texttt{use\_llm}\ \vee\ \texttt{LLMUnavailable},
\end{cases}$$

$$\text{result}=\operatorname{sort}_{\downarrow\,\mathrm{score}}(\text{candidates})\big[:\!k\big]\quad(\text{truncate only if } k>0).$$

The returned `RankResult` carries the candidate list, the `method` string
(`"llm"`, `"heuristic"`, or `"heuristic (llm unavailable: …)"`), and the digest
$\mathcal{D}$. The CLI serializes this to one JSON object on stdout
(see the README's *Output* section); exit code is $0$ if any candidate survived,
$2$ if none, $1$ on an IO error.
