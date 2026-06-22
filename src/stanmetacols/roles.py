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
