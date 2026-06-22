"""Deterministic feature extraction from a pandas .obs into an ObsDigest.

No LLM, no network, no mutation of the input.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import ObsDigest, ColumnProfile

_BARCODE_RE = re.compile(r"^[ACGTN]{8,}(-\d+)?$", re.IGNORECASE)


def _classify_dtype(s: pd.Series) -> str:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "string"


def _group_stats(counts: np.ndarray):
    if counts.size == 0:
        return {"min": 0, "max": 0, "median": 0.0}, 0.0
    mn, mx = int(counts.min()), int(counts.max())
    med = float(np.median(counts))
    balance = (mn / mx) if mx > 0 else 0.0
    return {"min": mn, "max": mx, "median": med}, balance


def _profile_column(name: str, s: pd.Series, n_obs: int, max_example_values: int) -> ColumnProfile:
    n_missing = int(s.isna().sum())
    vc = s.value_counts(dropna=True)
    n_unique = int(vc.size)
    cells_per_group, balance = _group_stats(vc.to_numpy())
    example_values = sorted(str(v) for v in vc.index)[:max_example_values]
    sample = [str(v) for v in vc.index[:1000]]
    frac_bc = (sum(1 for v in sample if _BARCODE_RE.match(v)) / len(sample)) if sample else 0.0
    return ColumnProfile(
        name=name,
        dtype=_classify_dtype(s),
        n_unique=n_unique,
        n_missing=n_missing,
        example_values=example_values,
        cells_per_group=cells_per_group,
        balance=balance,
        unique_per_cell=(n_obs > 0 and n_unique == n_obs),
        single_value=(n_unique <= 1),
        looks_like_barcode=(frac_bc > 0.5),
    )


def profile_obs(obs, obs_names=None, *, max_example_values: int = 8,
                max_composite_pairs: int = 8) -> ObsDigest:
    if obs_names is None:
        obs_names = list(obs.index)
    n_obs = len(obs)
    columns = [
        _profile_column(str(c), obs[c], n_obs, max_example_values)
        for c in obs.columns
    ]
    return ObsDigest(n_obs=n_obs, columns=columns, composite_candidates=[], barcode=None)
