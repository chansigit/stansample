"""Deterministic feature extraction from a pandas .obs into an ObsDigest.

No LLM, no network, no mutation of the input.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import ObsDigest, ColumnProfile, CompositeProfile, BarcodeProfile

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


def _composite_candidates(obs, profiles, n_obs, max_pairs):
    if max_pairs <= 0 or n_obs == 0:
        return []
    eligible = [
        p for p in profiles
        if not p.unique_per_cell and not p.single_value
        and p.dtype in ("categorical", "string", "integer", "bool")
        and 2 <= p.n_unique <= n_obs * 0.5
        and not p.looks_like_barcode
    ]
    eligible.sort(key=lambda p: p.balance, reverse=True)
    eligible = eligible[:12]  # bound the O(k^2) groupby work
    pairs = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i].name, eligible[j].name
            sizes = obs.groupby([a, b], observed=True).size()
            sizes = sizes[sizes > 0]
            n_unique = int(sizes.size)
            if not (2 <= n_unique < n_obs):
                continue
            cells_per_group, balance = _group_stats(sizes.to_numpy())
            pairs.append(CompositeProfile([a, b], n_unique, cells_per_group, balance))
    pairs.sort(key=lambda c: c.balance, reverse=True)
    return pairs[:max_pairs]


def _barcode_profile(obs_names, n_obs, max_example_groups=8):
    if n_obs == 0:
        return None
    s = pd.Series([str(x) for x in obs_names])
    options = []
    if s.str.contains("_", regex=False).mean() > 0.9:
        options.append(("_", "prefix", s.str.rsplit("_", n=1).str[0]))
    tail = s.str.rsplit("-", n=1).str[-1]
    if tail.str.fullmatch(r"\d+").mean() > 0.9:
        options.append(("-", "suffix", tail))
    best = None
    for delimiter, position, grp in options:
        vc = grp.value_counts()
        n_groups = int(vc.size)
        if not (2 <= n_groups < n_obs):
            continue
        cells_per_group, balance = _group_stats(vc.to_numpy())
        prof = BarcodeProfile(
            delimiter=delimiter, position=position, n_groups=n_groups,
            cells_per_group=cells_per_group, balance=balance,
            example_groups=sorted(str(v) for v in vc.index)[:max_example_groups],
        )
        if best is None or prof.balance > best.balance:
            best = prof
    return best


def profile_obs(obs, obs_names=None, *, max_example_values: int = 8,
                max_composite_pairs: int = 8) -> ObsDigest:
    if obs_names is None:
        obs_names = list(obs.index)
    n_obs = len(obs)
    columns = [
        _profile_column(str(c), obs[c], n_obs, max_example_values)
        for c in obs.columns
    ]
    return ObsDigest(
        n_obs=n_obs,
        columns=columns,
        composite_candidates=_composite_candidates(obs, columns, n_obs, max_composite_pairs),
        barcode=_barcode_profile(obs_names, n_obs),
    )
