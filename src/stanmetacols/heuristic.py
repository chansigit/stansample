"""Deterministic per-role scorer over an ObsDigest. No network, no API key."""

from __future__ import annotations

from .schema import ObsDigest, Candidate
from .roles import (ROLES, name_signal, value_check, celltype_value_frac,
                    celltype_name_base, vocab_value_frac, vocab_name_base)
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


_COARSE_CARD = (2, 25)
_FINE_CARD = (5, 200)


def _card_fit(n_unique: int, role_key: str) -> float:
    lo, hi = _COARSE_CARD if role_key == "cell_type_coarse" else _FINE_CARD
    return 1.0 if lo <= n_unique <= hi else 0.3


def _rank_celltype(digest: ObsDigest, role) -> list:
    out = []
    for c in digest.columns:
        if c.single_value or c.unique_per_cell:
            continue
        if c.dtype not in ("categorical", "string"):
            continue
        ns = name_signal(c.name, role)          # role-specific (coarse/fine) aliases
        base = celltype_name_base(c.name)       # generic "celltype"/"annotation" name
        vocab = celltype_value_frac(c)
        name_score = max(ns, 0.6 * base)
        if name_score <= 0 and vocab < 0.5:     # must look like a cell-type column
            continue
        card = _card_fit(c.n_unique, role.key)
        score = max(0.0, min(1.0, 0.4 * name_score + 0.4 * vocab + 0.2 * card))
        if score <= 0:
            continue
        out.append(Candidate(
            role=role.key, column=c.name, kind="single", score=score,
            source="heuristic",
            reason=(f"name={name_score:.1f}, celltype_vocab={vocab:.2f}, "
                    f"n_unique={c.n_unique}")))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


_VOCAB_CARD = (2, 50)


def _vocab_card_fit(n_unique: int) -> float:
    lo, hi = _VOCAB_CARD
    return 1.0 if lo <= n_unique <= hi else 0.3


def _rank_vocab(digest: ObsDigest, role) -> list:
    """Score categorical label columns (organ, tissue) by name + value vocabulary."""
    out = []
    for c in digest.columns:
        if c.single_value or c.unique_per_cell:
            continue
        if c.dtype not in ("categorical", "string"):
            continue
        ns = name_signal(c.name, role)
        base = vocab_name_base(c.name, role)
        vocab = vocab_value_frac(c, role)
        name_score = max(ns, 0.6 * base)
        if name_score <= 0 and vocab < 0.5:      # must look like the role
            continue
        card = _vocab_card_fit(c.n_unique)
        score = max(0.0, min(1.0, 0.4 * name_score + 0.4 * vocab + 0.2 * card))
        if score <= 0:
            continue
        out.append(Candidate(
            role=role.key, column=c.name, kind="single", score=score,
            source="heuristic",
            reason=(f"name={name_score:.1f}, {role.key}_vocab={vocab:.2f}, "
                    f"n_unique={c.n_unique}")))
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def rank_heuristic(digest: ObsDigest, roles) -> dict:
    out = {}
    for key in roles:
        role = ROLES[key]
        if role.type == "grouping":
            out[key] = _rank_sample(digest)
        elif role.type == "numeric":
            out[key] = _rank_numeric(digest, role)
        elif role.type in ("organ", "tissue"):
            out[key] = _rank_vocab(digest, role)
        else:                                    # "celltype"
            out[key] = _rank_celltype(digest, role)
    return out
