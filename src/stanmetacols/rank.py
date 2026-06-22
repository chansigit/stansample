"""Public orchestrator: build digest, rank per role (LLM stage 1 + heuristic
fallback). Numeric adjudication (stage 2) is wired in Task 6."""

from __future__ import annotations

from .schema import MetaColsResult, LLMUnavailable
from .profile import profile_obs
from .roles import ROLE_KEYS, NUMERIC_ROLE_KEYS
from .llm import rank_with_llm, adjudicate_numeric
from .heuristic import rank_heuristic

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
