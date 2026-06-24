"""LLM ranking over the digest. Provider plumbing lives in the shared
llm_client; this module is the stanmetacols-specific glue.

* "anthropic" (default): native client.messages.parse with a Pydantic
  output_format (strongest structured-output guarantee) — local _anthropic_parse.
* "openai": any OpenAI-compatible endpoint via the shared OpenAICompatClient +
  call_structured.
"""

from __future__ import annotations

from .schema import ObsDigest, Candidate, RankedCandidates, Adjudications
from .llm_client import LLMUnavailable, OpenAICompatClient, call_structured
from .prompts import (SYSTEM_PROMPT, build_user_prompt,
                      ADJUDICATION_SYSTEM_PROMPT, build_adjudication_prompt)


def _valid_labels(digest: ObsDigest) -> dict:
    """Map every real candidate label -> its kind."""
    labels = {c.name: "single" for c in digest.columns}
    for comp in digest.composite_candidates:
        labels[comp.label] = "composite"
    if digest.barcode is not None:
        labels[digest.barcode.label] = "barcode"
    return labels


def _anthropic_parse(system, user, schema, *, model, client, max_tokens):
    """Native anthropic structured output via messages.parse. Kept local so the
    shared llm_client stays SDK-free. Errors -> LLMUnavailable."""
    if client is None:
        try:
            import anthropic
        except Exception as exc:                # not installed
            raise LLMUnavailable(f"anthropic not installed: {exc}") from exc
        try:
            client = anthropic.Anthropic()
        except Exception as exc:                # no key, bad config
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc
    try:
        resp = client.messages.parse(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema)
    except Exception as exc:                     # any API/connection/parse error
        raise LLMUnavailable(str(exc)) from exc
    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("model returned no parseable structured output")
    return parsed


def _openai_client(model, client, base_url, api_key, max_tokens):
    """Use an injected .complete client, else build an OpenAICompatClient."""
    if client is not None:
        return client
    return OpenAICompatClient(base_url, api_key, model, max_tokens=max_tokens)


def rank_with_llm(digest: ObsDigest, roles, *, hint: str = "",
                  provider: str = "anthropic",
                  model: str = "claude-opus-4-8", client=None,
                  base_url: str | None = None, api_key: str | None = None,
                  max_tokens: int = 2048) -> dict:
    system = SYSTEM_PROMPT
    user = build_user_prompt(digest, roles, hint)
    if provider == "anthropic":
        parsed = _anthropic_parse(system, user, RankedCandidates,
                                  model=model, client=client, max_tokens=max_tokens)
    elif provider == "openai":
        parsed = call_structured(
            _openai_client(model, client, base_url, api_key, max_tokens),
            system, user, RankedCandidates.model_validate, list_key="candidates")
    else:
        raise LLMUnavailable(f"unknown provider: {provider!r}")

    labels = _valid_labels(digest)
    requested = set(roles)
    out = {k: [] for k in roles}
    for rc in parsed.candidates:
        if rc.role not in requested:
            continue
        kind = labels.get(rc.column)
        if kind is None:                          # hallucinated column -> drop
            continue
        score = max(0.0, min(1.0, float(rc.score)))
        out[rc.role].append(Candidate(role=rc.role, column=rc.column, kind=kind,
                                      score=score, reason=rc.reason, source="llm"))
    for k in out:
        out[k].sort(key=lambda c: c.score, reverse=True)
    return out


def adjudicate_numeric(digest, contention, *, hint: str = "",
                       provider: str = "anthropic",
                       model: str = "claude-opus-4-8", client=None,
                       base_url: str | None = None, api_key: str | None = None,
                       max_tokens: int = 1024) -> dict:
    system = ADJUDICATION_SYSTEM_PROMPT
    user = build_adjudication_prompt(digest, contention, hint)
    if provider == "anthropic":
        parsed = _anthropic_parse(system, user, Adjudications,
                                  model=model, client=client, max_tokens=max_tokens)
    elif provider == "openai":
        parsed = call_structured(
            _openai_client(model, client, base_url, api_key, max_tokens),
            system, user, Adjudications.model_validate, list_key="verdicts")
    else:
        raise LLMUnavailable(f"unknown provider: {provider!r}")
    offered = {role: {c.column for c in cands} for role, cands in contention.items()}
    verdicts = {}
    for v in parsed.verdicts:
        if v.role in offered and v.column in offered[v.role]:
            verdicts[v.role] = (v.column, v.reason)
    return verdicts
