"""Single structured LLM call over the digest, with a pluggable provider.

Two backends, selected by ``provider``:

* ``"anthropic"`` (default) — native ``client.messages.parse`` with a Pydantic
  output schema (strongest structured-output guarantee). Used for Claude.
* ``"openai"`` — any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
  Volcengine ARK, DeepSeek, vLLM, Ollama, …). The reply text is parsed as JSON
  and validated against the same Pydantic schema.

Both SDKs are imported lazily, so the package installs and the heuristic path
runs without either one.
"""

from __future__ import annotations

import json

from .schema import ObsDigest, Candidate, RankedCandidates, Adjudications, LLMUnavailable
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


def _extract_json(text: str | None) -> str:
    """Return the outermost JSON object/array in ``text``.

    Tolerates Markdown code fences and prose around the payload by slicing from
    the first opening bracket to the matching last closing bracket of the same
    kind. Robust enough for a model instructed to "return JSON only".
    """
    if not text or not text.strip():
        raise LLMUnavailable("empty model response")
    t = text.strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise LLMUnavailable("no JSON found in model response")
    start = min(starts)
    close = "}" if t[start] == "{" else "]"
    end = t.rfind(close)
    if end < start:
        raise LLMUnavailable("no JSON found in model response")
    return t[start:end + 1]


def _parse_ranked(text: str | None) -> RankedCandidates:
    """Parse a model's text reply into RankedCandidates (object or bare array)."""
    blob = _extract_json(text)
    try:
        data = json.loads(blob)
    except Exception as exc:
        raise LLMUnavailable(f"response is not valid JSON: {exc}") from exc
    if isinstance(data, list):           # a bare list of candidates
        data = {"candidates": data}
    try:
        return RankedCandidates.model_validate(data)
    except Exception as exc:
        raise LLMUnavailable(f"response does not match schema: {exc}") from exc


def _call_anthropic(digest: ObsDigest, roles, model: str, client, max_tokens: int) -> RankedCandidates:
    if client is None:
        try:
            import anthropic
        except Exception as exc:  # not installed
            raise LLMUnavailable(f"anthropic not installed: {exc}") from exc
        try:
            client = anthropic.Anthropic()
        except Exception as exc:  # no key, bad config
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc

    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(digest, roles)}],
            output_format=RankedCandidates,
        )
    except Exception as exc:  # any API/connection/parse error -> fallback
        raise LLMUnavailable(str(exc)) from exc

    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("model returned no parseable structured output")
    return parsed


def _call_openai(digest: ObsDigest, roles, model: str, client, base_url, api_key,
                 max_tokens: int) -> RankedCandidates:
    if client is None:
        try:
            from openai import OpenAI
        except Exception as exc:  # not installed
            raise LLMUnavailable(f"openai not installed: {exc}") from exc
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        try:
            # OpenAI() also reads OPENAI_API_KEY / OPENAI_BASE_URL from the env.
            client = OpenAI(**kwargs)
        except Exception as exc:  # no key, bad config
            raise LLMUnavailable(f"cannot construct client: {exc}") from exc

    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(digest, roles)},
            ],
        )
    except Exception as exc:  # any API/connection error -> fallback
        raise LLMUnavailable(str(exc)) from exc

    try:
        text = resp.choices[0].message.content
    except Exception as exc:
        raise LLMUnavailable(f"unexpected response shape: {exc}") from exc
    return _parse_ranked(text)


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
