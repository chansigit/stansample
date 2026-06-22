"""Single structured LLM call (claude-opus-4-8) over the digest.

anthropic is imported lazily so the package installs and the heuristic path
runs without it.
"""

from .schema import ObsDigest, Candidate, RankedCandidates, LLMUnavailable
from .prompts import SYSTEM_PROMPT, build_user_prompt


def _valid_labels(digest: ObsDigest) -> dict:
    """Map every real candidate label -> its kind."""
    labels = {c.name: "single" for c in digest.columns}
    for comp in digest.composite_candidates:
        labels[comp.label] = "composite"
    if digest.barcode is not None:
        labels[digest.barcode.label] = "barcode"
    return labels


def rank_with_llm(digest: ObsDigest, *, model: str = "claude-opus-4-8",
                  client=None, max_tokens: int = 2048) -> list:
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
            messages=[{"role": "user", "content": build_user_prompt(digest)}],
            output_format=RankedCandidates,
        )
    except LLMUnavailable:
        raise
    except Exception as exc:  # any API/connection/parse error -> fallback
        raise LLMUnavailable(str(exc)) from exc

    parsed = getattr(resp, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("model returned no parseable structured output")

    labels = _valid_labels(digest)
    out = []
    for rc in parsed.candidates:
        kind = labels.get(rc.column)
        if kind is None:           # hallucinated / unknown column -> drop
            continue
        score = max(0.0, min(1.0, float(rc.score)))
        out.append(Candidate(column=rc.column, kind=kind, score=score,
                             reason=rc.reason, source="llm"))
    out.sort(key=lambda c: c.score, reverse=True)
    return out
