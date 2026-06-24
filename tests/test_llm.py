# tests/test_llm.py
import json

import pandas as pd
import pytest

from stanmetacols.profile import profile_obs
from stanmetacols.schema import RankedCandidates, LLMUnavailable
from stanmetacols.llm import rank_with_llm


def _digest():
    return profile_obs(pd.DataFrame(
        {"sample_id": ["S1"] * 5 + ["S2"] * 5, "tissue": ["lung"] * 10},
        index=[f"c{i}" for i in range(10)]))


class _StubMessages:
    def __init__(self, parsed):
        self._parsed = parsed
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs

        class _Resp:
            parsed_output = self._parsed
        return _Resp()


class _StubClient:
    def __init__(self, parsed):
        self.messages = _StubMessages(parsed)


def test_maps_and_filters_hallucinations():
    parsed = RankedCandidates(candidates=[
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.9, "reason": "looks like a sample id"},
        {"role": "sample", "column": "made_up_column", "kind": "single", "score": 0.8, "reason": "hallucinated"},
    ])
    client = _StubClient(parsed)
    out = rank_with_llm(_digest(), ["sample"], client=client)
    cols = [c.column for c in out["sample"]]
    assert "sample_id" in cols
    assert "made_up_column" not in cols          # filtered: not in digest
    assert all(c.source == "llm" for c in out["sample"])
    # prompt carried the digest
    assert "sample_id" in client.messages.kwargs["messages"][0]["content"]
    assert client.messages.kwargs["model"] == "claude-opus-4-8"


def test_api_error_becomes_llm_unavailable():
    class _Boom:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("network down")
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], client=_Boom())


def test_none_parsed_output_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], client=_StubClient(None))


# --- OpenAI-compatible backend (OpenAI, Volcengine ARK, DeepSeek, vLLM, …) ---

class _StubChatClient:
    """Minimal .complete(system, user) -> str client for the openai path."""
    def __init__(self, content=None, raise_exc=None):
        self._content = content
        self._raise = raise_exc
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self._raise is not None:
            raise self._raise
        return self._content


def test_openai_parses_json_and_filters_hallucinations():
    content = json.dumps({"candidates": [
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.9, "reason": "ok"},
        {"role": "sample", "column": "made_up", "kind": "single", "score": 0.8, "reason": "halluc"},
    ]})
    client = _StubChatClient(content)
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=client)
    assert [c.column for c in out["sample"]] == ["sample_id"]
    assert all(c.source == "llm" for c in out["sample"])
    system, user = client.calls[0]
    assert "sample_id" in user                # digest carried in the user prompt


def test_openai_strips_markdown_code_fences():
    body = json.dumps({"candidates": [
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.7, "reason": "x"}]})
    client = _StubChatClient("```json\n" + body + "\n```")
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=client)
    assert [c.column for c in out["sample"]] == ["sample_id"]


def test_openai_accepts_bare_json_array():
    content = json.dumps([
        {"role": "sample", "column": "sample_id", "kind": "single", "score": 0.6, "reason": "x"}])
    out = rank_with_llm(_digest(), ["sample"], provider="openai", client=_StubChatClient(content))
    assert [c.column for c in out["sample"]] == ["sample_id"]


def test_openai_non_json_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="openai",
                      client=_StubChatClient("I cannot help with that."))


def test_openai_api_error_becomes_llm_unavailable():
    client = _StubChatClient(raise_exc=RuntimeError("network down"))
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="openai", client=client)


def test_unknown_provider_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), ["sample"], provider="cohere")


def test_hint_reaches_user_prompt():
    parsed = RankedCandidates(candidates=[])
    client = _StubClient(parsed)
    rank_with_llm(_digest(), ["sample"], hint="HINTTOKEN", client=client)
    content = client.messages.kwargs["messages"][0]["content"]
    assert "HINTTOKEN" in content


def test_hint_reaches_openai_user_prompt():
    client = _StubChatClient(json.dumps({"candidates": []}))
    rank_with_llm(_digest(), ["sample"], hint="OAHINT", provider="openai", client=client)
    system, user = client.calls[0]
    assert "OAHINT" in user
