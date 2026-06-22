# tests/test_llm.py
import pandas as pd
import pytest

from stansample.profile import profile_obs
from stansample.schema import RankedCandidates, LLMUnavailable
from stansample.llm import rank_with_llm


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
        {"column": "sample_id", "kind": "single", "score": 0.9, "reason": "looks like a sample id"},
        {"column": "made_up_column", "kind": "single", "score": 0.8, "reason": "hallucinated"},
    ])
    client = _StubClient(parsed)
    out = rank_with_llm(_digest(), client=client)
    cols = [c.column for c in out]
    assert "sample_id" in cols
    assert "made_up_column" not in cols          # filtered: not in digest
    assert all(c.source == "llm" for c in out)
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
        rank_with_llm(_digest(), client=_Boom())


def test_none_parsed_output_becomes_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        rank_with_llm(_digest(), client=_StubClient(None))
