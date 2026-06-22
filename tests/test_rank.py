import pandas as pd
from stanmetacols.rank import rank_meta_columns
from stanmetacols.schema import RankedCandidates, Adjudications


def _obs():
    n = 40
    return pd.DataFrame({
        "sample": ["S1"] * 20 + ["S2"] * 20,
        "pct_counts_mt": [i / 100 for i in range(n)],
        "total_counts": [1000 + 5 * i for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])


class _StubClient:
    def __init__(self, parsed):
        class _M:
            def parse(_s, **kw):
                class _R: parsed_output = parsed
                return _R()
        self.messages = _M()


class _Boom:
    class messages:
        @staticmethod
        def parse(**kw):
            raise RuntimeError("no network")


def test_no_llm_heuristic_groups_by_role():
    res = rank_meta_columns(_obs(), use_llm=False)
    assert res.method == "heuristic"
    assert res.top("sample").column == "sample"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("n_counts").column == "total_counts"


def test_roles_subset():
    res = rank_meta_columns(_obs(), use_llm=False, roles=["pct_mt"])
    assert set(res.roles) == {"pct_mt"}


def test_llm_path_with_mock_client():
    parsed = RankedCandidates(candidates=[
        {"role": "pct_mt", "column": "pct_counts_mt", "kind": "single",
         "score": 0.95, "reason": "ok"}])
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False,
                            client=_StubClient(parsed))
    assert res.method == "llm (anthropic)"
    assert res.top("pct_mt").column == "pct_counts_mt"
    assert res.top("pct_mt").source == "llm"


def test_llm_failure_falls_back():
    res = rank_meta_columns(_obs(), use_llm=True, adjudicate=False, client=_Boom())
    assert res.method.startswith("heuristic (llm unavailable")
    assert res.top("sample").column == "sample"


def test_top_k_truncation_per_role():
    res = rank_meta_columns(_obs(), use_llm=False, top_k=1)
    assert all(len(v) <= 1 for v in res.roles.values())


def test_input_not_mutated():
    obs = _obs(); before = obs.copy()
    rank_meta_columns(obs, use_llm=False)
    pd.testing.assert_frame_equal(obs, before)


class _TwoStageClient:
    """messages.parse returns the stage-1 ranking first, the adjudication second."""
    def __init__(self, stage1, stage2):
        self._responses = [stage1, stage2]
        class _M:
            def parse(_s, **kw):
                payload = self._responses.pop(0)
                class _R: parsed_output = payload
                return _R()
        self.messages = _M()


def _ambiguous_obs():
    n = 40
    return pd.DataFrame({
        "total_counts": [1000 + 5 * i for i in range(n)],
        "total_counts_mt": [10 + i for i in range(n)],
    }, index=[f"c{i}" for i in range(n)])


def test_adjudication_reorders_numeric_role():
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.80, "reason": "looks like counts"},
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.78, "reason": "also counts-like"}])
    stage2 = Adjudications(verdicts=[
        {"role": "n_counts", "column": "total_counts",
         "reason": "per-cell total, not the mt subset"}])
    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_TwoStageClient(stage1, stage2))
    assert res.method == "llm (anthropic) + adjudication"
    assert res.top("n_counts").column == "total_counts"
    assert "subset" in res.top("n_counts").reason


def test_no_adjudication_when_clear_winner():
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.95, "reason": "clear"},
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.40, "reason": "weak"}])
    # second response would raise if called
    class _OneCall:
        def __init__(self, payload):
            self._p = [payload]
            class _M:
                def parse(_s, **kw):
                    class _R: parsed_output = self._p.pop(0)
                    return _R()
            self.messages = _M()
    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_OneCall(stage1))
    assert res.method == "llm (anthropic)"     # no " + adjudication"
    assert res.top("n_counts").column == "total_counts"


def test_adjudication_failure_keeps_stage1():
    stage1 = RankedCandidates(candidates=[
        {"role": "n_counts", "column": "total_counts", "kind": "single",
         "score": 0.80, "reason": "a"},
        {"role": "n_counts", "column": "total_counts_mt", "kind": "single",
         "score": 0.78, "reason": "b"}])

    class _Stage2Boom:
        def __init__(self, s1):
            self._first = [s1]
            outer = self
            class _M:
                def parse(_s, **kw):
                    if outer._first:
                        payload = outer._first.pop(0)
                        class _R: parsed_output = payload
                        return _R()
                    raise RuntimeError("adjudication network error")
            self.messages = _M()

    res = rank_meta_columns(_ambiguous_obs(), roles=["n_counts"], use_llm=True,
                            client=_Stage2Boom(stage1))
    assert res.method == "llm (anthropic)"     # stage-1 kept, non-fatal
    assert res.top("n_counts").column == "total_counts"   # unchanged top
