import json
import anndata
import numpy as np
import pandas as pd

from stansample.__main__ import main


def _write_h5ad(path, obs, names):
    a = anndata.AnnData(X=np.zeros((len(obs), 2), dtype="float32"),
                        obs=obs.set_index(pd.Index(names)))
    a.write_h5ad(path)


def test_cli_emits_json_on_stdout(tmp_path, capsys):
    p = tmp_path / "x.h5ad"
    obs = pd.DataFrame({"sample_id": ["S1"] * 5 + ["S2"] * 5})
    _write_h5ad(p, obs, [f"c{i}" for i in range(10)])
    code = main([str(p), "--no-llm"])               # JSON is the only output mode
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["method"] == "heuristic"
    assert any(c["column"] == "sample_id" for c in out["candidates"])


def test_cli_exit_2_still_emits_valid_json(tmp_path, capsys):
    p = tmp_path / "y.h5ad"
    obs = pd.DataFrame({"tissue": ["lung"] * 5})         # single value -> no candidate
    _write_h5ad(p, obs, ["aa", "bb", "cc", "dd", "ee"])  # no barcode delimiter
    code = main([str(p), "--no-llm"])
    assert code == 2
    out = json.loads(capsys.readouterr().out)            # stdout parseable even when empty
    assert out["candidates"] == []


def test_cli_bad_path_exit_1(capsys):
    code = main(["/no/such/file.h5ad", "--no-llm"])
    assert code == 1
