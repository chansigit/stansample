import numpy as np
import pandas as pd

from stansample.profile import profile_obs


def _obs():
    n = 30
    return pd.DataFrame(
        {
            "sample_id": ["S1"] * 10 + ["S2"] * 10 + ["S3"] * 10,
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,
            "pct_mito": np.linspace(0.0, 10.0, n),
            "cell_id": [f"cell{i}" for i in range(n)],
        },
        index=[f"S{(i // 10) + 1}_AAAC{i:04d}-1" for i in range(n)],
    )


def _col(digest, name):
    return next(c for c in digest.columns if c.name == name)


def test_column_basic_features():
    d = profile_obs(_obs())
    assert d.n_obs == 30
    assert {c.name for c in d.columns} == {
        "sample_id", "donor", "timepoint", "tissue", "pct_mito", "cell_id"}
    assert _col(d, "sample_id").n_unique == 3
    assert _col(d, "sample_id").cells_per_group == {"min": 10, "max": 10, "median": 10.0}
    assert _col(d, "sample_id").balance == 1.0
    assert _col(d, "donor").balance == 1.0


def test_flags_and_dtypes():
    d = profile_obs(_obs())
    assert _col(d, "tissue").single_value is True
    assert _col(d, "cell_id").unique_per_cell is True
    assert _col(d, "pct_mito").dtype == "float"
    assert _col(d, "timepoint").dtype in ("string", "categorical")
    assert _col(d, "sample_id").example_values == sorted(["S1", "S2", "S3"])


def test_barcode_empty_at_this_stage():
    d = profile_obs(_obs())
    assert d.barcode is None


def test_empty_obs():
    d = profile_obs(pd.DataFrame())
    assert d.n_obs == 0
    assert d.columns == []
