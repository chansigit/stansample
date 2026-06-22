import numpy as np
import pandas as pd

from stanmetacols.profile import profile_obs


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


def test_barcode_detected_prefix_on_obs_fixture():
    d = profile_obs(_obs())
    assert d.barcode is not None
    assert d.barcode.position == "prefix"
    assert d.barcode.delimiter == "_"
    assert d.barcode.n_groups == 3


def test_empty_obs():
    d = profile_obs(pd.DataFrame())
    assert d.n_obs == 0
    assert d.columns == []


def test_numeric_stats_unit_float():
    obs = pd.DataFrame({"pct": [0.0, 0.1, 0.2, 0.9, 1.0]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is True
    assert col.is_integer_valued is False
    assert col.frac_unit == 1.0
    assert col.frac_nonneg == 1.0
    assert col.v_min == 0.0 and col.v_max == 1.0


def test_numeric_stats_integer_counts():
    obs = pd.DataFrame({"total_counts": [1000, 2000, 3000, 50000]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is True
    assert col.is_integer_valued is True
    assert col.frac_unit == 0.0           # none in [0,1]
    assert col.v_median == 2500.0


def test_numeric_stats_absent_for_categorical():
    obs = pd.DataFrame({"sample": ["A", "B", "A", "B"]})
    col = profile_obs(obs).columns[0]
    assert col.is_numeric is False
    assert col.frac_unit == 0.0 and col.is_integer_valued is False
