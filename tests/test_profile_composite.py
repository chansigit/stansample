import pandas as pd

from stansample.profile import profile_obs


def _obs():
    return pd.DataFrame(
        {
            "donor": ["D1"] * 15 + ["D2"] * 15,
            "timepoint": ["t0", "t1", "t2"] * 10,
            "tissue": ["lung"] * 30,            # single value, ineligible
            "cell_id": [f"cell{i}" for i in range(30)],  # unique-per-cell, ineligible
        },
        index=[f"c{i}" for i in range(30)],
    )


def test_composite_pair_generated():
    d = profile_obs(_obs())
    labels = {c.label for c in d.composite_candidates}
    assert "donor + timepoint" in labels
    comp = next(c for c in d.composite_candidates if c.label == "donor + timepoint")
    assert comp.n_unique == 6  # 2 donors x 3 timepoints


def test_ineligible_columns_excluded_from_composites():
    d = profile_obs(_obs())
    for c in d.composite_candidates:
        assert "tissue" not in c.columns      # single-value excluded
        assert "cell_id" not in c.columns     # unique-per-cell excluded


def test_cap_respected():
    d = profile_obs(_obs(), max_composite_pairs=0)
    assert d.composite_candidates == []
