import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.heuristic import rank_heuristic
from stanmetacols.roles import ROLE_KEYS

_FINE_TYPES = ["CD4 T cell", "CD8 T cell", "B cell", "NK cell", "Plasma cell",
               "Macrophage", "Monocyte", "Dendritic cell", "Mast cell",
               "Fibroblast", "Endothelial cell", "Epithelial cell"]


def _celltype_digest():
    n = 60
    coarse = (["Epithelial"] * 20 + ["Immune"] * 20 + ["Stromal"] * 20)
    fine = [_FINE_TYPES[i % len(_FINE_TYPES)] for i in range(n)]   # 12 distinct
    return profile_obs(pd.DataFrame({
        "cell_type": coarse,                 # 3 broad lineages, low cardinality
        "subtype": fine,                     # 12 specific cell-type names
        "tissue": ["lung"] * 30 + ["liver"] * 30,   # NOT a cell type
    }, index=[f"c{i}" for i in range(n)]))


def _digest():
    n = 60
    return profile_obs(pd.DataFrame({
        "sample": ["S1"] * 30 + ["S2"] * 30,
        "pct_counts_mt": [i / 100 for i in range(n)],          # [0,1) floats
        "pct_counts_hb": [i / 1000 for i in range(n)],
        "total_counts": [1000 + 10 * i for i in range(n)],     # large ints
        "n_genes_by_counts": [200 + i for i in range(n)],      # mid ints
        "doublet_score": [i / 100 for i in range(n)],
    }, index=[f"c{i}" for i in range(n)]))


def test_each_role_top_is_correct_column():
    out = rank_heuristic(_digest(), list(ROLE_KEYS))
    assert out["sample"][0].column == "sample"
    assert out["pct_mt"][0].column == "pct_counts_mt"
    assert out["pct_hb"][0].column == "pct_counts_hb"
    assert out["n_counts"][0].column == "total_counts"
    assert out["n_genes"][0].column == "n_genes_by_counts"
    assert out["doublet_score"][0].column == "doublet_score"


def test_numeric_role_requires_name_hit():
    # a bare [0,1] float column with no pct/doublet name must NOT appear for pct_mt
    d = profile_obs(pd.DataFrame({"score_x": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    assert rank_heuristic(d, ["pct_mt"])["pct_mt"] == []


def test_value_guard_rejects_unit_column_for_counts():
    # a [0,1] column literally named like counts should not win n_counts on value
    d = profile_obs(pd.DataFrame({"pct_counts": [i / 100 for i in range(50)]},
                                 index=[f"c{i}" for i in range(50)]))
    cands = rank_heuristic(d, ["n_counts"])["n_counts"]
    assert all(c.score < 0.7 for c in cands)    # name may hit, value does not


def test_celltype_roles_pick_right_columns():
    out = rank_heuristic(_celltype_digest(), ["cell_type_coarse", "cell_type_fine"])
    assert out["cell_type_coarse"][0].column == "cell_type"
    assert out["cell_type_fine"][0].column == "subtype"


def test_non_celltype_column_not_surfaced():
    out = rank_heuristic(_celltype_digest(), ["cell_type_coarse"])
    cols = [c.column for c in out["cell_type_coarse"]]
    assert "tissue" not in cols       # tissue values aren't cell-type names


def _organ_tissue_digest():
    n = 60
    return profile_obs(pd.DataFrame({
        "organ": ["heart"] * 20 + ["liver"] * 20 + ["lung"] * 20,
        "tissue": ["blood"] * 20 + ["PBMC"] * 20 + ["bone marrow"] * 20,
        "lineage_label": ["heart"] * 20 + ["liver"] * 20 + ["lung"] * 20,  # organ values, neutral name
    }, index=[f"c{i}" for i in range(n)]))


def test_organ_tissue_top_columns_and_no_crosstalk():
    out = rank_heuristic(_organ_tissue_digest(), ["organ", "tissue"])
    assert out["organ"][0].column == "organ"
    assert out["tissue"][0].column == "tissue"
    organ_cols = [c.column for c in out["organ"]]
    tissue_cols = [c.column for c in out["tissue"]]
    # disjoint vocabularies: neither column bleeds into the other role
    assert "tissue" not in organ_cols      # blood/PBMC/marrow are not organs
    assert "organ" not in tissue_cols      # heart/liver/lung are not tissues


def test_vocab_routing_by_value_not_just_name():
    # a neutrally-named column holding organ values routes to organ, not tissue
    out = rank_heuristic(_organ_tissue_digest(), ["organ", "tissue"])
    assert "lineage_label" in [c.column for c in out["organ"]]
    assert "lineage_label" not in [c.column for c in out["tissue"]]


def test_vocab_skips_single_value_column():
    # a single-valued tissue-named column must NOT surface (mirrors CLI exit-2 case)
    d = profile_obs(pd.DataFrame({"tissue": ["lung"] * 8},
                                 index=[f"c{i}" for i in range(8)]))
    assert rank_heuristic(d, ["organ", "tissue"])["tissue"] == []
    assert rank_heuristic(d, ["organ", "tissue"])["organ"] == []
