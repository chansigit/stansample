import pandas as pd
from stanmetacols.profile import profile_obs
from stanmetacols.prompts import SYSTEM_PROMPT, ADJUDICATION_SYSTEM_PROMPT, build_user_prompt, build_adjudication_prompt


def test_prompts_mention_roles_and_json():
    assert "json" in SYSTEM_PROMPT.lower()
    for token in ("sample", "pct_mt", "n_counts", "n_genes", "doublet"):
        assert token in SYSTEM_PROMPT
    assert "canonical" in ADJUDICATION_SYSTEM_PROMPT.lower()


def _d():
    return profile_obs(pd.DataFrame({"sample": ["A", "B"]}))


def test_user_prompt_includes_hint_block():
    p = build_user_prompt(_d(), ["sample"], hint="mito col is mt.frac")
    assert "User guidance" in p
    assert "mito col is mt.frac" in p


def test_user_prompt_omits_block_when_hint_empty():
    p = build_user_prompt(_d(), ["sample"])
    assert "User guidance" not in p


def test_adjudication_prompt_includes_hint():
    p = build_adjudication_prompt(_d(), {}, hint="counts are in total_umis")
    assert "User guidance" in p and "total_umis" in p


def test_prompts_discriminate_organ_and_tissue():
    sp = SYSTEM_PROMPT.lower()
    # phrases that exist ONLY in the discrimination paragraph, not the roles block
    assert "for the organ and tissue roles" in sp
    assert "they are distinct" in sp
    assert "organ, tissue, both, or neither" in sp
    # ordering: the paragraph sits between the cell-type guidance and the JSON instruction
    ct_end = sp.index("both, one, or neither.")
    disc = sp.index("they are distinct")
    json_start = sp.index("return json only")
    assert ct_end < disc < json_start
