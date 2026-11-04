from markersub.naming import is_static, parse_leg_plates


def test_leg_tokens():
    assert parse_leg_plates("S01_cut01_L3.c3d") == {"left": 3}
    assert parse_leg_plates("drop jump L1_R2.c3d") == {"left": 1, "right": 2}
    assert parse_leg_plates("S01-R12.c3d") == {"right": 12}
    assert parse_leg_plates("S01_walk.c3d") == {}
    assert parse_leg_plates("LR3_test.c3d") == {}           # not a separate token
    assert parse_leg_plates("R01_cut01_L3.c3d") == {"left": 3}   # subject code is not a leg token
    assert parse_leg_plates("S01_L3_walk.c3d") == {}        # leg token must be at the end


def test_invalid_leg_tokens():
    import pytest
    with pytest.raises(ValueError):
        parse_leg_plates("S01_cut01_L3_R3.c3d")             # both legs on one plate
    with pytest.raises(ValueError):
        parse_leg_plates("S01_cut01_L3_L2.c3d")             # leg named twice


def test_static_token():
    assert is_static("S01_Static_1.c3d", "static")
    assert not is_static("S01_cut01_L3.c3d", "static")
