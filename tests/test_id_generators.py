"""Tests for ID generator functions."""

from summit_sim.schemas import generate_class_id, generate_scenario_id


def test_generate_scenario_id():
    """Test scenario ID generation format."""
    scenario_id = generate_scenario_id()
    assert scenario_id.startswith("scn-")
    assert len(scenario_id) == 12  # "scn-" + 8 hex chars
    assert scenario_id[4:].isalnum()  # The rest should be alphanumeric


def test_generate_class_id():
    """Test class ID generation format."""
    class_id = generate_class_id()
    assert len(class_id) == 6  # 6 hex chars
    assert class_id.isalnum()  # Should be alphanumeric
