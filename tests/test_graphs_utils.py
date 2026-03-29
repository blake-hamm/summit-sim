"""Tests for graph utilities and AppState."""

from summit_sim.graphs.utils import AppState


class TestAppState:
    """Tests for AppState singleton container."""

    def test_app_state_default_values(self):
        """Test that AppState attributes default to None."""
        # Reset to ensure clean state
        AppState.store = None
        AppState.checkpointer = None
        AppState.author_graph = None
        AppState.simulation_graph = None

        assert AppState.store is None
        assert AppState.checkpointer is None
        assert AppState.author_graph is None
        assert AppState.simulation_graph is None

    def test_app_state_can_store_mock_values(self):
        """Test that AppState can store mock dependencies."""
        mock_store = {"test": "store"}
        mock_checkpointer = {"test": "checkpointer"}
        mock_author_graph = {"test": "author_graph"}
        mock_simulation_graph = {"test": "simulation_graph"}

        AppState.store = mock_store
        AppState.checkpointer = mock_checkpointer
        AppState.author_graph = mock_author_graph
        AppState.simulation_graph = mock_simulation_graph

        assert AppState.store == mock_store
        assert AppState.checkpointer == mock_checkpointer
        assert AppState.author_graph == mock_author_graph
        assert AppState.simulation_graph == mock_simulation_graph

        # Reset after test
        AppState.store = None
        AppState.checkpointer = None
        AppState.author_graph = None
        AppState.simulation_graph = None

    def test_app_state_is_class_level(self):
        """Test that AppState is shared at class level."""
        # Set a value
        AppState.store = "shared_value"

        # Access via class
        assert AppState.store == "shared_value"

        # Reset
        AppState.store = None
