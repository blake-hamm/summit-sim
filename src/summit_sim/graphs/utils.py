"""Shared utilities for LangGraph state definitions."""

from dataclasses import dataclass

from langgraph.store.memory import InMemoryStore


@dataclass
class TranscriptEntry:
    """Single entry in simulation transcript with full context.

    Captures complete information about a turn for debrief analysis.
    """

    turn_id: int
    turn_narrative: str
    choice_id: str
    choice_description: str
    was_correct: bool
    feedback: str
    learning_moments: list[str]
    next_turn_id: int | None


scenario_store = InMemoryStore()
