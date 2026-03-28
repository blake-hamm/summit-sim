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
    student_action: str
    was_correct: bool
    feedback: str
    learning_moments: list[str]


scenario_store = InMemoryStore()
