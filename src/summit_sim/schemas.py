"""Pydantic schemas for Summit-Sim data models."""

import uuid
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


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


def generate_scenario_id() -> str:
    """Generate unique scenario identifier."""
    return f"scn-{uuid.uuid4().hex[:8]}"


class ScenarioConfig(BaseModel):
    """Configuration provided by the author to generate a targeted WFR scenario.

    Replaces generic inputs with dimensions that directly impact WFR decision-making:
    patient assessment (Medical/Trauma/Environmental), resource management (group size),
    and evacuation logistics (distance/environment).
    """

    primary_focus: Literal["Trauma", "Medical", "Environmental", "Mixed"] = Field(
        ...,
        description="The core WFR syllabus category to test.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Curriculum Focus",
                "options": ["Trauma", "Medical", "Environmental", "Mixed"],
                "value": "Trauma",
            }
        },
    )
    environment: Literal[
        "Alpine/Mountain", "Desert", "Forest/Trail", "Water/River", "Winter/Snow"
    ] = Field(
        ...,
        description="The physical setting, driving MOI and weather risks.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Environment",
                "options": [
                    "Alpine/Mountain",
                    "Desert",
                    "Forest/Trail",
                    "Water/River",
                    "Winter/Snow",
                ],
                "value": "Alpine/Mountain",
            }
        },
    )
    available_personnel: Literal[
        "Solo Rescuer (1)", "Partner (2)", "Small Group (3-5)", "Large Expedition (6+)"
    ] = Field(
        ...,
        description="Total conscious people. Dictates litter carries or runners.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Group Size / Resources",
                "options": [
                    "Solo Rescuer (1)",
                    "Partner (2)",
                    "Small Group (3-5)",
                    "Large Expedition (6+)",
                ],
                "value": "Small Group (3-5)",
            }
        },
    )
    evac_distance: Literal[
        "Short (< 2 hours)", "Remote (1 day)", "Expedition (2+ days)"
    ] = Field(
        ...,
        description="Distance to definitive care. Key for stay vs go decisions.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Evacuation Distance",
                "options": [
                    "Short (< 2 hours)",
                    "Remote (1 day)",
                    "Expedition (2+ days)",
                ],
                "value": "Remote (1 day)",
            }
        },
    )
    complexity: Literal["Standard", "Complicated", "Critical"] = Field(
        ...,
        description="Complicated=underlying condition; Critical=deteriorating vitals.",
        json_schema_extra={
            "ui": {
                "type": "select",
                "label": "Patient Complexity",
                "options": ["Standard", "Complicated", "Critical"],
                "value": "Standard",
            }
        },
    )


class DynamicTurnResult(BaseModel):
    """Result from ActionResponder agent after evaluating student action.

    Single schema enforces evaluation → narrative → state evolution order.
    Generated dynamically for each student action in free-text simulation.
    """

    was_correct: bool = Field(
        ..., description="Whether the student's action was medically correct"
    )
    completion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Progress toward scenario completion (0.0-1.0 scale)",
    )

    feedback: str = Field(
        ..., description="AI-generated personalized feedback on the action"
    )
    narrative_text: str = Field(
        ...,
        description=(
            "Immersive narrative describing what the student discovers "
            "based on their action. Progressively reveal hidden information "
            "from hidden_truth/hidden_state as student performs assessments. "
            "3-5 sentences, end with open question inviting next action.\n\n"
            "EXAMPLE 1 - Vitals check reveals findings:\n"
            "Student: 'I check pulse and breathing.'\n"
            "AI: 'You check wrist pulse - rapid at 110 bpm. Breathing is "
            "quick and shallow at 24/min. Skin feels cool and clammy. "
            "What do you check next?'\n\n"
            "EXAMPLE 2 - Physical exam reveals injuries:\n"
            "Student: 'I do a head-to-toe exam.'\n"
            "AI: 'Head shows no trauma. Chest rises evenly. Abdomen soft. "
            "Right ankle has deformity, swelling, and bruising. "
            "Foot is cold. How do you proceed?'\n\n"
            "EXAMPLE 3 - SAMPLE history reveals info:\n"
            "Student: 'I ask about allergies and history.'\n"
            "AI: 'Patient reports penicillin allergy and carries EpiPen "
            "for bee stings. Takes asthma meds. Last ate 4 hours ago. "
            "Does this change your priorities?'\n\n"
            "EXAMPLE 4 - Scene assessment reveals changes:\n"
            "Student: 'I scan for dangers.'\n"
            "AI: 'Dark clouds approach from west - storm in 30 min. "
            "Temperature dropping. 2 hours of daylight left. Limited "
            "shelter on this ridge. How does this affect your plan?'\n\n"
            "EXAMPLE 5 - Progressive revelation over time:\n"
            "Turn 1: 'You check vitals - HR 110, RR 24. Skin pale.'\n"
            "Turn 2: 'You find burns on right calf. Neck tender at C4.'\n"
            "Turn 3: 'Patient has no recall, allergies, or meds.'\n"
            "Each narrative adds new discoveries without repeating facts."
        ),
    )


class ScenarioDraft(BaseModel):
    """Complete AI-generated wilderness rescue scenario.

    Generated from ScenarioConfig for dynamic open-ended simulation.
    Contains only initial setup - turns are generated dynamically
    based on student free-text actions.
    """

    title: str = Field(
        ...,
        description=(
            "A punchy, unique title for the scenario. "
            "Examples: 'Lightning Strike on the Grand Teton', "
            "'Anaphylaxis on the Pine Ridge Trail', 'HAPE at 14,000 Feet'"
        ),
    )
    setting: str = Field(
        ...,
        description=(
            "Specific location, weather, and time of day. Keep under 20 words. "
            "Example: 'Exposed rocky ridge at 13,000 ft. Incoming thunderstorm, "
            "dropping temperatures.'"
        ),
    )
    patient_summary: str = Field(
        ...,
        description=(
            "Age, sex, chief complaint, and visible mechanism of injury (MOI). "
            "Example: '28-year-old female, thrown 10 feet by indirect lightning "
            "strike. Conscious but confused.'"
        ),
    )
    hidden_truth: str = Field(
        ...,
        description=(
            "The actual medical diagnosis students must discover. Example: 'Patient "
            "has a minor burn on the right leg, but the critical hidden issue is a "
            "suspected cervical spine injury from the throw and developing "
            "hypothermia.'"
        ),
    )
    learning_objectives: list[str] = Field(
        ...,
        min_length=2,
        max_length=3,
        description=(
            "2-3 specific WFR skills tested from the curriculum (e.g., 'Spinal "
            "clearance protocol', 'Lightning strike safety/evacuation', "
            "'Hypothermia prevention'). Select from WFR learning objectives catalog."
        ),
    )
    initial_narrative: str = Field(
        ...,
        max_length=400,
        description=(
            "Immersive opening scene ending with a call to action. STRICTLY 2-4 "
            "sentences. DO NOT EXCEED 4 SENTENCES. Present tense, second person "
            "('You are...'). Must end with a question inviting action. Example: 'You "
            "are descending the Grand Teton when a loud crack echoes, and you see "
            "your climbing partner thrown against the rocks by an indirect lightning "
            "strike. The sky is dark, and the wind is picking up. She is groaning on "
            "the ground. What is your first move?'"
        ),
    )
    hidden_state: str = Field(
        ...,
        description=(
            "Comprehensive baseline medical data for AI reference. Must include "
            "complete initial vitals (HR, RR, BP, SCTM, AVPU), SAMPLE history, and "
            "hidden injuries. Write as a clinical summary. Example: 'Patient is "
            "A&O x2. HR 110, RR 24, BP 130/80. SCTM: Pale, cool, clammy. "
            "Superficial fern-like burn on right calf. Tenderness upon palpation "
            "of C4 vertebrae. No other major trauma. Cannot recall the incident.'"
        ),
    )
    scene_state: str = Field(
        ...,
        description=(
            "Environmental context, available gear, group dynamics, and exact "
            "distance/time to definitive care. Keep concise. Example: 'Group of 2. "
            "1 rope, standard rack, basic WFR first aid kit. 6 hours from the "
            "trailhead. Immediate danger of secondary lightning strikes.'"
        ),
    )


class RollupResult(BaseModel):
    """Final weighted score for optimization across judge criteria.

    Aggregates results from trace-level and session-level judges
    to produce an overall quality score for prompt optimization.
    """

    session_id: str = Field(..., description="Unique session identifier")
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted score from all criteria (0.0-1.0)"
    )
    trace_contribution: float = Field(
        ..., ge=0.0, le=0.85, description="Score from trace-level criteria only"
    )
    session_contribution: float = Field(
        ..., ge=0.0, le=0.20, description="Score from session-level criteria only"
    )
    breakdown: dict[str, bool] = Field(
        ..., description="Criterion name -> passed status mapping"
    )
    total_criteria: int = Field(..., description="Total number of criteria evaluated")
    passed_criteria: int = Field(..., description="Number of criteria that passed")


class DebriefReport(BaseModel):
    """Structured debrief report analyzing student simulation performance.

    Generated after simulation completion to provide comprehensive qualitative feedback
    on student decision-making and clinical reasoning against the hidden medical truth.
    """

    summary: str = Field(
        ...,
        description=(
            "Executive summary evaluating the student's overall clinical approach. "
            "Analyze how well they synthesized information from their typed actions "
            "against the hidden medical truth. 2-3 sentences."
        ),
    )
    clinical_reasoning: str = Field(
        ...,
        description=(
            "A 2-3 sentence paragraph analyzing how well the student synthesized "
            "clues, followed the PAS protocol, and demonstrated clinical reasoning. "
            "Focus on specific moments in their typed actions where they showed good "
            "judgment or missed critical cues."
        ),
    )
    key_mistakes: list[str] = Field(
        ...,
        description=(
            "Critical reasoning failures or missed opportunities identified in the "
            "student's typed actions. Focus on clinical judgment errors, premature "
            "interventions, or skipped assessment steps. Reference specific moments."
        ),
    )
    strong_actions: list[str] = Field(
        ...,
        description=(
            "Specific moments where the student's typed text demonstrated excellent "
            "clinical judgment, proper PAS protocol adherence, or sound reasoning. "
            "Highlight what they did right and why it mattered."
        ),
    )
    teaching_points: list[str] = Field(
        ...,
        description=(
            "Key WFR concepts to reinforce based on gaps observed in their "
            "performance. Connect to Scene Safety, PAS Assessment, Treatment "
            "decisions, and Evacuation planning dimensions."
        ),
    )
    best_next_actions: list[str] = Field(
        ...,
        description=(
            "Guidance on what competencies the student should develop next. "
            "Suggest specific skills, scenarios to practice, or knowledge areas "
            "to study to improve their wilderness first response capabilities."
        ),
    )
