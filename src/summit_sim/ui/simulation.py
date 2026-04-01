"""Simulation flow handlers for the Chainlit app."""

import base64
import logging
from typing import TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from summit_sim.agents.action_responder import ActionResponse
from summit_sim.graphs.simulation import (
    COMPLETION_THRESHOLD,
    SimulationState,
)
from summit_sim.graphs.utils import AppState
from summit_sim.schemas import DebriefReport, ScenarioDraft
from summit_sim.settings import settings
from summit_sim.ui.utils import format_scenario_details

logger = logging.getLogger(__name__)


MAX_ACTION_LENGTH = 500

if TYPE_CHECKING:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig
else:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig


async def start_simulation_session() -> None:
    """Start player session by loading scenario from store."""
    logger.info("Starting simulation session")
    scenario_id: str = cl.user_session.get("scenario_id") or ""

    if not scenario_id:
        await cl.Message(
            content="❌ No scenario ID found. Please check your link.",
        ).send()
        return

    store = AppState.store
    result = await store.aget(("scenarios",), scenario_id)

    if result is None:
        await cl.Message(
            content="❌ Scenario not found. Please check your link.",
        ).send()
        return

    scenario_data = result.value
    scenario = ScenarioDraft.model_validate(
        scenario_data.get("scenario_draft", scenario_data.get("scenario"))
    )
    cl.user_session.set("scenario", scenario)

    await show_scenario_intro(scenario)


async def show_scenario_intro(
    scenario: ScenarioDraft, enable_chat: bool = False
) -> None:
    """Display scenario intro with image below title, then details.

    Shows title + image first, then scenario details in a follow-up message.
    Optionally enables chat input for student mode.
    No placeholder - just shows nothing if no image.
    Fails fast if image_data is corrupt (base64 decode will error).
    """
    # Message 1: Title + Image (+ optional ChatEnabler)
    title_elements = []
    if scenario.image_data:
        image_bytes = base64.b64decode(scenario.image_data)
        title_elements.append(
            cl.Image(
                content=image_bytes,
                name="scenario_image",
                display="inline",
                size="large",
            )
        )

    if enable_chat:
        title_elements.append(
            cl.CustomElement(name="ChatEnabler", props={}, display="inline")
        )

    await cl.Message(content=f"## 🏔️ {scenario.title}", elements=title_elements).send()

    # Message 2: Scenario details
    details_content = format_scenario_details(scenario)
    await cl.Message(content=details_content).send()

    await run_simulation()


async def run_simulation() -> None:
    """Run the simulation graph with player interactions."""
    scenario = cl.user_session.get("scenario")
    scenario_id = cl.user_session.get("scenario_id") or ""

    if scenario is None:
        await cl.Message(content="❌ Error: No scenario found.").send()
        return

    assert isinstance(scenario, ScenarioDraft)

    graph = AppState.simulation_graph
    cl.user_session.set("simulation_graph", graph)

    thread_id = cl.user_session.get("id")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Retrieve authoring trace_id for correlation if this is student E2E flow
    authoring_trace_id = cl.user_session.get("authoring_trace_id")

    initial_state = SimulationState(
        scenario=scenario,
        transcript=[],
        turn_count=0,
        is_complete=False,
        action_response=None,
        scenario_id=scenario_id,
        hidden_state=scenario.hidden_state,
        current_trace_id=authoring_trace_id,
    )

    try:
        result = await graph.ainvoke(initial_state, config)
        await handle_simulation_loop(result, graph, config)
    except Exception as e:
        await cl.Message(content=f"❌ Error during simulation: {e!s}").send()


async def handle_simulation_loop(  # noqa: PLR0912
    state: SimulationState | dict,
    graph: CompiledStateGraph,
    config: RunnableConfig,
) -> None:
    """Handle simulation interrupts and player free-text actions."""
    if isinstance(state, dict):
        state = SimulationState.from_graph_result(state)

    while True:
        # Display feedback from previous action if available
        if state.action_response:
            result = ActionResponse.model_validate(state.action_response)

            await cl.Message(
                content=(
                    f"### Feedback\n\n"
                    f"{result.feedback}\n\n"
                    f"**Progress:** {result.completion_score:.0%} complete"
                ),
            ).send()

        if state.is_complete:
            await show_debrief(state)
            break

        # Get current narrative to display
        scenario = cl.user_session.get("scenario")
        if scenario is None:
            await cl.Message(content="❌ Error: Scenario lost.").send()
            break
        assert isinstance(scenario, ScenarioDraft)

        if state.turn_count == 0:
            # Initial turn - show the opening narrative
            current_narrative = scenario.initial_narrative
        else:
            # Subsequent turns - show narrative from last action result
            result = ActionResponse.model_validate(state.action_response)
            current_narrative = result.narrative_text

        # Scene conditions are shown in the intro; narrative is the prompt
        prompt_content = current_narrative

        # Get free-text action from student with character limit
        res = await cl.AskUserMessage(
            content=prompt_content,
            timeout=settings.ui_timeout,
        ).send()

        if not res or not res.get("output"):
            await cl.Message(content="⏱️ Simulation timed out.").send()
            break

        student_action = res.get("output", "").strip()

        # Validate action length
        if len(student_action) > MAX_ACTION_LENGTH:
            await cl.Message(
                content=(
                    f"⚠️ Action too long ({len(student_action)} chars). "
                    f"Please keep it under {MAX_ACTION_LENGTH} characters."
                ),
            ).send()
            continue

        if not student_action:
            await cl.Message(content="⚠️ Please enter an action.").send()
            continue

        loading_msg = await cl.Message(content="⏳ Evaluating your action...").send()

        # Resume graph with student action
        result = await graph.ainvoke(
            Command(resume={"action": student_action}),
            config=config,
        )

        await loading_msg.remove()

        state = SimulationState.from_graph_result(result)


async def show_debrief(state: SimulationState) -> None:
    """Display the final debrief report."""
    if state.debrief_report is None:
        await cl.Message(content="❌ Error: No debrief available.").send()
        return

    debrief = DebriefReport.model_validate(state.debrief_report)

    # Pull progressive completion_score from LangGraph state
    action_response = state.action_response or {}
    completion_score = action_response.get("completion_score", 0)
    score_percent = completion_score * 100
    score_emoji = "✅" if completion_score >= COMPLETION_THRESHOLD else "❌"

    content_parts = [
        f"## 🏁 Simulation Complete\n\n"
        f"**Score:** {score_emoji} **{score_percent:.0f}%**\n\n"
        f"**Summary:**\n{debrief.summary}",
        f"**Clinical Reasoning Analysis:**\n{debrief.clinical_reasoning}",
    ]

    if debrief.key_mistakes:
        mistakes = "\n".join(f"⚠️ {m}" for m in debrief.key_mistakes)
        content_parts.append(f"**Key Mistakes:**\n{mistakes}")

    if debrief.strong_actions:
        strong = "\n".join(f"✅ {s}" for s in debrief.strong_actions)
        content_parts.append(f"**Strong Actions:**\n{strong}")

    if debrief.teaching_points:
        points = "\n".join(f"📖 {p}" for p in debrief.teaching_points)
        content_parts.append(f"**Teaching Points:**\n{points}")

    if debrief.best_next_actions:
        next_actions = "\n".join(f"➡️ {n}" for n in debrief.best_next_actions)
        content_parts.append(f"**Recommendations:**\n{next_actions}")

    await cl.Message(content="\n\n".join(content_parts)).send()
