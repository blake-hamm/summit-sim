"""Simulation flow handlers for the Chainlit app."""

import logging
from typing import TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from summit_sim.graphs.simulation import (
    SimulationState,
    create_simulation_graph,
)
from summit_sim.graphs.utils import scenario_store
from summit_sim.schemas import DebriefReport, ScenarioDraft

logger = logging.getLogger(__name__)

PASS_SCORE_THRESHOLD = 70

if TYPE_CHECKING:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig
else:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig


async def start_simulation_session() -> None:
    """Start player session by loading scenario from store."""
    logger.info("Starting simulation session")
    scenario_id = cl.user_session.get("scenario_id")

    result = scenario_store.get(("scenarios",), scenario_id)

    if result is None:
        await cl.Message(
            content="❌ Scenario not found. Please check your link.",
        ).send()
        return

    scenario_data = result.value
    scenario = ScenarioDraft.model_validate(scenario_data["scenario_draft"])
    cl.user_session.set("scenario", scenario)

    await show_scenario_intro(scenario)


async def show_scenario_intro(scenario: ScenarioDraft) -> None:
    """Display scenario intro with start button."""
    objectives_text = "\n".join(f"• {obj}" for obj in scenario.learning_objectives)
    await cl.Message(
        content=(
            f"## 🏔️ {scenario.title}\n\n"
            f"**Setting:** {scenario.setting}\n\n"
            f"**Patient:** {scenario.patient_summary}\n\n"
            f"**Learning Objectives:**\n{objectives_text}"
        ),
    ).send()

    res = await cl.AskActionMessage(
        content="Ready to begin the simulation?",
        actions=[
            cl.Action(
                name="start_simulation",
                payload={"value": "start"},
                label="▶️ Start Scenario",
            ),
        ],
    ).send()

    if res and res.get("payload", {}).get("value") == "start":
        await run_simulation()


async def run_simulation() -> None:
    """Run the simulation graph with player interactions."""
    scenario = cl.user_session.get("scenario")
    scenario_id = cl.user_session.get("scenario_id") or ""

    if scenario is None:
        await cl.Message(content="❌ Error: No scenario found.").send()
        return

    assert isinstance(scenario, ScenarioDraft)

    graph = create_simulation_graph()
    cl.user_session.set("simulation_graph", graph)

    thread_id = cl.user_session.get("id")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    starting_turn = scenario.get_starting_turn()
    if starting_turn is None:
        await cl.Message(content="❌ Invalid scenario: no starting turn found.").send()
        return

    initial_state = SimulationState(
        scenario_draft=scenario.model_dump(),
        current_turn_id=starting_turn.turn_id,
        transcript=[],
        is_complete=False,
        key_learning_moments=[],
        last_selected_choice=None,
        simulation_result=None,
        scenario_id=scenario_id,
    )

    try:
        result = await graph.ainvoke(initial_state, config)
        await handle_simulation_loop(result, graph, config)
    except Exception as e:
        await cl.Message(content=f"❌ Error during simulation: {e!s}").send()


async def handle_simulation_loop(
    state: SimulationState | dict,
    graph: CompiledStateGraph,
    config: RunnableConfig,
) -> None:
    """Handle simulation interrupts and player choices."""
    if isinstance(state, dict):
        state = SimulationState.from_graph_result(state)

    while True:
        if state.simulation_result:
            result_dict = state.simulation_result
            feedback = result_dict.get("feedback", "")
            learning_moments = result_dict.get("learning_moments", [])
            moments_text = "\n".join(f"• {m}" for m in learning_moments) or "None"

            await cl.Message(
                content=(
                    f"### Feedback\n\n"
                    f"{feedback}\n\n"
                    f"### Learning Moments\n\n"
                    f"{moments_text}"
                ),
            ).send()

        if state.is_complete:
            await show_debrief(state)
            break

        scenario = cl.user_session.get("scenario")
        if scenario is None:
            await cl.Message(content="❌ Error: Scenario lost.").send()
            break
        assert isinstance(scenario, ScenarioDraft)

        current_turn = scenario.get_turn(state.current_turn_id)

        if current_turn is None:
            await cl.Message(content="❌ Error: Turn not found.").send()
            break

        if scene_state := current_turn.scene_state:
            scene_text = ", ".join(f"**{k}:** {v}" for k, v in scene_state.items())
            await cl.Message(
                content=(
                    f"**Conditions:** {scene_text}\n\n{current_turn.narrative_text}"
                ),
            ).send()
        else:
            await cl.Message(content=current_turn.narrative_text).send()

        actions = [
            cl.Action(
                name=choice.choice_id,
                payload={"choice_id": choice.choice_id},
                label=choice.description,
            )
            for choice in current_turn.choices
        ]

        res = await cl.AskActionMessage(
            content="What would you do?",
            actions=actions,
        ).send()

        if not res or not res.get("payload"):
            break

        choice_id = res.get("payload", {}).get("choice_id")

        loading_msg = await cl.Message(content="⏳ Processing choice...").send()

        result = await graph.ainvoke(
            Command(resume={"choice_id": choice_id}),
            config=config,
        )

        loading_msg.content = "✅ Choice recorded"
        await loading_msg.update()

        state = SimulationState.from_graph_result(result)


async def show_debrief(state: SimulationState) -> None:
    """Display the final debrief report."""
    if state.debrief_report is None:
        await cl.Message(content="❌ Error: No debrief available.").send()
        return

    debrief = DebriefReport.model_validate(state.debrief_report)

    score = debrief.final_score
    score_emoji = "✅" if score >= PASS_SCORE_THRESHOLD else "❌"

    content_parts = [
        f"## 🏁 Simulation Complete\n\n"
        f"**Score:** {score_emoji} **{score}%**\n"
        f"**Status:** {debrief.completion_status.upper()}\n\n"
        f"**Summary:**\n{debrief.summary}",
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
