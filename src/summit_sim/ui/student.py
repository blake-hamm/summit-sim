"""Student flow handlers for the Chainlit app."""

from typing import TYPE_CHECKING

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from summit_sim.graphs.student import (
    StudentState,
    create_student_graph,
)
from summit_sim.graphs.utils import scenario_store
from summit_sim.schemas import DebriefReport, ScenarioDraft

PASS_SCORE_THRESHOLD = 70

if TYPE_CHECKING:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig
else:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig


async def start_student_session() -> None:
    """Start student session by loading scenario from store."""
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
    await cl.Message(
        content=f"## 🏔️ {scenario.title}",
    ).send()

    await cl.Message(
        content=f"**Setting:** {scenario.setting}",
    ).send()

    await cl.Message(
        content=f"**Patient:** {scenario.patient_summary}",
    ).send()

    objectives_text = "\n".join(f"• {obj}" for obj in scenario.learning_objectives)
    await cl.Message(
        content=f"**Learning Objectives:**\n{objectives_text}",
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
    """Run the simulation graph with student interactions."""
    scenario = cl.user_session.get("scenario")
    scenario_id = cl.user_session.get("scenario_id") or ""
    class_id = cl.user_session.get("class_id")

    if scenario is None:
        await cl.Message(content="❌ Error: No scenario found.").send()
        return

    assert isinstance(scenario, ScenarioDraft)

    await cl.Message(content="⏳ Starting simulation...").send()

    graph = create_student_graph()
    cl.user_session.set("simulation_graph", graph)

    thread_id = cl.user_session.get("id")
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    initial_state = StudentState(
        scenario_draft=scenario.model_dump(),
        current_turn_id=scenario.starting_turn_id,
        transcript=[],
        is_complete=False,
        key_learning_moments=[],
        last_selected_choice=None,
        simulation_result=None,
        scenario_id=scenario_id,
        class_id=class_id,
    )

    try:
        result = await graph.ainvoke(initial_state, config)
        await handle_simulation_loop(result, graph, config)
    except Exception as e:
        await cl.Message(content=f"❌ Error during simulation: {e!s}").send()


async def handle_simulation_loop(
    state: StudentState | dict,
    graph: CompiledStateGraph,
    config: RunnableConfig,
) -> None:
    """Handle simulation interrupts and student choices."""
    if isinstance(state, dict):
        state = StudentState.from_graph_result(state)

    while True:
        if state.simulation_result:
            result_dict = state.simulation_result
            await cl.Message(
                content=f"💬 **{result_dict.get('feedback', '')}**",
            ).send()

            learning_moments = result_dict.get("learning_moments", [])
            for moment in learning_moments:
                await cl.Message(content=f"📚 {moment}").send()

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
            await cl.Message(content=f"**Conditions:** {scene_text}").send()

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

        result = await graph.ainvoke(
            Command(resume={"choice_id": choice_id}),
            config=config,
        )

        state = StudentState.from_graph_result(result)


async def show_debrief(state: StudentState) -> None:
    """Display the final debrief report."""
    if state.debrief_report is None:
        await cl.Message(content="❌ Error: No debrief available.").send()
        return

    debrief = DebriefReport.model_validate(state.debrief_report)

    score = debrief.final_score
    score_emoji = "✅" if score >= PASS_SCORE_THRESHOLD else "❌"

    await cl.Message(
        content=(
            f"## 🏁 Simulation Complete\n\n"
            f"**Score:** {score_emoji} **{score}%**\n"
            f"**Status:** {debrief.completion_status.upper()}"
        ),
    ).send()

    await cl.Message(content=f"**Summary:**\n{debrief.summary}").send()

    if debrief.key_mistakes:
        mistakes = "\n".join(f"⚠️ {m}" for m in debrief.key_mistakes)
        await cl.Message(content=f"**Key Mistakes:**\n{mistakes}").send()

    if debrief.strong_actions:
        strong = "\n".join(f"✅ {s}" for s in debrief.strong_actions)
        await cl.Message(content=f"**Strong Actions:**\n{strong}").send()

    if debrief.teaching_points:
        points = "\n".join(f"📖 {p}" for p in debrief.teaching_points)
        await cl.Message(content=f"**Teaching Points:**\n{points}").send()

    if debrief.best_next_actions:
        next_actions = "\n".join(f"➡️ {n}" for n in debrief.best_next_actions)
        await cl.Message(content=f"**Recommendations:**\n{next_actions}").send()
