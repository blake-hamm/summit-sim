"""Teacher flow handlers for the Chainlit app."""

from typing import TYPE_CHECKING, Awaitable, Callable

from langgraph.types import Command

from summit_sim.graphs.teacher import (
    TeacherState,
    create_teacher_graph,
)
from summit_sim.schemas import ScenarioDraft, TeacherConfig
from summit_sim.settings import settings

if TYPE_CHECKING:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig
else:
    import chainlit as cl

RestartFunc = Callable[[], Awaitable[None]]
_restart_func: RestartFunc | None = None


def set_restart_func(func: RestartFunc) -> None:
    """Set the restart function to be called on errors."""
    global _restart_func  # noqa: PLW0603
    _restart_func = func


async def _restart() -> None:
    """Restart the session."""
    if _restart_func is not None:
        await _restart_func()


async def ask_num_participants() -> None:
    """Ask for number of participants."""
    res = await cl.AskActionMessage(
        content=(
            "**Step 1/3: Number of Participants**\n\n"
            "How many people are in the rescue group?"
        ),
        actions=[
            cl.Action(name="p1", payload={"value": "1"}, label="1"),
            cl.Action(name="p2", payload={"value": "2"}, label="2"),
            cl.Action(name="p3", payload={"value": "3"}, label="3"),
            cl.Action(name="p4", payload={"value": "4"}, label="4"),
            cl.Action(name="p5", payload={"value": "5"}, label="5"),
            cl.Action(name="p6", payload={"value": "6"}, label="6+"),
        ],
    ).send()

    if res and res.get("payload"):
        value = res.get("payload", {}).get("value", "3")
        if value == "6+":
            value = "6"
        cl.user_session.set("num_participants", int(value))
        await ask_activity_type()


async def ask_activity_type() -> None:
    """Ask for activity type."""
    res = await cl.AskActionMessage(
        content="**Step 2/3: Activity Type**\n\nWhat activity is the group engaged in?",
        actions=[
            cl.Action(name="hiking", payload={"value": "hiking"}, label="Hiking"),
            cl.Action(name="skiing", payload={"value": "skiing"}, label="Skiing"),
            cl.Action(
                name="canyoneering",
                payload={"value": "canyoneering"},
                label="Canyoneering",
            ),
        ],
    ).send()

    if res and res.get("payload"):
        value = res.get("payload", {}).get("value", "hiking")
        cl.user_session.set("activity_type", value)
        await ask_difficulty()


async def ask_difficulty() -> None:
    """Ask for difficulty level."""
    res = await cl.AskActionMessage(
        content=(
            "**Step 3/3: Difficulty Level**\n\nHow challenging should this scenario be?"
        ),
        actions=[
            cl.Action(
                name="low", payload={"value": "low"}, label="Low - Basic first aid"
            ),
            cl.Action(name="med", payload={"value": "med"}, label="Medium - WFA level"),
            cl.Action(name="high", payload={"value": "high"}, label="High - WFR level"),
        ],
    ).send()

    if res and res.get("payload"):
        value = res.get("payload", {}).get("value", "med")
        cl.user_session.set("difficulty", value)
        await generate_scenario()


async def generate_scenario() -> None:
    """Generate scenario with collected config."""
    num_participants_val = cl.user_session.get("num_participants")
    activity_type_val = cl.user_session.get("activity_type")
    difficulty_val = cl.user_session.get("difficulty")

    num_participants = (
        int(num_participants_val) if num_participants_val is not None else 3
    )
    activity_type = (
        str(activity_type_val) if activity_type_val is not None else "hiking"
    )
    difficulty = str(difficulty_val) if difficulty_val is not None else "med"

    config = TeacherConfig(
        num_participants=num_participants,
        activity_type=activity_type,  # type: ignore[arg-type]
        difficulty=difficulty,  # type: ignore[arg-type]
    )

    cl.user_session.set("teacher_config", config)

    await cl.Message(content="⏳ Generating your scenario...").send()

    graph = create_teacher_graph()
    cl.user_session.set("graph", graph)

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    initial_state: TeacherState = TeacherState(
        teacher_config=config.model_dump(),
        scenario_draft=None,
        scenario_id="",
        class_id="",
        retry_count=0,
        feedback_history=[],
        approval_status=None,
    )

    try:
        result = await graph.ainvoke(
            initial_state,
            config=config_dict,
        )

        if result.get("scenario_draft"):
            state = TeacherState.from_graph_result(result)
            await show_review_screen(state)
        else:
            await cl.Message(
                content="❌ Error: Scenario generation failed. Please try again.",
            ).send()
            await _restart()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error during generation: {e!s}",
        ).send()
        await _restart()


async def show_review_screen(state: TeacherState) -> None:
    """Display the scenario review screen with approve button."""
    scenario_dict = state.scenario_draft
    scenario_id = state.scenario_id

    if scenario_dict is None:
        await cl.Message(
            content="❌ Error: No scenario to review.",
        ).send()
        return

    scenario = ScenarioDraft.model_validate(scenario_dict)
    await cl.Message(
        content=f"## 📋 Scenario Review\n\n**ID:** `{scenario_id}`",
    ).send()

    await cl.Message(
        content=f"**{scenario.title}**\n\n{scenario.setting}",
    ).send()

    await cl.Message(
        content=f"**Patient:** {scenario.patient_summary}",
    ).send()

    learning_obj_text = "\n".join(f"• {obj}" for obj in scenario.learning_objectives)
    await cl.Message(
        content=f"**Learning Objectives:**\n{learning_obj_text}",
    ).send()

    await cl.Message(
        content=f"**Total Turns:** {len(scenario.turns)}",
    ).send()

    res = await cl.AskActionMessage(
        content=(
            "Review the scenario above. "
            "Click **Approve** when ready to share with students."
        ),
        actions=[
            cl.Action(
                name="approve",
                payload={"value": "approve"},
                label="✅ Approve & Generate Link",
            ),
        ],
    ).send()

    if res and res.get("payload", {}).get("value") == "approve":
        await handle_approval(state)


async def handle_approval(state: TeacherState) -> None:
    """Handle scenario approval and generate shareable link."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        await _restart()
        return

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(
            Command(resume={"decision": "approve"}),
            config=config_dict,
        )

        final_state = TeacherState.from_graph_result(result)
        approval_status = final_state.approval_status or ""

        if approval_status == "approved":
            scenario_id = final_state.scenario_id or ""
            class_id = final_state.class_id or ""
            shareable_url = f"{settings.base_url}?scenario_id={scenario_id}"

            await cl.Message(
                content=(
                    f"## ✅ Scenario Approved!\n\n"
                    f"**Scenario ID:** `{scenario_id}`\n"
                    f"**Class ID:** `{class_id}`\n\n"
                    f"**Shareable URL:**\n{shareable_url}"
                ),
            ).send()

            await cl.Message(
                content=(
                    "Students can join by visiting the URL above. "
                    "The simulation is ready to run!"
                ),
            ).send()
        else:
            await cl.Message(
                content="❌ Error: Approval failed. Please try again.",
            ).send()
            await show_review_screen(state)

    except Exception as e:
        await cl.Message(
            content=f"❌ Error during approval: {e!s}",
        ).send()
        await show_review_screen(state)
